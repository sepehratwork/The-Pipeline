import os
import gc
import json
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk, Features, Value
import glob
import io
import multiprocessing as mp
import shutil
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
import zstandard as zstd


def check_and_load_cache(processed_path, current_config):
    """Checks if a processed dataset exists and matches the current configuration."""
    config_path = os.path.join(processed_path, "config.json")
    if os.path.exists(processed_path) and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                saved_config = json.load(f)
            if saved_config == current_config:
                print(f"📦 [Cache Hit] Loading processed dataset from: {processed_path}")
                ds = load_from_disk(processed_path)
                print(f"✓ Dataset loaded successfully ({len(ds):,} samples).")
                return ds
            else:
                print("🔄 [Cache Miss] Dataset configuration changed. Reprocessing dataset...")
        except Exception as e:
            print(f"⚠️ Failed to load cached dataset ({e}). Reprocessing from scratch...")
    else:
        print(f"ℹ️  [Cache Miss] No existing cache found at: {processed_path}. Processing new dataset...")
    return None


def save_cache(dataset, processed_path, current_config):
    """Saves the processed dataset and its configuration to disk."""
    print(f"💾 Saving processed dataset ({len(dataset):,} samples) to: {processed_path}...")
    dataset.save_to_disk(processed_path)
    with open(os.path.join(processed_path, "config.json"), "w") as f:
        json.dump(current_config, f, indent=2)
    print(f"✓ Successfully cached dataset to {processed_path}.")


def format_dpo_dataset(example):
    if isinstance(example["chosen"], list):
        if len(example["chosen"]) > 1:
            example["prompt"] = example["chosen"][:-1]
            example["chosen"] = example["chosen"][-1:]
        elif isinstance(example["prompt"], str):
            example["prompt"] = [{"role": "user", "content": example["prompt"]}]
        if isinstance(example["rejected"], list) and len(example["rejected"]) > 1:
            example["rejected"] = example["rejected"][-1:]
    return example


# ---------------------------------------------------------
# 1. STREAMING FILE PARSER (Zero-RAM decompression)
# ---------------------------------------------------------
def _stream_zst_records(file_path):
    """Streams JSON objects line-by-line from a .zst file without loading it all to RAM."""
    dctx = zstd.ZstdDecompressor()
    with open(file_path, "rb") as f:
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        if "text" in record and record["text"]:
                            yield record["text"]
                    except Exception:
                        continue


# ---------------------------------------------------------
# 2. WORKER PROCESS (Tokenize & write shards to Google Drive)
# ---------------------------------------------------------
def _worker_process_files(
    worker_id,
    file_subset,
    tokenizer_name,
    seq_len,
    processed_dir,
    shard_size_samples=10_000,
):
    """Each worker reads assigned files, tokenizes, and writes complete Parquet shards."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Use int32/int8 to cut disk and RAM usage by 50%
    schema = pa.schema(
        [
            ("input_ids", pa.list_(pa.int32())),
            ("attention_mask", pa.list_(pa.int8())),
            ("labels", pa.list_(pa.int32())),
        ]
    )

    batch_input_ids = []
    batch_att_mask = []
    batch_labels = []

    shard_idx = 0
    local_temp_dir = f"/tmp/worker_{worker_id}"
    os.makedirs(local_temp_dir, exist_ok=True)

    def _flush_shard():
        nonlocal shard_idx, batch_input_ids, batch_att_mask, batch_labels
        if not batch_input_ids:
            return

        table = pa.Table.from_arrays(
            [
                pa.array(batch_input_ids, type=pa.list_(pa.int32())),
                pa.array(batch_att_mask, type=pa.list_(pa.int8())),
                pa.array(batch_labels, type=pa.list_(pa.int32())),
            ],
            schema=schema,
        )

        # 1. Write locally to /tmp (fast NVMe, no Drive FUSE overhead)
        local_file = os.path.join(
            local_temp_dir, f"shard_{worker_id}_{shard_idx:05d}.parquet"
        )
        pq.write_table(table, local_file, compression="zstd")

        # 2. Move atomically to Google Drive
        dest_file = os.path.join(
            processed_dir, f"shard_{worker_id}_{shard_idx:05d}.parquet"
        )
        shutil.move(local_file, dest_file)

        # Clear batch
        batch_input_ids.clear()
        batch_att_mask.clear()
        batch_labels.clear()
        del table
        shard_idx += 1

    for file_path in file_subset:
        try:
            for text in _stream_zst_records(file_path):
                # Tokenize sample
                encoded = tokenizer(
                    text,
                    truncation=True,
                    max_length=seq_len,
                    padding="max_length",
                    return_attention_mask=True,
                )

                input_ids = encoded["input_ids"]
                batch_input_ids.append(input_ids)
                batch_att_mask.append(encoded["attention_mask"])
                batch_labels.append(input_ids.copy())

                if len(batch_input_ids) >= shard_size_samples:
                    _flush_shard()

        except Exception as e:
            print(f"⚠️ Worker {worker_id} error reading {file_path}: {e}")

    # Flush any remaining samples
    _flush_shard()

    # Cleanup local worker directory
    shutil.rmtree(local_temp_dir, ignore_errors=True)
    gc.collect()


# ---------------------------------------------------------
# 3. MAIN PREPARATION PIPELINE
# ---------------------------------------------------------
def prepare_pretrain_dataset(phase_path, tokenizer, seq_len):
    processed_dir = f"/content/drive/MyDrive/Original/{phase_path}/processed_parquet"
    data_dir = f"/content/drive/MyDrive/Original/{phase_path}/data"

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} not found.")

    os.makedirs(processed_dir, exist_ok=True)

    # Check if already processed
    existing_shards = glob.glob(os.path.join(processed_dir, "*.parquet"))
    if len(existing_shards) > 0:
        print(
            f"✓ Found {len(existing_shards):,} existing Parquet shards in {processed_dir}."
        )
        print("Skipping preprocessing.")
        return processed_dir

    # 1. Collect all raw files
    all_files = []
    for root, _, files in os.walk(data_dir):
        for file_name in files:
            if not file_name.startswith("."):
                all_files.append(os.path.join(root, file_name))
    all_files.sort()

    if not all_files:
        raise FileNotFoundError(f"No valid data files found in {data_dir}.")

    tokenizer_name = "OLMo-2-1124-13B"  # adjust if local

    num_proc = max(1, min(os.cpu_count() or 1, len(all_files)))
    file_chunks = [all_files[i::num_proc] for i in range(num_proc)]

    print(
        f"📂 Tokenizing & Sharding {len(all_files):,} files directly to Parquet on Google Drive..."
    )
    print(f"⚙️ Using {num_proc} worker processes. RAM usage will remain constant.")

    # 2. Run multi-process sharding
    ctx = mp.get_context("spawn")
    processes = []
    for worker_id in range(num_proc):
        p = ctx.Process(
            target=_worker_process_files,
            args=(
                worker_id,
                file_chunks[worker_id],
                tokenizer_name,
                seq_len,
                processed_dir,
                10_000,  # 10,000 samples per shard (~50-100MB per file)
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(
        f"✓ Successfully generated all Parquet shards in: {processed_dir}"
    )
    return processed_dir


def prepare_sft_dataset(dataset_name, tokenizer, seq_len):
    # processed_path = f"/content/drive/MyDrive/Simulated/{dataset_name}/processed"
    processed_path = f"/content/drive/MyDrive/Original/{dataset_name}/processed"
    current_config = {
        "seq_len": seq_len,
        "tokenizer": getattr(tokenizer, "name_or_path", str(tokenizer.__class__))
    }
    
    cached_ds = check_and_load_cache(processed_path, current_config)
    if cached_ds is not None:
        return cached_ds

    # dataset_source = f"/content/drive/MyDrive/Simulated/{dataset_name}"
    dataset_source = f"/content/drive/MyDrive/Original/{dataset_name}"
    print(f"📂 Loading SFT dataset from: {dataset_source}...")
    ds = load_dataset(dataset_source, split="train")
    print(f"✓ Loaded {len(ds):,} raw SFT examples.")

    def tokenize_function(examples):
        texts = []
        if "messages" in examples:
            for msgs in examples["messages"]:
                if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
                    texts.append(tokenizer.apply_chat_template(msgs, tokenize=False))
                else:
                    texts.append("".join([f"{m.get('role', 'user')}: {m.get('content', '')}\n" for m in msgs]))
        elif "conversations" in examples:
            for msgs in examples["conversations"]:
                if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
                    texts.append(tokenizer.apply_chat_template(msgs, tokenize=False))
                else:
                    texts.append("".join([f"{m.get('from', m.get('role', 'user'))}: {m.get('value', m.get('content', ''))}\n" for m in msgs]))
        elif "text" in examples:
            texts = [str(t) for t in examples["text"]]
        elif "prompt" in examples and "response" in examples:
            texts = [str(p) + str(r) for p, r in zip(examples["prompt"], examples["response"])]
        elif "chosen" in examples:
            texts = [str(p) + str(c) for p, c in zip(examples["prompt"], examples["chosen"])]
        else:
            texts = [str(x) for x in examples[list(examples.keys())[0]]]

        tokenized = tokenizer(texts, truncation=True, max_length=seq_len, padding="max_length")
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    num_proc = max(os.cpu_count(), 1)
    print(f"⚙️  Tokenizing SFT conversations (seq_len={seq_len}, workers={num_proc})...")
    tokenized_ds = ds.map(tokenize_function, batched=True, num_proc=num_proc, desc="Tokenizing SFT dataset")
    tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    save_cache(tokenized_ds, processed_path, current_config)
    return tokenized_ds


def prepare_dpo_dataset(dataset_name):
    # processed_path = f"/content/drive/MyDrive/Simulated/{dataset_name}/processed"
    processed_path = f"/content/drive/MyDrive/Original/{dataset_name}/processed"
    current_config = {"format": "dpo_standard"}
    
    cached_ds = check_and_load_cache(processed_path, current_config)
    if cached_ds is not None:
        return cached_ds

    # dataset_source = f"/content/drive/MyDrive/Simulated/{dataset_name}"
    dataset_source = f"/content/drive/MyDrive/Original/{dataset_name}"
    print(f"📂 Loading DPO dataset from: {dataset_source}...")
    ds = load_dataset(dataset_source, split="train")
    print(f"✓ Loaded {len(ds):,} raw preference pairs.")

    num_proc = max(os.cpu_count(), 1)
    print(f"⚙️  Formatting preference pairs into prompt/chosen/rejected triplets (workers={num_proc})...")
    formatted_ds = ds.map(format_dpo_dataset, num_proc=num_proc, desc="Formatting DPO dataset")
    
    save_cache(formatted_ds, processed_path, current_config)
    return formatted_ds


def prepare_rlvr_dataset(dataset_name, tokenizer):
    # processed_path = f"/content/drive/MyDrive/Simulated/{dataset_name}/processed"
    processed_path = f"/content/drive/MyDrive/Original/{dataset_name}/processed"
    current_config = {
        "tokenizer": getattr(tokenizer, "name_or_path", str(tokenizer.__class__)),
        "format": "rlvr_extracted"
    }
    
    cached_ds = check_and_load_cache(processed_path, current_config)
    if cached_ds is not None:
        return cached_ds

    # dataset_source = f"/content/drive/MyDrive/Simulated/{dataset_name}"
    dataset_source = f"/content/drive/MyDrive/Original/{dataset_name}"
    print(f"📂 Loading RLVR reasoning dataset from: {dataset_source}...")
    ds = load_dataset(dataset_source, split="train")
    print(f"✓ Loaded {len(ds):,} raw RLVR reasoning examples.")

    def extract_fields(example):
        if "prompt" in example: prompt_text = example["prompt"]
        elif "messages" in example:
            msgs = example["messages"][:-1] if len(example["messages"]) > 0 and example["messages"][-1]["role"] == "assistant" else example["messages"]
            prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else: prompt_text = str(example)

        ground_truth = example.get("ground_truth", example.get("answer", ""))
        if not ground_truth and "messages" in example and example["messages"][-1]["role"] == "assistant":
            ground_truth = example["messages"][-1]["content"]
        if ground_truth is None:
            ground_truth = ""

        return {"prompt_text": prompt_text, "ground_truth": ground_truth}

    num_proc = max(os.cpu_count(), 1)
    print(f"⚙️  Extracting prompt and ground truth reasoning targets (workers={num_proc})...")
    processed_ds = ds.map(extract_fields, num_proc=num_proc, desc="Preparing RLVR dataset")
    
    save_cache(processed_ds, processed_path, current_config)
    return processed_ds