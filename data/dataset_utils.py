import os
import gc
import json
import glob
import shutil
import tempfile
import queue
import traceback
from pathlib import Path
import multiprocessing as mp
import pandas as pd
from tqdm import tqdm

import datasets
import datasets.config
from datasets import Dataset, concatenate_datasets, load_from_disk, Features, Value, Sequence, load_dataset
from datasets.arrow_writer import ArrowWriter

# Prevent tokenizers deadlock when forking worker processes
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def configure_drive_caching(drive_base_path: str):
    """
    Redirects all Hugging Face, Transformers, PyArrow, and OS temporary
    directories to Google Drive to prevent filling up the Colab local disk.
    """
    drive_cache_dir = os.path.join(drive_base_path, "cache")
    drive_tmp_dir = os.path.join(drive_cache_dir, "tmp")
    drive_hf_home = os.path.join(drive_cache_dir, "huggingface")
    drive_hf_datasets = os.path.join(drive_cache_dir, "datasets")
    drive_hf_transformers = os.path.join(drive_cache_dir, "transformers")

    for path in [drive_cache_dir, drive_tmp_dir, drive_hf_home, drive_hf_datasets, drive_hf_transformers]:
        os.makedirs(path, exist_ok=True)

    # Export environment variables for the main process and child subprocesses
    os.environ["HF_HOME"] = drive_hf_home
    os.environ["HF_DATASETS_CACHE"] = drive_hf_datasets
    os.environ["TRANSFORMERS_CACHE"] = drive_hf_transformers
    os.environ["TMPDIR"] = drive_tmp_dir
    os.environ["TEMP"] = drive_tmp_dir
    os.environ["TMP"] = drive_tmp_dir

    # Point Python's internal tempfile directory to Drive
    tempfile.tempdir = drive_tmp_dir

    # Configure Hugging Face Datasets internal library cache paths
    datasets.config.HF_DATASETS_CACHE = Path(drive_hf_datasets)
    datasets.config.DOWNLOADED_DATASETS_PATH = Path(drive_hf_datasets) / "downloads"

    # Symlink ~/.cache/huggingface to Drive as an extra safeguard
    local_hf_cache = os.path.expanduser("~/.cache/huggingface")
    try:
        if os.path.islink(local_hf_cache):
            os.unlink(local_hf_cache)
        elif os.path.exists(local_hf_cache):
            shutil.rmtree(local_hf_cache, ignore_errors=True)
        os.makedirs(os.path.dirname(local_hf_cache), exist_ok=True)
        os.symlink(drive_hf_home, local_hf_cache)
    except Exception as e:
        print(f"ℹ️  [Cache Setup Note] Could not symlink local cache dir ({e}), using env vars.")

    return drive_tmp_dir, drive_hf_home, drive_hf_datasets, drive_hf_transformers


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


def save_cache_metadata(processed_path, current_config, num_shards):
    """
    Writes standard Hugging Face state.json and cache_config.json directly
    to avoid re-copying hundreds of gigabytes over Google Drive FUSE.
    """
    state = {
        "_data_files": [{"filename": f"data-{i:05d}-of-{num_shards:05d}.arrow"} for i in range(num_shards)],
        "_fingerprint": "custom_pretrain_tokenized",
        "_format_columns": ["input_ids", "attention_mask", "labels"],
        "_format_type": "torch",
        "_output_all_columns": False,
        "_split": None,
    }
    with open(os.path.join(processed_path, "state.json"), "w") as f:
        json.dump(state, f, indent=2)

    with open(os.path.join(processed_path, "cache_config.json"), "w") as f:
        json.dump(current_config, f, indent=2)


def _worker_stream_and_tokenize(
    worker_id,
    assigned_files,
    is_arrow_input,
    output_arrow_path,
    tokenizer,
    seq_len,
    progress_queue,
    drive_env_paths,
    batch_size=2048,
):
    """
    Worker process: reads assigned files in batches, tokenizes them,
    and writes directly to its own consolidated Arrow file on Google Drive.
    """
    # Enforce Drive cache paths inside child process
    drive_tmp_dir, drive_hf_home, drive_hf_datasets, drive_hf_transformers = drive_env_paths
    os.environ["HF_HOME"] = drive_hf_home
    os.environ["HF_DATASETS_CACHE"] = drive_hf_datasets
    os.environ["TRANSFORMERS_CACHE"] = drive_hf_transformers
    os.environ["TMPDIR"] = drive_tmp_dir
    os.environ["TEMP"] = drive_tmp_dir
    os.environ["TMP"] = drive_tmp_dir
    tempfile.tempdir = drive_tmp_dir
    datasets.config.HF_DATASETS_CACHE = Path(drive_hf_datasets)

    features = Features({
        "input_ids": Sequence(Value("int64")),
        "attention_mask": Sequence(Value("int64")),
        "labels": Sequence(Value("int64")),
    })

    writer = ArrowWriter(
        features=features,
        path=output_arrow_path,
        writer_batch_size=batch_size,
    )

    def _flush_batch(texts):
        if not texts:
            return
        outputs = tokenizer(
            texts,
            truncation=True,
            max_length=seq_len,
            padding="max_length",
            return_attention_mask=True,
        )
        batch_dict = {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
            "labels": outputs["input_ids"],  # Standard CLM targets
        }
        writer.write_batch(batch_dict)
        num_written = len(batch_dict["input_ids"])
        if progress_queue is not None:
            progress_queue.put(("batch", num_written))

    try:
        if is_arrow_input:
            # SCENARIO A: Read from pre-existing shards
            for shard_path in assigned_files:
                try:
                    ds_shard = Dataset.from_file(shard_path)
                    num_rows = len(ds_shard)
                    for i in range(0, num_rows, batch_size):
                        batch = ds_shard[i : i + batch_size]
                        texts = [t for t in batch.get("text", []) if t]
                        _flush_batch(texts)
                    del ds_shard
                except Exception as e:
                    print(f"⚠️ Worker {worker_id}: Error reading shard {shard_path}: {e}")
                finally:
                    if progress_queue is not None:
                        progress_queue.put(("file_done", 1))
                    gc.collect()
        else:
            # SCENARIO B: Stream directly from raw .jsonl.zst files
            buffer = []
            for file_path in assigned_files:
                df = None
                try:
                    df = pd.read_json(file_path, lines=True, compression="zstd")
                    if "text" in df.columns:
                        for text in df["text"].dropna():
                            if text:
                                buffer.append(text)
                                if len(buffer) >= batch_size:
                                    _flush_batch(buffer)
                                    buffer = []
                except Exception as e:
                    print(f"⚠️ Worker {worker_id}: Error reading {file_path}: {e}")
                finally:
                    if df is not None:
                        del df
                    if progress_queue is not None:
                        progress_queue.put(("file_done", 1))
                    gc.collect()

            if buffer:
                _flush_batch(buffer)

    except Exception as e:
        if progress_queue is not None:
            progress_queue.put(("worker_error", (worker_id, traceback.format_exc())))
        raise e
    finally:
        writer.finalize()
        if progress_queue is not None:
            progress_queue.put(("worker_done", worker_id))
        print(f"✓ Worker {worker_id} successfully finalized: {os.path.basename(output_arrow_path)}")


def prepare_pretrain_dataset(phase_path, tokenizer, seq_len, batch_size=2048):
    base_drive_path = f"/content/drive/MyDrive/Original/{phase_path}"
    processed_path = os.path.join(base_drive_path, "processed")
    os.makedirs(processed_path, exist_ok=True)

    # Configure all caching and temp storage strictly to Google Drive
    drive_env_paths = configure_drive_caching(base_drive_path)

    current_config = {
        "seq_len": seq_len,
        "tokenizer": getattr(tokenizer, "name_or_path", str(tokenizer.__class__)),
    }

    # 1. Fast Cache Check
    cached_ds = check_and_load_cache(processed_path, current_config)
    if cached_ds is not None:
        return cached_ds

    print(f"ℹ️  [Cache Miss] No valid cache found at {processed_path}. Preparing dataset...")

    # 2. Check if the existing shards exist from previous steps
    shard_cache_dir = os.path.join(base_drive_path, "shard_cache")
    existing_arrow_shards = []
    if os.path.exists(shard_cache_dir):
        existing_arrow_shards = sorted(glob.glob(os.path.join(shard_cache_dir, "*.arrow")))

    is_arrow_input = len(existing_arrow_shards) > 0
    if is_arrow_input:
        print(f"⚡ Detected {len(existing_arrow_shards):,} existing raw shards in {shard_cache_dir}.")
        print("   Directly tokenizing from existing shards to save hours of re-extraction!")
        files_to_process = existing_arrow_shards
    else:
        data_dir = os.path.join(base_drive_path, "data")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Neither shard_cache nor data directory {data_dir} found.")

        files_to_process = []
        for root, _, files in os.walk(data_dir):
            for file_name in files:
                if not file_name.startswith("."):
                    files_to_process.append(os.path.join(root, file_name))
        files_to_process.sort()

        if not files_to_process:
            raise FileNotFoundError(f"No valid files found in {data_dir}.")
        print(f"📂 Streaming {len(files_to_process):,} raw files directly into tokenized Arrow shards...")

    # 3. Utilize full hardware: 8 CPU cores & distributed batches for 50GB RAM
    total_cpus = os.cpu_count() or 8
    num_proc = min(8, total_cpus, len(files_to_process))

    # Interleaved round-robin partitioning for balanced worker load
    file_chunks = [files_to_process[i::num_proc] for i in range(num_proc)]

    # Consolidated destination paths: 1 Arrow file per worker
    output_arrow_paths = [
        os.path.join(processed_path, f"data-{i:05d}-of-{num_proc:05d}.arrow")
        for i in range(num_proc)
    ]

    print(f"⚙️  Spawning {num_proc} workers (utilizing 8 cores & 50GB RAM capacity)...")

    # 4. Multiprocessing execution with isolated process memory & real-time progress bar
    progress_queue = mp.Queue()
    processes = []
    for worker_id in range(num_proc):
        p = mp.Process(
            target=_worker_stream_and_tokenize,
            args=(
                worker_id,
                file_chunks[worker_id],
                is_arrow_input,
                output_arrow_paths[worker_id],
                tokenizer,
                seq_len,
                progress_queue,
                drive_env_paths,
                batch_size,
            ),
        )
        p.start()
        processes.append(p)

    # Real-time multi-worker progress tracking
    total_files = len(files_to_process)
    total_tokenized_samples = 0
    active_workers = num_proc

    with tqdm(
        total=total_files,
        desc="🚀 Tokenizing & Sharding (Step 4)",
        unit="file",
        dynamic_ncols=True,
    ) as pbar:
        while active_workers > 0:
            try:
                msg_type, val = progress_queue.get(timeout=0.2)
                if msg_type == "batch":
                    total_tokenized_samples += val
                    pbar.set_postfix({"tokenized_samples": f"{total_tokenized_samples:,}"})
                elif msg_type == "file_done":
                    pbar.update(val)
                elif msg_type == "worker_done":
                    active_workers -= 1
                elif msg_type == "worker_error":
                    w_id, err_trace = val
                    print(f"\n❌ Worker {w_id} encountered an error:\n{err_trace}")
            except queue.Empty:
                # Check for crashed workers
                for i, p in enumerate(processes):
                    if not p.is_alive() and p.exitcode not in (0, None):
                        raise RuntimeError(f"Worker {i} exited unexpectedly with code {p.exitcode}")

    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker process failed with exit code {p.exitcode}")

    # 5. Build and verify the consolidated Hugging Face Dataset
    print("🔗 Linking consolidated shards into memory-mapped Hugging Face Dataset...")
    valid_shards = [p for p in output_arrow_paths if os.path.exists(p) and os.path.getsize(p) > 0]
    if not valid_shards:
        raise RuntimeError("No tokenized data was produced by workers.")

    shard_datasets = [Dataset.from_file(shard_p) for shard_p in valid_shards]
    tokenized_ds = concatenate_datasets(shard_datasets)

    features = Features({
        "input_ids": Sequence(Value("int64")),
        "attention_mask": Sequence(Value("int64")),
        "labels": Sequence(Value("int64")),
    })

    # 6. Save metadata and configs so check_and_load_cache works instantly next time
    save_cache_metadata(processed_path, current_config, len(valid_shards))
    tokenized_ds.info.features = features
    tokenized_ds.info.write_to_directory(processed_path)

    tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    print(f"✓ Dataset ready with {len(tokenized_ds):,} tokenized sequences.")

    # 7. Cleanup old temporary raw shards to recover Google Drive storage
    if is_arrow_input and os.path.exists(shard_cache_dir):
        print(f"🧹 Cleaning up intermediate raw shard directory: {shard_cache_dir}...")
        shutil.rmtree(shard_cache_dir, ignore_errors=True)

    gc.collect()
    return tokenized_ds


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
    
    save_cache_metadata(tokenized_ds, processed_path, current_config)
    return tokenized_ds


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
    
    save_cache_metadata(formatted_ds, processed_path, current_config)
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
    
    save_cache_metadata(processed_ds, processed_path, current_config)
    return processed_ds