import os
import gc
import json
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk, Features, Value


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


def _text_generator(file_paths):
    """
    Worker generator that streams one text at a time from assigned files.
    Robustly flattens any nesting so pd.read_json always receives a string.
    """
    # 1. Flatten whatever data structure HF datasets passes to this worker
    files_to_read = []

    def _flatten(item):
        if isinstance(item, (list, tuple)):
            for sub in item:
                _flatten(sub)
        elif isinstance(item, str):
            files_to_read.append(item)

    _flatten(file_paths)

    # 2. Stream individual files one by one per worker process
    for file_path in files_to_read:
        df = None
        try:
            df = pd.read_json(file_path, lines=True, compression="zstd")
            if "text" in df.columns:
                for text in df["text"].dropna():
                    if text:  # Filter out empty strings
                        yield {"text": text}
        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")
        finally:
            if df is not None:
                del df  # Clean up memory immediately per file


def prepare_pretrain_dataset(phase_path, tokenizer, seq_len):
    processed_path = f"/content/drive/MyDrive/Original/{phase_path}/processed"
    current_config = {
        "seq_len": seq_len,
        "tokenizer": getattr(tokenizer, "name_or_path", str(tokenizer.__class__)),
    }

    cached_ds = check_and_load_cache(processed_path, current_config)
    if cached_ds is not None:
        return cached_ds

    data_dir = f"/content/drive/MyDrive/Original/{phase_path}/data"
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} not found.")

    # 1. Collect and sort all files for deterministic sharding
    all_files = []
    for root, _, files in os.walk(data_dir):
        for file_name in files:
            if not file_name.startswith("."):  # Ignore hidden/metadata files
                all_files.append(os.path.join(root, file_name))
    all_files.sort()

    if not all_files:
        raise FileNotFoundError(f"No valid files found in {data_dir}.")

    # 2. Determine worker count and partition files across all CPU cores
    total_cpus = os.cpu_count() or 1
    num_proc = max(1, min(total_cpus, len(all_files)))

    # Interleaved round-robin partitioning so each worker gets balanced data
    file_chunks = [all_files[i::num_proc] for i in range(num_proc)]

    print(
        f"📂 Streaming {len(all_files):,} files across {num_proc} CPU cores "
        f"directly to Arrow table..."
    )

    # Note: Using /tmp for intermediate shard cache avoids Google Drive FUSE write latency
    local_cache_dir = f"/tmp/shard_cache/{phase_path}"
    os.makedirs(local_cache_dir, exist_ok=True)

    # Explicit schema prevents schema-inference conflicts across parallel workers
    features = Features({"text": Value("string")})

    # 3. Parallel extraction across CPU cores
    ds = Dataset.from_generator(
        _text_generator,
        gen_kwargs={"file_paths": file_chunks},
        num_proc=num_proc,
        features=features,
        cache_dir=local_cache_dir,
    )
    print(f"✓ Total raw pre-training samples loaded: {len(ds):,}")

    # 4. Tokenize and assign labels in a single pass
    def tokenize_and_label(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=seq_len,
            padding="max_length",
        )
        outputs["labels"] = [ids.copy() for ids in outputs["input_ids"]]
        return outputs

    print(f"⚙️  Tokenizing & labeling dataset (seq_len={seq_len}, workers={num_proc})...")

    gc.collect()

    tokenized_ds = ds.map(
        tokenize_and_label,
        batched=True,
        batch_size=1000,
        writer_batch_size=1000,
        remove_columns=["text"],
        num_proc=num_proc,
        desc="Tokenizing & Labeling",
    )

    tokenized_ds.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    print("💾 Saving cache...")
    save_cache(tokenized_ds, processed_path, current_config)

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