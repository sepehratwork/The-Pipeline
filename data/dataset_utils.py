import os
import gc
import json
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk


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


def _text_generator(data_dir):
    """
    Generator that streams one text at a time directly from compressed files.
    Keeps only 1 file in memory at any given time.
    """
    shards = [s for s in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, s))]
    for shard in shards:
        shard_path = os.path.join(data_dir, shard)
        for file_name in os.listdir(shard_path):
            file_path = os.path.join(shard_path, file_name)
            try:
                # Read single compressed file
                df = pd.read_json(file_path, lines=True, compression='zstd')
                if "text" in df.columns:
                    for text in df["text"].dropna():
                        if text:  # Filter empty strings
                            yield {"text": text}
            except Exception as e:
                print(f"⚠️ Error reading {file_path}: {e}")
            finally:
                del df  # Explicit cleanup per file


def prepare_pretrain_dataset(phase_path, tokenizer, seq_len):
    # processed_path = f"/content/drive/MyDrive/Simulated/{phase_path}/processed"
    processed_path = f"/content/drive/MyDrive/Original/{phase_path}/processed"
    current_config = {
        "seq_len": seq_len,
        "tokenizer": getattr(tokenizer, "name_or_path", str(tokenizer.__class__))
    }
    
    cached_ds = check_and_load_cache(processed_path, current_config)
    if cached_ds is not None:
        return cached_ds

    # data_dir = f"/content/drive/MyDrive/Simulated/{phase_path}/data"
    data_dir = f"/content/drive/MyDrive/Original/{phase_path}/data"

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} not found.")

    print(f"📂 Streaming data shards from {data_dir} directly to Arrow table...")
    
    # 1. Stream data directly to disk without storing raw DataFrames in RAM
    # Using local /tmp/hf_cache prevents slow FUSE / Google Drive I/O crashes
    local_cache_dir = "/tmp/hf_cache"
    os.makedirs(local_cache_dir, exist_ok=True)
    
    ds = Dataset.from_generator(
        _text_generator, 
        gen_kwargs={"data_dir": data_dir},
        cache_dir=local_cache_dir
    )
    print(f"✓ Total raw pre-training samples loaded: {len(ds):,}")

    # 2. Tokenize and assign labels in a single pass
    def tokenize_and_label(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=seq_len,
            padding="max_length"
        )
        # Create labels immediately in memory without a second .map()
        outputs["labels"] = [ids.copy() for ids in outputs["input_ids"]]
        return outputs

    # Bound worker count to prevent CPU processes from duplicating memory
    num_proc = max(1, os.cpu_count())
    print(f"⚙️  Tokenizing & labeling dataset (seq_len={seq_len}, workers={num_proc})...")
    
    gc.collect()

    tokenized_ds = ds.map(
        tokenize_and_label,
        batched=True,
        batch_size=1000,           # Tokenize in controlled batch sizes
        writer_batch_size=1000,    # Regularly flushes Arrow buffers from RAM to disk
        remove_columns=["text"],
        num_proc=num_proc,
        desc="Tokenizing & Labeling"
    )

    tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
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