import os
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset

def load_stage_dataset(phase_path, tokenizer, seq_len):
    dss = []
    data_dir = f"../{phase_path}/data"

    if os.path.exists(data_dir):
        for shard in os.listdir(data_dir):
            shard_path = f"{data_dir}/{shard}"
            if os.path.isdir(shard_path):
                for j in os.listdir(shard_path):
                    df = pd.read_json(f"{shard_path}/{j}", lines=True, compression='zstd')
                    df = df.drop(columns=[col for col in df.columns if col != "text"])
                    dss.append(Dataset.from_pandas(df))
        ds = concatenate_datasets(dss)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=seq_len, padding="max_length")

    tokenized_ds = ds.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_ds = tokenized_ds.map(lambda e: {"labels": e["input_ids"].copy()}, batched=True)
    tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized_ds

def prepare_sft_dataset(dataset_name, tokenizer, seq_len):
    ds = load_dataset(dataset_name, split="train")

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

    tokenized_ds = ds.map(tokenize_function, batched=True)
    tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
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
