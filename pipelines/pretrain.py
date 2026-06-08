import os
import shutil
import torch
from transformers import Trainer, TrainingArguments

from models import get_model_classes
from data import load_pretrain_phase_dataset
from utils import GradientMetricsCallback, get_latest_checkpoint, clear_all_checkpoints


def _run_pretrain_stage(stage_name, model_type, tokenizer, dataset_path, seq_len, output_dir, config_kwargs, train_args_kwargs, resume_model_path=None):
    if not os.path.exists(os.path.join(output_dir, "final_model", "model.safetensors")):
        print(f"=== Starting {stage_name} ===")
        os.makedirs(output_dir, exist_ok=True)
        ConfigClass, ModelClass = get_model_classes(model_type)
        if resume_model_path:
            config = ConfigClass.from_pretrained(resume_model_path)
            for k, v in config_kwargs.items():
                setattr(config, k, v)
            model = ModelClass.from_pretrained(resume_model_path, config=config, ignore_mismatched_sizes=True)
        else:
            config = ConfigClass(vocab_size=len(tokenizer), **config_kwargs)
            model = ModelClass(config)
        ds = load_pretrain_phase_dataset(dataset_path, tokenizer, seq_len=seq_len)
        args = TrainingArguments(
            output_dir=output_dir,
            report_to="none",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            save_total_limit=2, # Keep only the last 2 checkpoints
            **train_args_kwargs
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds,
            callbacks=[GradientMetricsCallback(log_file=os.path.join(output_dir, "training_log.jsonl"), plot_dir=output_dir)]
        )
        # Robust resumption loop
        while True:
            ckpt = get_latest_checkpoint(output_dir)
            try:
                trainer.train(resume_from_checkpoint=ckpt)
                break
            except Exception as e:
                if ckpt is not None:
                    print(f"Checkpoint {ckpt} corrupted or failed to load: {e}. Deleting and trying previous.")
                    shutil.rmtree(ckpt, ignore_errors=True)
                else:
                    raise e

        model.save_pretrained(os.path.join(output_dir, "final_model"))
        clear_all_checkpoints(output_dir)
    return os.path.join(output_dir, "final_model")


def run_stage1_pretraining(model_type, tokenizer, base_dir):
    return _run_pretrain_stage(
        # "Stage 1: Pretraining", model_type, tokenizer, "dolma3_mix-150B-1025", 8192,
        "Stage 1: Pretraining", model_type, tokenizer, "dolma3_mix-150B-1025", 1024,
        os.path.join(base_dir, "Stage1"),
        {"max_position_embeddings": 8192, "use_yarn": False},
        # {"max_steps": 10000, "per_device_train_batch_size": 1, "learning_rate": 3.0e-4, "lr_scheduler_type": "cosine", "warmup_steps": 2000, "logging_steps": 1, "save_steps": 5}
        {"max_steps": 4, "per_device_train_batch_size": 1, "learning_rate": 3.0e-4, "lr_scheduler_type": "cosine", "warmup_steps": 2000, "logging_steps": 1, "save_steps": 2}
    )


def run_stage2_midtraining(model_type, tokenizer, base_dir, stage1_model_path):
    return _run_pretrain_stage(
        # "Stage 2: Midtraining", model_type, tokenizer, "dolma3_dolmino_mix-100B-1125", 8192,
        "Stage 2: Midtraining", model_type, tokenizer, "dolma3_dolmino_mix-100B-1125", 1024,
        os.path.join(base_dir, "Stage2"),
        {"max_position_embeddings": 8192, "use_yarn": False},
        # {"max_steps": 5000, "per_device_train_batch_size": 1, "learning_rate": 2.074e-4, "lr_scheduler_type": "linear", "warmup_steps": 0, "logging_steps": 1, "save_steps": 5},
        {"max_steps": 4, "per_device_train_batch_size": 1, "learning_rate": 2.074e-4, "lr_scheduler_type": "linear", "warmup_steps": 0, "logging_steps": 1, "save_steps": 2},
        resume_model_path=stage1_model_path
    )


def run_stage3_long_context(model_type, tokenizer, base_dir, stage2_model_path):
    return _run_pretrain_stage(
        # "Stage 3: Long-context Extension", model_type, tokenizer, "dolma3_longmino_mix-100B-1125", 65536,
        "Stage 3: Long-context Extension", model_type, tokenizer, "dolma3_longmino_mix-100B-1125", 2048,
        os.path.join(base_dir, "Stage3"),
        {"max_position_embeddings": 65536, "use_yarn": True},
        # {"max_steps": 2000, "per_device_train_batch_size": 1, "learning_rate": 2.074e-4, "lr_scheduler_type": "linear", "warmup_steps": 200, "logging_steps": 1, "save_steps": 5},
        {"max_steps": 4, "per_device_train_batch_size": 1, "learning_rate": 2.074e-4, "lr_scheduler_type": "linear", "warmup_steps": 200, "logging_steps": 1, "save_steps": 2},
        resume_model_path=stage2_model_path
    )
