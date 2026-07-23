import os
import shutil
import gc
import torch
from transformers import Trainer, TrainingArguments

from models import get_model_classes
from data import load_pretrain_phase_dataset
from utils import GradientMetricsCallback, get_latest_checkpoint, clear_all_checkpoints, save_to_hf_hub
from utils.callbacks import StageTimer


def _run_pretrain_stage(stage_name, architecture, tokenizer, dataset_path, seq_len, output_dir, config_kwargs, train_args_kwargs, resume_model_path=None):
    if not os.path.exists(os.path.join(output_dir, "final_model", "model.safetensors")) and not os.path.exists(os.path.join(output_dir, "final_model", "pytorch_model.bin")):
        print(f"=== Starting {stage_name} ===")
        os.makedirs(output_dir, exist_ok=True)
        
        # Start Stage Timing
        base_dir = os.path.dirname(output_dir)
        timer = StageTimer(base_dir)
        start_t = timer.start_stage(stage_name)
        
        ConfigClass, ModelClass = get_model_classes(architecture)
        
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        if resume_model_path:
            config = ConfigClass.from_pretrained(resume_model_path)
            for k, v in config_kwargs.items():
                setattr(config, k, v)
            # Load directly with target dtype and low CPU memory footprint
            model = ModelClass.from_pretrained(
                resume_model_path, 
                config=config, 
                ignore_mismatched_sizes=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            )
        else:
            config = ConfigClass(vocab_size=len(tokenizer), **config_kwargs)
            model = ModelClass(config).to(dtype)

        if hasattr(model, "tie_weights"):
            model.tie_weights()

        ds = load_pretrain_phase_dataset(dataset_path, tokenizer, seq_len=seq_len)
        args = TrainingArguments(
            output_dir=output_dir,
            report_to="none",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},  # Recommended non-reentrant checkpointing
            max_grad_norm=1.0,
            optim="adamw_torch_fused",
            save_total_limit=2,
            save_safetensors=False,  # Prevents RuntimeError with shared embedding tensors
            **train_args_kwargs
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds,
            # Passed model to callback init
            callbacks=[GradientMetricsCallback(model=model, log_file=os.path.join(output_dir, "training_log.jsonl"), plot_dir=output_dir)]
        )
        
        # Robust resumption loop
        while True:
            ckpt = get_latest_checkpoint(output_dir)
            if ckpt is None:
                print("No valid checkpoint found. Starting training from the beginning.")
                trainer.train()
                break
            try:
                print(f"Attempting to resume from checkpoint: {ckpt}")
                trainer.train(resume_from_checkpoint=ckpt)
                break
            except Exception as e:
                print(f"Checkpoint {ckpt} corrupted or failed to load: {e}. Deleting and trying previous.")
                shutil.rmtree(ckpt, ignore_errors=True)
                
        model.save_pretrained(os.path.join(output_dir, "final_model"), safe_serialization=False)
        clear_all_checkpoints(output_dir) # Remove all checkpoints after phase finishes

        del model, trainer, ds
        gc.collect()
        torch.cuda.empty_cache()
        
        # End Stage Timing
        timer.end_stage(stage_name, start_t)
        
    clear_all_checkpoints(output_dir) # Failsafe cleanup
    return os.path.join(output_dir, "final_model")


def run_stage1_pretraining(architecture, tokenizer, base_dir):
    return _run_pretrain_stage(
        "Stage 1: Pretraining", architecture, tokenizer, "dolma3_mix-150B-1025", 1024,
        os.path.join(base_dir, "Stage1"),
        {"max_position_embeddings": 8192, "use_yarn": False},
        {"max_steps": 6, "per_device_train_batch_size": 1, "learning_rate": 3.0e-4, "lr_scheduler_type": "cosine", "warmup_steps": 2000, "logging_steps": 1, "save_steps": 2}
    )


def run_stage2_midtraining(architecture, tokenizer, base_dir, stage1_model_path):
    return _run_pretrain_stage(
        "Stage 2: Midtraining", architecture, tokenizer, "dolma3_dolmino_mix-100B-1125", 1024,
        os.path.join(base_dir, "Stage2"),
        {"max_position_embeddings": 8192, "use_yarn": False},
        {"max_steps": 6, "per_device_train_batch_size": 1, "learning_rate": 2.074e-4, "lr_scheduler_type": "linear", "warmup_steps": 0, "logging_steps": 1, "save_steps": 2},
        resume_model_path=stage1_model_path
    )


def run_stage3_long_context(architecture, tokenizer, base_dir, stage2_model_path, hf_username=None):
    stage3_model_path = _run_pretrain_stage(
        "Stage 3: Long-context Extension", architecture, tokenizer, "dolma3_longmino_mix-100B-1125", 2048,
        os.path.join(base_dir, "Stage3"),
        {"max_position_embeddings": 65536, "use_yarn": True},
        {"max_steps": 6, "per_device_train_batch_size": 1, "learning_rate": 2.074e-4, "lr_scheduler_type": "linear", "warmup_steps": 200, "logging_steps": 1, "save_steps": 2},
        resume_model_path=stage2_model_path
    )
    
    # Save model to Hugging Face Hub with format: f"{architecture}_base"
    repo_name = f"{architecture}_base"
    save_to_hf_hub(stage3_model_path, tokenizer, repo_name, hf_username=hf_username)

    return stage3_model_path