import os
import gc
import shutil
import torch
from transformers import Trainer, TrainingArguments
from trl import DPOTrainer, DPOConfig

from datasets import load_dataset
from models import get_model_classes
from data import prepare_sft_dataset, prepare_dpo_dataset
from utils import GradientMetricsCallback, get_latest_checkpoint, clear_all_checkpoints
from utils.callbacks import StageTimer


def run_stage4_sft(architecture, tokenizer, base_dir, stage3_model_path):
    stage4_dir = os.path.join(base_dir, "Stage4")
    if not os.path.exists(os.path.join(stage4_dir, "final_model", "model.safetensors")) and not os.path.exists(os.path.join(stage4_dir, "final_model", "pytorch_model.bin")):
        print("=== Starting Stage 4: Supervised Finetuning (SFT) ===")
        os.makedirs(stage4_dir, exist_ok=True)

        # Start Stage Timing
        timer = StageTimer(base_dir)
        start_t = timer.start_stage("Stage 4: Supervised Finetuning (SFT)")

        ConfigClass, ModelClass = get_model_classes(architecture)
        config = ConfigClass.from_pretrained(stage3_model_path)

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        # Load directly in correct precision to avoid overhead during casting
        model = ModelClass.from_pretrained(
            stage3_model_path, 
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        if hasattr(model, "tie_weights"):
            model.tie_weights()

        ds = prepare_sft_dataset("../Dolci-Think-SFT-32B", tokenizer, seq_len=1024)

        args = TrainingArguments(
            max_steps=6,
            save_total_limit=2, 
            output_dir=stage4_dir, per_device_train_batch_size=1,
            gradient_accumulation_steps=4, learning_rate=5.0e-5, logging_steps=1, save_steps=2,
            report_to="none", bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_torch_fused",
            save_safetensors=False,  # Prevents RuntimeError with shared embedding tensors
        )

        trainer = Trainer(
            model=model, args=args, train_dataset=ds,
            callbacks=[GradientMetricsCallback(model=model, log_file=os.path.join(stage4_dir, "training_log.jsonl"), plot_dir=stage4_dir)]
        )

        # Robust resumption loop
        while True:
            ckpt = get_latest_checkpoint(stage4_dir)
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
                    
        model.save_pretrained(os.path.join(stage4_dir, "final_model"), safe_serialization=False)
        clear_all_checkpoints(stage4_dir) # Remove all checkpoints after phase finishes
        
        del model, trainer, ds
        gc.collect()
        torch.cuda.empty_cache()

        # End Stage Timing
        timer.end_stage("Stage 4: Supervised Finetuning (SFT)", start_t)

    return os.path.join(stage4_dir, "final_model")


def run_stage5_dpo(architecture, tokenizer, base_dir, stage4_model_path):
    stage5_dir = os.path.join(base_dir, "Stage5")
    if not os.path.exists(os.path.join(stage5_dir, "final_model", "model.safetensors")) and not os.path.exists(os.path.join(stage5_dir, "final_model", "pytorch_model.bin")):
        print("=== Starting Stage 5: Direct Preference Optimization (DPO) ===")
        os.makedirs(stage5_dir, exist_ok=True)

        # Start Stage Timing
        timer = StageTimer(base_dir)
        start_t = timer.start_stage("Stage 5: Direct Preference Optimization (DPO)")

        ConfigClass, ModelClass = get_model_classes(architecture)
        config = ConfigClass.from_pretrained(stage4_model_path)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # Optimized loading for both model instances
        model = ModelClass.from_pretrained(
            stage4_model_path, 
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        ref_model = ModelClass.from_pretrained(
            stage4_model_path, 
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        if hasattr(model, "tie_weights"):
            model.tie_weights()
            ref_model.tie_weights()
        
        # Deactive gradient tracking natively
        ref_model.requires_grad_(False)
        ref_model.eval()

        ds = prepare_dpo_dataset("../Dolci-Think-DPO-32B")

        args = DPOConfig(
            max_steps=6,
            save_total_limit=2,
            output_dir=stage5_dir, per_device_train_batch_size=1,
            max_grad_norm=1.0,
            gradient_accumulation_steps=4, learning_rate=8.0e-8, lr_scheduler_type="linear", warmup_ratio=0.1,
            logging_steps=1, save_steps=2, report_to="none", bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(), 
            gradient_checkpointing=True, 
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_torch_fused",
            beta=5.0, max_length=2048,
            save_safetensors=False,  # Prevents RuntimeError with shared embedding tensors
        )

        trainer = DPOTrainer(
            model=model, ref_model=ref_model, args=args, train_dataset=ds, processing_class=tokenizer,
            callbacks=[GradientMetricsCallback(model=model, log_file=os.path.join(stage5_dir, "training_log.jsonl"), plot_dir=stage5_dir)]
        )

        # Robust resumption loop
        while True:
            ckpt = get_latest_checkpoint(stage5_dir)
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
                    
        model.save_pretrained(os.path.join(stage5_dir, "final_model"), safe_serialization=False)
        clear_all_checkpoints(stage5_dir)

        del model, ref_model, trainer, ds
        gc.collect()
        torch.cuda.empty_cache()

        # End Stage Timing
        timer.end_stage("Stage 5: Direct Preference Optimization (DPO)", start_t)

    return os.path.join(stage5_dir, "final_model")