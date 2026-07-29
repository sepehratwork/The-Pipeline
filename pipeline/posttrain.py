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


def handle_weight_tying(model, config):
    """
    Ensures PyTorch tensor sharing matches Hugging Face safetensors configuration.
    """
    is_tied = getattr(config, "tie_word_embeddings", False)
    
    if is_tied:
        if hasattr(model, "tie_weights"):
            model.tie_weights()
        # Explicitly register dynamic tied weights keys for safetensors validation
        model._dynamic_tied_weights_keys = ["lm_head.weight", "model.embed_tokens.weight"]
    else:
        # If config specifies untied weights, ensure they do not share memory storage
        input_embeds = model.get_input_embeddings()
        output_embeds = model.get_output_embeddings()
        
        if input_embeds is not None and output_embeds is not None:
            if input_embeds.weight.data_ptr() == output_embeds.weight.data_ptr():
                output_embeds.weight = torch.nn.Parameter(output_embeds.weight.clone())


def run_stage4_sft(architecture, tokenizer, base_dir, stage3_model_path):
    stage4_dir = os.path.join(base_dir, "Stage4")
    final_model_dir = os.path.join(stage4_dir, "final_model")

    # Robust check for Hugging Face single-file or sharded model checkpoints
    is_already_saved = any(
        os.path.exists(os.path.join(final_model_dir, fname))
        for fname in ["model.safetensors", "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json"]
    )

    if not is_already_saved:
        print("=== Starting Stage 4: Supervised Finetuning (SFT) ===")
        os.makedirs(stage4_dir, exist_ok=True)

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
        
        # Safely align weight tying with safetensors requirements
        handle_weight_tying(model, config)

        ds = prepare_sft_dataset("Dolci-Think-SFT-32B", tokenizer, seq_len=1024)

        args = TrainingArguments(
            max_steps=6,
            save_total_limit=2, 
            output_dir=stage4_dir, per_device_train_batch_size=1,
            gradient_accumulation_steps=4, learning_rate=5.0e-5, logging_steps=1, save_steps=2,
            report_to="none", bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adamw_torch_fused",
            # save_safetensors=True,  # Standard Hugging Face safetensors format
            max_length=1024
        )

        trainer = Trainer(
            model=model, args=args, train_dataset=ds,
            processing_class=tokenizer,  # Standard HF Trainer tokenizer binding
            callbacks=[GradientMetricsCallback(model=model, log_file=os.path.join(stage4_dir, "training_log.jsonl"), plot_dir=stage4_dir)]
        )

        # Start Stage Timing
        timer = StageTimer(base_dir)
        start_t = timer.start_stage("Stage 4: Supervised Finetuning (SFT)")
        
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
                    
        # End Stage Timing
        timer.end_stage("Stage 4: Supervised Finetuning (SFT)", start_t)

        # Save model, tokenizer, and generation config according to HF standards
        os.makedirs(final_model_dir, exist_ok=True)
        handle_weight_tying(model, config)

        model.save_pretrained(final_model_dir, safe_serialization=True)
        tokenizer.save_pretrained(final_model_dir)
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.save_pretrained(final_model_dir)

        clear_all_checkpoints(stage4_dir) # Remove intermediate checkpoints after phase finishes
        
        del model, trainer, ds
        gc.collect()
        torch.cuda.empty_cache()

    return final_model_dir


def run_stage5_dpo(architecture, tokenizer, base_dir, stage4_model_path):
    stage5_dir = os.path.join(base_dir, "Stage5")
    final_model_dir = os.path.join(stage5_dir, "final_model")
    print(final_model_dir)

    # Robust check for Hugging Face single-file or sharded model checkpoints
    is_already_saved = any(
        os.path.exists(os.path.join(final_model_dir, fname))
        for fname in ["model.safetensors", "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json"]
    )

    if not is_already_saved:
        print("=== Starting Stage 5: Direct Preference Optimization (DPO) ===")
        os.makedirs(stage5_dir, exist_ok=True)

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
        
        # Safely align weight tying for model and ref_model
        handle_weight_tying(model, config)
        handle_weight_tying(ref_model, config)
        
        # Deactive gradient tracking natively
        ref_model.requires_grad_(False)
        ref_model.eval()

        ds = prepare_dpo_dataset("Dolci-Think-DPO-32B")

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
            beta=5.0, 
            max_length=2048,
            save_safetensors=True,  # Standard Hugging Face safetensors format
        )

        trainer = DPOTrainer(
            model=model, ref_model=ref_model, args=args, train_dataset=ds, processing_class=tokenizer,
            callbacks=[GradientMetricsCallback(model=model, log_file=os.path.join(stage5_dir, "training_log.jsonl"), plot_dir=stage5_dir)]
        )

        # Start Stage Timing
        timer = StageTimer(base_dir)
        start_t = timer.start_stage("Stage 5: Direct Preference Optimization (DPO)")
        
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
                    
        # End Stage Timing
        timer.end_stage("Stage 5: Direct Preference Optimization (DPO)", start_t)

        # Save model, tokenizer, and generation config according to HF standards
        os.makedirs(final_model_dir, exist_ok=True)
        handle_weight_tying(model, config)

        model.save_pretrained(final_model_dir, safe_serialization=True)
        tokenizer.save_pretrained(final_model_dir)
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.save_pretrained(final_model_dir)

        clear_all_checkpoints(stage5_dir)

        del model, ref_model, trainer, ds
        gc.collect()
        torch.cuda.empty_cache()

    return final_model_dir