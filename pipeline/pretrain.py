import os
import shutil
import gc
import torch
from transformers import Trainer, TrainingArguments

from models import get_model_classes
from data import prepare_pretrain_dataset
from utils import GradientMetricsCallback, get_latest_checkpoint, clear_all_checkpoints, save_to_hf_hub
from utils.callbacks import StageTimer


def _print_pretrain_stage_banner(stage_name, architecture, seq_len, train_args, config_kwargs):
    width = 75
    print("\n" + "=" * width)
    print(f"🎯 {stage_name.upper()} :: {architecture.upper()}".center(width))
    print("=" * width)
    print(f" • Sequence Length       : {seq_len}")
    print(f" • Max Training Steps    : {train_args.get('max_steps', 'N/A')}")
    print(f" • Per-Device Batch Size : {train_args.get('per_device_train_batch_size', 'N/A')}")
    print(f" • Learning Rate         : {train_args.get('learning_rate', 'N/A')}")
    print(f" • LR Scheduler          : {train_args.get('lr_scheduler_type', 'cosine')}")
    print(f" • Max Position Embeds   : {config_kwargs.get('max_position_embeddings', 'N/A')}")
    print(f" • YaRN Rope Scaling     : {config_kwargs.get('use_yarn', False)}")
    print("=" * width + "\n")


def _run_pretrain_stage(stage_name, architecture, tokenizer, dataset_path, seq_len, output_dir, config_kwargs, train_args_kwargs, resume_model_path=None):
    final_model_dir = os.path.join(output_dir, "final_model")
    
    # Robust check for Hugging Face single-file or sharded model checkpoints
    is_already_saved = any(
        os.path.exists(os.path.join(final_model_dir, fname))
        for fname in ["model.safetensors", "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json"]
    )

    if not is_already_saved:
        _print_pretrain_stage_banner(stage_name, architecture, seq_len, train_args_kwargs, config_kwargs)
        print(f"📁 Output Directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        ConfigClass, ModelClass = get_model_classes(architecture)
        
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"⚙️  Model Precision: {dtype} | CUDA BF16 Support: {torch.cuda.is_bf16_supported()}")
        
        if resume_model_path:
            print(f"🔄 Loading base weights from previous stage checkpoint: {resume_model_path}")
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
            print(f"🌱 Initializing model {architecture} from scratch with vocab size {len(tokenizer):,}...")
            config = ConfigClass(vocab_size=len(tokenizer), **config_kwargs)
            model = ModelClass(config).to(dtype)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.grad is not None or p.requires_grad)
        print(f"✓ Model loaded: {total_params:,} parameters ({trainable_params:,} trainable).")

        if hasattr(model, "tie_weights"):
            model.tie_weights()

        print(f"📥 Preparing pretrain dataset '{dataset_path}'...")
        ds = prepare_pretrain_dataset(dataset_path, tokenizer, seq_len=seq_len)
        print(f"✓ Dataset ready with {len(ds):,} tokenized sequences.")

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
            # save_safetensors=True,  # Standard Hugging Face safetensors format
            **train_args_kwargs
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds,
            processing_class=tokenizer,  # Standard HF Trainer tokenizer binding
            callbacks=[GradientMetricsCallback(model=model, log_file=os.path.join(output_dir, f"training_log_{stage_name}.jsonl"), plot_dir=output_dir)]
        )
        
        # Start Stage Timing
        base_dir = os.path.dirname(output_dir)
        timer = StageTimer(base_dir)
        start_t = timer.start_stage(stage_name)
        
        # Robust resumption loop
        while True:
            ckpt = get_latest_checkpoint(output_dir)
            if ckpt is None:
                print("🏁 No existing checkpoint found. Starting training from step 0.")
                trainer.train()
                break
            try:
                print(f"🔄 Resuming {stage_name} from checkpoint: {ckpt}")
                trainer.train(resume_from_checkpoint=ckpt)
                break
            except Exception as e:
                print(f"⚠️ Checkpoint {ckpt} corrupted or failed to load: {e}. Deleting and checking previous...")
                shutil.rmtree(ckpt, ignore_errors=True)
                
        # End Stage Timing
        timer.end_stage(stage_name, start_t)
        
        # Save model, tokenizer, and generation config according to HF standards
        print(f"💾 Saving stage final model and tokenizer to: {final_model_dir}...")
        os.makedirs(final_model_dir, exist_ok=True)
        if hasattr(model, "tie_weights"):
            model.tie_weights()
            
        model.save_pretrained(final_model_dir, safe_serialization=True)
        tokenizer.save_pretrained(final_model_dir)
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.save_pretrained(final_model_dir)
        print(f"✓ Model successfully saved to {final_model_dir}.")

        clear_all_checkpoints(output_dir) # Remove intermediate checkpoints after phase finishes

        del model, trainer, ds
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print(f"⏭️  [Skipped] {stage_name} already completed. Checkpoint found at {final_model_dir}")
        
    clear_all_checkpoints(output_dir) # Failsafe cleanup
    return final_model_dir


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
    
    # Save model to Hugging Face Hub"
    repo_name = f"{architecture}_base"
    print(f"\n🚀 Initiating Hugging Face Hub publication for Base Model: '{repo_name}'...")
    save_to_hf_hub(stage3_model_path, repo_name, hf_username=hf_username)

    return stage3_model_path