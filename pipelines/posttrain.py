import os
import gc
import torch
from transformers import Trainer, TrainingArguments
from trl import DPOTrainer, DPOConfig

from datasets import load_dataset
from models import get_model_classes
from data import prepare_sft_dataset, format_dpo_dataset
from utils import GradientMetricsCallback, get_latest_checkpoint


def run_stage4_sft(model_type, tokenizer, base_dir, stage3_model_path):
    if not os.path.exists(os.path.join(base_dir, "final_model", "model.safetensors")):
        print("=== Starting Stage 4: Supervised Finetuning (SFT) ===")
        stage4_dir = os.path.join(base_dir, "Stage4")
        os.makedirs(stage4_dir, exist_ok=True)

        ConfigClass, ModelClass = get_model_classes(model_type)
        config = ConfigClass.from_pretrained(stage3_model_path)

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = ModelClass.from_pretrained(stage3_model_path, config=config).to(dtype)
        # ds = prepare_sft_dataset("../Dolci-Think-SFT-32B", tokenizer, seq_len=32768)
        ds = prepare_sft_dataset("../Dolci-Think-SFT-32B", tokenizer, seq_len=1024)

        args = TrainingArguments(
            # num_train_epochs=2,
            max_steps=3,
            output_dir=stage4_dir, per_device_train_batch_size=1,
            gradient_accumulation_steps=4, learning_rate=5.0e-5, logging_steps=1, save_steps=1,
            report_to="none", bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True, optim="adamw_torch_fused"
        )

        trainer = Trainer(
            model=model, args=args, train_dataset=ds,
            callbacks=[GradientMetricsCallback(log_file=os.path.join(stage4_dir, "training_log.jsonl"), plot_dir=stage4_dir)]
        )

        ckpt = get_latest_checkpoint(stage4_dir)
        trainer.train(resume_from_checkpoint=ckpt)
        model.save_pretrained(os.path.join(stage4_dir, "final_model"))
        
        del model, trainer, ds
        gc.collect()
        torch.cuda.empty_cache()
    return os.path.join(stage4_dir, "final_model")


def run_stage5_dpo(model_type, tokenizer, base_dir, stage4_model_path):
    if not os.path.exists(os.path.join(base_dir, "final_model", "model.safetensors")):
        print("=== Starting Stage 5: Direct Preference Optimization (DPO) ===")
        stage5_dir = os.path.join(base_dir, "Stage5")
        os.makedirs(stage5_dir, exist_ok=True)

        ConfigClass, ModelClass = get_model_classes(model_type)
        config = ConfigClass.from_pretrained(stage4_model_path)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        model = ModelClass.from_pretrained(stage4_model_path, config=config).to(dtype)
        ref_model = ModelClass.from_pretrained(stage4_model_path, config=config).to(dtype)
        ref_model.eval()
        for param in ref_model.parameters(): param.requires_grad = False

        ds = load_dataset("../Dolci-Think-DPO-32B", split="train").map(format_dpo_dataset, desc="Formatting DPO dataset")

        args = DPOConfig(
            # num_train_epochs=2,
            max_steps=3,
            output_dir=stage5_dir, per_device_train_batch_size=1,
            gradient_accumulation_steps=4, learning_rate=8.0e-8, lr_scheduler_type="linear", warmup_ratio=0.1,
            logging_steps=1, save_steps=1, report_to="none", bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(), gradient_checkpointing=True, optim="adamw_torch_fused",
            # beta=5.0, max_length=16384, max_prompt_length=2048
            beta=5.0, max_length=2048, max_prompt_length=1024
        )

        trainer = DPOTrainer(
            model=model, ref_model=ref_model, args=args, train_dataset=ds, processing_class=tokenizer,
            callbacks=[GradientMetricsCallback(log_file=os.path.join(stage5_dir, "training_log.jsonl"), plot_dir=stage5_dir)]
        )

        ckpt = get_latest_checkpoint(stage5_dir)
        trainer.train(resume_from_checkpoint=ckpt)
        model.save_pretrained(os.path.join(stage5_dir, "final_model"))
    return os.path.join(stage5_dir, "final_model")
