import os
from transformers import Trainer, TrainingArguments
from models import get_model_classes
from data import load_stage_dataset
from utils import GradientMetricsCallback, get_latest_checkpoint

def _run_pretrain_stage(stage_name, model_type, tokenizer, dataset_path, seq_len, output_dir, config_kwargs, train_args_kwargs, resume_model_path=None):
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

    ds = load_stage_dataset(dataset_path, tokenizer, seq_len=seq_len)

    args = TrainingArguments(
        output_dir=output_dir,
        report_to="none",
        fp16=True,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        **train_args_kwargs
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        callbacks=[GradientMetricsCallback(log_file=os.path.join(output_dir, "training_log.jsonl"), plot_dir=output_dir)]
    )

    ckpt = get_latest_checkpoint(output_dir)
    trainer.train(resume_from_checkpoint=ckpt)
    model.save_pretrained(os.path.join(output_dir, "final_model"))
    return os.path.join(output_dir, "final_model")

def run_stage1_pretraining(model_type, tokenizer, base_dir):
    return _run_pretrain_stage(
        "Stage 1: Pretraining", model_type, tokenizer, "dolma3_mix-150B-1025", 2048,
        os.path.join(base_dir, "Stage1"),
        {"max_position_embeddings": 2048, "use_yarn": False},
        {"max_steps": 10, "per_device_train_batch_size": 1, "learning_rate": 3e-4, "logging_steps": 1, "save_steps": 5}
    )

def run_stage2_midtraining(model_type, tokenizer, base_dir, stage1_model_path):
    return _run_pretrain_stage(
        "Stage 2: Midtraining", model_type, tokenizer, "dolma3_dolmino_mix-100B-1125", 2048,
        os.path.join(base_dir, "Stage2"),
        {},
        {"max_steps": 10, "per_device_train_batch_size": 1, "learning_rate": 2e-4, "logging_steps": 1, "save_steps": 5},
        resume_model_path=stage1_model_path
    )

def run_stage3_long_context(model_type, tokenizer, base_dir, stage2_model_path):
    return _run_pretrain_stage(
        "Stage 3: Long-context Extension", model_type, tokenizer, "dolma3_longmino_mix-100B-1125", 4096,
        os.path.join(base_dir, "Stage3"),
        {"max_position_embeddings": 4096, "use_yarn": True},
        {"max_steps": 10, "per_device_train_batch_size": 1, "learning_rate": 2e-4, "logging_steps": 1, "save_steps": 5},
        resume_model_path=stage2_model_path
    )
