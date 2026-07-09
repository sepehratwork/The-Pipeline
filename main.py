import os
from transformers import AutoTokenizer
from pipeline import (
    run_stage1_pretraining,
    run_stage2_midtraining,
    run_stage3_long_context,
    run_stage4_sft,
    run_stage5_dpo,
    run_stage6_rlvr
)
from pipeline.evaluation import evaluate_base_model, evaluate_post_trained_model

def main():
    # --- CONFIGURATION ---
    # Choose your LLM Judge API here: "gemini" or "cloudflare"
    JUDGE_API_CHOICE = "cloudflare" 
    # ---------------------

    model_type = "olmo3"
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}"
            "{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + '<|endoftext|>\n' }}"
            "{% else %}{{ '<|' + message['role'] + '|>\n' + message['content'] + '\n' }}{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
        )

    pretrain_dir = "../ModelsCheckpoints/OLMo3/Pre-Training"
    posttrain_dir = "../ModelsCheckpoints/OLMo3/Post-Training"

    # Ensure report directory exists
    os.makedirs("reports", exist_ok=True)

    # ==========================================
    # Pre-training Stages & OLMES Evaluation
    # ==========================================
    
    stage1_model = run_stage1_pretraining(model_type, tokenizer, pretrain_dir)
    evaluate_base_model(stage1_model, tokenizer, "reports/Stage1_Pretraining")

    stage2_model = run_stage2_midtraining(model_type, tokenizer, pretrain_dir, stage1_model)
    evaluate_base_model(stage2_model, tokenizer, "reports/Stage2_Midtraining")

    stage3_model = run_stage3_long_context(model_type, tokenizer, pretrain_dir, stage2_model)
    evaluate_base_model(stage3_model, tokenizer, "reports/Stage3_LongContext")

    # ==========================================
    # Post-training Stages & OLMo 3 Evaluation
    # ==========================================
    
    stage4_model = run_stage4_sft(model_type, tokenizer, posttrain_dir, stage3_model)
    evaluate_post_trained_model(stage4_model, tokenizer, "reports/Stage4_SFT", judge_api=JUDGE_API_CHOICE)

    stage5_model = run_stage5_dpo(model_type, tokenizer, posttrain_dir, stage4_model)
    evaluate_post_trained_model(stage5_model, tokenizer, "reports/Stage5_DPO", judge_api=JUDGE_API_CHOICE)

    stage6_model = run_stage6_rlvr(model_type, tokenizer, posttrain_dir, stage5_model)
    evaluate_post_trained_model(stage6_model, tokenizer, "reports/Stage6_RLVR", judge_api=JUDGE_API_CHOICE)

    print("Pipeline and all evaluations completed successfully!")

if __name__ == "__main__":
    main()