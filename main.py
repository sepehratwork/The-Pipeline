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
from pipeline.evaluation import run_olmes_evaluation

def main():
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
    eval_dir = "../EvaluationResults"

    # API keys and tokens for OLMES evaluations (retrieved from system environment)
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    hf_token = os.environ.get("HF_TOKEN")

    # Pre-training Stages
    stage1_model = run_stage1_pretraining(model_type, tokenizer, pretrain_dir)
    run_olmes_evaluation(
        model_path=stage1_model, 
        output_dir=os.path.join(eval_dir, "Stage1_Pretraining"), 
        stage="pretraining",
        openai_api_key=openai_api_key,
        hf_token=hf_token
    )

    stage2_model = run_stage2_midtraining(model_type, tokenizer, pretrain_dir, stage1_model)
    run_olmes_evaluation(
        model_path=stage2_model, 
        output_dir=os.path.join(eval_dir, "Stage2_Midtraining"), 
        stage="midtraining",
        openai_api_key=openai_api_key,
        hf_token=hf_token
    )

    stage3_model = run_stage3_long_context(model_type, tokenizer, pretrain_dir, stage2_model)
    run_olmes_evaluation(
        model_path=stage3_model, 
        output_dir=os.path.join(eval_dir, "Stage3_LongContext"), 
        stage="long_context",
        openai_api_key=openai_api_key,
        hf_token=hf_token
    )

    # Post-training Stages
    stage4_model = run_stage4_sft(model_type, tokenizer, posttrain_dir, stage3_model)
    run_olmes_evaluation(
        model_path=stage4_model, 
        output_dir=os.path.join(eval_dir, "Stage4_SFT"), 
        stage="sft",
        openai_api_key=openai_api_key,
        hf_token=hf_token
    )

    stage5_model = run_stage5_dpo(model_type, tokenizer, posttrain_dir, stage4_model)
    run_olmes_evaluation(
        model_path=stage5_model, 
        output_dir=os.path.join(eval_dir, "Stage5_DPO"), 
        stage="dpo",
        openai_api_key=openai_api_key,
        hf_token=hf_token
    )

    stage6_model = run_stage6_rlvr(model_type, tokenizer, posttrain_dir, stage5_model)
    run_olmes_evaluation(
        model_path=stage6_model, 
        output_dir=os.path.join(eval_dir, "Stage6_RLVR"), 
        stage="rlvr",
        openai_api_key=openai_api_key,
        hf_token=hf_token
    )

if __name__ == "__main__":
    main()