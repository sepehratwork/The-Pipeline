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


def main():

    architecture = "olmo3"
    hf_username = "SepehrKerachi"
    pretrain_dir = "../ModelsCheckpoints/OLMo3/Pre-Training"
    posttrain_dir = "../ModelsCheckpoints/OLMo3/Post-Training"

    # ==========================================
    # Setting up the tokenizer
    # ==========================================
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

    # ==========================================
    # Pre-training Stages & OLMES Evaluation
    # ==========================================
    
    stage1_model = run_stage1_pretraining(architecture, tokenizer, pretrain_dir)

    stage2_model = run_stage2_midtraining(architecture, tokenizer, pretrain_dir, stage1_model)

    stage3_model = run_stage3_long_context(architecture, tokenizer, pretrain_dir, stage2_model, hf_username=hf_username)

    # ==========================================
    # Post-training Stages & OLMo 3 Evaluation
    # ==========================================
    
    stage4_model = run_stage4_sft(architecture, tokenizer, posttrain_dir, stage3_model)

    stage5_model = run_stage5_dpo(architecture, tokenizer, posttrain_dir, stage4_model)

    stage6_model = run_stage6_rlvr(architecture, tokenizer, posttrain_dir, stage5_model, hf_username=hf_username)

    print("Pipeline and all evaluations completed successfully!")

if __name__ == "__main__":
    main()