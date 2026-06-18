from transformers import AutoTokenizer
from pipelines import (
    run_stage1_pretraining,
    run_stage2_midtraining,
    run_stage3_long_context,
    run_stage4_sft,
    run_stage5_dpo,
    run_stage6_rlvr
)

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

    # Pre-training Stages
    stage1_model = run_stage1_pretraining(model_type, tokenizer, pretrain_dir)
    stage2_model = run_stage2_midtraining(model_type, tokenizer, pretrain_dir, stage1_model)
    stage3_model = run_stage3_long_context(model_type, tokenizer, pretrain_dir, stage2_model)

    # Post-training Stages
    stage4_model = run_stage4_sft(model_type, tokenizer, posttrain_dir, stage3_model)
    stage5_model = run_stage5_dpo(model_type, tokenizer, posttrain_dir, stage4_model)
    run_stage6_rlvr(model_type, tokenizer, posttrain_dir, stage5_model)

if __name__ == "__main__":
    main()
