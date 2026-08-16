import os
import argparse
import sys
import time

from huggingface_hub import login
from transformers import AutoTokenizer

from models import MODEL_REGISTRY
from pipeline import (
    run_stage1_pretraining,
    run_stage2_midtraining,
    run_stage3_long_context,
    run_stage4_sft,
    run_stage5_dpo,
    run_stage6_rlvr
)


def print_main_banner(architecture, hf_username, pretrain_dir, posttrain_dir):
    width = 80
    print("\n" + "=" * width)
    print("🚀  SMALL LANGUAGE MODEL (SLM) MULTI-STAGE TRAINING PIPELINE".center(width))
    print("=" * width)
    print(f" • Architecture       : {architecture}")
    print(f" • Hugging Face User  : {hf_username}")
    print(f" • Pre-Train Dir      : {pretrain_dir}")
    print(f" • Post-Train Dir     : {posttrain_dir}")
    print("=" * width + "\n")


def main(hf_token, architecture, hf_username = "SepehrKerachi"):

    pretrain_dir = f"/content/drive/MyDrive/Simulated/ModelsCheckpoints/{architecture}/Pre-Training"
    posttrain_dir = f"/content/drive/MyDrive/Simulated/ModelsCheckpoints/{architecture}/Post-Training"

    print_main_banner(architecture, hf_username, pretrain_dir, posttrain_dir)

    # ==========================================
    # Setting up the tokenizer
    # ==========================================
    print("🔑 [1/7] Authenticating with Hugging Face Hub...")
    login(token=hf_token)
    print("✓ Successfully authenticated with Hugging Face Hub.")

    print(f"📦 [2/7] Loading tokenizer for 'allenai/OLMo-2-1124-13B'...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-13B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("ℹ️  Tokenizer pad_token was None. Assigned eos_token as pad_token.")

    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}"
            "{% elif message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + '<|endoftext|>\n' }}"
            "{% else %}{{ '<|' + message['role'] + '|>\n' + message['content'] + '\n' }}{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
        )
        print("ℹ️  Applied default chat template to tokenizer.")

    print(f"✓ Tokenizer ready. Vocab size: {len(tokenizer):,} | Pad Token ID: {tokenizer.pad_token_id} | EOS Token ID: {tokenizer.eos_token_id}\n")

    # ==========================================
    # Pre-training Stages & OLMES Evaluation
    # ==========================================
    
    print("\n" + "#" * 80)
    print("  PHASE 1: PRE-TRAINING SUITE (Stages 1 to 3)".center(80))
    print("#" * 80 + "\n")

    stage1_model = run_stage1_pretraining(architecture, tokenizer, pretrain_dir)

    stage2_model = run_stage2_midtraining(architecture, tokenizer, pretrain_dir, stage1_model)

    stage3_model = run_stage3_long_context(architecture, tokenizer, pretrain_dir, stage2_model, hf_username=hf_username)

    # ==========================================
    # Post-training Stages & OLMo 3 Evaluation
    # ==========================================
    
    print("\n" + "#" * 80)
    print("  PHASE 2: POST-TRAINING SUITE (Stages 4 to 6)".center(80))
    print("#" * 80 + "\n")

    stage4_model = run_stage4_sft(architecture, tokenizer, posttrain_dir, stage3_model, hf_username=hf_username)

    stage5_model = run_stage5_dpo(architecture, tokenizer, posttrain_dir, stage4_model, hf_username=hf_username)

    stage6_model = run_stage6_rlvr(architecture, tokenizer, posttrain_dir, stage5_model, hf_username=hf_username)

    print("\n" + "=" * 80)
    print("🎉 Training pipeline completed successfully!".center(80))
    print("=" * 80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Train a language model in the size of 1 billion parameters in different architectures..."
        )

    parser.add_argument(
        "--architecture", "-a",
        type=str,
        default="olmo_3_test",
        help=f"Choose the name of the architecture: {list(MODEL_REGISTRY.keys())}"
    )

    parser.add_argument(
        "--hf-username", "-u",
        type=str,
        default="SepehrKerachi",
        help="Your username in hugging face"
    )

    parser.add_argument(
        "--hf-token", "-t",
        type=str,
        help="Your hugging face token"
    )

    args = parser.parse_args()

    main(
        architecture=args.architecture,
        hf_username=args.hf_username,
        hf_token=args.hf_token
    )