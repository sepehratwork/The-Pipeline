import os
import json
import shutil
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset
from models import get_model_classes
from rl_algorithms import get_rl_algorithm
from utils import generate_completions, get_resume_state, get_latest_checkpoint, cleanup_checkpoints, clear_all_checkpoints


def run_stage6_rlvr(model_type, tokenizer, base_dir, stage5_model_path, rl_algo_name="grpo"):
    print(f"=== Starting Stage 6: RLVR with {rl_algo_name.upper()} ===")
    stage6_dir = os.path.join(base_dir, "Stage6")
    os.makedirs(stage6_dir, exist_ok=True)
    log_file = os.path.join(stage6_dir, "training_log.jsonl")

    ds = load_dataset("../Dolci-Think-RL-32B", split="train")
    ConfigClass, ModelClass = get_model_classes(model_type)
    
    config = ConfigClass.from_pretrained(stage5_model_path)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Robust resumption loop for custom training loop
    while True:
        ckpt_dir = get_latest_checkpoint(stage6_dir)
        if ckpt_dir:
            print(f"Attempting to resume RLVR from {ckpt_dir}")
            try:
                model = ModelClass.from_pretrained(ckpt_dir, config=config).to(dtype).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-6, fused=torch.cuda.is_available())
                opt_path = os.path.join(ckpt_dir, "optimizer.pt")
                if os.path.exists(opt_path):
                    optimizer.load_state_dict(torch.load(opt_path))
                start_step = get_resume_state(log_file) + 1
                break
            except Exception as e:
                print(f"Failed to load checkpoint {ckpt_dir}: {e}. Deleting and trying previous.")
                shutil.rmtree(ckpt_dir, ignore_errors=True)
        else:
            print("No valid checkpoint found. Starting training from the beginning.")
            model = ModelClass.from_pretrained(stage5_model_path, config=config).to(dtype).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-6, fused=torch.cuda.is_available())
            start_step = 0
            break

    ref_model = ModelClass.from_pretrained(stage5_model_path, config=config).to(dtype).to(device)
    model.gradient_checkpointing = True
    ref_model.eval()
    for param in ref_model.parameters(): param.requires_grad = False

    rl_algo = get_rl_algorithm(rl_algo_name)

    max_steps, group_size, gradient_accumulation_steps = 4, 2, 4
    max_prompt_length, max_completion_length = 1024, 2048

    steps_list, variances, entropies, means, losses, flops_list = [], [], [], [], [], []
    total_flops = 0

    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    steps_list.append(data['step'])
                    variances.append(data['variance'])
                    entropies.append(data['entropy'])
                    means.append(data['mean'])
                    losses.append(data['loss'])
                    flops_list.append(data.get('flops', 0))
                    total_flops = data.get('flops', 0)
            f.close()

    model.train()
    optimizer.zero_grad()

    for step in range(start_step, max_steps):
        example = ds[step % len(ds)]
        
        if "prompt" in example: prompt_text = example["prompt"]
        elif "messages" in example:
            msgs = example["messages"][:-1] if len(example["messages"]) > 0 and example["messages"][-1]["role"] == "assistant" else example["messages"]
            prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else: prompt_text = str(example)

        ground_truth = example.get("ground_truth", example.get("answer", ""))
        if not ground_truth and "messages" in example and example["messages"][-1]["role"] == "assistant":
            ground_truth = example["messages"][-1]["content"]

        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_length).to(device)
        input_ids = inputs.input_ids.repeat(group_size, 1)
        attention_mask = inputs.attention_mask.repeat(group_size, 1)

        completions = generate_completions(model, input_ids, attention_mask, max_completion_length, tokenizer.pad_token_id, tokenizer.eos_token_id)
        prompt_len = input_ids.size(1)
        decoded_completions = tokenizer.batch_decode(completions, skip_special_tokens=True)

        rewards = []
        for comp in decoded_completions:
            reward = 0.5 if "<think>" in comp and "</think>" in comp else 0.0
            if ground_truth and str(ground_truth).lower() in comp.lower(): reward += 1.0
            rewards.append(reward)

        rewards = torch.tensor(rewards, dtype=dtype, device=device)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        full_ids = torch.cat([input_ids, completions], dim=1)
        full_mask = torch.cat([attention_mask, (completions != tokenizer.pad_token_id).long()], dim=1)

        policy_outputs = model(input_ids=full_ids, attention_mask=full_mask)
        policy_logprobs = F.log_softmax(policy_outputs.logits[:, prompt_len-1:-1, :], dim=-1)
        policy_token_logprobs = torch.gather(policy_logprobs, 2, completions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            ref_outputs = ref_model(input_ids=full_ids, attention_mask=full_mask)
            ref_logprobs = F.log_softmax(ref_outputs.logits[:, prompt_len-1:-1, :], dim=-1)
            ref_token_logprobs = torch.gather(ref_logprobs, 2, completions.unsqueeze(-1)).squeeze(-1)

        comp_mask = (completions != tokenizer.pad_token_id).float()
        loss = rl_algo.compute_loss(policy_token_logprobs, ref_token_logprobs, advantages, comp_mask) / gradient_accumulation_steps
        loss.backward()
        
        loss_val = loss.item() * gradient_accumulation_steps
        
        N, P = full_ids.size(1) * group_size, sum(p.numel() for p in model.parameters())
        total_flops += 8 * N * P + (2 * max_completion_length * group_size * P)

        del policy_outputs, policy_logprobs, policy_token_logprobs
        del ref_outputs, ref_logprobs, ref_token_logprobs
        del full_ids, full_mask, loss
        torch.cuda.empty_cache()

        if (step + 1) % gradient_accumulation_steps == 0:
            total_elements, sum_grads, sum_sq_grads, sum_abs_grads = 0, 0.0, 0.0, 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad = p.grad.float()
                    total_elements += grad.numel()
                    sum_grads += grad.sum().item()
                    sum_sq_grads += (grad ** 2).sum().item()
                    sum_abs_grads += grad.abs().sum().item()

            if total_elements > 0:
                mean = sum_grads / total_elements
                var = (sum_sq_grads / total_elements) - (mean ** 2)
                entropy = 0.0
                sum_abs_grads += 1e-8
                for p in model.parameters():
                    if p.grad is not None:
                        prob = p.grad.float().abs() / sum_abs_grads
                        prob = prob[prob > 0]
                        entropy -= (prob * torch.log(prob)).sum().item()
            else:
                mean, var, entropy = 0.0, 0.0, 0.0

            optimizer.step()
            optimizer.zero_grad()

            steps_list.append(step)
            variances.append(var)
            entropies.append(entropy)
            means.append(mean)
            losses.append(loss_val)
            flops_list.append(total_flops)

            with open(log_file, 'a') as f:
                f.write(json.dumps({'step': step, 'variance': var, 'entropy': entropy, 'mean': mean, 'loss': loss_val, 'flops': total_flops}) + '\n')
                f.close()

            plt.figure(figsize=(25, 5))
            for i, (data, title, color) in enumerate(zip(
                [variances, entropies, means, losses, flops_list],
                ['Gradient Variance', 'Gradient Entropy', 'Gradient Mean', 'Training Loss', 'Cumulative FLOPs'],
                ['blue', 'green', 'orange', 'red', 'purple']
            )):
                plt.subplot(1, 5, i+1)
                plt.plot(steps_list, data, color=color)
                plt.title(title)
                plt.xlabel('Steps')
            plt.tight_layout()
            plt.savefig(os.path.join(stage6_dir, 'training_metrics.png'))
            plt.close()

            # Save checkpoint and optimizer state for resumption
            ckpt_path = os.path.join(stage6_dir, f"checkpoint-{step}")
            os.makedirs(ckpt_path, exist_ok=True)
            model.save_pretrained(ckpt_path)
            torch.save(optimizer.state_dict(), os.path.join(ckpt_path, "optimizer.pt"))
            
            # Keep only the last 2 checkpoints
            cleanup_checkpoints(stage6_dir, keep=2)

    model.save_pretrained(os.path.join(stage6_dir, "final_model"))
    clear_all_checkpoints(stage6_dir) # Remove all checkpoints after phase finishes
    print("=== Stage 6 Completed Successfully ===")
