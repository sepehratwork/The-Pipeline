import os
import json
import shutil
import gc          # Added GC to perform final cleans on algorithm switch
import inspect     # Safely inspect method signatures
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data import prepare_rlvr_dataset
from models import get_model_classes
from rl_algorithms import get_rl_algorithm, RL_ALGO_REGISTRY
from utils import generate_completions, get_resume_state, get_latest_checkpoint, cleanup_checkpoints, clear_all_checkpoints


def run_stage6_rlvr(model_type, tokenizer, base_dir, stage5_model_path):
    print("=== Starting Stage 6: RLVR with ALL Algorithms ===")
    stage6_dir = os.path.join(base_dir, "Stage6")
    os.makedirs(stage6_dir, exist_ok=True)

    ds = prepare_rlvr_dataset("../Dolci-Think-RL-32B", tokenizer)
    
    ConfigClass, ModelClass = get_model_classes(model_type)
    config = ConfigClass.from_pretrained(stage5_model_path)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize standard AMP GradScaler if using float16
    use_scaler = (dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    # Iterate over all available RL algorithms
    for algo_name in RL_ALGO_REGISTRY.keys():
        print(f"\n--- Starting RLVR Training with {algo_name.upper()} ---")
        algo_dir = os.path.join(stage6_dir, algo_name)
        os.makedirs(algo_dir, exist_ok=True)
        
        final_model_path = os.path.join(algo_dir, "final_model")
        
        # Skip if this algorithm has already finished training
        if os.path.exists(final_model_path):
            print(f"Algorithm {algo_name.upper()} already completed. Skipping to next.")
            continue

        log_file = os.path.join(algo_dir, "training_log.jsonl")

        # Robust resumption loop for custom training loop per algorithm
        while True:
            ckpt_dir = get_latest_checkpoint(algo_dir)
            if ckpt_dir:
                print(f"Attempting to resume {algo_name.upper()} from {ckpt_dir}")
                try:
                    # Optimized model loading
                    model = ModelClass.from_pretrained(
                        ckpt_dir, 
                        config=config,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=True
                    ).to(device)
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
                print(f"No valid checkpoint found for {algo_name.upper()}. Starting training from the beginning.")
                model = ModelClass.from_pretrained(
                    stage5_model_path, 
                    config=config,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True
                ).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-6, fused=torch.cuda.is_available())
                start_step = 0
                break

        ref_model = ModelClass.from_pretrained(
            stage5_model_path, 
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to(device)
        
        # Configure non-reentrant gradient checkpointing
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
        # Free up reference model gradient tracking
        ref_model.requires_grad_(False)
        ref_model.eval()

        rl_algo = get_rl_algorithm(algo_name)

        max_steps, group_size, gradient_accumulation_steps = 6, 2, 4
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

        # Set training mode first
        model.train()
        optimizer.zero_grad(set_to_none=True)

        vocab_size = model.config.vocab_size

        for step in range(start_step, max_steps):
            example = ds[step % len(ds)]
            
            # Directly use the pre-extracted fields from the cached dataset
            prompt_text = example["prompt_text"]
            ground_truth = example["ground_truth"]

            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_length).to(device)
            input_ids = inputs.input_ids.repeat(group_size, 1)
            attention_mask = inputs.attention_mask.repeat(group_size, 1)

            # model.eval() is correctly set after model.train()
            model.eval()
            model.config.use_cache = True
            with torch.no_grad():
                # Perform generation inside autocast context
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    completions = generate_completions(model, input_ids, attention_mask, max_completion_length, tokenizer.pad_token_id, tokenizer.eos_token_id)
            model.train()
            model.config.use_cache = False
            
            # Flush generation and KV cache memory before training passes
            torch.cuda.empty_cache()
            
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

            # Clamp target token indices to be strictly within vocabulary range to avoid IndexErrors.
            # This is mathematically sound as pad tokens are masked out by comp_mask during loss computation.
            safe_completions = torch.clamp(completions, min=0, max=vocab_size - 1)

            # 1. Compute reference token logprobs first (no gradients needed)
            # This avoids keeping reference forward activations in memory during the policy forward pass
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    ref_outputs = ref_model(input_ids=full_ids, attention_mask=full_mask)
                    ref_logits = ref_outputs.logits[:, prompt_len-1:-1, :].float()
                    
                    # Memory optimization: use cross_entropy to avoid allocating massive [B, L, V] logprobs tensor
                    ref_token_logprobs = -F.cross_entropy(
                        ref_logits.transpose(1, 2), 
                        safe_completions, 
                        reduction="none"
                    )

            # Immediately delete reference variables and clean cache before policy forward
            del ref_outputs, ref_logits
            gc.collect()
            torch.cuda.empty_cache()

            # 2. Compute policy token logprobs (gradients needed)
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                policy_outputs = model(input_ids=full_ids, attention_mask=full_mask)
                policy_logits = policy_outputs.logits[:, prompt_len-1:-1, :].float()
                
                # Delete the original output wrapper early to free references
                del policy_outputs
                
                # Memory optimization: use cross_entropy to avoid allocating massive [B, L, V] logprobs tensor
                policy_token_logprobs = -F.cross_entropy(
                    policy_logits.transpose(1, 2), 
                    safe_completions, 
                    reduction="none"
                )

                comp_mask = (completions != tokenizer.pad_token_id).float()

                # Inspect compute_loss signature to dynamically pass old_logprobs if supported
                loss_kwargs = {}
                sig = inspect.signature(rl_algo.compute_loss)
                if "old_logprobs" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    loss_kwargs["old_logprobs"] = policy_token_logprobs.detach()

                loss = rl_algo.compute_loss(
                    policy_token_logprobs, 
                    ref_token_logprobs, 
                    advantages, 
                    comp_mask, 
                    **loss_kwargs
                ) / gradient_accumulation_steps
            
            loss_val = loss.item() * gradient_accumulation_steps
            
            # 3. Calculate FLOPs metrics before deleting any tensors
            N, P = full_ids.size(1) * group_size, sum(p.numel() for p in model.parameters())
            total_flops += 8 * N * P + (2 * max_completion_length * group_size * P)

            # 4. Clean up Python-side references to intermediate tensors before backward pass.
            # The autograd graph attached to `loss` keeps the required underlying tensors intact.
            del policy_logits, policy_token_logprobs
            del ref_token_logprobs, advantages, comp_mask
            del full_ids, full_mask, inputs, input_ids, attention_mask, completions, decoded_completions, rewards, safe_completions
            
            # Force clean up to maximize contiguous GPU memory for backward pass
            gc.collect()
            torch.cuda.empty_cache()

            # 5. Perform backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Free loss tensor reference as it is no longer needed
            del loss

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == max_steps:
                # If using a scaler, unscale the gradients before calculating metrics and clipping
                if scaler is not None:
                    scaler.unscale_(optimizer)

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

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step with scalability check
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                steps_list.append(step)
                variances.append(var)
                entropies.append(entropy)
                means.append(mean)
                losses.append(loss_val)
                flops_list.append(total_flops)

                with open(log_file, 'a') as f:
                    f.write(json.dumps({'step': step, 'variance': var, 'entropy': entropy, 'mean': mean, 'loss': loss_val, 'flops': total_flops}) + '\n')

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
                plt.savefig(os.path.join(algo_dir, 'training_metrics.png'))
                plt.close()

                ckpt_path = os.path.join(algo_dir, f"checkpoint-{step}")
                os.makedirs(ckpt_path, exist_ok=True)
                model.save_pretrained(ckpt_path)
                torch.save(optimizer.state_dict(), os.path.join(ckpt_path, "optimizer.pt"))
                
                cleanup_checkpoints(algo_dir, keep=2)

        model.save_pretrained(final_model_path)
        clear_all_checkpoints(algo_dir)
        print(f"=== {algo_name.upper()} Training Completed ===")
        
        # Free memory at the algorithm boundaries
        del model, ref_model, optimizer, rl_algo
        gc.collect()
        torch.cuda.empty_cache()

    print("=== Stage 6 Completed Successfully ===")
