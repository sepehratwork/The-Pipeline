import os
import json
import shutil
import gc          # Added GC to perform final cleans on algorithm switch
import inspect     # Safely inspect method signatures
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

from data import prepare_rlvr_dataset
from models import get_model_classes
from rl_algorithms import get_rl_algorithm, RL_ALGO_REGISTRY
from utils import generate_completions, get_resume_state, get_latest_checkpoint, cleanup_checkpoints, clear_all_checkpoints
from utils.callbacks import StageTimer


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

    # Initialize Cumulative Stage 6 Timer
    global_timer = StageTimer(base_dir)

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

        # Start Stage Timing for the active algorithm
        stage_key = f"Stage 6: RLVR ({algo_name.upper()})"
        start_t = global_timer.start_stage(stage_key)

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
        
        # Configure non-reentrant gradient checkpointing and input requirements to allow backward tracking
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        
        # Free up reference model gradient tracking
        ref_model.requires_grad_(False)
        ref_model.eval()

        rl_algo = get_rl_algorithm(algo_name)

        max_steps, group_size, gradient_accumulation_steps = 6, 2, 4
        max_prompt_length, max_completion_length = 1024, 2048

        steps_list, variances, entropies, means, losses, flops_list = [], [], [], [], [], []
        tokens_per_sec_list = []
        tokens_per_sec_buffer = []  # Accumulate tokens per sec for averaging across accumulation steps
        vram_allocated_list = []
        vram_reserved_list = []
        learning_rates = []  # Captured Learning Rate List
        cot_lengths_list = []  # Captured Step-Averaged CoT Length List
        cot_lengths_buffer = []  # Accumulate CoT lengths for averaging across accumulation steps
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
                        tokens_per_sec_list.append(data.get('tokens_per_sec', 0.0))
                        vram_allocated_list.append(data.get('vram_allocated', 0.0))
                        vram_reserved_list.append(data.get('vram_reserved', 0.0))
                        learning_rates.append(data.get('learning_rate', 0.0))
                        cot_lengths_list.append(data.get('cot_length', 0.0))
                        total_flops = data.get('flops', 0)

        # Set training mode first
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

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
            
            # Measure generation time for tokens per second calculation
            start_gen_time = time.time()
            with torch.no_grad():
                # Perform generation inside autocast context
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    completions = generate_completions(model, input_ids, attention_mask, max_completion_length, tokenizer.pad_token_id, tokenizer.eos_token_id)
            gen_duration = time.time() - start_gen_time
            
            # Calculate generated tokens per second (excluding padding tokens)
            non_pad_tokens = (completions != tokenizer.pad_token_id).sum().item()
            tokens_per_sec = non_pad_tokens / gen_duration if gen_duration > 0 else 0.0
            tokens_per_sec_buffer.append(tokens_per_sec)
            
            print(f"[{algo_name.upper()}] Step {step}: Generated {non_pad_tokens} tokens in {gen_duration:.2f}s ({tokens_per_sec:.2f} tokens/s)")

            model.train()
            model.config.use_cache = False
            
            # Flush generation and KV cache memory before training passes
            torch.cuda.empty_cache()
            
            prompt_len = input_ids.size(1)
            decoded_completions = tokenizer.batch_decode(completions, skip_special_tokens=True)

            # Calculate and store CoT lengths for this step's generated episodes
            step_cot_lengths = []
            for comp in decoded_completions:
                if "<think>" in comp and "</think>" in comp:
                    start_idx = comp.find("<think>") + len("<think>")
                    end_idx = comp.find("</think>", start_idx)
                    if end_idx != -1:
                        cot_text = comp[start_idx:end_idx]
                        # Measure sequence length using token count from the tokenizer
                        cot_len = len(tokenizer.encode(cot_text, add_special_tokens=False))
                    else:
                        cot_len = 0
                else:
                    cot_len = 0
                step_cot_lengths.append(cot_len)
            
            avg_cot_step = sum(step_cot_lengths) / len(step_cot_lengths) if step_cot_lengths else 0.0
            cot_lengths_buffer.append(avg_cot_step)

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

                # Optimised vector-wise calculation of gradient mean, variance, and entropy
                grads = [p.grad.view(-1).float() for p in model.parameters() if p.grad is not None]
                if grads:
                    all_grads = torch.cat(grads)
                    total_elements = all_grads.numel()
                    
                    if total_elements > 0:
                        sum_grads = all_grads.sum().item()
                        sum_sq_grads = (all_grads ** 2).sum().item()
                        
                        mean = sum_grads / total_elements
                        var = (sum_sq_grads / total_elements) - (mean ** 2)
                        
                        abs_grads = all_grads.abs()
                        sum_abs_grads = abs_grads.sum().item() + 1e-8
                        prob = abs_grads / sum_abs_grads
                        prob = prob[prob > 0]
                        entropy = -torch.sum(prob * torch.log(prob)).item()
                    else:
                        mean, var, entropy = 0.0, 0.0, 0.0
                else:
                    mean, var, entropy = 0.0, 0.0, 0.0

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step with scalability check
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                # Get learning rate from optimizer
                lr = 0.0
                for param_group in optimizer.param_groups:
                    lr = param_group.get('lr', 0.0)
                    break

                optimizer.zero_grad(set_to_none=True)

                # Average the tokens per second throughput over the accumulation step
                avg_tokens_per_sec = sum(tokens_per_sec_buffer) / len(tokens_per_sec_buffer) if tokens_per_sec_buffer else 0.0
                tokens_per_sec_buffer = []

                # Average the CoT length over the accumulation step
                avg_cot_len = sum(cot_lengths_buffer) / len(cot_lengths_buffer) if cot_lengths_buffer else 0.0
                cot_lengths_buffer = []

                # Capture peak VRAM consumption since last reset
                vram_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
                vram_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0.0
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                steps_list.append(step)
                variances.append(var)
                entropies.append(entropy)
                means.append(mean)
                losses.append(loss_val)
                flops_list.append(total_flops)
                tokens_per_sec_list.append(avg_tokens_per_sec)
                vram_allocated_list.append(vram_allocated)
                vram_reserved_list.append(vram_reserved)
                learning_rates.append(lr)
                cot_lengths_list.append(avg_cot_len)

                with open(log_file, 'a') as f:
                    f.write(json.dumps({
                        'step': step, 
                        'variance': var, 
                        'entropy': entropy, 
                        'mean': mean, 
                        'loss': loss_val, 
                        'flops': total_flops,
                        'tokens_per_sec': avg_tokens_per_sec,
                        'vram_allocated': vram_allocated,
                        'vram_reserved': vram_reserved,
                        'learning_rate': lr,
                        'cot_length': avg_cot_len
                    }) + '\n')

                # Render metrics over 9 subplot panels (Added Learning Rate & CoT Length)
                plt.figure(figsize=(45, 5))
                for i, (data, title, color) in enumerate(zip(
                    [variances, entropies, means, losses, flops_list, tokens_per_sec_list, vram_allocated_list, learning_rates, cot_lengths_list],
                    ['Gradient Variance', 'Gradient Entropy', 'Gradient Mean', 'Training Loss', 'Cumulative FLOPs', 'Inference Tokens/sec', 'Peak VRAM (GB)', 'Learning Rate', 'CoT Length (Tokens)'],
                    ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'magenta', 'cyan', 'olive']
                )):
                    plt.subplot(1, 9, i+1)
                    plt.plot(steps_list, data, color=color)
                    if title == 'Peak VRAM (GB)' and len(vram_reserved_list) > 0:
                        plt.plot(steps_list, vram_reserved_list, color='purple', linestyle='--', label='Reserved')
                        plt.legend()
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

        # End timing for the algorithm stage
        global_timer.end_stage(stage_key, start_t)

    print("=== Stage 6 Completed Successfully ===")