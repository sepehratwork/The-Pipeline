import os
# Prevent CUDA memory fragmentation before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import shutil
import gc          
import inspect     
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from data import prepare_rlvr_dataset
from models import get_model_classes
from rl_algorithms import get_rl_algorithm, RL_ALGO_REGISTRY
from utils import generate_completions, get_resume_state, get_latest_checkpoint, cleanup_checkpoints, clear_all_checkpoints, save_to_hf_hub
from utils.callbacks import StageTimer


def compute_token_logprobs_chunked(logits, labels, chunk_size=128):
    """
    Computes token log probabilities in flat token chunks without non-contiguous 
    3D transpositions to eliminate large VRAM spikes.
    
    Args:
        logits: [B, S, V] (Bfloat16 or Float16)
        labels: [B, S] (Long)
        chunk_size: Number of tokens per cross-entropy chunk (default: 128)
    """
    orig_shape = labels.shape  # (B, S)
    vocab_size = logits.size(-1)
    
    # Flatten to 2D view (N, V) - avoids costly 3D transpose memory duplication
    flat_logits = logits.reshape(-1, vocab_size)
    flat_labels = labels.reshape(-1)
    total_tokens = flat_labels.size(0)

    if total_tokens <= chunk_size:
        chunk_lp = -F.cross_entropy(flat_logits.float(), flat_labels, reduction="none")
        return chunk_lp.view(orig_shape)

    logprobs_list = []
    for i in range(0, total_tokens, chunk_size):
        chunk_l = flat_logits[i:i + chunk_size]
        chunk_lab = flat_labels[i:i + chunk_size]
        # Memory per chunk is only: chunk_size * vocab_size * 4 bytes (~50-65 MB)
        chunk_lp = -F.cross_entropy(chunk_l.float(), chunk_lab, reduction="none")
        logprobs_list.append(chunk_lp)

    return torch.cat(logprobs_list, dim=0).view(orig_shape)


def run_stage6_rlvr(architecture, tokenizer, base_dir, stage5_model_path, hf_username, seq_len_scale_factor):
    width = 75
    print("\n" + "=" * width)
    print(f"🎯 STAGE 6: RLVR SUITE (ALL REINFORCEMENT ALGORITHMS) :: {architecture.upper()}".center(width))
    print("=" * width)
    print(f" • Input Preference Model : {stage5_model_path}")
    print(f" • Target RL Algorithms   : {list(RL_ALGO_REGISTRY.keys())}")
    print("=" * width + "\n")
    del width

    stage6_dir = os.path.join(base_dir, "Stage6")
    os.makedirs(stage6_dir, exist_ok=True)

    print("📥 Loading reasoning target dataset for RLVR...")
    ds = prepare_rlvr_dataset("Dolci-Think-RL-32B", tokenizer)
    print(f"✓ RLVR dataset ready with {len(ds):,} prompts/verification examples.")
    
    ConfigClass, ModelClass = get_model_classes(architecture)
    config = ConfigClass.from_pretrained(stage5_model_path)
    del ConfigClass
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  Execution Device: {device} | Precision: {dtype}")

    # Initialize standard AMP GradScaler if using float16
    use_scaler = (dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None
    del use_scaler

    # Initialize Cumulative Stage 6 Timer
    output_dir = os.path.dirname(base_dir)
    del base_dir
    global_timer = StageTimer(output_dir)
    del output_dir

    # Iterate over all available RL algorithms
    for rl_algo_name in RL_ALGO_REGISTRY.keys():
        print("\n" + "-" * 75)
        print(f"🔄 Starting RLVR Optimization with Algorithm: [{rl_algo_name.upper()}]".center(75))
        print("-" * 75)
        algo_dir = os.path.join(stage6_dir, rl_algo_name)
        os.makedirs(algo_dir, exist_ok=True)
        
        final_model_path = os.path.join(algo_dir, "final_model")
        repo_name = f"{architecture}_{rl_algo_name}"
        
        # Skip if this algorithm has already finished training
        is_already_saved = any(
            os.path.exists(os.path.join(final_model_path, fname))
            for fname in ["model.safetensors", "model.safetensors.index.json", "pytorch_model.bin", "pytorch_model.bin.index.json"]
        )
        if is_already_saved:
            del is_already_saved
            print(f"⏭️  [Skipped] Algorithm {rl_algo_name.upper()} already completed locally at {final_model_path}.")
            save_to_hf_hub(final_model_path, repo_name, hf_username=hf_username)
            del final_model_path, repo_name, algo_dir
            continue
        del is_already_saved

        stage_key = f"Stage 6: RLVR ({rl_algo_name.upper()})"
        start_t = global_timer.start_stage(stage_key)

        log_file = os.path.join(algo_dir, "training_log.jsonl")

        # Checkpoint loading
        while True:
            ckpt_dir = get_latest_checkpoint(algo_dir)
            if ckpt_dir:
                print(f"🔄 Resuming {rl_algo_name.upper()} from checkpoint: {ckpt_dir}")
                try:
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
                        print("✓ Optimizer state successfully restored.")
                    del opt_path
                    start_step = get_resume_state(log_file) + 1
                    print(f"✓ Resuming training loop at step {start_step}.")
                    del ckpt_dir
                    break
                except Exception as e:
                    print(f"⚠️ Failed to load checkpoint {ckpt_dir}: {e}. Deleting and checking previous...")
                    shutil.rmtree(ckpt_dir, ignore_errors=True)
                    del ckpt_dir, e
            else:
                del ckpt_dir
                print(f"🌱 No checkpoint found for {rl_algo_name.upper()}. Starting training from step 0.")
                model = ModelClass.from_pretrained(
                    stage5_model_path, 
                    config=config,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True
                ).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-6, fused=torch.cuda.is_available())
                start_step = 0
                break

        print(f"📦 Loading frozen reference model for {rl_algo_name.upper()}...")
        ref_model = ModelClass.from_pretrained(
            stage5_model_path, 
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to(device)
        
        if hasattr(model, "tie_weights"):
            model.config.tie_word_embeddings = True
            model.tie_weights()
            ref_model.config.tie_word_embeddings = True
            ref_model.tie_weights()

        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        
        ref_model.requires_grad_(False)
        ref_model.eval()

        rl_algo = get_rl_algorithm(rl_algo_name)

        max_steps, group_size, gradient_accumulation_steps = 1400, 8, 64
        max_steps = 2
        max_prompt_length, max_completion_length = 2048, 32768
        max_completion_length = int(max_completion_length / (2 ** seq_len_scale_factor))

        steps_list, variances, entropies, means, losses, flops_list = [], [], [], [], [], []
        tokens_per_sec_list = []
        tokens_per_sec_buffer = []
        vram_allocated_list = []
        vram_reserved_list = []
        learning_rates = []
        cot_lengths_list = []
        cot_lengths_buffer = []
        
        confidences_list = []
        confidences_buffer = []

        rewards_list = []
        reward_means = []
        reward_variances = []
        reward_entropies = []
        rewards_buffer = []

        advantages_list = []
        advantage_means = []
        advantage_variances = []
        advantage_entropies = []
        advantages_buffer = []

        total_flops = 0

        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        del line
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
                        confidences_list.append(data.get('confidence', 0.0))
                        rewards_list.append(data.get('reward', 0.0))
                        reward_means.append(data.get('reward_mean', 0.0))
                        reward_variances.append(data.get('reward_variance', 0.0))
                        reward_entropies.append(data.get('reward_entropy', 0.0))
                        advantages_list.append(data.get('advantage', 0.0))
                        advantage_means.append(data.get('advantage_mean', 0.0))
                        advantage_variances.append(data.get('advantage_variance', 0.0))
                        advantage_entropies.append(data.get('advantage_entropy', 0.0))
                        total_flops = data.get('flops', 0)
                        del data
                    else:
                        del line

        model.train()
        optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        vocab_size = model.config.vocab_size

        step_pbar = tqdm(range(start_step, max_steps), desc=f"🚀 RLVR [{rl_algo_name.upper()}]", unit="step", dynamic_ncols=True)
        del start_step
        for step in step_pbar:
            example = ds[step % len(ds)]
            prompt_text = example["prompt_text"]
            ground_truth = example["ground_truth"]
            del example

            inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_prompt_length).to(device)
            del prompt_text
            input_ids = inputs.input_ids.repeat(group_size, 1)
            attention_mask = inputs.attention_mask.repeat(group_size, 1)
            del inputs

            model.eval()
            model.config.use_cache = True
            
            start_gen_time = time.time()
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    completions = generate_completions(model, input_ids, attention_mask, max_completion_length, tokenizer.pad_token_id, tokenizer.eos_token_id)
            gen_duration = time.time() - start_gen_time
            del start_gen_time
            
            non_pad_tokens = (completions != tokenizer.pad_token_id).sum().item()
            tokens_per_sec = non_pad_tokens / gen_duration if gen_duration > 0 else 0.0
            del non_pad_tokens, gen_duration
            tokens_per_sec_buffer.append(tokens_per_sec)
            del tokens_per_sec

            model.train()
            model.config.use_cache = False
            
            # --- MEMORY OPTIMIZATION 1: Trim completions to longest active token length ---
            non_pad_positions = (completions != tokenizer.pad_token_id)
            actual_max_comp_len = non_pad_positions.sum(dim=1).max().item()
            actual_max_comp_len = max(actual_max_comp_len, 1)  # Guard against empty sequence
            completions = completions[:, :actual_max_comp_len]
            del non_pad_positions

            prompt_len = input_ids.size(1)
            decoded_completions = tokenizer.batch_decode(completions, skip_special_tokens=True)

            step_cot_lengths = []
            for comp in decoded_completions:
                if "<think>" in comp and "</think>" in comp:
                    start_idx = comp.find("<think>") + len("<think>")
                    end_idx = comp.find("</think>", start_idx)
                    if end_idx != -1:
                        cot_text = comp[start_idx:end_idx]
                        del start_idx, end_idx
                        cot_len = len(tokenizer.encode(cot_text, add_special_tokens=False))
                        del cot_text
                    else:
                        del start_idx, end_idx
                        cot_len = 0
                else:
                    cot_len = 0
                step_cot_lengths.append(cot_len)
                del comp, cot_len
            
            avg_cot_step = sum(step_cot_lengths) / len(step_cot_lengths) if step_cot_lengths else 0.0
            del step_cot_lengths
            cot_lengths_buffer.append(avg_cot_step)
            del avg_cot_step

            rewards = []
            for comp in decoded_completions:
                reward = 0.5 if "<think>" in comp and "</think>" in comp else 0.0
                if ground_truth and str(ground_truth).lower() in comp.lower(): 
                    reward += 1.0
                rewards.append(reward)
                del comp, reward
            del ground_truth, decoded_completions

            rewards = torch.tensor(rewards, dtype=dtype, device=device)
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            rewards_buffer.extend(rewards.detach().float().cpu().tolist())
            del rewards
            advantages_buffer.extend(advantages.detach().float().cpu().tolist())

            full_ids = torch.cat([input_ids, completions], dim=1)
            del input_ids
            comp_mask = (completions != tokenizer.pad_token_id).float()
            full_mask = torch.cat([attention_mask, comp_mask.long()], dim=1)
            del attention_mask

            safe_completions = torch.clamp(completions, min=0, max=vocab_size - 1)
            del completions

            torch.cuda.empty_cache()

            # --- MEMORY OPTIMIZATION 2: Micro-batched Reference Logprobs ---
            ref_logprobs_list = []
            ref_micro_batch_size = 2  # Feed 2 sequences at a time
            for mb_start in range(0, group_size, ref_micro_batch_size):
                mb_end = mb_start + ref_micro_batch_size
                mb_full_ids = full_ids[mb_start:mb_end]
                mb_full_mask = full_mask[mb_start:mb_end]
                mb_safe_comp = safe_completions[mb_start:mb_end]

                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=dtype):
                        ref_outputs = ref_model(input_ids=mb_full_ids, attention_mask=mb_full_mask)
                        # .contiguous() breaks the storage link to the prompt logits immediately
                        ref_logits_slice = ref_outputs.logits[:, prompt_len-1:-1, :].contiguous()
                        del ref_outputs
                        
                        mb_ref_lp = compute_token_logprobs_chunked(ref_logits_slice, mb_safe_comp, chunk_size=128)
                        del ref_logits_slice
                        ref_logprobs_list.append(mb_ref_lp)

                del mb_full_ids, mb_full_mask, mb_safe_comp
            
            ref_token_logprobs = torch.cat(ref_logprobs_list, dim=0)
            del ref_logprobs_list

            gc.collect()
            torch.cuda.empty_cache()

            # --- MEMORY OPTIMIZATION 3: Micro-batched Policy Forward Pass ---
            policy_logprobs_list = []
            policy_micro_batch_size = 2  # Match micro-batch size to keep activation memory tiny
            for mb_start in range(0, group_size, policy_micro_batch_size):
                mb_end = mb_start + policy_micro_batch_size
                mb_full_ids = full_ids[mb_start:mb_end]
                mb_full_mask = full_mask[mb_start:mb_end]
                mb_safe_comp = safe_completions[mb_start:mb_end]

                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    mb_policy_outputs = model(input_ids=mb_full_ids, attention_mask=mb_full_mask)
                    # .contiguous() frees prompt logits memory when mb_policy_outputs is deleted
                    mb_policy_logits = mb_policy_outputs.logits[:, prompt_len-1:-1, :].contiguous()
                    del mb_policy_outputs
                    
                    mb_policy_lp = compute_token_logprobs_chunked(mb_policy_logits, mb_safe_comp, chunk_size=128)
                    del mb_policy_logits
                    policy_logprobs_list.append(mb_policy_lp)

                del mb_full_ids, mb_full_mask, mb_safe_comp

            # Reconstruct the full policy logprobs tensor across the group (preserves autograd graph)
            policy_token_logprobs = torch.cat(policy_logprobs_list, dim=0)
            del policy_logprobs_list, safe_completions, full_mask

            with torch.no_grad():
                token_probs = torch.exp(policy_token_logprobs.detach())
                step_confidence = ((token_probs * comp_mask).sum() / (comp_mask.sum() + 1e-8)).item()
                del token_probs
            confidences_buffer.append(step_confidence)
            del step_confidence

            loss_kwargs = {}
            sig = inspect.signature(rl_algo.compute_loss)
            if "old_logprobs" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                loss_kwargs["old_logprobs"] = policy_token_logprobs.detach()
            del sig

            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                loss = rl_algo.compute_loss(
                    policy_token_logprobs, 
                    ref_token_logprobs, 
                    advantages, 
                    comp_mask, 
                    **loss_kwargs
                ) / gradient_accumulation_steps
            del policy_token_logprobs, ref_token_logprobs, advantages, comp_mask, loss_kwargs
            
            loss_val = loss.item() * gradient_accumulation_steps
            
            N, P = full_ids.size(1) * group_size, sum(p.numel() for p in model.parameters())
            del full_ids, prompt_len
            total_flops += 8 * N * P + (2 * actual_max_comp_len * group_size * P)
            del N, P, actual_max_comp_len

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            del loss

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == max_steps:
                if scaler is not None:
                    scaler.unscale_(optimizer)

                grads = [p.grad.view(-1).float() for p in model.parameters() if p.grad is not None]
                if grads:
                    all_grads = torch.cat(grads)
                    del grads
                    total_elements = all_grads.numel()
                    
                    if total_elements > 0:
                        sum_grads = all_grads.sum().item()
                        sum_sq_grads = (all_grads ** 2).sum().item()
                        
                        mean = sum_grads / total_elements
                        del sum_grads
                        var = (sum_sq_grads / total_elements) - (mean ** 2)
                        del sum_sq_grads
                        
                        abs_grads = all_grads.abs()
                        del all_grads
                        sum_abs_grads = abs_grads.sum().item() + 1e-8
                        prob = abs_grads / sum_abs_grads
                        del abs_grads, sum_abs_grads
                        prob = prob[prob > 0]
                        entropy = -torch.sum(prob * torch.log(prob)).item()
                        del prob
                    else:
                        del all_grads
                        mean, var, entropy = 0.0, 0.0, 0.0
                    del total_elements
                else:
                    del grads
                    mean, var, entropy = 0.0, 0.0, 0.0

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                lr = 0.0
                for param_group in optimizer.param_groups:
                    lr = param_group.get('lr', 0.0)
                    del param_group
                    break

                optimizer.zero_grad(set_to_none=True)

                avg_tokens_per_sec = sum(tokens_per_sec_buffer) / len(tokens_per_sec_buffer) if tokens_per_sec_buffer else 0.0
                tokens_per_sec_buffer = []

                avg_cot_len = sum(cot_lengths_buffer) / len(cot_lengths_buffer) if cot_lengths_buffer else 0.0
                cot_lengths_buffer = []

                avg_confidence = sum(confidences_buffer) / len(confidences_buffer) if confidences_buffer else 0.0
                confidences_buffer = []

                if rewards_buffer:
                    r_tensor = torch.tensor(rewards_buffer, dtype=torch.float32)
                    r_mean = r_tensor.mean().item()
                    r_var = torch.var(r_tensor, unbiased=False).item()
                    abs_r = r_tensor.abs()
                    del r_tensor
                    sum_abs_r = abs_r.sum().item() + 1e-8
                    prob_r = abs_r / sum_abs_r
                    del abs_r, sum_abs_r
                    prob_r = prob_r[prob_r > 0]
                    r_entropy = -torch.sum(prob_r * torch.log(prob_r)).item() if prob_r.numel() > 0 else 0.0
                    del prob_r
                else:
                    r_mean, r_var, r_entropy = 0.0, 0.0, 0.0
                rewards_buffer = []

                if advantages_buffer:
                    adv_tensor = torch.tensor(advantages_buffer, dtype=torch.float32)
                    adv_mean = adv_tensor.mean().item()
                    adv_var = torch.var(adv_tensor, unbiased=False).item()
                    abs_adv = adv_tensor.abs()
                    del adv_tensor
                    sum_abs_adv = abs_adv.sum().item() + 1e-8
                    prob_adv = abs_adv / sum_abs_adv
                    del abs_adv, sum_abs_adv
                    prob_adv = prob_adv[prob_adv > 0]
                    adv_entropy = -torch.sum(prob_adv * torch.log(prob_adv)).item() if prob_adv.numel() > 0 else 0.0
                    del prob_adv
                else:
                    adv_mean, adv_var, adv_entropy = 0.0, 0.0, 0.0
                advantages_buffer = []

                vram_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
                vram_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0.0
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                step_pbar.set_postfix({
                    "loss": f"{loss_val:.4f}",
                    "tok/s": f"{avg_tokens_per_sec:.1f}",
                    "cot": f"{avg_cot_len:.0f}tok",
                    "conf": f"{avg_confidence:.3f}",
                    "rew": f"{r_mean:.2f}",
                    "vram": f"{vram_allocated:.2f}GB",
                    "lr": f"{lr:.1e}"
                })

                tqdm.write(
                    f"[{rl_algo_name.upper()}] Step {step:02d} | "
                    f"Loss: {loss_val:.6f} | Var: {var:.4e} | Ent: {entropy:.4f} | "
                    f"Speed: {avg_tokens_per_sec:.1f} tok/s | CoT: {avg_cot_len:.1f} tok | "
                    f"Conf: {avg_confidence:.4f} | Rew: {r_mean:.4f} | Adv: {adv_mean:.4f} | VRAM: {vram_allocated:.2f}GB"
                )

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
                confidences_list.append(avg_confidence)
                rewards_list.append(r_mean)
                reward_means.append(r_mean)
                reward_variances.append(r_var)
                reward_entropies.append(r_entropy)
                advantages_list.append(adv_mean)
                advantage_means.append(adv_mean)
                advantage_variances.append(adv_var)
                advantage_entropies.append(adv_entropy)

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
                        'cot_length': avg_cot_len,
                        'confidence': avg_confidence,
                        'reward': r_mean,
                        'reward_mean': r_mean,
                        'reward_variance': r_var,
                        'reward_entropy': r_entropy,
                        'advantage': adv_mean,
                        'advantage_mean': adv_mean,
                        'advantage_variance': adv_var,
                        'advantage_entropy': adv_entropy
                    }) + '\n')
                del var, entropy, mean, loss_val, avg_tokens_per_sec, vram_allocated, vram_reserved
                del lr, avg_cot_len, avg_confidence, r_mean, r_var, r_entropy, adv_mean, adv_var, adv_entropy

                plot_data = [
                    variances, entropies, means, losses, flops_list, 
                    tokens_per_sec_list, vram_allocated_list, learning_rates, cot_lengths_list,
                    confidences_list, rewards_list, reward_means, reward_variances, reward_entropies,
                    advantages_list, advantage_means, advantage_variances, advantage_entropies
                ]
                plot_titles = [
                    'Gradient Variance', 'Gradient Entropy', 'Gradient Mean', 'Training Loss', 'Cumulative FLOPs',
                    'Inference Tokens/sec', 'Peak VRAM (GB)', 'Learning Rate', 'CoT Length (Tokens)',
                    'Model Confidence', 'Reward', 'Reward Mean', 'Reward Variance', 'Reward Entropy',
                    'Advantage', 'Advantage Mean', 'Advantage Variance', 'Advantage Entropy'
                ]
                plot_colors = [
                    'blue', 'green', 'orange', 'red', 'purple', 
                    'brown', 'magenta', 'cyan', 'olive',
                    'teal', 'gold', 'darkorange', 'salmon', 'crimson',
                    'deepskyblue', 'steelblue', 'navy', 'indigo'
                ]

                plt.figure(figsize=(5 * len(plot_data), 5))
                for i, (data, title, color) in enumerate(zip(plot_data, plot_titles, plot_colors)):
                    plt.subplot(1, len(plot_data), i+1)
                    plt.plot(steps_list, data, color=color)
                    if title == 'Peak VRAM (GB)' and len(vram_reserved_list) > 0:
                        plt.plot(steps_list, vram_reserved_list, color='purple', linestyle='--', label='Reserved')
                        plt.legend()
                    plt.title(title)
                    plt.xlabel('Steps')
                    del i, data, title, color
                del plot_data, plot_titles, plot_colors
                plt.tight_layout()
                plt.savefig(os.path.join(algo_dir, 'training_metrics.png'))
                plt.close()

                ckpt_path = os.path.join(algo_dir, f"checkpoint-{step}")
                os.makedirs(ckpt_path, exist_ok=True)
                if hasattr(model, "tie_weights"):
                    model.config.tie_word_embeddings = True
                    model.tie_weights()

                try:
                    model.save_pretrained(ckpt_path, safe_serialization=True)
                except RuntimeError:
                    model.save_pretrained(ckpt_path, safe_serialization=False)

                tokenizer.save_pretrained(ckpt_path)
                if hasattr(model, "generation_config") and model.generation_config is not None:
                    model.generation_config.save_pretrained(ckpt_path)

                torch.save(optimizer.state_dict(), os.path.join(ckpt_path, "optimizer.pt"))
                del ckpt_path
                
                cleanup_checkpoints(algo_dir, keep=2)
            else:
                del loss_val

        del step_pbar
        if 'step' in locals():
            del step
        del max_steps, group_size, gradient_accumulation_steps
        del max_prompt_length, max_completion_length
        del steps_list, variances, entropies, means, losses, flops_list
        del tokens_per_sec_list, tokens_per_sec_buffer
        del vram_allocated_list, vram_reserved_list
        del learning_rates, cot_lengths_list, cot_lengths_buffer
        del confidences_list, confidences_buffer
        del rewards_list, reward_means, reward_variances, reward_entropies, rewards_buffer
        del advantages_list, advantage_means, advantage_variances, advantage_entropies, advantages_buffer
        del total_flops, vocab_size, log_file

        print(f"💾 Saving final RLVR {rl_algo_name.upper()} model to: {final_model_path}...")
        os.makedirs(final_model_path, exist_ok=True)
        if hasattr(model, "tie_weights"):
            model.config.tie_word_embeddings = True
            model.tie_weights()

        try:
            model.save_pretrained(final_model_path, safe_serialization=True)
        except RuntimeError:
            model.save_pretrained(final_model_path, safe_serialization=False)

        tokenizer.save_pretrained(final_model_path)
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.save_pretrained(final_model_path)

        clear_all_checkpoints(algo_dir)
        del algo_dir
        print(f"✓ {rl_algo_name.upper()} Training Completed Successfully.")
        
        del model, ref_model, optimizer, rl_algo
        gc.collect()
        torch.cuda.empty_cache()

        print(f"🚀 Publishing RLVR '{repo_name}' to Hugging Face Hub...")
        save_to_hf_hub(final_model_path, repo_name, hf_username=hf_username)
        del final_model_path, repo_name

        global_timer.end_stage(stage_key, start_t)
        del stage_key, start_t

    del rl_algo_name
    del architecture, tokenizer, stage5_model_path, hf_username, seq_len_scale_factor
    del stage6_dir, ds, ModelClass, config, dtype, device, scaler, global_timer

    print("\n" + "=" * 75)
    print("✅ Stage 6 Completed Successfully for All Algorithms!".center(75))
    print("=" * 75 + "\n")