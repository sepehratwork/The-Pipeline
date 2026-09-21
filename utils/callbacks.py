import os
import json
import time
import math
import torch
import matplotlib.pyplot as plt
from transformers import TrainerCallback


class StageTimer:
    """
    Timer utility to record the training time taken by each stage.
    Saves and accumulates times inside stage_times.json to support resumption.
    """
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.log_file = os.path.join(base_dir, "stage_times.json")
        os.makedirs(base_dir, exist_ok=True)

    def _load_times(self):
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_times(self, times):
        try:
            with open(self.log_file, "w") as f:
                json.dump(times, f, indent=4)
        except Exception as e:
            print(f"⚠️ [TIMER] Error saving stage times: {e}")

    def start_stage(self, stage_name):
        print(f"\n⏱️  [TIMER] >>> Initiating execution timer for: {stage_name}")
        return time.time()

    def end_stage(self, stage_name, start_time):
        elapsed = time.time() - start_time
        times = self._load_times()
        times[stage_name] = times.get(stage_name, 0.0) + elapsed
        self._save_times(times)

        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = elapsed % 60
        print(f"⏱️  [TIMER] <<< Completed {stage_name} in {hours:02d}h {minutes:02d}m {seconds:05.2f}s (Current run: {elapsed:.2f}s).")
        self.print_summary()

    def print_summary(self):
        times = self._load_times()
        if not times:
            return
        print("\n" + "┌" + "─" * 73 + "┐")
        print("│" + " 📊 CUMULATIVE TRAINING TIME BREAKDOWN".center(72) + "│")
        print("├" + "─" * 50 + "┬" + "─" * 22 + "┤")
        print(f"│ {'Stage Name':<48} │ {'Duration':<20} │")
        print("├" + "─" * 50 + "┼" + "─" * 22 + "┤")
        total_time = 0.0
        for stage, duration in times.items():
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = duration % 60
            dur_str = f"{hours:02d}h {minutes:02d}m {seconds:05.2f}s"
            print(f"│ {stage[:48]:<48} │ {dur_str:<20} │")
            total_time += duration

        tot_hours = int(total_time // 3600)
        tot_minutes = int((total_time % 3600) // 60)
        tot_seconds = total_time % 60
        tot_str = f"{tot_hours:02d}h {tot_minutes:02d}m {tot_seconds:05.2f}s"
        print("├" + "─" * 50 + "┴" + "─" * 22 + "┤")
        print(f"│ {'🌟 TOTAL ELAPSED TIME':<47}   {tot_str:<20} │")
        print("└" + "─" * 73 + "┘\n")


class GradientMetricsCallback(TrainerCallback):
    """
    Memory-efficient, zero-allocation gradient metrics callback.
    Tracks gradient mean, variance, Shannon empirical entropy, next-token prediction
    confidence, LR, loss, and VRAM.
    """
    def __init__(
        self, 
        log_file, 
        plot_dir, 
        model=None, 
        entropy_mode="distribution", 
        confidence_mode="top1",
        num_bins=100
    ):
        self.model = model
        self.optimizer = None
        self.log_file = log_file
        self.plot_dir = plot_dir
        self.entropy_mode = entropy_mode      # "distribution" or "coordinate"
        self.confidence_mode = confidence_mode  # "top1" (argmax token) or "target" (ground-truth label)
        self.num_bins = num_bins

        self.steps, self.variances, self.entropies, self.means, self.losses = [], [], [], [], []
        self.confidences, self.flops = [], []
        self.vram_allocated = []
        self.vram_reserved = []
        self.learning_rates = []
        os.makedirs(self.plot_dir, exist_ok=True)

        self._temp_mean = 0.0
        self._temp_var = 0.0
        self._temp_entropy = 0.0
        self._grad_captured = False

        # Next-token confidence accumulators across gradient accumulation micro-steps
        self._accumulated_conf = 0.0
        self._accumulated_conf_count = 0
        self._hook_handle = None

        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            self.steps.append(data['step'])
                            self.variances.append(data['variance'])
                            self.entropies.append(data['entropy'])
                            self.means.append(data['mean'])
                            self.losses.append(data['loss'])
                            self.confidences.append(data.get('confidence', 0.0))
                            self.flops.append(data.get('flops', 0))
                            self.vram_allocated.append(data.get('vram_allocated', 0.0))
                            self.vram_reserved.append(data.get('vram_reserved', 0.0))
                            self.learning_rates.append(data.get('learning_rate', 0.0))
            except Exception as e:
                print(f"⚠️ [METRICS] Could not read existing log file: {e}")

        if self.model is not None:
            self._register_hook(self.model)

    def _register_hook(self, model):
        """Attaches a forward hook to intercept logits without storing large tensors."""
        if self._hook_handle is not None:
            try:
                self._hook_handle.remove()
            except Exception:
                pass
            self._hook_handle = None

        try:
            # PyTorch >= 2.0 supports with_kwargs=True
            self._hook_handle = model.register_forward_hook(self._forward_hook, with_kwargs=True)
        except TypeError:
            # PyTorch < 2.0 fallback
            self._hook_handle = model.register_forward_hook(self._forward_hook_legacy)

    def _forward_hook(self, module, args, kwargs, output):
        attn_mask = kwargs.get("attention_mask", None) if isinstance(kwargs, dict) else None
        labels = kwargs.get("labels", None) if isinstance(kwargs, dict) else None
        self._calculate_confidence(module, output, attn_mask, labels)

    def _forward_hook_legacy(self, module, args, output):
        self._calculate_confidence(module, output, None, None)

    @torch.no_grad()
    def _calculate_confidence(self, module, output, attention_mask=None, labels=None):
        # Ignore forward passes during evaluation or non-training steps
        if not module.training:
            return

        logits = None
        if hasattr(output, "logits") and output.logits is not None:
            logits = output.logits
        elif isinstance(output, (tuple, list)):
            for item in output:
                if isinstance(item, torch.Tensor) and item.dim() == 3:
                    logits = item
                    break

        if logits is None:
            return

        logits_f = logits.detach().float()

        if self.confidence_mode == "target" and labels is not None:
            # Ground-truth next-token confidence: P(y_t | x_<t)
            shift_logits = logits_f[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            valid_mask = (shift_labels != -100)

            if valid_mask.sum() > 0:
                target_tokens = shift_labels.clamp(min=0).unsqueeze(-1)
                target_logits = torch.gather(shift_logits, -1, target_tokens).squeeze(-1)
                lse = torch.logsumexp(shift_logits, dim=-1)
                token_conf = torch.exp(target_logits - lse)
                step_conf = token_conf[valid_mask].mean().item()
            else:
                step_conf = 0.0
        else:
            # Model Top-1 confidence: max_v P(v | x_<t) = exp(max(z) - logsumexp(z))
            # Zero tensor allocation over vocabulary dimension
            max_logits = logits_f.max(dim=-1).values
            lse = torch.logsumexp(logits_f, dim=-1)
            token_conf = torch.exp(max_logits - lse)  # Shape: [batch_size, seq_len]

            if attention_mask is not None and attention_mask.shape == token_conf.shape:
                mask = attention_mask.bool()
                step_conf = token_conf[mask].mean().item() if mask.sum() > 0 else token_conf.mean().item()
            else:
                step_conf = token_conf.mean().item()

        self._accumulated_conf += step_conf
        self._accumulated_conf_count += 1

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if model is not None:
            self.model = model
            self._register_hook(model)
        if optimizer is not None:
            self.optimizer = optimizer

    def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
        active_model = model if model is not None else self.model
        if active_model is not None:
            self._calculate_and_store_gradients(active_model)
            self._grad_captured = True

    def on_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
        if not self._grad_captured:
            active_model = model if model is not None else self.model
            if active_model is not None:
                self._calculate_and_store_gradients(active_model)
                self._grad_captured = True

    @torch.no_grad()
    def _calculate_and_store_gradients(self, model):
        total_elements = 0
        sum_grads = 0.0
        sum_sq_grads = 0.0
        min_val = float('inf')
        max_val = float('-inf')

        has_grads = False
        for p in model.parameters():
            if p.grad is not None:
                has_grads = True
                g = p.grad.detach()
                n = g.numel()
                total_elements += n
                sum_grads += g.sum(dtype=torch.float64).item()
                sum_sq_grads += g.float().pow(2).sum(dtype=torch.float64).item()
                min_val = min(min_val, g.min().item())
                max_val = max(max_val, g.max().item())

        if not has_grads or total_elements == 0:
            return

        mean = sum_grads / total_elements
        var = max(0.0, (sum_sq_grads / total_elements) - (mean ** 2))

        if self.entropy_mode == "distribution":
            if min_val >= max_val:
                entropy = 0.0
            else:
                hist = torch.zeros(self.num_bins, dtype=torch.float64)
                for p in model.parameters():
                    if p.grad is not None:
                        h = torch.histc(p.grad.detach().float(), bins=self.num_bins, min=min_val, max=max_val)
                        hist += h.cpu().to(torch.float64)

                probs = hist / total_elements
                probs = probs[probs > 0]
                shannon_bits = -(probs * torch.log2(probs)).sum().item()
                entropy = shannon_bits / math.log2(self.num_bins)
        else:
            total_abs = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_abs += p.grad.detach().abs().sum(dtype=torch.float64).item()

            if total_abs > 0:
                sum_x_ln_x = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        g_abs = p.grad.detach().abs().float()
                        g_pos = g_abs[g_abs > 0]
                        if g_pos.numel() > 0:
                            sum_x_ln_x += (g_pos * torch.log(g_pos)).sum(dtype=torch.float64).item()
                raw_h = math.log(total_abs) - (sum_x_ln_x / total_abs)
                entropy = max(0.0, min(1.0, raw_h / math.log(total_elements)))
            else:
                entropy = 0.0

        self._temp_mean = mean
        self._temp_var = var
        self._temp_entropy = entropy

    def on_step_end(self, args, state, control, model=None, **kwargs):
        self._grad_captured = False

        mean = self._temp_mean
        var = self._temp_var
        entropy = self._temp_entropy

        # Average next-token confidence across gradient accumulation micro-batches
        if self._accumulated_conf_count > 0:
            confidence = self._accumulated_conf / self._accumulated_conf_count
        else:
            confidence = 0.0
        self._accumulated_conf = 0.0
        self._accumulated_conf_count = 0

        loss = 0.0
        if state.log_history:
            for entry in reversed(state.log_history):
                if 'loss' in entry:
                    loss = entry['loss']
                    break

        current_flops = state.total_flos
        step = state.global_step

        vram_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
        vram_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0.0
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        lr = 0.0
        opt = kwargs.get("optimizer", self.optimizer)
        if opt is not None and hasattr(opt, "param_groups"):
            for param_group in opt.param_groups:
                lr = param_group.get('lr', 0.0)
                break

        print(
            f"  📈 [Step {step:03d}] Loss: {loss:.5f} | Conf: {confidence:.4f} | "
            f"Grad Var: {var:.4e} | Grad Ent: {entropy:.4f} | LR: {lr:.2e} | "
            f"VRAM: {vram_allocated:.2f}GB (Alloc) / {vram_reserved:.2f}GB (Res)"
        )

        self.steps.append(step)
        self.variances.append(var)
        self.entropies.append(entropy)
        self.means.append(mean)
        self.losses.append(loss)
        self.confidences.append(confidence)
        self.flops.append(current_flops)
        self.vram_allocated.append(vram_allocated)
        self.vram_reserved.append(vram_reserved)
        self.learning_rates.append(lr)

        with open(self.log_file, 'a') as f:
            f.write(json.dumps({
                'step': step, 
                'confidence': confidence,
                'variance': var, 
                'entropy': entropy, 
                'mean': mean, 
                'loss': loss, 
                'flops': current_flops,
                'vram_allocated': vram_allocated,
                'vram_reserved': vram_reserved,
                'learning_rate': lr
            }) + '\n')

        should_plot = (
            step == 1 or 
            (args.logging_steps > 0 and step % args.logging_steps == 0) or 
            (state.max_steps > 0 and step >= state.max_steps)
        )
        if should_plot:
            self._save_plot()

    def on_train_end(self, args, state, control, **kwargs):
        if self._hook_handle is not None:
            try:
                self._hook_handle.remove()
            except Exception:
                pass
            self._hook_handle = None
        self._save_plot()

    def _save_plot(self):
        if not self.steps:
            return
        try:
            # 7 distinct subplots to include Token Confidence
            fig, axes = plt.subplots(1, 7, figsize=(35, 4))
            metrics = [
                (self.losses, 'Training Loss', 'red'),
                (self.confidences, f'Token Confidence ({self.confidence_mode.capitalize()})', 'purple'),
                (self.variances, 'Gradient Variance', 'blue'),
                (self.entropies, 'Gradient Entropy', 'green'),
                (self.means, 'Gradient Mean', 'orange'),
                (self.vram_allocated, 'Peak VRAM (GB)', 'magenta'),
                (self.learning_rates, 'Learning Rate', 'cyan')
            ]
            for ax, (data, title, color) in zip(axes, metrics):
                ax.plot(self.steps, data, color=color)
                if title == 'Peak VRAM (GB)' and len(self.vram_reserved) > 0:
                    ax.plot(self.steps, self.vram_reserved, color='indigo', linestyle='--', label='Reserved')
                    ax.legend()
                if 'Confidence' in title or 'Entropy' in title:
                    ax.set_ylim(-0.05, 1.05)
                ax.set_title(title)
                ax.set_xlabel('Steps')
                ax.grid(True, linestyle=':', alpha=0.6)

            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, 'training_metrics.png'), dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"⚠️ [METRICS] Error saving metrics plot: {e}")