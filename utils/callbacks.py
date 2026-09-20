import os
import json
import time
import math
import functools
import torch
import matplotlib.pyplot as plt
from transformers import TrainerCallback


class StageTimer:
    """
    Timer utility to record the training time taken by each of the 6 stages.
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
    def __init__(self, log_file, plot_dir, model=None):
        self.model = model
        self.optimizer = None
        self.log_file = log_file
        self.plot_dir = plot_dir
        
        self.steps, self.variances, self.entropies, self.means, self.losses, self.flops = [], [], [], [], [], []
        self.vram_allocated = []
        self.vram_reserved = []
        self.learning_rates = []
        os.makedirs(self.plot_dir, exist_ok=True)

        # Temporary variables stored in CPU RAM
        self._temp_mean = 0.0
        self._temp_var = 0.0
        self._temp_entropy = 0.0
        self._metrics_computed_for_step = False

        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self.steps.append(data['step'])
                        self.variances.append(data['variance'])
                        self.entropies.append(data['entropy'])
                        self.means.append(data['mean'])
                        self.losses.append(data['loss'])
                        self.flops.append(data.get('flops', 0))
                        self.vram_allocated.append(data.get('vram_allocated', 0.0))
                        self.vram_reserved.append(data.get('vram_reserved', 0.0))
                        self.learning_rates.append(data.get('learning_rate', 0.0))

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        if model is not None:
            self.model = model
            
        if optimizer is not None:
            self.optimizer = optimizer
            self._hook_optimizer(optimizer)

    def _hook_optimizer(self, optimizer):
        if hasattr(optimizer, '_is_gradient_metrics_hooked') and optimizer._is_gradient_metrics_hooked:
            return
            
        original_step = optimizer.step
        
        @functools.wraps(original_step)
        def hooked_step(*args, **kwargs):
            # Fallback in case on_pre_optimizer_step is not invoked by Trainer
            if not self._metrics_computed_for_step and hasattr(self, 'model') and self.model is not None:
                self._calculate_and_store_gradients(self.model)
            return original_step(*args, **kwargs)
            
        # Prevent PyTorch lr_scheduler UserWarning about overridden step()
        hooked_step._with_counter = getattr(original_step, '_with_counter', True)
        optimizer.step = hooked_step
        optimizer._is_gradient_metrics_hooked = True

    def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
        """
        Native Hugging Face Trainer callback triggered after gradient accumulation 
        and clipping, right before optimizer.step(). Runs exactly once per global step.
        """
        target_model = model if model is not None else self.model
        if target_model is not None and not self._metrics_computed_for_step:
            self._calculate_and_store_gradients(target_model)

    def on_substep_end(self, args, state, control, **kwargs):
        """
        No-op during micro-batch accumulation to prevent running expensive 
        gradient calculations 512 times per global step.
        """
        pass

    @torch.no_grad()
    def _calculate_and_store_gradients(self, model):
        """
        Calculates gradient metrics (mean, variance, entropy) in CPU RAM.
        No large tensors are allocated on the GPU, completely eliminating CUDA OOM.
        """
        total_elements = 0
        min_val = float('inf')
        max_val = float('-inf')

        active_params = []
        # Pass 1: Global min/max determination across active gradients
        for p in model.parameters():
            if p.grad is not None:
                p_min = p.grad.min().item()
                p_max = p.grad.max().item()
                if p_min < min_val:
                    min_val = p_min
                if p_max > max_val:
                    max_val = p_max
                active_params.append(p)

        if not active_params or not (math.isfinite(min_val) and math.isfinite(max_val)):
            self._metrics_computed_for_step = True
            return

        # Pass 2: Streaming accumulation in CPU RAM
        num_bins = 100
        sum_grads = 0.0
        sum_sq_grads = 0.0
        hist_counts = torch.zeros(num_bins, dtype=torch.float64, device="cpu")
        is_flat_distribution = (min_val >= max_val)

        for p in active_params:
            # Transfer only one layer's gradient to CPU RAM in float32; 0 GPU memory is retained
            g_cpu = p.grad.detach().to(device="cpu", dtype=torch.float32)
            n = g_cpu.numel()
            if n == 0:
                del g_cpu
                continue

            total_elements += n
            sum_grads += g_cpu.sum().item()
            sum_sq_grads += (g_cpu ** 2).sum().item()

            if not is_flat_distribution:
                # Compute binned histogram on CPU
                layer_hist = torch.histc(g_cpu, bins=num_bins, min=min_val, max=max_val).to(torch.float64)
                hist_counts += layer_hist
                del layer_hist

            del g_cpu

        if total_elements == 0:
            self._metrics_computed_for_step = True
            return

        # 1. Gradient Mean
        mean = sum_grads / total_elements

        # 2. Gradient Variance
        var = max(0.0, (sum_sq_grads / total_elements) - (mean ** 2))

        # 3. Gradient Entropy (Shannon entropy of empirical distribution of values)
        if is_flat_distribution:
            entropy = 0.0
        else:
            probs = hist_counts / total_elements
            pos_probs = probs[probs > 0]
            # Shannon entropy: -sum(p * ln(p)) over non-empty histogram bins
            entropy = -torch.sum(pos_probs * torch.log(pos_probs)).item()

        self._temp_mean = mean
        self._temp_var = var
        self._temp_entropy = entropy
        self._metrics_computed_for_step = True

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """
        Retrieves cached metrics from CPU RAM, logs them, and generates plots.
        """
        mean = self._temp_mean
        var = self._temp_var
        entropy = self._temp_entropy

        # Reset temporary variables and calculation flag for the next step
        self._temp_mean = 0.0
        self._temp_var = 0.0
        self._temp_entropy = 0.0
        self._metrics_computed_for_step = False

        loss = state.log_history[-1].get('loss', 0.0) if len(state.log_history) > 0 else 0.0
        current_flops = state.total_flos
        step = state.global_step

        # Measure peak memory usage in GB
        vram_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
        vram_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0.0
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        lr = 0.0
        optimizer = getattr(self, 'optimizer', None) or kwargs.get('optimizer', None)
        if optimizer is not None:
            for param_group in optimizer.param_groups:
                lr = param_group.get('lr', 0.0)
                break

        print(
            f"  📈 [Step {step:03d}] Loss: {loss:.5f} | Grad Var: {var:.4e} | Grad Ent: {entropy:.4f} | "
            f"LR: {lr:.2e} | VRAM: {vram_allocated:.2f}GB (Alloc) / {vram_reserved:.2f}GB (Res)"
        )

        self.steps.append(step)
        self.variances.append(var)
        self.entropies.append(entropy)
        self.means.append(mean)
        self.losses.append(loss)
        self.flops.append(current_flops)
        self.vram_allocated.append(vram_allocated)
        self.vram_reserved.append(vram_reserved)
        self.learning_rates.append(lr)

        with open(self.log_file, 'a') as f:
            f.write(json.dumps({
                'step': step, 
                'variance': var, 
                'entropy': entropy, 
                'mean': mean, 
                'loss': loss, 
                'flops': current_flops,
                'vram_allocated': vram_allocated,
                'vram_reserved': vram_reserved,
                'learning_rate': lr
            }) + '\n')

        plt.figure(figsize=(30, 4))
        for i, (data, title, color) in enumerate(zip(
            [self.variances, self.entropies, self.means, self.losses, self.vram_allocated, self.learning_rates],
            ['Gradient Variance', 'Gradient Entropy', 'Gradient Mean', 'Training Loss', 'Peak VRAM (GB)', 'Learning Rate'],
            ['blue', 'green', 'orange', 'red', 'magenta', 'cyan']
        )):
            plt.subplot(1, 6, i + 1)
            plt.plot(self.steps, data, color=color)
            if title == 'Peak VRAM (GB)' and len(self.vram_reserved) > 0:
                plt.plot(self.steps, self.vram_reserved, color='purple', linestyle='--', label='Reserved')
                plt.legend()
            plt.title(title)
            plt.xlabel('Steps')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'training_metrics.png'))
        plt.close()