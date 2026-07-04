import os
import json
import time
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
        print(f"\n⏱️ [TIMER] Starting timing for: {stage_name}...")
        return time.time()

    def end_stage(self, stage_name, start_time):
        elapsed = time.time() - start_time
        times = self._load_times()
        times[stage_name] = times.get(stage_name, 0.0) + elapsed
        self._save_times(times)

        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = elapsed % 60
        print(f"⏱️ [TIMER] Completed {stage_name} in {hours}h {minutes}m {seconds:.2f}s (Elapsed: {elapsed:.2f}s).")
        self.print_summary()

    def print_summary(self):
        times = self._load_times()
        if not times:
            return
        print("\n" + "=" * 60)
        print("📊 CUMULATIVE TRAINING TIME SUMMARY (All Stages)")
        print("=" * 60)
        total_time = 0.0
        for stage, duration in times.items():
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = duration % 60
            print(f" - {stage:35}: {hours:02d}h {minutes:02d}m {seconds:05.2f}s (Total: {duration:.2f}s)")
            total_time += duration

        tot_hours = int(total_time // 3600)
        tot_minutes = int((total_time % 3600) // 60)
        tot_seconds = total_time % 60
        print("-" * 60)
        print(f" 🌟 TOTAL ELAPSED TIME FOR ALL STAGES: {tot_hours:02d}h {tot_minutes:02d}m {tot_seconds:05.2f}s ({total_time:.2f}s)")
        print("=" * 60 + "\n")


class GradientMetricsCallback(TrainerCallback):
    def __init__(self, log_file, plot_dir, model=None):
        self.model = model
        self.optimizer = None
        self.log_file = log_file
        self.plot_dir = plot_dir
        
        self.steps, self.variances, self.entropies, self.means, self.losses, self.flops = [], [], [], [], [], []
        self.vram_allocated = []
        self.vram_reserved = []
        self.learning_rates = []  # List to collect learning rates
        os.makedirs(self.plot_dir, exist_ok=True)

        # Temporary variables to store gradient metrics calculated right after backward (before optimizer.zero_grad())
        self._temp_mean = 0.0
        self._temp_var = 0.0
        self._temp_entropy = 0.0

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
                f.close()

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        """
        Resets peak memory stats and hooks key training items.
        """
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        if model is not None:
            self.model = model
            
        if optimizer is not None:
            self.optimizer = optimizer
            self._hook_optimizer(optimizer)

    def _hook_optimizer(self, optimizer):
        # Prevent double-hooking the optimizer during resumptions
        if hasattr(optimizer, '_is_gradient_metrics_hooked') and optimizer._is_gradient_metrics_hooked:
            return
            
        original_step = optimizer.step
        
        def hooked_step(*args, **kwargs):
            # Capture the gradient metrics right before weights are updated and gradients are cleared
            if hasattr(self, 'model') and self.model is not None:
                self._calculate_and_store_gradients(self.model)
            return original_step(*args, **kwargs)
            
        optimizer.step = hooked_step
        optimizer._is_gradient_metrics_hooked = True

    def _calculate_and_store_gradients(self, model):
        """
        Calculates gradient metrics right after backward but before zeroing.
        """
        grads = [p.grad.view(-1).float() for p in model.parameters() if p.grad is not None]
        if not grads:
            return

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

            self._temp_mean = mean
            self._temp_var = var
            self._temp_entropy = entropy

    def on_substep_end(self, args, state, control, model=None, **kwargs):
        """
        Fall-back capture step for gradient accumulation loops where on_substep_end is explicitly run.
        """
        if model is None:
            model = self.model
        if model is None:
            return

        self._calculate_and_store_gradients(model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """
        Retrieves the cached metrics, measures learning rate and VRAM usage, logs them, and saves the plot.
        """
        mean = self._temp_mean
        var = self._temp_var
        entropy = self._temp_entropy

        # Reset temporary variables for the next step
        self._temp_mean = 0.0
        self._temp_var = 0.0
        self._temp_entropy = 0.0

        loss = state.log_history[-1].get('loss', 0.0) if len(state.log_history) > 0 else 0.0
        current_flops = state.total_flos
        step = state.global_step

        # Measure peak memory usage in GB
        vram_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
        vram_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0.0
        
        # Reset peak memory statistics for the next step
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Extract current learning rate from the active optimizer
        lr = 0.0
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                lr = param_group.get('lr', 0.0)
                break

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

        # Extended plotting with 6 subplots including VRAM and Learning Rate
        plt.figure(figsize=(30, 4))
        for i, (data, title, color) in enumerate(zip(
            [self.variances, self.entropies, self.means, self.losses, self.vram_allocated, self.learning_rates],
            ['Gradient Variance', 'Gradient Entropy', 'Gradient Mean', 'Training Loss', 'Peak VRAM (GB)', 'Learning Rate'],
            ['blue', 'green', 'orange', 'red', 'magenta', 'cyan']
        )):
            plt.subplot(1, 6, i+1)
            plt.plot(self.steps, data, color=color)
            if title == 'Peak VRAM (GB)' and len(self.vram_reserved) > 0:
                plt.plot(self.steps, self.vram_reserved, color='purple', linestyle='--', label='Reserved')
                plt.legend()
            plt.title(title)
            plt.xlabel('Steps')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'training_metrics.png'))
        plt.close()