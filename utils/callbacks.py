import os
import json
import torch
import matplotlib.pyplot as plt
from transformers import TrainerCallback

class GradientMetricsCallback(TrainerCallback):
    def __init__(self, log_file, plot_dir):
        self.log_file = log_file
        self.plot_dir = plot_dir
        self.steps, self.variances, self.entropies, self.means, self.losses, self.flops = [], [], [], [], [], []
        os.makedirs(self.plot_dir, exist_ok=True)

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
                f.close()

    def on_step_end(self, args, state, control, model, **kwargs):
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

        loss = state.log_history[-1].get('loss', 0.0) if len(state.log_history) > 0 else 0.0
        current_flops = state.total_flos
        step = state.global_step

        self.steps.append(step)
        self.variances.append(var)
        self.entropies.append(entropy)
        self.means.append(mean)
        self.losses.append(loss)
        self.flops.append(current_flops)

        with open(self.log_file, 'a') as f:
            f.write(json.dumps({'step': step, 'variance': var, 'entropy': entropy, 'mean': mean, 'loss': loss, 'flops': current_flops}) + '\n')
            f.close()

        plt.figure(figsize=(25, 4))
        for i, (data, title, color) in enumerate(zip(
            [self.variances, self.entropies, self.means, self.losses],
            ['Gradient Variance', 'Gradient Entropy', 'Gradient Mean', 'Training Loss'],
            ['blue', 'green', 'orange', 'red', 'purple']
        )):
            plt.subplot(1, 5, i+1)
            plt.plot(self.steps, data, color=color)
            plt.title(title)
            plt.xlabel('Steps')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'training_metrics.png'))
        plt.close()
