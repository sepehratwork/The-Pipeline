# rl_algorithms/sapo.py
import torch
from .base import RLAlgorithm

class SAPOAlgorithm(RLAlgorithm):
    """
    Soft Adaptive Policy Optimization (SAPO)
    Paper: Soft Adaptive Policy Optimization (arXiv:2511.20347v2)
    """
    def __init__(self, tau_pos=1.0, tau_neg=1.05, beta=0.01):
        # Asymmetric temperatures based on Section 5.1 of the paper
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.beta = beta

    def compute_loss(self, policy_logprobs, ref_logprobs, advantages, comp_mask, **kwargs):
        # SAPO uses the ratio between the current policy and the old (behavior) policy.
        # We extract old_logprobs passed from the training loop.
        old_logprobs = kwargs.get('old_logprobs', ref_logprobs)
        
        # r_{i,t}(\theta) = \pi_\theta / \pi_{\theta_{old}}
        ratio = torch.exp(policy_logprobs - old_logprobs)
        
        # Group-normalized advantage (shared across tokens within a response)
        adv = advantages.unsqueeze(1)
        
        # Equation (6): Asymmetric temperatures for positive and negative advantages
        # \tau_{i,t} = \tau_{pos} if \hat{A}_{i,t} > 0 else \tau_{neg}
        tau = torch.where(
            adv > 0, 
            torch.tensor(self.tau_pos, device=adv.device, dtype=adv.dtype), 
            torch.tensor(self.tau_neg, device=adv.device, dtype=adv.dtype)
        )
        
        # Equation (6): Soft gate function
        # f_{i,t}(x) = \sigma(\tau_{i,t} * (x - 1)) * (4 / \tau_{i,t})
        x = tau * (ratio - 1.0)
        sigmoid_x = torch.sigmoid(x)
        f_val = sigmoid_x * (4.0 / tau)
        
        # Equation (5): Surrogate objective to maximize: f_{i,t}(r_{i,t}(\theta)) * \hat{A}_{i,t}
        # We negate it because PyTorch optimizers minimize the loss
        policy_loss = -(f_val * adv)
        
        # KL divergence penalty (Standard in RLVR to stay close to the reference model)
        # kl = \pi_{ref} / \pi_\theta - \log(\pi_{ref} / \pi_\theta) - 1
        kl = torch.exp(ref_logprobs - policy_logprobs) - (ref_logprobs - policy_logprobs) - 1.0
        
        # Total loss
        loss = policy_loss + self.beta * kl
        
        # Mask out padding tokens and compute the mean over valid tokens
        return (loss * comp_mask).sum() / (comp_mask.sum() + 1e-8)