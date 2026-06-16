import torch
import torch.nn.functional as F
from .base import RLAlgorithm

class PPOAlgorithm(RLAlgorithm):
    def __init__(self, epsilon=0.2, beta=0.01, vf_coef=0.1):
        self.epsilon = epsilon
        self.beta = beta
        self.vf_coef = vf_coef

    def compute_loss(self, policy_logprobs, ref_logprobs, advantages, comp_mask, **kwargs):
        """
        Computes the PPO loss including the clipped surrogate objective and KL penalty.
        Optionally includes value loss if value predictions are provided.
        """
        # PPO uses the ratio between the current policy and the old policy.
        # In a single-pass setup without a separate old policy forward pass,
        # the old policy logprobs are the detached current logprobs.
        old_logprobs = kwargs.get('old_logprobs', policy_logprobs.detach())
        
        ratio = torch.exp(policy_logprobs - old_logprobs)
        adv = advantages.unsqueeze(1)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv

        policy_loss = -torch.min(surr1, surr2)
        
        # KL penalty to keep the policy close to the reference model
        # Using the standard KL approximation: log(pi_theta) - log(pi_ref)
        kl = policy_logprobs - ref_logprobs
        
        loss = policy_loss + self.beta * kl
        
        # If value predictions and returns are provided (Actor-Critic PPO), compute value loss
        if 'values' in kwargs and 'returns' in kwargs:
            values = kwargs['values']
            returns = kwargs['returns'].unsqueeze(1)
            value_loss = F.mse_loss(values, returns, reduction='none')
            loss = loss + self.vf_coef * value_loss

        return (loss * comp_mask).sum() / (comp_mask.sum() + 1e-8)
