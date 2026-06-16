# rl_algorithms/dapo.py
import torch
from .base import RLAlgorithm

class DAPOAlgorithm(RLAlgorithm):
    def __init__(self, epsilon_low=0.2, epsilon_high=0.28):
        """
        DAPO: Decoupled Clip and Dynamic sAmpling Policy Optimization
        """
        # 1. Raise the Ceiling: Clip-Higher (Section 3.1)
        # Decoupled clipping ranges to prevent entropy collapse
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high

    def compute_loss(self, policy_logprobs, ref_logprobs, advantages, comp_mask, **kwargs):
        # DAPO uses pi_theta / pi_old for the ratio.
        old_logprobs = kwargs.get('old_logprobs', ref_logprobs)
        
        ratio = torch.exp(policy_logprobs - old_logprobs)
        adv = advantages.unsqueeze(1)

        surr1 = ratio * adv
        # Apply decoupled asymmetric clipping
        surr2 = torch.clamp(ratio, 1.0 - self.epsilon_low, 1.0 + self.epsilon_high) * adv

        # 3. Removing KL Divergence (Section 2.3)
        # DAPO explicitly removes the KL penalty term to allow the model to diverge 
        # from the reference policy during long-CoT reasoning.
        policy_loss = -torch.min(surr1, surr2)

        # 2. Rebalancing Act: Token-Level Policy Gradient Loss (Section 3.3)
        # Instead of averaging loss per sample and then across the batch (like standard GRPO),
        # DAPO aggregates the loss across all valid tokens directly.
        loss = (policy_loss * comp_mask).sum() / (comp_mask.sum() + 1e-8)
        
        return loss

    @staticmethod
    def compute_soft_overlong_punishment(completion_lengths, l_max, l_cache):
        """
        4. Hide and Seek: Soft Overlong Punishment (Section 3.4, Equation 13)
        Applies a length-aware penalty to truncated samples.
        """
        punishments = torch.zeros_like(completion_lengths, dtype=torch.float32)
        
        for i, length in enumerate(completion_lengths):
            if length <= l_max - l_cache:
                punishments[i] = 0.0
            elif l_max - l_cache < length <= l_max:
                punishments[i] = ((l_max - l_cache) - length) / l_cache
            else:
                punishments[i] = -1.0
                
        return punishments