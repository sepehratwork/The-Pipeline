# rl_algorithms/gspo.py
import torch
from .base import RLAlgorithm

class GSPOAlgorithm(RLAlgorithm):
    """
    Group Sequence Policy Optimization (GSPO)
    Based on the paper: "Group Sequence Policy Optimization" (Qwen Team, Alibaba Inc.)
    """
    def __init__(self, epsilon_left=3e-4, epsilon_right=4e-4, beta=0.01, use_token_level=False):
        # GSPO uses asymmetric, much smaller clipping ranges compared to GRPO (Section 5.1)
        self.epsilon_left = epsilon_left
        self.epsilon_right = epsilon_right
        self.beta = beta
        self.use_token_level = use_token_level

    def compute_loss(self, policy_logprobs, ref_logprobs, advantages, comp_mask, **kwargs):
        # GSPO requires old_logprobs to compute the sequence-level importance ratio
        old_logprobs = kwargs.get('old_logprobs')
        if old_logprobs is None:
            raise ValueError("GSPO requires 'old_logprobs' to compute the importance ratio.")

        # Calculate sequence lengths |y_i|
        seq_lengths = comp_mask.sum(dim=1)
        seq_lengths_clamped = torch.clamp(seq_lengths, min=1.0) # Prevent division by zero

        # Calculate sequence-level importance ratio s_i(theta) (Equation 7)
        # s_i(theta) = exp( (1 / |y_i|) * sum_{t=1}^{|y_i|} (log pi_theta - log pi_old) )
        sum_logp_policy = (policy_logprobs * comp_mask).sum(dim=1)
        sum_logp_old = (old_logprobs * comp_mask).sum(dim=1)
        
        s_i = torch.exp((sum_logp_policy - sum_logp_old) / seq_lengths_clamped)

        if self.use_token_level:
            # GSPO-token variant (Equation 13 & 14)
            # s_{i,t}(theta) = sg[s_i(theta)] * (pi_theta / sg[pi_theta])
            s_i_sg = s_i.detach().unsqueeze(1)
            pi_theta = torch.exp(policy_logprobs)
            pi_theta_sg = pi_theta.detach()
            
            # Avoid division by zero
            pi_theta_sg = torch.clamp(pi_theta_sg, min=1e-8)
            s_i_t = s_i_sg * (pi_theta / pi_theta_sg)
            
            adv = advantages.unsqueeze(1)
            surr1 = s_i_t * adv
            surr2 = torch.clamp(s_i_t, 1.0 - self.epsilon_left, 1.0 + self.epsilon_right) * adv
            
            # Token-level policy loss, averaged per sequence then over the batch
            policy_loss_per_token = -torch.min(surr1, surr2)
            policy_loss_per_seq = (policy_loss_per_token * comp_mask).sum(dim=1) / seq_lengths_clamped
            policy_loss = policy_loss_per_seq.mean()
        else:
            # Standard GSPO (Sequence-level objective, Equation 5)
            surr1 = s_i * advantages
            surr2 = torch.clamp(s_i, 1.0 - self.epsilon_left, 1.0 + self.epsilon_right) * advantages
            
            # Sequence-level policy loss
            policy_loss = -torch.min(surr1, surr2).mean()

        # Token-level KL penalty (standard practice, omitted for brevity in paper but necessary for stability)
        # kl = exp(ref_logprobs - policy_logprobs) - (ref_logprobs - policy_logprobs) - 1
        kl = torch.exp(ref_logprobs - policy_logprobs) - (ref_logprobs - policy_logprobs) - 1
        kl_per_seq = (kl * comp_mask).sum(dim=1) / seq_lengths_clamped
        kl_loss = kl_per_seq.mean()

        loss = policy_loss + self.beta * kl_loss
        return loss