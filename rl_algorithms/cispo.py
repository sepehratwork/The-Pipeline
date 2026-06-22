import torch
from .base import RLAlgorithm


class CISPOAlgorithm(RLAlgorithm):
    def __init__(self, eps_low: float = 10.0, eps_high: float = 0.2, **kwargs):
        """
        Clipped IS-weight Policy Optimization (CISPO) Algorithm.
        
        Args:
            eps_low (float): Lower clipping parameter for the IS weight (epsilon_low^IS).
                             Defaults to 10.0, which effectively disables lower clipping 
                             since 1 - 10.0 < 0 (matching the paper's default setup).
            eps_high (float): Upper clipping parameter for the IS weight (epsilon_high^IS).
                              Defaults to 0.2.
        """
        super().__init__()
        self.eps_low = eps_low
        self.eps_high = eps_high

    def compute_loss(self, policy_logprobs, ref_logprobs, advantages, comp_mask, old_logprobs=None, **kwargs):
        """
        Computes the CISPO loss according to Equations 4 and 5 of the paper.
        
        Args:
            policy_logprobs (torch.Tensor): Log probabilities of the current policy model 
                                            for the generated tokens. Shape: (group_size, seq_len)
            ref_logprobs (torch.Tensor): Log probabilities of the reference model 
                                         for the generated tokens. Shape: (group_size, seq_len)
            advantages (torch.Tensor): Advantages for each completion. Shape: (group_size,)
            comp_mask (torch.Tensor): Completion mask (1.0 for valid tokens, 0.0 for padding).
                                      Shape: (group_size, seq_len)
            old_logprobs (torch.Tensor, optional): Log probabilities from the old policy at generation time.
                                                   Shape: (group_size, seq_len). If None, defaults to 
                                                   detached policy_logprobs.
        
        Returns:
            torch.Tensor: Scalar loss value (negative of the CISPO objective).
        """
        # Default old_logprobs to detached current policy logprobs if not provided
        if old_logprobs is None:
            old_logprobs = policy_logprobs.detach()

        # 1. Compute the importance sampling ratio: r_t(theta) = exp(log_pi_theta - log_pi_old)
        ratio = torch.exp(policy_logprobs - old_logprobs)
        
        # Clamp for numerical stability
        ratio = torch.clamp(ratio, min=1e-8, max=100.0)

        # 2. Apply clipping to the IS ratio (Equation 5)
        # r_hat = clip(r_t(theta), 1 - eps_low, 1 + eps_high)
        low_bound = 1.0 - self.eps_low
        high_bound = 1.0 + self.eps_high
        clipped_ratio = torch.clamp(ratio, min=low_bound, max=high_bound)

        # 3. Stop-gradient on the clipped IS weight (Equation 4)
        clipped_ratio_sg = clipped_ratio.detach()

        # 4. Compute token-level objective (Equation 4)
        # sg(r_hat) * A_t * log_pi_theta
        # Align advantages from (group_size,) to (group_size, 1) for broadcasting
        advantages_expanded = advantages.unsqueeze(-1)
        token_objective = clipped_ratio_sg * advantages_expanded * policy_logprobs

        # 5. Mask out padded tokens
        masked_objective = token_objective * comp_mask

        # 6. Aggregate and normalize by total completion tokens (sum of the mask)
        # We minimize the negative objective to maximize the expected reward
        total_tokens = comp_mask.sum()
        loss = -masked_objective.sum() / (total_tokens + 1e-8)

        return loss
