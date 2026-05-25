import torch
from .base import RLAlgorithm

class GRPOAlgorithm(RLAlgorithm):
    def __init__(self, epsilon=0.2, beta=0.01):
        self.epsilon = epsilon
        self.beta = beta

    def compute_loss(self, policy_logprobs, ref_logprobs, advantages, comp_mask):
        ratio = torch.exp(policy_logprobs - ref_logprobs)
        adv = advantages.unsqueeze(1)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv

        policy_loss = -torch.min(surr1, surr2)
        kl = torch.exp(ref_logprobs - policy_logprobs) - (ref_logprobs - policy_logprobs) - 1

        loss = policy_loss + self.beta * kl
        return (loss * comp_mask).sum() / (comp_mask.sum() + 1e-8)
