class RLAlgorithm:
    """Base class for RL algorithms to ensure generalization."""
    def compute_loss(self, policy_logprobs, ref_logprobs, advantages, comp_mask):
        raise NotImplementedError
