from .base import RLAlgorithm
from .grpo import GRPOAlgorithm

RL_ALGO_REGISTRY = {
    "grpo": GRPOAlgorithm,
    # "ppo": PPOAlgorithm,
    # "dapo": DAPOAlgorithm,
}

def get_rl_algorithm(algo_name: str, **kwargs):
    if algo_name not in RL_ALGO_REGISTRY:
        raise ValueError(f"RL Algorithm {algo_name} not supported.")
    return RL_ALGO_REGISTRY[algo_name](**kwargs)
