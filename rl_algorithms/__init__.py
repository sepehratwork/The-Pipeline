# rl_algorithms/__init__.py
from .base import RLAlgorithm
from .grpo import GRPOAlgorithm
from .ppo import PPOAlgorithm
from .dapo import DAPOAlgorithm
from .gspo import GSPOAlgorithm
from .sapo import SAPOAlgorithm

RL_ALGO_REGISTRY = {
    "grpo": GRPOAlgorithm,
    "ppo": PPOAlgorithm,
    "dapo": DAPOAlgorithm,
    "gspo": GSPOAlgorithm,
    "sapo": SAPOAlgorithm,
}

def get_rl_algorithm(algo_name: str, **kwargs):
    if algo_name not in RL_ALGO_REGISTRY:
        raise ValueError(f"RL Algorithm {algo_name} not supported.")
    return RL_ALGO_REGISTRY[algo_name](**kwargs)