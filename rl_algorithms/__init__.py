from .base import RLAlgorithm
from .grpo import GRPOAlgorithm
from .gspo import GSPOAlgorithm
from .dapo import DAPOAlgorithm
from .sapo import SAPOAlgorithm
from .cispo import CISPOAlgorithm


RL_ALGO_REGISTRY = {
    "grpo": GRPOAlgorithm,
    "gspo": GSPOAlgorithm,
    "dapo": DAPOAlgorithm,
    "sapo": SAPOAlgorithm,
    "cispo": CISPOAlgorithm,
}

def get_rl_algorithm(algo_name: str, **kwargs):
    if algo_name not in RL_ALGO_REGISTRY:
        raise ValueError(f"RL Algorithm {algo_name} not supported.")
    return RL_ALGO_REGISTRY[algo_name](**kwargs)