from .olmo_3 import OLMo3Config, OLMo3ForCausalLM
from .qwen_3 import Qwen3Config, Qwen3ForCausalLM
from .deepseek_v4 import DeepSeekV4Config, DeepSeekV4ForCausalLM

# Registry mapping model architecture names to Config and Model classes
MODEL_REGISTRY = {
    "olmo_3": (OLMo3Config, OLMo3ForCausalLM),
    "qwen_3": (Qwen3Config, Qwen3ForCausalLM),
    "deepseek_v4": (DeepSeekV4Config, DeepSeekV4ForCausalLM),
}


def get_model_classes(architecture: str):
    if architecture not in MODEL_REGISTRY:
        raise ValueError(f"Model type {architecture} not supported. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[architecture]