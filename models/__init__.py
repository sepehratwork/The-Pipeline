from .olmo3 import OLMo3Config, OLMo3ForCausalLM
from .normalization import RMSNorm
from .positional_encoding import RotaryEmbedding, apply_rotary_pos_emb, rotate_half
from .mlp import SwiGLUMLP
from .attention import GroupedQueryAttention

# Future models can be added here and mapped
MODEL_REGISTRY = {
    "olmo3": (OLMo3Config, OLMo3ForCausalLM),
    # "deepseek": (DeepseekConfig, DeepseekForCausalLM),
    # "qwen": (QwenConfig, QwenForCausalLM),
}


def get_model_classes(model_type: str):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Model type {model_type} not supported. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type]
