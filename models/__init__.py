from .architectures import OLMo3Config, OLMo3ForCausalLM
from .utils import (
    RMSNorm,
    RotaryPositionalEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
    SwiGLUMLP,
    GroupedQueryAttention,
)

# Future models can be added here and mapped
MODEL_REGISTRY = {
    "olmo3": (OLMo3Config, OLMo3ForCausalLM),
    # "deepseek": (DeepseekConfig, DeepseekForCausalLM),
    # "qwen": (QwenConfig, QwenForCausalLM),
}


def get_model_classes(architecture: str):
    if architecture not in MODEL_REGISTRY:
        raise ValueError(f"Model type {architecture} not supported. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[architecture]