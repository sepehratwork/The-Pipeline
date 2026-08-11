from .architectures import (
    OLMo3Config, 
    OLMo3ForCausalLM, 
    Qwen3Config, 
    Qwen3ForCausalLM,
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    GLM5Config,
    GLM5ForCausalLM,
    KimiK3Config,
    KimiK3ForCausalLM,
    MagistralConfig,
    MagistralForCausalLM,
)
    
from .utils import (
    RMSNorm,
    RotaryPositionalEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
    SwiGLUMLP,
    SiTUGLU,
    DeepSeekMoE,
    StableLatentMoE,
    TopKMoE,
    GroupedQueryAttention,
    MultiLatentAttention,
    KimiDeltaAttention,
    CompressedSparseAttention,
    HeavilyCompressedAttention,
    ManifoldConstrainedHyperConnections,
    FP4Quantizer,
)

# Registry mapping model architecture names to Config and Model classes
MODEL_REGISTRY = {
    "olmo_3": (OLMo3Config, OLMo3ForCausalLM),
    "qwen_3": (Qwen3Config, Qwen3ForCausalLM),
    "deepseek_v4": (DeepSeekV4Config, DeepSeekV4ForCausalLM),
    "glm_5": (GLM5Config, GLM5ForCausalLM),
    "kimi_k3": (KimiK3Config, KimiK3ForCausalLM),
    "magistral": (MagistralConfig, MagistralForCausalLM),
}


def get_model_classes(architecture: str):
    if architecture not in MODEL_REGISTRY:
        raise ValueError(f"Model type {architecture} not supported. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[architecture]