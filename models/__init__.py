from .architectures import (
    OLMo3Config, 
    OLMo3ForCausalLM,
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    GLM5Config,
    GLM5ForCausalLM,
    KimiK3Config,
    KimiK3ForCausalLM,
    MagistralConfig,
    MagistralForCausalLM,
    MiniMaxM2Config,
    MiniMaxM2ForCausalLM,
    MobileLLMProConfig,
    MobileLLMProForCausalLM,
    Nemotron3Config,
    Nemotron3ForCausalLM,
    Nemotron3DenseForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoEForCausalLM,
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
    FineGrainedSigmoidMoE,
    FineGrainedMoE,
    LatentMoE,
    GroupedQueryAttention,
    MultiLatentAttention,
    KimiDeltaAttention,
    CompressedSparseAttention,
    HeavilyCompressedAttention,
    NoRopeGroupedQueryAttention,
    Mamba2Layer,
    ManifoldConstrainedHyperConnections,
    FP4Quantizer,
)

# Registry mapping model architecture names to Config and Model classes
MODEL_REGISTRY = {
    "olmo_3": (OLMo3Config, OLMo3ForCausalLM),
    "deepseek_v4": (DeepSeekV4Config, DeepSeekV4ForCausalLM),
    "glm_5": (GLM5Config, GLM5ForCausalLM),
    "kimi_k3": (KimiK3Config, KimiK3ForCausalLM),
    "magistral": (MagistralConfig, MagistralForCausalLM),
    "minimax_m2": (MiniMaxM2Config, MiniMaxM2ForCausalLM),
    "mobilellm_pro": (MobileLLMProConfig, MobileLLMProForCausalLM),
    "nemotron_3": (Nemotron3Config, Nemotron3ForCausalLM),
    "nemotron_3_dense": (Nemotron3Config, Nemotron3DenseForCausalLM),
    "qwen_3": (Qwen3Config, Qwen3ForCausalLM),
    "qwen_3_moe": (Qwen3Config, Qwen3MoEForCausalLM),
}


def get_model_classes(architecture: str):
    if architecture not in MODEL_REGISTRY:
        raise ValueError(f"Model type {architecture} not supported. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[architecture]