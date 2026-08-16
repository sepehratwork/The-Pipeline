from .architectures import (
    # OLMo 3
    OLMo3TestConfig, 
    OLMo3ForCausalLM,
    # DeepSeek V4
    DeepSeekV4Config,
    DeepSeekV4TestConfig,
    DeepSeekV4ForCausalLM,
    # GLM 5
    GLM5Config,
    GLM5TestConfig,
    GLM5ForCausalLM,
    # Kimi K3
    KimiK3Config,
    KimiK3TestConfig,
    KimiK3ForCausalLM,
    # Magistral
    MagistralConfig,
    MagistralTestConfig,
    MagistralForCausalLM,
    # MiniMax M2
    MiniMaxM2Config,
    MiniMaxM2TestConfig,
    MiniMaxM2ForCausalLM,
    # MobileLLM
    MobileLLMProConfig,
    MobileLLMProTestConfig,
    MobileLLMProForCausalLM,
    # Nemotron 3
    Nemotron3Config,
    Nemotron3TestConfig,
    Nemotron3ForCausalLM,
    Nemotron3DenseForCausalLM,
    # Qwen 3
    Qwen3Config,
    Qwen3TestConfig,
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
    # OLMo 3
    "olmo_3_test": (OLMo3TestConfig, OLMo3ForCausalLM),
    # DeepSeek V4
    "deepseek_v4": (DeepSeekV4Config, DeepSeekV4ForCausalLM),
    "deepseek_v4_test": (DeepSeekV4TestConfig, DeepSeekV4ForCausalLM),
    # GLM 5
    "glm_5": (GLM5Config, GLM5ForCausalLM),
    "glm_5_test": (GLM5TestConfig, GLM5ForCausalLM),
    # Kimi K3
    "kimi_k3": (KimiK3Config, KimiK3ForCausalLM),
    "kimi_k3_test": (KimiK3TestConfig, KimiK3ForCausalLM),
    # Magistral
    "magistral": (MagistralConfig, MagistralForCausalLM),
    "magistral_test": (MagistralTestConfig, MagistralForCausalLM),
    # MiniMax M2
    "minimax_m2": (MiniMaxM2Config, MiniMaxM2ForCausalLM),
    "minimax_m2_test": (MiniMaxM2TestConfig, MiniMaxM2ForCausalLM),
    # MobileLLM Pro
    "mobilellm_pro": (MobileLLMProConfig, MobileLLMProForCausalLM),
    "mobilellm_pro_test": (MobileLLMProTestConfig, MobileLLMProForCausalLM),
    # Nemotron 3
    "nemotron_3": (Nemotron3Config, Nemotron3ForCausalLM),
    "nemotron_3_test": (Nemotron3TestConfig, Nemotron3ForCausalLM),
    "nemotron_3_dense": (Nemotron3Config, Nemotron3DenseForCausalLM),
    "nemotron_3_dense_test": (Nemotron3TestConfig, Nemotron3DenseForCausalLM),
    # Qwen 3
    "qwen_3": (Qwen3Config, Qwen3ForCausalLM),
    "qwen_3_test": (Qwen3TestConfig, Qwen3ForCausalLM),
    "qwen_3_moe": (Qwen3Config, Qwen3MoEForCausalLM),
    "qwen_3_moe_test": (Qwen3TestConfig, Qwen3MoEForCausalLM),
}


def get_model_classes(architecture: str):
    if architecture not in MODEL_REGISTRY:
        raise ValueError(f"Model type {architecture} not supported. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[architecture]