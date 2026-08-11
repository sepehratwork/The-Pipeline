import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..utils.normalization import RMSNorm
from ..utils.mlp import SwiGLUMLP, DeepSeekMoE
from ..utils.attention import MultiLatentAttention


class GLM5Config(PretrainedConfig):
    """
    Configuration class for the GLM-5 model.
    
    Hyperparameters are scaled according to scaling laws to target 1.0 Billion Active Parameters (~1000M active parameters).
    Target parameters calculation:
    - Vocab embedding (154,880 x 1,536, tied): ~237.89M
    - Dense Layers (2 layers): ~53.74M
    - MoE Layers (14 layers, 8 active experts out of 64 routed + 1 shared): ~707.98M
    - Total Active Parameters = 237.89M + 53.74M + 707.98M = ~999.61M Active Parameters (~1.0 Billion)
    """
    architecture = "glm_5"

    def __init__(
        self,
        vocab_size: int = 154880,
        hidden_size: int = 1536,               # Hidden dimension for 1B active target
        intermediate_size: int = 4096,         # SwiGLU intermediate size for dense layers
        moe_intermediate_size: int = 1024,     # Per-expert intermediate size
        num_hidden_layers: int = 16,           # Total transformer layers
        num_dense_layers: int = 2,             # Dense layers at the bottom of the stack
        num_attention_heads: int = 16,         # Number of MLA attention heads
        qk_head_dim: int = 128,                # QK head dimension
        v_head_dim: int = 128,                 # V head dimension
        rope_head_dim: int = 64,               # Decoupled RoPE head dimension
        q_lora_rank: int = 512,                # Low-rank compression for query
        kv_lora_rank: int = 256,               # Low-rank compression for key-value
        num_routed_experts: int = 64,          # Total routed experts (set to 0 for pure dense model)
        num_active_experts: int = 8,           # Active routed experts per token
        num_shared_experts: int = 1,           # Number of shared experts
        hash_routing_layers: int = 0,          # Layers utilizing deterministic hash routing
        max_position_embeddings: int = 202752, # Extended max position embeddings (GLM-5 SFT max)
        rope_theta: float = 1000000.0,         # Base RoPE theta
        use_yarn: bool = True,                 # YaRN RoPE extension
        original_max_position_embeddings: int = 8192,
        z_loss_weight: float = 1e-5,           # Logit Z-loss weight
        use_dsa: bool = False,                 # DeepSeek Sparse Attention flag
        topk_indexer: int = 64,                # Top-k sparse indexer for DSA
        rms_norm_eps: float = 1e-6,
        tie_word_embeddings: bool = True,      # Tied embeddings flag
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_dense_layers = num_dense_layers
        self.num_attention_heads = num_attention_heads
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.rope_head_dim = rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_routed_experts = num_routed_experts
        self.num_active_experts = num_active_experts
        self.num_shared_experts = num_shared_experts
        self.hash_routing_layers = hash_routing_layers
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.use_yarn = use_yarn
        self.original_max_position_embeddings = original_max_position_embeddings
        self.z_loss_weight = z_loss_weight
        self.use_dsa = use_dsa
        self.topk_indexer = topk_indexer
        self.rms_norm_eps = rms_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class GLM5Block(nn.Module):
    """
    GLM-5 Transformer Block.
    
    Contains Multi-Latent Attention (MLA) and either a Dense SwiGLU MLP or a DeepSeekMoE block.
    """
    def __init__(self, config: GLM5Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = MultiLatentAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Bottom layers use dense SwiGLU MLP, upper layers use MoE (or if num_routed_experts == 0, pure dense)
        is_dense = (layer_idx < config.num_dense_layers) or (config.num_routed_experts == 0)
        
        if is_dense:
            self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size)
        else:
            # Re-map config attributes expected by DeepSeekMoE
            moe_config = config
            moe_config.intermediate_size = config.moe_intermediate_size
            self.mlp = DeepSeekMoE(moe_config, layer_idx)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        residual = hidden_states
        hidden_states, present_kv = self.self_attn(
            self.input_layernorm(hidden_states), attention_mask, position_ids, past_key_value
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states, present_kv


class GLM5PreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and standard Hugging Face interface."""
    config_class = GLM5Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class GLM5Model(GLM5PreTrainedModel):
    """
    The bare GLM-5 Model transformer outputting raw hidden-states without standard LM head.
    """
    def __init__(self, config: GLM5Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([GLM5Block(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self, 
        input_ids, 
        attention_mask=None, 
        position_ids=None, 
        past_key_values=None, 
        use_cache=None, 
        **kwargs
    ):
        if use_cache is None:
            use_cache = getattr(self.config, "use_cache", False)

        if position_ids is None:
            past_length = past_key_values[0][0].shape[-2] if past_key_values else 0
            position_ids = torch.arange(past_length, input_ids.shape[1] + past_length, device=input_ids.device).unsqueeze(0)

        hidden_states = self.embed_tokens(input_ids)
        next_decoder_cache = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    return lambda *inputs: module(*inputs)
                hidden_states, present_kv = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer), hidden_states, attention_mask, position_ids, past_kv, use_reentrant=False
                )
            else:
                hidden_states, present_kv = layer(hidden_states, attention_mask, position_ids, past_kv)
                
            if use_cache:
                next_decoder_cache.append(present_kv)

        return self.norm(hidden_states), next_decoder_cache


class GLM5ForCausalLM(GLM5PreTrainedModel, GenerationMixin):
    """
    GLM-5 Model with a Causal Language Modeling Head on top.
    """
    def __init__(self, config: GLM5Config):
        super().__init__(config)
        self.model = GLM5Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self, 
        input_ids, 
        attention_mask=None, 
        position_ids=None, 
        past_key_values=None, 
        labels=None, 
        use_cache=None, 
        **kwargs
    ):
        outputs, past_kv = self.model(
            input_ids, 
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            past_key_values=past_key_values, 
            use_cache=use_cache, 
            **kwargs
        )
        logits = self.lm_head(outputs)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = nn.CrossEntropyLoss()(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
            # Z-loss regularization as specified in GLM-5 paper
            z_loss = (torch.logsumexp(shift_logits, dim=-1) ** 2).mean()
            loss = ce_loss + self.config.z_loss_weight * z_loss

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_kv)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[-2]
            remove_prefix_length = past_length if input_ids.shape[1] > past_length else input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )