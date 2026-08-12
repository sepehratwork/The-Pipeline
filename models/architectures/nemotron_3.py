import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..utils.normalization import RMSNorm
from ..utils.mlp import SwiGLUMLP, LatentMoE
from ..utils.attention import NoRopeGroupedQueryAttention
from ..utils.mamba import Mamba2Layer


class Nemotron3Config(PretrainedConfig):
    """
    Configuration class for NVIDIA Nemotron 3 family models.
    Supports both MoE Hybrid (LatentMoE) and Dense Hybrid (Dense Mamba-Transformer) architectures.
    Hyperparameters default to 1 Billion Active Parameters scaling budget.
    """
    architecture = "nemotron_3"

    def __init__(
        self,
        vocab_size: int = 100278,
        hidden_size: int = 2048,
        intermediate_size: int = 4096,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 2,           # GQA with 2 KV heads per paper spec
        attn_layer_indices: Optional[List[int]] = None,
        max_position_embeddings: int = 1048576, # 1M context length support
        is_moe: bool = True,                     # Toggle between MoE Hybrid and Dense Hybrid
        latent_dim: int = 512,                  # Latent dimension for LatentMoE (d/4 = 2048/4 = 512)
        num_routed_experts: int = 32,            # Total LatentMoE experts
        num_active_experts: int = 4,             # Active LatentMoE experts per token
        use_mtp: bool = True,                    # Multi-Token Prediction layers
        num_mtp_tokens: int = 2,                 # Predict 2 future draft tokens
        z_loss_weight: float = 1e-5,
        tie_word_embeddings: bool = True,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        
        # Interleaved layer layout: Attention placed selectively at layers 7, 15, 23 by default
        if attn_layer_indices is None:
            self.attn_layer_indices = [7, 15, 23]
        else:
            self.attn_layer_indices = attn_layer_indices

        self.max_position_embeddings = max_position_embeddings
        self.is_moe = is_moe
        self.latent_dim = latent_dim
        self.num_routed_experts = num_routed_experts
        self.num_active_experts = num_active_experts
        self.use_mtp = use_mtp
        self.num_mtp_tokens = num_mtp_tokens
        self.z_loss_weight = z_loss_weight
        self.tie_word_embeddings = tie_word_embeddings

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Nemotron3TestConfig(Nemotron3Config):
    """
    Test configuration for Nemotron 3 scaled to ~200M active parameters.
    """
    architecture = "nemotron_3_test"

    def __init__(
        self,
        vocab_size: int = 100278,
        hidden_size: int = 768,
        intermediate_size: int = 2048,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        attn_layer_indices: Optional[List[int]] = None,
        max_position_embeddings: int = 131072,
        is_moe: bool = True,
        latent_dim: int = 192,
        num_routed_experts: int = 16,
        num_active_experts: int = 2,
        use_mtp: bool = True,
        num_mtp_tokens: int = 2,
        z_loss_weight: float = 1e-5,
        tie_word_embeddings: bool = True,
        **kwargs
    ):
        if attn_layer_indices is None:
            attn_layer_indices = [3, 7, 11]

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attn_layer_indices=attn_layer_indices,
            max_position_embeddings=max_position_embeddings,
            is_moe=is_moe,
            latent_dim=latent_dim,
            num_routed_experts=num_routed_experts,
            num_active_experts=num_active_experts,
            use_mtp=use_mtp,
            num_mtp_tokens=num_mtp_tokens,
            z_loss_weight=z_loss_weight,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )


class Nemotron3Block(nn.Module):
    """
    Nemotron 3 Hybrid Decoder Block.
    Predominantly interleaves Mamba-2 layers with MoE/MLP layers, with a select few
    self-attention layers (No-RoPE GQA) to achieve high efficiency and context scaling.
    """
    def __init__(self, config: Nemotron3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_attn_layer = layer_idx in config.attn_layer_indices

        self.input_layernorm = RMSNorm(config.hidden_size)

        # 1. Sequence Modeling Layer (Mamba-2 or No-RoPE GQA)
        if self.is_attn_layer:
            self.mixer = NoRopeGroupedQueryAttention(config, layer_idx)
        else:
            self.mixer = Mamba2Layer(d_model=config.hidden_size)

        self.post_mixer_layernorm = RMSNorm(config.hidden_size)

        # 2. Feed-Forward Layer (LatentMoE or SwiGLUMLP)
        if config.is_moe:
            self.mlp = LatentMoE(config, layer_idx)
        else:
            self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        # 1. Sequence Mixer Stage
        residual = hidden_states
        mixer_states, present_kv = self.mixer(
            self.input_layernorm(hidden_states), 
            attention_mask=attention_mask if self.is_attn_layer else None,
            position_ids=position_ids if self.is_attn_layer else None, 
            past_key_value=past_key_value
        )
        hidden_states = residual + mixer_states

        # 2. Feed-Forward Stage
        residual = hidden_states
        hidden_states = self.mlp(self.post_mixer_layernorm(hidden_states))
        hidden_states = residual + hidden_states

        return hidden_states, present_kv


class MultiTokenPredictionHead(nn.Module):
    """
    Multi-Token Prediction (MTP) module (Section 2.3).
    Auxiliary token prediction layers generating predictions for future tokens
    to accelerate long-form text generation via native speculative decoding.
    """
    def __init__(self, hidden_size: int, vocab_size: int, num_mtp_tokens: int = 2):
        super().__init__()
        self.num_mtp_tokens = num_mtp_tokens
        self.mtp_heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size, bias=False)
            for _ in range(num_mtp_tokens)
        ])

    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        return [head(hidden_states) for head in self.mtp_heads]


class Nemotron3PreTrainedModel(PreTrainedModel):
    """Base class for Nemotron 3 pretrained models handling weights initialization."""
    config_class = Nemotron3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _check_and_adjust_experts_implementation(self, experts_implementation):
        return experts_implementation

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


class Nemotron3Model(Nemotron3PreTrainedModel):
    """Core Nemotron 3 Transformer stack."""
    def __init__(self, config: Nemotron3Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Nemotron3Block(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.Tensor] = None, 
        past_key_values: Optional[List[torch.Tensor]] = None, 
        use_cache: Optional[bool] = None, 
        **kwargs
    ):
        if use_cache is None:
            use_cache = getattr(self.config, "use_cache", False)

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


class Nemotron3ForCausalLM(Nemotron3PreTrainedModel, GenerationMixin):
    """
    Nemotron 3 Language Model for Causal Language Modeling with MoE Hybrid (LatentMoE).
    Features granular reasoning budget control and optional Multi-Token Prediction (MTP).
    Scaled to ~1 Billion Active Parameters per token.
    """
    def __init__(self, config: Nemotron3Config):
        config.is_moe = True
        super().__init__(config)
        self.model = Nemotron3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.use_mtp:
            self.mtp = MultiTokenPredictionHead(config.hidden_size, config.vocab_size, config.num_mtp_tokens)
        else:
            self.mtp = None

        self.post_init()

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.Tensor] = None, 
        past_key_values: Optional[List[torch.Tensor]] = None, 
        labels: Optional[torch.Tensor] = None, 
        use_cache: Optional[bool] = None, 
        **kwargs
    ):
        outputs, past_kv = self.model(
            input_ids, attention_mask=attention_mask, position_ids=position_ids, 
            past_key_values=past_key_values, use_cache=use_cache, **kwargs
        )
        logits = self.lm_head(outputs)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            
            # Primary Cross Entropy Loss
            ce_loss = nn.CrossEntropyLoss()(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            z_loss = (torch.logsumexp(shift_logits, dim=-1) ** 2).mean()
            loss = ce_loss + self.config.z_loss_weight * z_loss

            # Multi-Token Prediction (MTP) Loss integration
            if self.mtp is not None and self.training:
                mtp_logits_list = self.mtp(outputs)
                for k, mtp_logits in enumerate(mtp_logits_list):
                    shift_k = k + 2
                    if labels.shape[1] >= shift_k:
                        m_logits = mtp_logits[..., :-shift_k, :].contiguous().float()
                        m_labels = labels[..., shift_k:].contiguous()
                        loss += 0.3 * nn.CrossEntropyLoss()(m_logits.view(-1, self.config.vocab_size), m_labels.view(-1))

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_kv)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[-2] if isinstance(past_key_values[0], tuple) else 0
            if input_ids.shape[1] > past_length:
                input_ids = input_ids[:, past_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

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


class Nemotron3DenseForCausalLM(Nemotron3PreTrainedModel, GenerationMixin):
    """
    Nemotron 3 Language Model for Causal Language Modeling with Dense Hybrid Architecture
    (Mamba-2 + Transformer Dense Feed-Forward layers).
    Scaled to ~1 Billion Parameters.
    """
    def __init__(self, config: Nemotron3Config):
        config.is_moe = False
        super().__init__(config)
        self.model = Nemotron3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.Tensor] = None, 
        past_key_values: Optional[List[torch.Tensor]] = None, 
        labels: Optional[torch.Tensor] = None, 
        use_cache: Optional[bool] = None, 
        **kwargs
    ):
        outputs, past_kv = self.model(
            input_ids, attention_mask=attention_mask, position_ids=position_ids, 
            past_key_values=past_key_values, use_cache=use_cache, **kwargs
        )
        logits = self.lm_head(outputs)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = nn.CrossEntropyLoss()(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            z_loss = (torch.logsumexp(shift_logits, dim=-1) ** 2).mean()
            loss = ce_loss + self.config.z_loss_weight * z_loss

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_kv)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[-2] if isinstance(past_key_values[0], tuple) else 0
            if input_ids.shape[1] > past_length:
                input_ids = input_ids[:, past_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

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