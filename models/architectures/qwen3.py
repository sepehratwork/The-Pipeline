"""
Qwen3 Language Model Architecture Implementation.

This module implements the Qwen3 causal language model architecture based on the
Qwen3 Technical Report (2025). Key features include Grouped Query Attention (GQA),
QK-Normalization for training stability, SwiGLU activations, RMSNorm pre-normalization,
and RoPE positional embeddings with extended context scaling.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..utils.normalization import RMSNorm
from ..utils.mlp import SwiGLUMLP
from ..utils.attention import GroupedQueryAttention


class Qwen3Config(PretrainedConfig):
    """
    Configuration class for Qwen3 model architecture.
    
    Hyperparameters are sized for ~1.0 Billion parameters (Wider & Shallower depth/width ratio
    optimized for inference latency and throughput per Scaling Laws).
    """
    architecture = "qwen3"

    def __init__(
        self,
        vocab_size: int = 151669,           # Standard Qwen BBPE Tokenizer vocabulary size
        hidden_size: int = 1536,            # Hidden dimension (~1B parameter configuration)
        intermediate_size: int = 4864,      # SwiGLU intermediate dimension
        num_hidden_layers: int = 28,        # Number of Transformer decoder layers
        num_attention_heads: int = 12,      # Number of Query attention heads (head_dim = 128)
        num_key_value_heads: int = 4,       # Grouped Query Attention (3:1 ratio)
        max_position_embeddings: int = 32768,# Standard context window length
        rope_theta: float = 1000000.0,      # RoPE base frequency (ABF tech from paper)
        rms_norm_eps: float = 1e-6,         # RMSNorm epsilon
        use_qk_norm: bool = True,           # Enable QK-Norm for training stability
        use_yarn: bool = False,             # YaRN long-context extension flag
        original_max_position_embeddings: int = 32768,
        tie_word_embeddings: bool = True,   # Tied embeddings to fit parameter budget
        z_loss_weight: float = 0.0,         # Optional auxiliary loss weight
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.use_qk_norm = use_qk_norm
        self.use_yarn = use_yarn
        self.original_max_position_embeddings = original_max_position_embeddings
        self.tie_word_embeddings = tie_word_embeddings
        self.z_loss_weight = z_loss_weight

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Qwen3Block(nn.Module):
    """
    Single Decoder Block for Qwen3 Transformer architecture.
    
    Composes RMSNorm Pre-normalization, Grouped Query Attention with QK-Norm,
    and SwiGLU MLP with residual skip connections.
    """
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Self-Attention with pre-normalization & residual connection
        residual = hidden_states
        normed_hidden_states = self.input_layernorm(hidden_states)
        attn_outputs, present_kv = self.self_attn(
            normed_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )
        hidden_states = residual + attn_outputs

        # Feed-Forward Network with pre-normalization & residual connection
        residual = hidden_states
        normed_hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(normed_hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states, present_kv


class Qwen3PreTrainedModel(PreTrainedModel):
    """
    Base class for Qwen3 models handling weight initialization and HF integration.
    """
    config_class = Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: nn.Module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Qwen3Model(Qwen3PreTrainedModel):
    """
    Core Qwen3 Transformer Decoder Backbone.
    """
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Qwen3Block(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        use_cache = use_cache if use_cache is not None else getattr(self.config, "use_cache", False)

        if input_ids is None:
            raise ValueError("input_ids cannot be None.")

        batch_size, seq_length = input_ids.shape

        if position_ids is None:
            past_length = past_key_values[0][0].shape[-2] if past_key_values is not None else 0
            position_ids = torch.arange(
                past_length, seq_length + past_length, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0)

        hidden_states = self.embed_tokens(input_ids)
        next_decoder_cache = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    return lambda *inputs: module(*inputs)
                hidden_states, present_kv = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_kv,
                    use_reentrant=False,
                )
            else:
                hidden_states, present_kv = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_kv,
                )

            if use_cache:
                next_decoder_cache.append(present_kv)

        hidden_states = self.norm(hidden_states)
        return hidden_states, next_decoder_cache


class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    """
    Qwen3 Model with Causal Language Modeling Head.
    """
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights if specified by configuration
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        outputs, past_kv = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        logits = self.lm_head(outputs)

        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            ce_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
            loss = ce_loss
            if getattr(self.config, "z_loss_weight", 0.0) > 0:
                z_loss = (torch.logsumexp(shift_logits, dim=-1) ** 2).mean()
                loss = loss + self.config.z_loss_weight * z_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_kv,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> dict:
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[-2]
            remove_prefix_length = past_length if input_ids.shape[1] > past_length else input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

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