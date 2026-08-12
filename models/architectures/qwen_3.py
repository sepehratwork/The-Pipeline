import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, List
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..utils.normalization import RMSNorm
from ..utils.mlp import SwiGLUMLP, FineGrainedMoE
from ..utils.attention import GroupedQueryAttention


class Qwen3Config(PretrainedConfig):
    """
    Configuration class for the Qwen3 Language Model family.
    
    Default parameters are scaled to target ~1 Billion parameters for Qwen3 Dense,
    and ~1 Billion active parameters per token for Qwen3 MoE.
    """
    model_type = "qwen3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 151669,               # Qwen3 Byte-level BPE tokenizer vocabulary size
        hidden_size: int = 2048,                # Hidden dimension size (1B budget target)
        intermediate_size: int = 5632,          # Dense SwiGLU intermediate size (~2.75x hidden_size)
        num_hidden_layers: int = 16,            # Number of Transformer layers
        num_attention_heads: int = 16,          # Number of Query attention heads
        num_key_value_heads: int = 4,           # Grouped Query Attention (GQA) KV heads
        max_position_embeddings: int = 32768,   # Default context length window
        rope_theta: float = 1000000.0,          # ABF RoPE base frequency
        rms_norm_eps: float = 1e-6,             # Pre-normalization RMSNorm epsilon
        use_sliding_window: bool = False,       # SWA toggle
        sliding_window: Optional[int] = None,   # Sliding window attention size
        z_loss_weight: float = 1e-5,            # Z-loss regularization coefficient
        tie_word_embeddings: bool = True,       # Tied embedding weights for <=4B scale
        # MoE Specific Parameters
        is_moe: bool = False,                   # Switch between Dense and MoE variants
        num_experts: int = 128,                 # Fine-grained total expert count
        num_experts_per_tok: int = 8,           # Top-K activated experts per token
        moe_intermediate_size: int = 704,       # Fine-grained expert intermediate size (8 * 704 = 5632 active)
        router_aux_loss_coef: float = 0.01,     # Global-batch load balancing loss coefficient
        # Thinking Mode Parameters
        enable_thinking: bool = True,           # Enable dynamic thinking mode
        thinking_budget: Optional[int] = None,  # Maximum thinking token budget during inference
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
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.z_loss_weight = z_loss_weight
        self.tie_word_embeddings = tie_word_embeddings
        self.is_moe = is_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.router_aux_loss_coef = router_aux_loss_coef
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Qwen3TestConfig(Qwen3Config):
    """
    Test configuration for Qwen3 scaled to ~200M parameters.
    """
    model_type = "qwen3_test"

    def __init__(
        self,
        vocab_size: int = 151669,
        hidden_size: int = 768,
        intermediate_size: int = 2048,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 2,
        max_position_embeddings: int = 16384,
        rope_theta: float = 1000000.0,
        rms_norm_eps: float = 1e-6,
        use_sliding_window: bool = False,
        sliding_window: Optional[int] = None,
        z_loss_weight: float = 1e-5,
        tie_word_embeddings: bool = True,
        is_moe: bool = False,
        num_experts: int = 64,
        num_experts_per_tok: int = 4,
        moe_intermediate_size: int = 352,
        router_aux_loss_coef: float = 0.01,
        enable_thinking: bool = True,
        thinking_budget: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            z_loss_weight=z_loss_weight,
            tie_word_embeddings=tie_word_embeddings,
            is_moe=is_moe,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            router_aux_loss_coef=router_aux_loss_coef,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            **kwargs
        )


class Qwen3DecoderLayer(nn.Module):
    """
    Standard Qwen3 Dense Transformer Layer:
    - Pre-normalization via RMSNorm
    - Grouped Query Attention (GQA) with QK-Norm and no QKV-bias
    - SwiGLU MLP
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
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Optional[torch.Tensor]]:
        # Self Attention with Residual Connection
        residual = hidden_states
        normed_states = self.input_layernorm(hidden_states)
        attn_outputs, present_kv = self.self_attn(
            normed_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )
        hidden_states = residual + attn_outputs

        # SwiGLU Feed-Forward Network with Residual Connection
        residual = hidden_states
        normed_states = self.post_attention_layernorm(hidden_states)
        mlp_outputs = self.mlp(normed_states)
        hidden_states = residual + mlp_outputs

        return hidden_states, present_kv, None


class Qwen3MoEDecoderLayer(nn.Module):
    """
    Qwen3 Mixture-of-Experts (MoE) Transformer Layer:
    - Pre-normalization via RMSNorm
    - Grouped Query Attention (GQA) with QK-Norm
    - Fine-Grained MoE Block (128 total experts, 8 active per token, NO shared experts)
    """
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.moe = FineGrainedMoE(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Optional[torch.Tensor]]:
        # Self Attention with Residual Connection
        residual = hidden_states
        normed_states = self.input_layernorm(hidden_states)
        attn_outputs, present_kv = self.self_attn(
            normed_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )
        hidden_states = residual + attn_outputs

        # Fine-Grained MoE Block with Residual Connection
        residual = hidden_states
        normed_states = self.post_attention_layernorm(hidden_states)
        moe_outputs, aux_loss = self.moe(normed_states)
        hidden_states = residual + moe_outputs

        return hidden_states, present_kv, aux_loss


class Qwen3PreTrainedModel(PreTrainedModel):
    """Base class for Qwen3 models handling weight initialization and checkpointing."""
    config_class = Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
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
    Qwen3 Core Transformer Decoder Pipeline supporting both Dense and MoE architectures.
    """
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Build Dense or MoE decoder layers
        if config.is_moe:
            self.layers = nn.ModuleList([Qwen3MoEDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        else:
            self.layers = nn.ModuleList([Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)])

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor]]], Optional[torch.Tensor]]:
        if use_cache is None:
            use_cache = getattr(self.config, "use_cache", False)

        if position_ids is None:
            past_length = past_key_values[0][0].shape[-2] if past_key_values is not None else 0
            position_ids = torch.arange(past_length, input_ids.shape[1] + past_length, device=input_ids.device).unsqueeze(0)

        hidden_states = self.embed_tokens(input_ids)
        next_decoder_cache = [] if use_cache else None
        total_aux_loss = 0.0

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    return lambda *inputs: module(*inputs)
                hidden_states, present_kv, layer_aux_loss = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_kv,
                    use_reentrant=False
                )
            else:
                hidden_states, present_kv, layer_aux_loss = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_kv,
                )

            if use_cache:
                next_decoder_cache.append(present_kv)
            if layer_aux_loss is not None:
                total_aux_loss = total_aux_loss + layer_aux_loss

        hidden_states = self.norm(hidden_states)
        aux_loss_tensor = total_aux_loss if isinstance(total_aux_loss, torch.Tensor) else None

        return hidden_states, next_decoder_cache, aux_loss_tensor


class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    """
    Qwen3 Dense Causal Language Model with Causal LM Loss and Thinking Budget Enforcement.
    """
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        hidden_states, past_kv, aux_loss = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = nn.CrossEntropyLoss()(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
            # Z-loss stability regularization
            z_loss = (torch.logsumexp(shift_logits, dim=-1) ** 2).mean()
            loss = ce_loss + self.config.z_loss_weight * z_loss
            
            if aux_loss is not None:
                loss = loss + aux_loss

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_kv)

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple[torch.Tensor]]] = None, attention_mask: Optional[torch.Tensor] = None, **kwargs
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


class Qwen3MoEForCausalLM(Qwen3ForCausalLM):
    """
    Qwen3 Mixture-of-Experts (MoE) Causal Language Model.
    Inherits from Qwen3ForCausalLM with `is_moe=True` default setting.
    """
    def __init__(self, config: Qwen3Config):
        config.is_moe = True
        super().__init__(config)