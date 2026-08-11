import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..utils.normalization import RMSNorm
from ..utils.mlp import SwiGLUMLP
from ..utils.attention import GroupedQueryAttention


class Qwen3Config(PretrainedConfig):
    """
    Configuration class for the Qwen3 language model.
    
    This configuration defines the model hyperparameters scaled to approximately
    1 Billion parameters (~0.994B) following the Qwen3 Technical Report and scaling laws:
    - Grouped Query Attention (GQA) with 12 Query heads and 4 Key-Value heads.
    - SwiGLU feed-forward network with intermediate dimension of 4096.
    - 28 Transformer decoder layers with hidden dimension of 1536.
    - Qwen byte-level BPE vocabulary size of 151,669.
    - Rotary Positional Embeddings (RoPE) with a base frequency of 1,000,000 (1M).
    - RMSNorm with epsilon of 1e-6 and QK-Normalization in attention modules.
    - No QKV bias across linear projections (removed in Qwen3 for stability).
    """
    model_type = "qwen3"
    architecture = "qwen3"

    def __init__(
        self,
        vocab_size: int = 151669,                # Qwen3 default vocabulary size
        hidden_size: int = 1536,                 # Scaled for ~1B parameter budget
        intermediate_size: int = 4096,           # SwiGLU intermediate dimension (~8/3 * hidden_size)
        num_hidden_layers: int = 28,             # Total Transformer decoder layers
        num_attention_heads: int = 12,           # Query attention heads (head_dim = 128)
        num_key_value_heads: int = 4,            # Key/Value heads for Grouped Query Attention (GQA)
        max_position_embeddings: int = 32768,    # 32K default sequence context window
        rope_theta: float = 1000000.0,           # 1M base frequency for RoPE (Qwen3 Sec 3.2)
        rms_norm_eps: float = 1e-6,               # Pre-normalization RMSNorm epsilon
        z_loss_weight: float = 1e-5,             # Auxiliary z-loss weight for large vocab stability
        use_yarn: bool = False,                  # YaRN scaling factor flag for context extension
        original_max_position_embeddings: int = 32768,
        tie_word_embeddings: bool = True,        # Tied input/output embeddings for ~1B budget
        initializer_range: float = 0.02,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = 151643,
        eos_token_id: Optional[int] = 151643,
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
        self.z_loss_weight = z_loss_weight
        self.use_yarn = use_yarn
        self.original_max_position_embeddings = original_max_position_embeddings
        self.initializer_range = initializer_range

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )


class Qwen3Block(nn.Module):
    """
    Qwen3 Decoder Transformer Block.
    
    Structure:
    1. Pre-RMSNorm -> Grouped Query Attention (GQA) with QK-Norm and RoPE -> Residual Connection
    2. Pre-RMSNorm -> SwiGLU Feed-Forward Network (MLP) -> Residual Connection
    """
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config, layer_idx=layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Attention block with Pre-RMSNorm and Residual Connection
        residual = hidden_states
        normed_hidden_states = self.input_layernorm(hidden_states)
        attn_outputs, present_kv = self.self_attn(
            normed_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )
        hidden_states = residual + attn_outputs

        # MLP Feed-Forward block with Pre-RMSNorm and Residual Connection
        residual = hidden_states
        normed_hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_outputs = self.mlp(normed_hidden_states)
        hidden_states = residual + mlp_outputs

        return hidden_states, present_kv


class Qwen3PreTrainedModel(PreTrainedModel):
    """
    Abstract base class for Qwen3 models to handle weight initialization and Hugging Face integration.
    """
    config_class = Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: nn.Module):
        """Initializes weight parameters according to Qwen3 specification."""
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
    Core Qwen3 Transformer Backbone Model.
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
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: Optional[bool] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]]:
        if use_cache is None:
            use_cache = getattr(self.config, "use_cache", False)

        if position_ids is None:
            past_length = past_key_values[0][0].shape[-2] if past_key_values is not None else 0
            position_ids = torch.arange(
                past_length, input_ids.shape[1] + past_length, device=input_ids.device
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
                    use_reentrant=False
                )
            else:
                hidden_states, present_kv = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_kv
                )

            if use_cache:
                next_decoder_cache.append(present_kv)

        hidden_states = self.norm(hidden_states)
        return hidden_states, next_decoder_cache


class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    """
    Qwen3 Language Model with a Causal Language Modeling head for Next-Token Prediction.
    """
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder: Qwen3Model):
        self.model = decoder

    def get_decoder(self) -> Qwen3Model:
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
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
            **kwargs
        )

        logits = self.lm_head(outputs)

        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()

            # Cross-Entropy Loss
            ce_loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )

            # Auxiliary z-loss to prevent logit drift over large vocabulary (151,669)
            z_loss = (torch.logsumexp(shift_logits, dim=-1) ** 2).mean()
            loss = ce_loss + getattr(self.config, "z_loss_weight", 1e-5) * z_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_kv,
        )

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, past_key_values: Optional[Tuple] = None, attention_mask: Optional[torch.Tensor] = None, **kwargs
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

    def _reorder_cache(self, past_key_values: Tuple, beam_idx: torch.LongTensor) -> Tuple:
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )