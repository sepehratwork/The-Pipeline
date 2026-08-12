import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..utils.normalization import RMSNorm
from ..utils.mlp import SwiGLUMLP
from ..utils.attention import GroupedQueryAttention


class MobileLLMProConfig(PretrainedConfig):
    """
    Configuration class for MobileLLM-Pro (1.08B Parameters).
    
    Ref: MobileLLM-Pro Technical Report (Meta Reality Labs, 2025)
    - 30 Transformer layers
    - Hidden dim: 1280, FFN dim: 6144 (4.8x scaling)
    - 20 Query Heads, 4 KV Heads (GQA)
    - Vocab size: 202,048 with shared input/output embeddings
    - Context length: 128,000 tokens (128k)
    - Interleaved Local-Global Attention (512 local sliding window)
    """
    architecture = "mobilellm_pro"
    model_type = "mobilellm_pro"

    def __init__(
        self,
        vocab_size=202048,
        hidden_size=1280,
        intermediate_size=6144,
        num_hidden_layers=30,
        num_attention_heads=20,
        num_key_value_heads=4,
        max_position_embeddings=128000,
        sliding_window=512,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.sliding_window = sliding_window
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class MobileLLMProBlock(nn.Module):
    """Individual MobileLLM-Pro Transformer Layer with RMSNorm, GQA/SWA, and SwiGLU FFN."""
    def __init__(self, config: MobileLLMProConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        residual = hidden_states
        hidden_states, present_kv = self.self_attn(
            self.input_layernorm(hidden_states), attention_mask, position_ids, past_key_value
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states, present_kv


class MobileLLMProPreTrainedModel(PreTrainedModel):
    """Base PreTrainedModel class handling weight initialization for MobileLLM-Pro."""
    config_class = MobileLLMProConfig
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


class MobileLLMProModel(MobileLLMProPreTrainedModel):
    """MobileLLM-Pro Transformer Backbone."""
    def __init__(self, config: MobileLLMProConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MobileLLMProBlock(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=None, **kwargs):
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


class MobileLLMProForCausalLM(MobileLLMProPreTrainedModel, GenerationMixin):
    """MobileLLM-Pro Model with Causal Language Modeling Head for On-Device Deployment."""
    def __init__(self, config: MobileLLMProConfig):
        super().__init__(config)
        self.model = MobileLLMProModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, labels=None, use_cache=None, **kwargs):
        outputs, past_kv = self.model(input_ids, attention_mask, position_ids, past_key_values, use_cache, **kwargs)
        logits = self.lm_head(outputs)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

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