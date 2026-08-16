import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..utils.normalization import RMSNorm
from ..utils.mlp import SwiGLUMLP, TopKMoE
from ..utils.attention import GroupedQueryAttention


class MagistralConfig(PretrainedConfig):
    """
    Configuration class for the Magistral reasoning language model (arXiv:2506.10910v1).
    
    Default parameters are scaled to target a ~1 Billion parameter budget (or 1 Billion active
    parameters in MoE mode) while preserving the architectural design of Magistral Small (24B)
    and Magistral Medium (Mistral Small 3 / Mistral Medium 3 base).
    """
    architecture = "magistral"

    def __init__(
        self,
        vocab_size: int = 32768,
        hidden_size: int = 2048,
        intermediate_size: int = 5632,        # SwiGLU intermediate size for ~1B parameter budget
        num_hidden_layers: int = 22,          # 22 layers to achieve ~1.05B parameters
        num_attention_heads: int = 16,        # 16 Query heads (head_dim = 128)
        num_key_value_heads: int = 4,         # GQA with 4:1 ratio
        head_dim: int = 128,
        max_position_embeddings: int = 32768, # Long-context capability up to 32k/40k tokens
        sliding_window: int = 4096,           # Sliding Window Attention (SWA) limit
        rope_theta: float = 1000000.0,        # High rope_theta for reasoning long context
        rms_norm_eps: float = 1e-6,
        use_moe: bool = False,                 # Set True for Mixture-of-Experts variant
        num_local_experts: int = 8,           # Total routed experts in MoE mode
        num_experts_per_tok: int = 2,         # Top-2 active routed experts
        moe_intermediate_size: int = 1408,    # Expert size for 1B active parameter MoE budget
        num_shared_experts: int = 1,          # Shared expert count
        z_loss_weight: float = 1e-5,          # Logit stability z-loss factor
        use_yarn: bool = False,
        original_max_position_embeddings: int = 8192,
        tie_word_embeddings: bool = True,     # Tied embeddings standard for 1B scale
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.sliding_window = sliding_window
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.use_moe = use_moe
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.num_shared_experts = num_shared_experts
        self.z_loss_weight = z_loss_weight
        self.use_yarn = use_yarn
        self.original_max_position_embeddings = original_max_position_embeddings
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class MagistralTestConfig(MagistralConfig):
    """
    100M Parameter Test Configuration for Magistral.
    """
    architecture = "magistral_test"

    def __init__(
        self,
        vocab_size: int = 32768,
        hidden_size: int = 768,
        intermediate_size: int = 2048,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 3,
        head_dim: int = 64,
        max_position_embeddings: int = 8192,
        sliding_window: int = 2048,
        rope_theta: float = 1000000.0,
        rms_norm_eps: float = 1e-6,
        use_moe: bool = False,
        num_local_experts: int = 8,
        num_experts_per_tok: int = 2,
        moe_intermediate_size: int = 512,
        num_shared_experts: int = 1,
        z_loss_weight: float = 1e-5,
        use_yarn: bool = False,
        original_max_position_embeddings: int = 8192,
        tie_word_embeddings: bool = True,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            sliding_window=sliding_window,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            use_moe=use_moe,
            num_local_experts=num_local_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            num_shared_experts=num_shared_experts,
            z_loss_weight=z_loss_weight,
            use_yarn=use_yarn,
            original_max_position_embeddings=original_max_position_embeddings,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )


class MagistralDecoderLayer(nn.Module):
    """
    Decoder layer for Magistral language model.
    Composes pre-layernorm RMSNorm, Grouped Query Attention (GQA), and SwiGLU MLP / TopKMoE.
    """
    def __init__(self, config: MagistralTestConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if getattr(config, "use_moe", False):
            self.mlp = TopKMoE(config, layer_idx)
        else:
            self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        past_key_value: tuple = None
    ):
        # Attention block with residual connection
        residual = hidden_states
        normed_hidden = self.input_layernorm(hidden_states)
        attn_out, present_kv = self.self_attn(
            normed_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value
        )
        hidden_states = residual + attn_out

        # Feed-forward block with residual connection
        residual = hidden_states
        normed_hidden = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(normed_hidden)
        hidden_states = residual + mlp_out

        return hidden_states, present_kv


class MagistralPreTrainedModel(PreTrainedModel):
    """
    Base PreTrainedModel class for Magistral architecture weight initialization
    and Hugging Face compatibility.
    """
    config_class = MagistralTestConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _check_and_adjust_experts_implementation(self, experts_implementation):
        return experts_implementation

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


class MagistralModel(MagistralPreTrainedModel):
    """
    Transformer core decoder for Magistral reasoning language model.
    """
    def __init__(self, config: MagistralTestConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            MagistralDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values: tuple = None,
        use_cache: bool = None,
        **kwargs
    ):
        if use_cache is None:
            use_cache = getattr(self.config, "use_cache", False)

        if position_ids is None:
            past_length = past_key_values[0][0].shape[-2] if past_key_values else 0
            position_ids = torch.arange(
                past_length, input_ids.shape[1] + past_length, device=input_ids.device
            ).unsqueeze(0)

        hidden_states = self.embed_tokens(input_ids)
        next_decoder_cache = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
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


class MagistralForCausalLM(MagistralPreTrainedModel, GenerationMixin):
    """
    Magistral Causal Language Model with LM Head for auto-regressive generation
    and reinforcement learning from verifiable rewards (RLVR).
    """
    def __init__(self, config: MagistralTestConfig):
        super().__init__(config)
        self.model = MagistralModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values: tuple = None,
        labels: torch.LongTensor = None,
        use_cache: bool = None,
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
            ce_loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            )
            # Logit regularization (z-loss)
            z_loss = (torch.logsumexp(shift_logits, dim=-1) ** 2).mean()
            loss = ce_loss + getattr(self.config, "z_loss_weight", 1e-5) * z_loss

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_kv)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
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