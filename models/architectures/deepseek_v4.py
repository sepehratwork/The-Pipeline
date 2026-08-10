import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..utils.normalization import RMSNorm
from ..utils.mlp import DeepSeekMoE
from ..utils.attention import DeepSeekV4HybridAttention
from ..utils.mhc import ManifoldConstrainedHyperConnections


class DeepSeekV4Config(PretrainedConfig):
    """
    Configuration for DeepSeek-V4 model (scaled to ~1 Billion active parameters).
    """
    architecture = "deepseek_v4"

    def __init__(
        self,
        vocab_size=100278,
        hidden_size=1536,                # Hidden dimension d
        intermediate_size=1024,          # Expert intermediate dimension
        num_hidden_layers=24,            # Total Transformer blocks
        num_attention_heads=12,          # Query heads
        query_compression_dim=512,       # Latent query compression dimension d_c
        n_hc=4,                          # mHC expansion factor
        num_routed_experts=32,           # Total fine-grained routed experts
        num_active_experts=4,            # Active experts per token
        num_shared_experts=1,            # Shared experts count
        csa_compression_rate=4,          # CSA compression rate m
        hca_compression_rate=16,         # HCA compression rate m'
        window_size=128,                 # SWA window size n_win
        rope_theta=500000.0,
        max_position_embeddings=8192,
        z_loss_weight=1e-5,
        tie_word_embeddings=True,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.query_compression_dim = query_compression_dim
        self.n_hc = n_hc
        self.num_routed_experts = num_routed_experts
        self.num_active_experts = num_active_experts
        self.num_shared_experts = num_shared_experts
        self.csa_compression_rate = csa_compression_rate
        self.hca_compression_rate = hca_compression_rate
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.z_loss_weight = z_loss_weight
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class DeepSeekV4Block(nn.Module):
    """
    DeepSeek-V4 Transformer Block integrating Manifold-Constrained Hyper-Connections (mHC),
    Hybrid Attention (CSA/HCA/SWA), and DeepSeekMoE.
    """
    def __init__(self, config: DeepSeekV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        # Pre-attention and Pre-MoE RMSNorms
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)

        # Hybrid Attention (CSA/HCA/SWA)
        self.self_attn = DeepSeekV4HybridAttention(config, layer_idx)

        # DeepSeekMoE (Hash routing for initial 3 layers, Sqrt(Softplus) for remainder)
        use_hash = (layer_idx < 3)
        self.moe = DeepSeekMoE(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_routed_experts=config.num_routed_experts,
            num_active_experts=config.num_active_experts,
            num_shared_experts=config.num_shared_experts,
            use_hash_routing=use_hash
        )

        # Manifold-Constrained Hyper-Connections (mHC)
        self.mhc_attn = ManifoldConstrainedHyperConnections(config.hidden_size, n_hc=config.n_hc)
        self.mhc_moe = ManifoldConstrainedHyperConnections(config.hidden_size, n_hc=config.n_hc)

    def _attn_wrapper(self, x, attention_mask, position_ids, past_key_value):
        norm_x = self.input_layernorm(x)
        return self.self_attn(norm_x, attention_mask, position_ids, past_key_value)

    def _moe_wrapper(self, x, input_ids):
        norm_x = self.post_attention_layernorm(x)
        return self.moe(norm_x, input_ids=input_ids)

    def forward(self, residual_state, attention_mask=None, position_ids=None, past_key_value=None, input_ids=None):
        # 1. Attention Pass with mHC residual routing
        res_out = self.mhc_attn(
            residual_state, self._attn_wrapper, attention_mask, position_ids, past_key_value
        )
        if isinstance(res_out, tuple):
            residual_state, present_kv = res_out[0], res_out[1]
        else:
            residual_state, present_kv = res_out, None

        # 2. MoE Pass with mHC residual routing
        residual_state = self.mhc_moe(
            residual_state, self._moe_wrapper, input_ids
        )

        return residual_state, present_kv


class DeepSeekV4PreTrainedModel(PreTrainedModel):
    config_class = DeepSeekV4Config
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


class DeepSeekV4Model(DeepSeekV4PreTrainedModel):
    def __init__(self, config: DeepSeekV4Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DeepSeekV4Block(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=None, **kwargs):
        if use_cache is None:
            use_cache = getattr(self.config, "use_cache", False)

        bsz, seq_len = input_ids.shape
        if position_ids is None:
            past_length = past_key_values[0][0].shape[-2] if past_key_values else 0
            position_ids = torch.arange(past_length, seq_len + past_length, device=input_ids.device).unsqueeze(0)

        hidden_states = self.embed_tokens(input_ids)

        # Expand input to mHC residual stream [bsz, seq_len, n_hc, hidden_size]
        residual_state = hidden_states.unsqueeze(2).repeat(1, 1, self.config.n_hc, 1)

        next_decoder_cache = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    return lambda *inputs: module(*inputs)
                residual_state, present_kv = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    residual_state, attention_mask, position_ids, past_kv, input_ids,
                    use_reentrant=False
                )
            else:
                residual_state, present_kv = layer(
                    residual_state, attention_mask, position_ids, past_kv, input_ids
                )

            if use_cache:
                next_decoder_cache.append(present_kv)

        # Collapse mHC stream back to standard hidden size
        output_states = residual_state.mean(dim=2)
        return self.norm(output_states), next_decoder_cache


class DeepSeekV4ForCausalLM(DeepSeekV4PreTrainedModel, GenerationMixin):
    def __init__(self, config: DeepSeekV4Config):
        super().__init__(config)
        self.model = DeepSeekV4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, labels=None, use_cache=None, **kwargs):
        outputs, past_kv = self.model(
            input_ids, attention_mask, position_ids, past_key_values, use_cache, **kwargs
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