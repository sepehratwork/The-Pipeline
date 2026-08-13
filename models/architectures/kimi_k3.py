import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..utils.normalization import RMSNorm
from ..utils.mlp import SiTUGLU, StableLatentMoE
from ..utils.attention import KimiDeltaAttention, MultiLatentAttention


class BlockAttentionResiduals(nn.Module):
    """
    Block Attention Residuals (Block AttnRes) as described in Kimi K3 (Section 2.2, Eq. 8-10).
    Partition L layers into N blocks. Each layer selectively retrieves representations across
    preceding block outputs using learnable layer-specific pseudo-queries w_l and softmax attention.
    """
    def __init__(self, hidden_size: int, num_layers: int, num_blocks: int = 8, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.norm = RMSNorm(hidden_size, eps=eps)

        # Learnable layer-specific pseudo-queries w_l
        self.pseudo_queries = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_size) * 0.02) for _ in range(num_layers)
        ])

    def forward(
        self,
        layer_idx: int,
        block_representations: List[torch.Tensor],
        current_intra_sum: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        keys = list(block_representations)
        if current_intra_sum is not None:
            keys.append(current_intra_sum)

        K = torch.stack(keys, dim=0)  # (K_len, bsz, seq_len, hidden_size)
        q = self.pseudo_queries[layer_idx]  # (hidden_size,)

        K_normed = self.norm(K)
        logits = torch.einsum("d, kbsd -> kbs", q, K_normed)  # (K_len, bsz, seq_len)
        attn_weights = F.softmax(logits, dim=0).unsqueeze(-1)  # (K_len, bsz, seq_len, 1)

        return (attn_weights * K).sum(dim=0)


class KimiK3Config(PretrainedConfig):
    """
    Configuration class for the Kimi K3 Model.
    
    Configured for ~1 Billion Active Parameters using scaling laws and paper specs:
    - 21 Layers (5 Blocks of 4 layers + 1 final Gated MLA layer = 21 layers: 15 KDA + 6 Gated MLA)
    - 1M Context Length capability with NoPE (No Position Encoding)
    - Hybrid KDA-MLA Attention (3:1 ratio)
    - Stable LatentMoE or Dense SiTU-GLU FFN option
    """
    architecture = "kimi_k3"

    def __init__(
        self,
        vocab_size: int = 160000,
        hidden_size: int = 1536,
        num_hidden_layers: int = 21,
        num_attention_heads: int = 12,
        head_dim: int = 128,
        kda_ratio: int = 3,  # 3 KDA layers to 1 Gated MLA layer
        latent_dim: int = 768,  # 0.5 * hidden_size
        num_routed_experts: int = 64,
        num_active_experts: int = 4,
        num_shared_experts: int = 1,
        moe_intermediate_size: int = 1024,
        shared_intermediate_size: int = 2048,
        dense_intermediate_size: int = 4096,
        use_moe: bool = True,  # True for MoE, False for Dense version
        attn_res_num_blocks: int = 8,
        max_position_embeddings: int = 1000000,
        rms_norm_eps: float = 1e-6,
        z_loss_weight: float = 1e-5,
        tie_word_embeddings: bool = True,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.kda_ratio = kda_ratio
        self.latent_dim = latent_dim
        self.num_routed_experts = num_routed_experts
        self.num_active_experts = num_active_experts
        self.num_shared_experts = num_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_intermediate_size = shared_intermediate_size
        self.dense_intermediate_size = dense_intermediate_size
        self.use_moe = use_moe
        self.attn_res_num_blocks = attn_res_num_blocks
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.z_loss_weight = z_loss_weight
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class KimiK3TestConfig(KimiK3Config):
    """
    100M Active Parameter Test Configuration for Kimi K3.
    """
    def __init__(
        self,
        vocab_size: int = 160000,
        hidden_size: int = 512,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 8,
        head_dim: int = 64,
        kda_ratio: int = 3,
        latent_dim: int = 256,
        num_routed_experts: int = 32,
        num_active_experts: int = 4,
        num_shared_experts: int = 1,
        moe_intermediate_size: int = 384,
        shared_intermediate_size: int = 768,
        dense_intermediate_size: int = 1536,
        use_moe: bool = True,
        attn_res_num_blocks: int = 4,
        max_position_embeddings: int = 128000,
        rms_norm_eps: float = 1e-6,
        z_loss_weight: float = 1e-5,
        tie_word_embeddings: bool = True,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            kda_ratio=kda_ratio,
            latent_dim=latent_dim,
            num_routed_experts=num_routed_experts,
            num_active_experts=num_active_experts,
            num_shared_experts=num_shared_experts,
            moe_intermediate_size=moe_intermediate_size,
            shared_intermediate_size=shared_intermediate_size,
            dense_intermediate_size=dense_intermediate_size,
            use_moe=use_moe,
            attn_res_num_blocks=attn_res_num_blocks,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            z_loss_weight=z_loss_weight,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )


class KimiK3Block(nn.Module):
    """
    Building block for Kimi K3, integrating Attention (KDA or Gated MLA),
    Feed-Forward Network (Stable LatentMoE or Dense SiTU-GLU), and RMSNorm.
    """
    def __init__(self, config: KimiK3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Hybrid Attention: Every 4th layer (3:1 ratio) or final layer uses Gated MLA, others use KDA
        is_mla = ((layer_idx + 1) % (config.kda_ratio + 1) == 0) or (layer_idx == config.num_hidden_layers - 1)
        self.is_mla = is_mla

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if is_mla:
            self.self_attn = MultiLatentAttention(config, layer_idx)
        else:
            self.self_attn = KimiDeltaAttention(config, layer_idx)

        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # FFN: Stable LatentMoE or Dense SiTU-GLU
        if config.use_moe:
            self.mlp = StableLatentMoE(config, layer_idx)
        else:
            self.mlp = SiTUGLU(config.hidden_size, config.dense_intermediate_size)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        residual = hidden_states
        hidden_states, present_kv = self.self_attn(
            self.input_layernorm(hidden_states), attention_mask, position_ids, past_key_value
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = residual + hidden_states

        return hidden_states, present_kv


class KimiK3PreTrainedModel(PreTrainedModel):
    config_class = KimiK3Config
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


class KimiK3Model(KimiK3PreTrainedModel):
    """
    Kimi K3 Backbone Model with Block Attention Residuals (AttnRes) and Hybrid KDA-MLA Attention.
    """
    def __init__(self, config: KimiK3Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([KimiK3Block(config, i) for i in range(config.num_hidden_layers)])
        self.attn_res = BlockAttentionResiduals(config.hidden_size, config.num_hidden_layers, config.attn_res_num_blocks, eps=config.rms_norm_eps)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=None, **kwargs):
        if use_cache is None:
            use_cache = getattr(self.config, "use_cache", False)

        hidden_states = self.embed_tokens(input_ids)
        next_decoder_cache = [] if use_cache else None

        # AttnRes State Initialization
        block_size = max(1, self.config.num_hidden_layers // self.config.attn_res_num_blocks)
        block_representations = [hidden_states]
        current_block_sum = None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None

            # Block AttnRes Retrieval (Eq. 10)
            residual_input = self.attn_res(i, block_representations, current_block_sum)

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    return lambda *inputs: module(*inputs)
                hidden_states, present_kv = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer), residual_input, attention_mask, position_ids, past_kv, use_reentrant=False
                )
            else:
                hidden_states, present_kv = layer(residual_input, attention_mask, position_ids, past_kv)

            if use_cache:
                next_decoder_cache.append(present_kv)

            # Update Intra-Block & Inter-Block AttnRes representations
            if current_block_sum is None:
                current_block_sum = hidden_states
            else:
                current_block_sum = current_block_sum + hidden_states

            if (i + 1) % block_size == 0 or (i + 1) == self.config.num_hidden_layers:
                block_representations.append(current_block_sum)
                current_block_sum = None

        return self.norm(hidden_states), next_decoder_cache


class KimiK3ForCausalLM(KimiK3PreTrainedModel, GenerationMixin):
    """
    Kimi K3 Language Model for Causal LM tasks.
    """
    def __init__(self, config: KimiK3Config):
        super().__init__(config)
        self.model = KimiK3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, labels=None, use_cache=None, **kwargs):
        outputs, past_kv = self.model(input_ids, attention_mask, position_ids, past_key_values, use_cache, **kwargs)
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