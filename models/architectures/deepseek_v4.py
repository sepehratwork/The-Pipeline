import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..utils.normalization import RMSNorm
from ..utils.mlp import DeepSeekMoE
from ..utils.attention import GroupedQueryAttention, CompressedSparseAttention, HeavilyCompressedAttention
from ..utils.mhc import ManifoldConstrainedHyperConnections


class DeepSeekV4Config(PretrainedConfig):
    """
    Configuration for DeepSeek-V4 scaled to ~1B active parameters.
    """
    model_type = "deepseek_v4"

    def __init__(
        self,
        vocab_size=100278,
        hidden_size=1536,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_position_embeddings=8192,
        rope_theta=500000.0,
        n_hc=4,
        t_max=20,
        compression_rate=4,
        heavy_compression_rate=128,
        head_dim=256,
        attention_topk=256,
        q_lora_rank=512,
        indexer_heads=16,
        indexer_dim=64,
        num_projection_groups=4,
        group_intermediate_dim=512,
        window_size=128,
        num_routed_experts=64,
        num_active_experts=6,
        num_shared_experts=1,
        hash_routing_layers=3,
        z_loss_weight=1e-5,
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
        self.rope_theta = rope_theta
        self.n_hc = n_hc
        self.t_max = t_max
        self.compression_rate = compression_rate
        self.heavy_compression_rate = heavy_compression_rate
        self.head_dim = head_dim
        self.attention_topk = attention_topk
        self.q_lora_rank = q_lora_rank
        self.indexer_heads = indexer_heads
        self.indexer_dim = indexer_dim
        self.num_projection_groups = num_projection_groups
        self.group_intermediate_dim = group_intermediate_dim
        self.window_size = window_size
        self.num_routed_experts = num_routed_experts
        self.num_active_experts = num_active_experts
        self.num_shared_experts = num_shared_experts
        self.hash_routing_layers = hash_routing_layers
        self.z_loss_weight = z_loss_weight
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class DeepSeekV4TestConfig(DeepSeekV4Config):
    """
    100M Parameter Test Configuration for DeepSeek-V4.
    """
    def __init__(
        self,
        vocab_size=100278,
        hidden_size=512,
        intermediate_size=1536,
        num_hidden_layers=12,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=8192,
        rope_theta=500000.0,
        n_hc=4,
        t_max=20,
        compression_rate=4,
        heavy_compression_rate=128,
        head_dim=64,
        attention_topk=64,
        q_lora_rank=256,
        indexer_heads=8,
        indexer_dim=32,
        num_projection_groups=2,
        group_intermediate_dim=256,
        window_size=128,
        num_routed_experts=32,
        num_active_experts=4,
        num_shared_experts=1,
        hash_routing_layers=2,
        z_loss_weight=1e-5,
        tie_word_embeddings=True,
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
            n_hc=n_hc,
            t_max=t_max,
            compression_rate=compression_rate,
            heavy_compression_rate=heavy_compression_rate,
            head_dim=head_dim,
            attention_topk=attention_topk,
            q_lora_rank=q_lora_rank,
            indexer_heads=indexer_heads,
            indexer_dim=indexer_dim,
            num_projection_groups=num_projection_groups,
            group_intermediate_dim=group_intermediate_dim,
            window_size=window_size,
            num_routed_experts=num_routed_experts,
            num_active_experts=num_active_experts,
            num_shared_experts=num_shared_experts,
            hash_routing_layers=hash_routing_layers,
            z_loss_weight=z_loss_weight,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )


class DeepSeekV4Block(nn.Module):
    """
    Transformer Block for DeepSeek-V4 with mHC residual connections and Hybrid Attention/MoE.
    """
    def __init__(self, config: DeepSeekV4TestConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_hc = config.n_hc

        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)

        # Hybrid Attention Interleaving
        if layer_idx < 2:
            self.self_attn = GroupedQueryAttention(config, layer_idx)
        elif layer_idx % 2 == 0:
            self.self_attn = CompressedSparseAttention(config, layer_idx)
        else:
            self.self_attn = HeavilyCompressedAttention(config, layer_idx)

        self.mhc = ManifoldConstrainedHyperConnections(config.hidden_size, n_hc=config.n_hc, t_max=config.t_max)
        self.mlp = DeepSeekMoE(config, layer_idx)

    def forward(self, x_l, attention_mask=None, position_ids=None, past_key_value=None):
        # x_l shape: (bsz, seq_len, n_hc, hidden_size)
        a_l, b_l, c_l = self.mhc(x_l)

        # Compute layer input via mHC input mapping
        h_in = torch.matmul(a_l, x_l).squeeze(-2)  # (bsz, seq_len, hidden_size)

        # Attention sub-layer
        attn_out, present_kv = self.self_attn(
            self.input_layernorm(h_in), attention_mask, position_ids, past_key_value
        )
        x_mid = torch.matmul(b_l, x_l) + torch.matmul(c_l, attn_out.unsqueeze(-2))

        # Re-compute mHC for FFN sub-layer
        a_l2, b_l2, c_l2 = self.mhc(x_mid)
        h_in2 = torch.matmul(a_l2, x_mid).squeeze(-2)

        mlp_out = self.mlp(self.post_attention_layernorm(h_in2))
        x_next = torch.matmul(b_l2, x_mid) + torch.matmul(c_l2, mlp_out.unsqueeze(-2))

        return x_next, present_kv


class DeepSeekV4PreTrainedModel(PreTrainedModel):
    config_class = DeepSeekV4TestConfig
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


class DeepSeekV4Model(DeepSeekV4PreTrainedModel):
    def __init__(self, config: DeepSeekV4TestConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DeepSeekV4Block(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=None, **kwargs):
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], device=input_ids.device).unsqueeze(0)

        inputs_embeds = self.embed_tokens(input_ids)
        
        # Expand residual stream for mHC: (bsz, seq_len, n_hc, hidden_size)
        x_l = inputs_embeds.unsqueeze(-2).repeat(1, 1, self.config.n_hc, 1)

        next_decoder_cache = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            x_l, present_kv = layer(x_l, attention_mask, position_ids, past_kv)
            if use_cache:
                next_decoder_cache.append(present_kv)

        # Collapse residual stream back to hidden_size
        final_hidden = self.norm(x_l.mean(dim=-2))
        return final_hidden, next_decoder_cache


class DeepSeekV4ForCausalLM(DeepSeekV4PreTrainedModel, GenerationMixin):
    """
    DeepSeek-V4 Causal Language Model with Multi-Token Prediction (MTP) support.
    """
    def __init__(self, config: DeepSeekV4TestConfig):
        super().__init__(config)
        self.model = DeepSeekV4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Depth=1 MTP Module
        self.mtp_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
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
            
            # MTP Auxiliary Loss
            mtp_logits = self.mtp_head(outputs[..., :-2, :]).float()
            mtp_labels = labels[..., 2:].contiguous()
            mtp_loss = nn.CrossEntropyLoss()(mtp_logits.view(-1, self.config.vocab_size), mtp_labels.view(-1)) if mtp_logits.size(1) > 0 else 0.0

            loss = ce_loss + self.config.z_loss_weight * z_loss + 0.3 * mtp_loss

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_kv)