import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalization import RMSNorm
from .positional_encoding import RotaryPositionalEmbedding, apply_rotary_pos_emb


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) with optional Sliding Window Attention (SWA)"""
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.num_heads * self.head_dim)
        self.k_norm = RMSNorm(self.num_key_value_heads * self.head_dim)

        # OLMo 3 uses full attention on 1 out of 4 layers and the last layer
        self.is_full_attention = (layer_idx % 4 == 3) or (layer_idx == config.num_hidden_layers - 1)
        self.is_swa = not self.is_full_attention

        use_yarn_here = config.use_yarn and self.is_full_attention
        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim, max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta, use_yarn=use_yarn_here, original_max=config.original_max_position_embeddings
        )

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        bsz, q_len, _ = hidden_states.size()

        q = self.q_norm(self.q_proj(hidden_states)).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states)).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = k.shape[-2] + (past_key_value[0].shape[-2] if past_key_value is not None else 0)
        cos, sin = self.rotary_emb(v, kv_seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        past_key_value = (k, v)

        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        use_custom_mask = self.is_swa or (attention_mask is not None and not torch.all(attention_mask))

        if not use_custom_mask and q_len == kv_seq_len:
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            if q_len == kv_seq_len:
                causal_mask = torch.ones((bsz, 1, q_len, kv_seq_len), device=q.device, dtype=torch.bool).tril_()
                if self.is_swa: causal_mask.triu_(diagonal=-self.config.sliding_window + 1)
            else:
                row_idx = torch.arange(kv_seq_len - q_len, kv_seq_len, device=q.device).unsqueeze(1)
                col_idx = torch.arange(kv_seq_len, device=q.device).unsqueeze(0)
                causal_mask = row_idx >= col_idx
                if self.is_swa: causal_mask &= (row_idx < col_idx + self.config.sliding_window)
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, q_len, kv_seq_len).clone()

            if attention_mask is not None:
                causal_mask &= attention_mask.unsqueeze(1).unsqueeze(2).bool()

            float_mask = torch.zeros_like(causal_mask, dtype=q.dtype)
            float_mask.masked_fill_(~causal_mask, torch.finfo(q.dtype).min)
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=float_mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output), past_key_value
