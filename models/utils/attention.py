import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .normalization import RMSNorm
from .positional_encoding import RotaryPositionalEmbedding, apply_rotary_pos_emb


def apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int = 64) -> torch.Tensor:
    """Applies Rotary Positional Embedding to the last `rope_dim` dimensions of a tensor."""
    x_pass = x[..., :-rope_dim]
    x_rope = x[..., -rope_dim:]

    cos = cos[..., -rope_dim:]
    sin = sin[..., -rope_dim:]

    x1 = x_rope[..., : rope_dim // 2]
    x2 = x_rope[..., rope_dim // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)

    x_embed = (x_rope * cos) + (rotated * sin)
    return torch.cat([x_pass, x_embed], dim=-1)


class GroupedQueryAttention(nn.Module):
    """Standard Grouped Query Attention (GQA) with optional QK-Norm and SWA."""
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

        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.q_norm = RMSNorm(self.num_heads * self.head_dim, eps=eps)
        self.k_norm = RMSNorm(self.num_key_value_heads * self.head_dim, eps=eps)

        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim,
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            base=getattr(config, "rope_theta", 500000.0),
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

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output), past_key_value


class CompressedSparseAttention(nn.Module):
    """
    Compressed Sparse Attention (CSA) from DeepSeek-V4.
    Compresses KV entries of every m tokens, uses a Lightning Indexer for top-k selection,
    applies RMSNorm on queries/KV, partial RoPE, Attention Sink, and Grouped Output Projection.
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.m = getattr(config, "csa_compression_rate", 4)
        self.rope_dim = getattr(config, "rope_dim", 64)

        # Compression Projections & Positional Biases
        self.w_a_kv = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.w_b_kv = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.w_a_z = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.w_b_z = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.b_a = nn.Parameter(torch.zeros(self.m, self.head_dim))
        self.b_b = nn.Parameter(torch.zeros(self.m, self.head_dim))

        # Query Projections
        q_comp_dim = getattr(config, "query_compression_dim", 512)
        self.w_dq = nn.Linear(self.hidden_size, q_comp_dim, bias=False)
        self.w_uq = nn.Linear(q_comp_dim, self.num_heads * self.head_dim, bias=False)

        # Query & KV RMSNorms
        self.q_norm = RMSNorm(self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim)

        # Learnable Attention Sink
        self.sink_logits = nn.Parameter(torch.zeros(self.num_heads))

        # Grouped Output Projection
        self.num_groups = getattr(config, "num_output_groups", 8)
        self.d_g = getattr(config, "intermediate_group_dim", 512)
        heads_per_group = self.num_heads // self.num_groups
        self.group_projections = nn.ModuleList([
            nn.Linear(heads_per_group * self.head_dim, self.d_g, bias=False)
            for _ in range(self.num_groups)
        ])
        self.o_proj = nn.Linear(self.num_groups * self.d_g, self.hidden_size, bias=False)

        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim,
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            base=getattr(config, "rope_theta", 500000.0)
        )

    def _compress_kv(self, h: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = h.shape
        c_a, c_b = self.w_a_kv(h), self.w_b_kv(h)
        z_a, z_b = self.w_a_z(h), self.w_b_z(h)

        num_blocks = seq_len // self.m
        if num_blocks == 0:
            return c_a.mean(dim=1, keepdim=True)

        c_a_b = c_a[:, :num_blocks * self.m].view(bsz, num_blocks, self.m, self.head_dim)
        c_b_b = c_b[:, :num_blocks * self.m].view(bsz, num_blocks, self.m, self.head_dim)
        z_a_b = (z_a[:, :num_blocks * self.m].view(bsz, num_blocks, self.m, self.head_dim) + self.b_a).softmax(dim=-2)
        z_b_b = (z_b[:, :num_blocks * self.m].view(bsz, num_blocks, self.m, self.head_dim) + self.b_b).softmax(dim=-2)

        return (z_a_b * c_a_b + z_b_b * c_b_b).sum(dim=-2)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        bsz, q_len, _ = hidden_states.shape

        # Compress KV entries
        compressed_kv = self.kv_norm(self._compress_kv(hidden_states))  # [bsz, num_blocks, head_dim]

        # Generate and normalize queries
        c_q = self.w_dq(hidden_states)
        q = self.w_uq(c_q).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)

        cos, sin = self.rotary_emb(q, q_len)
        q = apply_partial_rope(q, cos, sin, rope_dim=self.rope_dim)

        k = compressed_kv.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        v = k

        # Scaled dot-product scores
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)

        # Incorporate Attention Sink into Softmax
        sink = self.sink_logits.view(1, self.num_heads, 1, 1).expand(bsz, -1, q_len, 1)
        scores_with_sink = torch.cat([attn_scores, sink], dim=-1)
        probs_with_sink = F.softmax(scores_with_sink, dim=-1)
        attn_probs = probs_with_sink[..., :-1]

        attn_out = torch.matmul(attn_probs, v)
        attn_out = apply_partial_rope(attn_out, cos, -sin, rope_dim=self.rope_dim)

        # Grouped Output Projection
        attn_out = attn_out.transpose(1, 2).contiguous()
        heads_per_group = self.num_heads // self.num_groups
        group_outputs = [
            self.group_projections[g](attn_out[:, :, g * heads_per_group : (g + 1) * heads_per_group, :].reshape(bsz, q_len, -1))
            for g in range(self.num_groups)
        ]

        final_out = self.o_proj(torch.cat(group_outputs, dim=-1))
        return final_out, past_key_value


class HeavilyCompressedAttention(nn.Module):
    """
    Heavily Compressed Attention (HCA) from DeepSeek-V4.
    Applies aggressive compression (m' = 16 or 128) for dense attention over compressed blocks.
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.m_prime = getattr(config, "hca_compression_rate", 16)
        self.rope_dim = getattr(config, "rope_dim", 64)

        self.w_kv = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.w_z = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.b_z = nn.Parameter(torch.zeros(self.m_prime, self.head_dim))

        q_comp_dim = getattr(config, "query_compression_dim", 512)
        self.w_dq = nn.Linear(self.hidden_size, q_comp_dim, bias=False)
        self.w_uq = nn.Linear(q_comp_dim, self.num_heads * self.head_dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim)

        self.sink_logits = nn.Parameter(torch.zeros(self.num_heads))

        self.num_groups = getattr(config, "num_output_groups", 8)
        self.d_g = getattr(config, "intermediate_group_dim", 512)
        heads_per_group = self.num_heads // self.num_groups
        self.group_projections = nn.ModuleList([
            nn.Linear(heads_per_group * self.head_dim, self.d_g, bias=False)
            for _ in range(self.num_groups)
        ])
        self.o_proj = nn.Linear(self.num_groups * self.d_g, self.hidden_size, bias=False)

        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim,
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            base=getattr(config, "rope_theta", 500000.0)
        )

    def _compress_kv(self, h: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = h.shape
        c = self.w_kv(h)
        z = self.w_z(h)

        num_blocks = seq_len // self.m_prime
        if num_blocks == 0:
            return c.mean(dim=1, keepdim=True)

        c_b = c[:, :num_blocks * self.m_prime].view(bsz, num_blocks, self.m_prime, self.head_dim)
        z_b = (z[:, :num_blocks * self.m_prime].view(bsz, num_blocks, self.m_prime, self.head_dim) + self.b_z).softmax(dim=-2)

        return (z_b * c_b).sum(dim=-2)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        bsz, q_len, _ = hidden_states.shape

        compressed_kv = self.kv_norm(self._compress_kv(hidden_states))

        c_q = self.w_dq(hidden_states)
        q = self.q_norm(self.w_uq(c_q).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2))

        cos, sin = self.rotary_emb(q, q_len)
        q = apply_partial_rope(q, cos, sin, rope_dim=self.rope_dim)

        k = compressed_kv.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        v = k

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)

        sink = self.sink_logits.view(1, self.num_heads, 1, 1).expand(bsz, -1, q_len, 1)
        scores_with_sink = torch.cat([attn_scores, sink], dim=-1)
        probs_with_sink = F.softmax(scores_with_sink, dim=-1)
        attn_probs = probs_with_sink[..., :-1]

        attn_out = torch.matmul(attn_probs, v)
        attn_out = apply_partial_rope(attn_out, cos, -sin, rope_dim=self.rope_dim)

        attn_out = attn_out.transpose(1, 2).contiguous()
        heads_per_group = self.num_heads // self.num_groups
        group_outputs = [
            self.group_projections[g](attn_out[:, :, g * heads_per_group : (g + 1) * heads_per_group, :].reshape(bsz, q_len, -1))
            for g in range(self.num_groups)
        ]

        final_out = self.o_proj(torch.cat(group_outputs, dim=-1))
        return final_out, past_key_value


class DeepSeekV4HybridAttention(nn.Module):
    """
    Hybrid Attention architecture for DeepSeek-V4.
    Interleaves Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA),
    while using Sliding Window Attention (SWA) for initial layers.
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Layers 0 and 1 use pure SWA / HCA; subsequent layers interleave CSA and HCA
        if layer_idx < 2:
            self.attn = GroupedQueryAttention(config, layer_idx)
        elif layer_idx % 2 == 0:
            self.attn = CompressedSparseAttention(config, layer_idx)
        else:
            self.attn = HeavilyCompressedAttention(config, layer_idx)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        return self.attn(hidden_states, attention_mask, position_ids, past_key_value)