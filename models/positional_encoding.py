import math
import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) with optional YaRN scaling"""
    def __init__(self, dim, max_position_embeddings=8192, base=500000.0, use_yarn=False, original_max=8192):
        super().__init__()
        self.dim = dim
        self.base = base
        self.use_yarn = use_yarn
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))

        if self.use_yarn:
            scale = max_position_embeddings / original_max
            mscale = 0.1 * math.log(scale) + 1.0
            inv_freq = inv_freq / scale
            self.mscale = mscale
        else:
            self.mscale = 1.0

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.mscale
        sin = emb.sin() * self.mscale
        return cos.to(x.dtype), sin.to(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
