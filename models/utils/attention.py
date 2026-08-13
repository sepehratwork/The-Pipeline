import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .normalization import RMSNorm
from .positional_encoding import RotaryPositionalEmbedding, apply_rotary_pos_emb


def apply_partial_rope(
    x: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor, 
    position_ids: torch.Tensor, 
    rope_dim: int = 64
) -> torch.Tensor:
    """Applies Rotary Positional Embedding to the last `rope_dim` dimensions of a tensor."""
    x_pass = x[..., :-rope_dim]
    x_rope = x[..., -rope_dim:]

    cos_p = cos[position_ids].unsqueeze(1)[..., -rope_dim:]
    sin_p = sin[position_ids].unsqueeze(1)[..., -rope_dim:]

    x1 = x_rope[..., : rope_dim // 2]
    x2 = x_rope[..., rope_dim // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)

    x_embed = (x_rope * cos_p) + (rotated * sin_p)
    return torch.cat([x_pass, x_embed], dim=-1)


class GroupedQueryAttention(nn.Module):
    """
    Standard Grouped Query Attention (GQA) with optional QK-Norm and
    support for interleaved Local-Global Sliding Window Attention (SWA).
    """
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
            max_position_embeddings=getattr(config, "max_position_embeddings", 128000),
            base=getattr(config, "rope_theta", 500000.0),
        )

        # Interleaved Local-Global Sliding Window Attention (SWA) Configuration
        sliding_window = getattr(config, "sliding_window", None)
        num_layers = getattr(config, "num_hidden_layers", 30)

        if sliding_window is not None:
            # 3 to 1 local-global ratio: first (0) and last (num_layers - 1) layers are always global;
            # every 4th layer is global, and remaining layers use local sliding window attention.
            if layer_idx == 0 or layer_idx == (num_layers - 1) or (layer_idx % 4 == 0):
                self.sliding_window = None
            else:
                self.sliding_window = sliding_window
        else:
            self.sliding_window = None

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

        # Construct Causal / Sliding Window / Padding Attention Mask for PyTorch SDPA
        attn_mask = None
        is_causal = False

        has_padding = False
        padding_mask = None

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                if (attention_mask == 0).any():
                    has_padding = True
                    padding_mask = (attention_mask != 0).unsqueeze(1).unsqueeze(2)  # (bsz, 1, 1, kv_seq_len)
            elif attention_mask.ndim == 4:
                has_padding = True
                if attention_mask.dtype == torch.bool:
                    padding_mask = attention_mask
                elif torch.is_floating_point(attention_mask):
                    padding_mask = (attention_mask == 0.0)
                else:
                    padding_mask = (attention_mask != 0)

        device = hidden_states.device
        if self.sliding_window is not None:
            q_idx = torch.arange(kv_seq_len - q_len, kv_seq_len, device=device).unsqueeze(1)
            k_idx = torch.arange(kv_seq_len, device=device).unsqueeze(0)
            distance = q_idx - k_idx
            swa_mask = (distance >= 0) & (distance < self.sliding_window)
            swa_mask = swa_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, q_len, kv_seq_len)

            if has_padding and padding_mask is not None:
                attn_mask = padding_mask & swa_mask
            else:
                attn_mask = swa_mask
            is_causal = False
        else:
            if has_padding and padding_mask is not None:
                q_idx = torch.arange(kv_seq_len - q_len, kv_seq_len, device=device).unsqueeze(1)
                k_idx = torch.arange(kv_seq_len, device=device).unsqueeze(0)
                causal_mask = (q_idx >= k_idx).unsqueeze(0).unsqueeze(0)
                attn_mask = padding_mask & causal_mask
                is_causal = False
            else:
                if q_len == 1:
                    attn_mask = None
                    is_causal = False
                elif q_len == kv_seq_len:
                    attn_mask = None
                    is_causal = True
                else:
                    q_idx = torch.arange(kv_seq_len - q_len, kv_seq_len, device=device).unsqueeze(1)
                    k_idx = torch.arange(kv_seq_len, device=device).unsqueeze(0)
                    attn_mask = (q_idx >= k_idx).unsqueeze(0).unsqueeze(0)
                    is_causal = False

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output), past_key_value


class MultiLatentAttention(nn.Module):
    """Multi-Latent Attention (MLA) module as specified in DeepSeek-V2/GLM-5/Kimi K3."""
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qk_head_dim = getattr(config, "qk_head_dim", 128)
        self.v_head_dim = getattr(config, "v_head_dim", 128)
        self.rope_head_dim = getattr(config, "rope_head_dim", 0)
        self.use_nope = getattr(config, "use_nope", True)

        self.q_lora_rank = getattr(config, "q_lora_rank", 512)
        self.kv_lora_rank = getattr(config, "kv_lora_rank", 256)
        eps = getattr(config, "rms_norm_eps", 1e-6)

        if self.q_lora_rank > 0:
            self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=eps)
            self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.qk_head_dim, bias=False)

        if not self.use_nope and self.rope_head_dim > 0:
            self.q_rope_proj = nn.Linear(
                self.q_lora_rank if self.q_lora_rank > 0 else self.hidden_size,
                self.num_heads * self.rope_head_dim,
                bias=False
            )

        kv_dim = self.kv_lora_rank + (0 if self.use_nope else self.rope_head_dim)
        self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size, kv_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_head_dim + self.v_head_dim),
            bias=False
        )

        self.use_gated = getattr(config, "use_gated_mla", True)
        if self.use_gated:
            self.g_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if not self.use_nope and self.rope_head_dim > 0:
            self.rotary_emb = RotaryPositionalEmbedding(
                self.rope_head_dim,
                max_position_embeddings=getattr(config, "max_position_embeddings", 1000000),
                base=getattr(config, "rope_theta", 500000.0),
            )

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank > 0:
            compressed_q = self.q_a_layernorm(self.q_a_proj(hidden_states))
            q_content = self.q_b_proj(compressed_q).view(bsz, q_len, self.num_heads, self.qk_head_dim)
        else:
            q_content = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.qk_head_dim)

        kv_compressed = self.kv_a_proj_with_mqa(hidden_states)
        if not self.use_nope and self.rope_head_dim > 0:
            compressed_kv, k_rope = torch.split(kv_compressed, [self.kv_lora_rank, self.rope_head_dim], dim=-1)
            q_rope = self.q_rope_proj(compressed_q if self.q_lora_rank > 0 else hidden_states).view(bsz, q_len, self.num_heads, self.rope_head_dim)
        else:
            compressed_kv = kv_compressed

        compressed_kv = self.kv_a_layernorm(compressed_kv)
        kv_uncompressed = self.kv_b_proj(compressed_kv).view(bsz, q_len, self.num_heads, self.qk_head_dim + self.v_head_dim)
        k_content, v_content = torch.split(kv_uncompressed, [self.qk_head_dim, self.v_head_dim], dim=-1)

        if not self.use_nope and self.rope_head_dim > 0:
            k_rope = k_rope.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
            kv_seq_len = q_len + (past_key_value[0].shape[-2] if past_key_value is not None else 0)
            cos, sin = self.rotary_emb(v_content, kv_seq_len)
            q_rope, k_rope = apply_rotary_pos_emb(q_rope.transpose(1, 2), k_rope.transpose(1, 2), cos, sin, position_ids)
            q = torch.cat([q_content.transpose(1, 2), q_rope], dim=-1)
            k = torch.cat([k_content.transpose(1, 2), k_rope], dim=-1)
        else:
            q = q_content.transpose(1, 2)
            k = k_content.transpose(1, 2)

        v = v_content.transpose(1, 2)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        past_key_value = (k, v)

        attn_mask = None
        is_causal = False

        has_padding = False
        padding_mask = None

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                if (attention_mask == 0).any():
                    has_padding = True
                    padding_mask = (attention_mask != 0).unsqueeze(1).unsqueeze(2)
            elif attention_mask.ndim == 4:
                has_padding = True
                if attention_mask.dtype == torch.bool:
                    padding_mask = attention_mask
                elif torch.is_floating_point(attention_mask):
                    padding_mask = (attention_mask == 0.0)
                else:
                    padding_mask = (attention_mask != 0)

        kv_seq_len = k.shape[-2]
        if has_padding and padding_mask is not None:
            device = hidden_states.device
            q_idx = torch.arange(kv_seq_len - q_len, kv_seq_len, device=device).unsqueeze(1)
            k_idx = torch.arange(kv_seq_len, device=device).unsqueeze(0)
            causal_mask = (q_idx >= k_idx).unsqueeze(0).unsqueeze(0)
            attn_mask = padding_mask & causal_mask
            is_causal = False
        else:
            if q_len == 1:
                attn_mask = None
                is_causal = False
            elif q_len == kv_seq_len:
                attn_mask = None
                is_causal = True
            else:
                device = hidden_states.device
                q_idx = torch.arange(kv_seq_len - q_len, kv_seq_len, device=device).unsqueeze(1)
                k_idx = torch.arange(kv_seq_len, device=device).unsqueeze(0)
                attn_mask = (q_idx >= k_idx).unsqueeze(0).unsqueeze(0)
                is_causal = False

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.num_heads * self.v_head_dim)

        if self.use_gated:
            gate = torch.sigmoid(self.g_proj(hidden_states))
            attn_output = gate * attn_output

        return self.o_proj(attn_output), past_key_value


class KimiDeltaAttention(nn.Module):
    """Kimi Delta Attention (KDA) module."""
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = getattr(config, "num_attention_heads", 16)
        self.head_dim = getattr(config, "head_dim", 128)
        self.kernel_size = getattr(config, "short_conv_kernel_size", 4)
        self.g_min = getattr(config, "kda_g_min", -5.0)

        total_dim = self.num_heads * self.head_dim

        self.q_proj = nn.Linear(self.hidden_size, total_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, total_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, total_dim, bias=False)
        self.beta_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        z_dim = getattr(config, "kda_z_dim", self.hidden_size // 4)
        self.z_down = nn.Linear(self.hidden_size, z_dim, bias=False)
        self.z_up = nn.Linear(z_dim, total_dim, bias=True)

        self.A_h = nn.Parameter(torch.zeros(self.num_heads, 1))

        self.q_conv = nn.Conv1d(total_dim, total_dim, kernel_size=self.kernel_size, groups=total_dim, padding=self.kernel_size - 1)
        self.k_conv = nn.Conv1d(total_dim, total_dim, kernel_size=self.kernel_size, groups=total_dim, padding=self.kernel_size - 1)
        self.v_conv = nn.Conv1d(total_dim, total_dim, kernel_size=self.kernel_size, groups=total_dim, padding=self.kernel_size - 1)

        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.o_norm = RMSNorm(total_dim, eps=eps)
        self.g_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        bsz, q_len, _ = hidden_states.size()

        q_raw = self.q_proj(hidden_states)
        k_raw = self.k_proj(hidden_states)
        v_raw = self.v_proj(hidden_states)

        q_conv_out = F.silu(self.q_conv(q_raw.transpose(1, 2))[:, :, :q_len].transpose(1, 2))
        k_conv_out = F.silu(self.k_conv(k_raw.transpose(1, 2))[:, :, :q_len].transpose(1, 2))
        v_conv_out = F.silu(self.v_conv(v_raw.transpose(1, 2))[:, :, :q_len].transpose(1, 2))

        q = F.normalize(q_conv_out.view(bsz, q_len, self.num_heads, self.head_dim), p=2, dim=-1)
        k = F.normalize(k_conv_out.view(bsz, q_len, self.num_heads, self.head_dim), p=2, dim=-1)
        v = v_conv_out.view(bsz, q_len, self.num_heads, self.head_dim)

        beta = torch.sigmoid(self.beta_proj(hidden_states))

        z = self.z_up(self.z_down(hidden_states)).view(bsz, q_len, self.num_heads, self.head_dim)
        g = self.g_min * torch.sigmoid(torch.exp(self.A_h).unsqueeze(0).unsqueeze(0) * z)
        alpha = torch.exp(g)

        if past_key_value is not None and isinstance(past_key_value, tuple) and len(past_key_value) > 0:
            S = past_key_value[0]
        else:
            S = torch.zeros(bsz, self.num_heads, self.head_dim, self.head_dim, device=hidden_states.device, dtype=hidden_states.dtype)

        o_outputs = []
        for t in range(q_len):
            q_t = q[:, t, :, :]
            k_t = k[:, t, :, :]
            v_t = v[:, t, :, :]
            b_t = beta[:, t, :].unsqueeze(-1)
            a_t = alpha[:, t, :, :]

            S = S * a_t.unsqueeze(-1)
            kt_S = torch.matmul(k_t.unsqueeze(-2), S)
            S = S - b_t.unsqueeze(-1) * torch.matmul(k_t.unsqueeze(-1), kt_S) + b_t.unsqueeze(-1) * torch.matmul(k_t.unsqueeze(-1), v_t.unsqueeze(-2))
            o_t = torch.matmul(S.transpose(-1, -2), q_t.unsqueeze(-1)).squeeze(-1)
            o_outputs.append(o_t)

        o_tilde = torch.stack(o_outputs, dim=1).view(bsz, q_len, self.num_heads * self.head_dim)
        present_kv = (S,)

        gate = torch.sigmoid(self.g_proj(hidden_states))
        output = self.o_proj(gate * self.o_norm(o_tilde))

        return output, present_kv


class CompressedSparseAttention(nn.Module):
    """Compressed Sparse Attention (CSA) module."""
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.m = getattr(config, "compression_rate", 4)
        self.head_dim = getattr(config, "head_dim", 512)
        self.num_heads = getattr(config, "num_attention_heads", 64)
        self.top_k = getattr(config, "attention_topk", 512)
        self.q_lora_rank = getattr(config, "q_lora_rank", 1024)
        self.indexer_heads = getattr(config, "indexer_heads", 64)
        self.indexer_dim = getattr(config, "indexer_dim", 128)
        self.n_groups = getattr(config, "num_projection_groups", 8)
        self.d_g = getattr(config, "group_intermediate_dim", 1024)

        self.w_a_kv = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.w_b_kv = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.w_a_z = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.w_b_z = nn.Linear(self.hidden_size, self.head_dim, bias=False)

        self.b_a = nn.Parameter(torch.zeros(self.m, self.head_dim))
        self.b_b = nn.Parameter(torch.zeros(self.m, self.head_dim))

        self.w_dq = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.w_iuq = nn.Linear(self.q_lora_rank, self.indexer_heads * self.indexer_dim, bias=False)
        self.w_uq = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.w_w = nn.Linear(self.hidden_size, self.indexer_heads, bias=False)

        self.group_projs = nn.ModuleList([
            nn.Linear((self.num_heads // self.n_groups) * self.head_dim, self.d_g, bias=False)
            for _ in range(self.n_groups)
        ])
        self.out_proj = nn.Linear(self.n_groups * self.d_g, self.hidden_size, bias=False)

        self.query_norm = RMSNorm(self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim)
        self.sink_logits = nn.Parameter(torch.zeros(self.num_heads))

    def _compress_kv(self, h: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = h.size()
        pad_len = (self.m - (seq_len % self.m)) % self.m
        if pad_len > 0:
            h = F.pad(h, (0, 0, 0, pad_len))
            seq_len += pad_len

        n_blocks = seq_len // self.m
        c_a = self.w_a_kv(h).view(bsz, n_blocks, self.m, self.head_dim)
        c_b = self.w_b_kv(h).view(bsz, n_blocks, self.m, self.head_dim)
        z_a = self.w_a_z(h).view(bsz, n_blocks, self.m, self.head_dim) + self.b_a
        z_b = self.w_b_z(h).view(bsz, n_blocks, self.m, self.head_dim) + self.b_b

        scores = torch.softmax(torch.cat([z_a, z_b], dim=-2), dim=-2)
        s_a, s_b = torch.split(scores, self.m, dim=-2)

        return (s_a * c_a + s_b * c_b).sum(dim=-2)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        bsz, q_len, _ = hidden_states.size()

        c_comp_norm = self.kv_norm(self._compress_kv(hidden_states))
        c_q = self.w_dq(hidden_states)

        queries = self.query_norm(self.w_uq(c_q).view(bsz, q_len, self.num_heads, self.head_dim))
        scores = torch.matmul(queries, c_comp_norm.transpose(-1, -2)) / math.sqrt(self.head_dim)
        sink_exp = torch.exp(self.sink_logits).view(1, 1, self.num_heads, 1)
        attn_weights = torch.softmax(scores + sink_exp, dim=-1)

        attn_out = torch.matmul(attn_weights, c_comp_norm).view(bsz, q_len, self.n_groups, -1)
        grouped_outputs = [self.group_projs[g](attn_out[:, :, g, :]) for g in range(self.n_groups)]
        final_output = self.out_proj(torch.cat(grouped_outputs, dim=-1))

        return final_output, past_key_value


class HeavilyCompressedAttention(nn.Module):
    """Heavily Compressed Attention (HCA) module."""
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.m_prime = getattr(config, "heavy_compression_rate", 128)
        self.head_dim = getattr(config, "head_dim", 512)
        self.num_heads = getattr(config, "num_attention_heads", 64)
        self.q_lora_rank = getattr(config, "q_lora_rank", 1024)
        self.n_groups = getattr(config, "num_projection_groups", 8)
        self.d_g = getattr(config, "group_intermediate_dim", 1024)

        self.w_kv = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.w_z = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.b_z = nn.Parameter(torch.zeros(self.m_prime, self.head_dim))

        self.w_dq = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.w_uq = nn.Linear(self.q_lora_rank, self.num_heads * self.head_dim, bias=False)

        self.group_projs = nn.ModuleList([
            nn.Linear((self.num_heads // self.n_groups) * self.head_dim, self.d_g, bias=False)
            for _ in range(self.n_groups)
        ])
        self.out_proj = nn.Linear(self.n_groups * self.d_g, self.hidden_size, bias=False)

        self.query_norm = RMSNorm(self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        bsz, q_len, _ = hidden_states.size()

        pad_len = (self.m_prime - (q_len % self.m_prime)) % self.m_prime
        h = F.pad(hidden_states, (0, 0, 0, pad_len)) if pad_len > 0 else hidden_states
        n_blocks = h.size(1) // self.m_prime

        c = self.w_kv(h).view(bsz, n_blocks, self.m_prime, self.head_dim)
        z = torch.softmax(self.w_z(h).view(bsz, n_blocks, self.m_prime, self.head_dim) + self.b_z, dim=-2)
        c_comp = self.kv_norm((z * c).sum(dim=-2))

        queries = self.query_norm(self.w_uq(self.w_dq(hidden_states)).view(bsz, q_len, self.num_heads, self.head_dim))

        attn_weights = torch.softmax(torch.matmul(queries, c_comp.transpose(-1, -2)) / math.sqrt(self.head_dim), dim=-1)
        attn_out = torch.matmul(attn_weights, c_comp).view(bsz, q_len, self.n_groups, -1)

        grouped_outputs = [self.group_projs[i](attn_out[:, :, i, :]) for i in range(self.n_groups)]
        return self.out_proj(torch.cat(grouped_outputs, dim=-1)), past_key_value


class NoRopeGroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) WITHOUT Rotary Positional Embeddings (No-RoPE).
    Used in Nemotron 3 as positional information is implicitly handled by interleaved Mamba-2 layers.
    Prevents out-of-distribution RoPE issues during context extension up to 1M tokens.
    """
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

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        bsz, q_len, _ = hidden_states.size()

        q = self.q_norm(self.q_proj(hidden_states)).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states)).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = k.shape[-2] + (past_key_value[0].shape[-2] if past_key_value is not None else 0)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        past_key_value = (k, v)

        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_mask = None
        is_causal = False

        has_padding = False
        padding_mask = None

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                if (attention_mask == 0).any():
                    has_padding = True
                    padding_mask = (attention_mask != 0).unsqueeze(1).unsqueeze(2)
            elif attention_mask.ndim == 4:
                has_padding = True
                if attention_mask.dtype == torch.bool:
                    padding_mask = attention_mask
                elif torch.is_floating_point(attention_mask):
                    padding_mask = (attention_mask == 0.0)
                else:
                    padding_mask = (attention_mask != 0)

        if has_padding and padding_mask is not None:
            device = hidden_states.device
            q_idx = torch.arange(kv_seq_len - q_len, kv_seq_len, device=device).unsqueeze(1)
            k_idx = torch.arange(kv_seq_len, device=device).unsqueeze(0)
            causal_mask = (q_idx >= k_idx).unsqueeze(0).unsqueeze(0)
            attn_mask = padding_mask & causal_mask
            is_causal = False
        else:
            if q_len == 1:
                attn_mask = None
                is_causal = False
            elif q_len == kv_seq_len:
                attn_mask = None
                is_causal = True
            else:
                device = hidden_states.device
                q_idx = torch.arange(kv_seq_len - q_len, kv_seq_len, device=device).unsqueeze(1)
                k_idx = torch.arange(kv_seq_len, device=device).unsqueeze(0)
                attn_mask = (q_idx >= k_idx).unsqueeze(0).unsqueeze(0)
                is_causal = False

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output), past_key_value