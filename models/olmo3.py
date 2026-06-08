import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


class OLMo3Config(PretrainedConfig):
    model_type = "olmo3"
    def __init__(
        self, 
        vocab_size=100278, 
        hidden_size=1024, 
        intermediate_size=2816,          # Adjusted for SwiGLU optimal ratio (approx 8/3 * hidden_size)
        num_hidden_layers=8,             # Shallow depth respects "Inverse Depth Scaling" & perfectly fits the 1-in-4 Full Attention rule
        num_attention_heads=8,           # hidden_size // 128 head_dim = 8
        num_key_value_heads=4,           # GQA for efficient inference
        max_position_embeddings=8192,    # OLMo 3 standard context
        sliding_window=4096,             # OLMo 3 standard SWA
        rope_theta=500000.0,
        z_loss_weight=1e-5, 
        use_yarn=False, 
        original_max_position_embeddings=8192, 
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
        self.z_loss_weight = z_loss_weight
        self.use_yarn = use_yarn
        self.original_max_position_embeddings = original_max_position_embeddings
        super().__init__(**kwargs)


class OLMo3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class OLMo3RotaryEmbedding(nn.Module):
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


class OLMo3Attention(nn.Module):
    def __init__(self, config: OLMo3Config, layer_idx: int):
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

        self.q_norm = OLMo3RMSNorm(self.num_heads * self.head_dim)
        self.k_norm = OLMo3RMSNorm(self.num_key_value_heads * self.head_dim)

        self.is_full_attention = (layer_idx % 4 == 3) or (layer_idx == config.num_hidden_layers - 1)
        self.is_swa = not self.is_full_attention

        use_yarn_here = config.use_yarn and self.is_full_attention
        self.rotary_emb = OLMo3RotaryEmbedding(
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


class OLMo3MLP(nn.Module):
    def __init__(self, config: OLMo3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class OLMo3Block(nn.Module):
    def __init__(self, config: OLMo3Config, layer_idx: int):
        super().__init__()
        self.input_layernorm = OLMo3RMSNorm(config.hidden_size)
        self.self_attn = OLMo3Attention(config, layer_idx)
        self.post_attention_layernorm = OLMo3RMSNorm(config.hidden_size)
        self.mlp = OLMo3MLP(config)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        residual = hidden_states
        hidden_states, present_kv = self.self_attn(
            self.input_layernorm(hidden_states), attention_mask, position_ids, past_key_value
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states, present_kv


class OLMo3PreTrainedModel(PreTrainedModel):
    config_class = OLMo3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _check_and_adjust_experts_implementation(self, experts_implementation):
        return experts_implementation

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None: module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None: module.weight.data[module.padding_idx].zero_()


class OLMo3Model(OLMo3PreTrainedModel):
    def __init__(self, config: OLMo3Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([OLMo3Block(config, i) for i in range(config.num_hidden_layers)])
        self.norm = OLMo3RMSNorm(config.hidden_size)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=None, **kwargs):
        if use_cache is None: use_cache = getattr(self.config, "use_cache", False)

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
            if use_cache: next_decoder_cache.append(present_kv)

        return self.norm(hidden_states), next_decoder_cache


class OLMo3ForCausalLM(OLMo3PreTrainedModel, GenerationMixin):
    def __init__(self, config: OLMo3Config):
        super().__init__(config)
        self.model = OLMo3Model(config)
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
            past_length = past_key_values[0][0].shape[-2]
            remove_prefix_length = past_length if input_ids.shape[1] > past_length else input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values: position_ids = position_ids[:, -input_ids.shape[1] :]

        return {
            "input_ids": input_ids, "position_ids": position_ids,
            "past_key_values": past_key_values, "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        return tuple(tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past) for layer_past in past_key_values)
