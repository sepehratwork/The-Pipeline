import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..utils.normalization import RMSNorm
from ..utils.mlp import DeepSeekMoE
from ..utils.attention import GroupedQueryAttention, CompressedSparseAttention, HeavilyCompressedAttention
from ..utils.mhc import ManifoldConstrainedHyperConnections


class DeepSeekV4Config(PretrainedConfig):
    """
    100M Parameter Test Configuration for DeepSeek-V4.
    """
    architecture = "deepseek_v4_test"

    def __init__(
        self,
        vocab_size: int = 100278,
        hidden_size: int = 512,
        intermediate_size: int = 1536,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        max_position_embeddings: int = 8192,
        rope_theta: float = 500000.0,
        n_hc: int = 4,
        t_max: int = 20,
        compression_rate: int = 4,
        heavy_compression_rate: int = 128,
        head_dim: int = 64,
        attention_topk: int = 64,
        q_lora_rank: int = 256,
        indexer_heads: int = 8,
        indexer_dim: int = 32,
        num_projection_groups: int = 2,
        group_intermediate_dim: int = 256,
        window_size: int = 128,
        num_routed_experts: int = 32,
        num_active_experts: int = 4,
        num_shared_experts: int = 1,
        hash_routing_layers: int = 2,
        z_loss_weight: float = 1e-5,
        mtp_loss_weight: float = 0.3,
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
            mtp_loss_weight=mtp_loss_weight,
            use_yarn=use_yarn,
            original_max_position_embeddings=original_max_position_embeddings,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )


class DeepSeekV4TestConfig(PretrainedConfig):
    """
    100M Parameter Configuration for DeepSeek-V4.
    Calibrated strictly according to DeepSeek-V4 architectural specifications
    and scaling law constraints.
    """
    architecture = "deepseek_v4_test"

    def __init__(
        self,
        vocab_size: int = 32000,                  # Scaled down to prevent embedding parameter starvation (16.38M)
        hidden_size: int = 512,                   # Hidden dimension d = 512
        intermediate_size: int = 448,             # Fine-grained intermediate dim per expert d_ff = 448
        num_hidden_layers: int = 12,              # Depth L = 12 (balances R_{D/W} and inference latency)
        num_attention_heads: int = 8,             # n_h = 8 (head_dim * n_h = 64 * 8 = 512 = d)
        num_key_value_heads: int = 2,
        max_position_embeddings: int = 8192,
        rope_theta: float = 500000.0,
        n_hc: int = 4,                            # mHC stream expansion factor n_hc = 4
        t_max: int = 20,                          # Sinkhorn-Knopp iterations t_max = 20
        compression_rate: int = 4,                # CSA compression rate m = 4
        heavy_compression_rate: int = 128,        # HCA compression rate m' = 128
        head_dim: int = 64,                       # Head dimension c = 64
        attention_topk: int = 32,                 # Sparse attention top-k for compressed tokens
        q_lora_rank: int = 256,                   # Query compression rank d_c = d / 2 = 256
        indexer_heads: int = 4,                   # Indexer query heads n_h^I = 4
        indexer_dim: int = 32,                    # Indexer head dim c^I = 32
        num_projection_groups: int = 2,           # g = 2 groups for grouped output projection
        group_intermediate_dim: int = 128,        # d_g = 128 (satisfies d_g < c * n_h / g = 256)
        window_size: int = 128,                   # Sliding window attention size n_win = 128
        num_routed_experts: int = 8,              # Fine-grained routed experts N_routed = 8
        num_active_experts: int = 2,              # Top-2 activated routed experts
        num_shared_experts: int = 1,              # 1 shared expert (DeepSeekMoE standard)
        hash_routing_layers: int = 2,             # Hash routing for the initial 2 layers
        z_loss_weight: float = 1e-5,
        mtp_loss_weight: float = 0.3,             # MTP loss weight as in DeepSeek-V4 pre-training
        use_yarn: bool = False,
        original_max_position_embeddings: int = 8192,
        tie_word_embeddings: bool = True,         # Tied word embeddings
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
            mtp_loss_weight=mtp_loss_weight,
            use_yarn=use_yarn,
            original_max_position_embeddings=original_max_position_embeddings,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )


class DeepSeekV4Block(nn.Module):
    """
    Transformer Block for DeepSeek-V4 with mHC residual connections and Hybrid Attention/MoE.
    """
    def __init__(self, config: DeepSeekV4Config, layer_idx: int):
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

    def forward(
        self,
        x_l: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
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

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor]]]]:
        if use_cache is None:
            use_cache = getattr(self.config, "use_cache", False)

        if position_ids is None:
            past_length = past_key_values[0][0].shape[-2] if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None else 0
            position_ids = torch.arange(past_length, input_ids.shape[1] + past_length, device=input_ids.device).unsqueeze(0)

        inputs_embeds = self.embed_tokens(input_ids)
        
        # Expand residual stream for mHC: (bsz, seq_len, n_hc, hidden_size)
        x_l = inputs_embeds.unsqueeze(-2).repeat(1, 1, self.config.n_hc, 1)

        next_decoder_cache = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    return lambda *inputs: module(*inputs)

                x_l, present_kv = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    x_l,
                    attention_mask,
                    position_ids,
                    past_kv,
                    use_reentrant=False
                )
            else:
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
    def __init__(self, config: DeepSeekV4Config):
        super().__init__(config)
        self.model = DeepSeekV4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Depth=1 MTP Module
        self.mtp_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
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
            
            loss_fct = nn.CrossEntropyLoss()
            ce_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
            # Label-masked Z-Loss
            valid_mask = (shift_labels != -100)
            if valid_mask.any():
                z_loss = ((torch.logsumexp(shift_logits, dim=-1) ** 2) * valid_mask).sum() / valid_mask.sum().clamp(min=1)
            else:
                z_loss = (torch.logsumexp(shift_logits, dim=-1) ** 2).mean()
            
            # Multi-Token Prediction (MTP) Auxiliary Loss
            mtp_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
            if outputs.size(1) > 2:
                mtp_logits = self.mtp_head(outputs[..., :-2, :]).contiguous().float()
                mtp_labels = labels[..., 2:].contiguous()
                mtp_loss = loss_fct(mtp_logits.view(-1, self.config.vocab_size), mtp_labels.view(-1))

            loss = ce_loss + self.config.z_loss_weight * z_loss + self.config.mtp_loss_weight * mtp_loss

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_kv)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> dict:
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[-2]
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