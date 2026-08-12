import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..utils.normalization import RMSNorm
from ..utils.mlp import FineGrainedSigmoidMoE
from ..utils.attention import GroupedQueryAttention


class MiniMaxM2Config(PretrainedConfig):
    """
    Configuration class for MiniMax-M2 language model.
    
    Hyperparameters scaled to ~1 Billion Active Parameters:
    - hidden_size: 2048
    - intermediate_size (per fine-grained expert): 512
    - num_hidden_layers: 24
    - num_attention_heads: 16 (head_dim = 128)
    - num_key_value_heads: 2 (GQA ratio 8:1)
    - num_experts: 64 total fine-grained experts
    - num_experts_per_tok: 8 active experts per token
    - num_shared_experts: 1 (size 1024)
    - Active params per token: ~1.035 Billion
    """
    architecture = "minimax_m2"

    def __init__(
        self,
        vocab_size=100278,
        hidden_size=2048,
        intermediate_size=512,            # Per fine-grained expert FFN size
        num_hidden_layers=24,             # Total decoder layers
        num_attention_heads=16,           # Query heads (head_dim = 128)
        num_key_value_heads=2,            # Key/Value heads (GQA 8:1)
        num_experts=64,                   # Fine-grained experts total
        num_experts_per_tok=8,            # Activated experts per token
        num_shared_experts=1,             # Shared experts count
        shared_expert_intermediate_size=1024, # Shared expert FFN size
        max_position_embeddings=192000,   # MiniMax-M2 native 192K context window
        rope_theta=500000.0,
        rms_norm_eps=1e-6,
        num_mtp_modules=1,                # Multi-Token Prediction (MTP) depth K
        mtp_loss_factor=0.3,              # MTP loss weight
        tie_word_embeddings=True,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.num_mtp_modules = num_mtp_modules
        self.mtp_loss_factor = mtp_loss_factor
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class MiniMaxM2TestConfig(MiniMaxM2Config):
    """
    Test configuration for MiniMax-M2 scaled to ~200M active parameters.
    """
    architecture = "minimax_m2_test"

    def __init__(
        self,
        vocab_size=100278,
        hidden_size=768,
        intermediate_size=256,
        num_hidden_layers=12,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_experts=32,
        num_experts_per_tok=4,
        num_shared_experts=1,
        shared_expert_intermediate_size=512,
        max_position_embeddings=32768,
        rope_theta=500000.0,
        rms_norm_eps=1e-6,
        num_mtp_modules=1,
        mtp_loss_factor=0.3,
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
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            num_shared_experts=num_shared_experts,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            num_mtp_modules=num_mtp_modules,
            mtp_loss_factor=mtp_loss_factor,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )


class MiniMaxM2Block(nn.Module):
    """
    Transformer block for MiniMax-M2 containing Full Multi-Head Self-Attention
    with Grouped Query Attention (GQA) and Fine-Grained MoE with Sigmoid Gating.
    """
    def __init__(self, config: MiniMaxM2Config, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_sparse_moe = FineGrainedSigmoidMoE(config, layer_idx)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None):
        residual = hidden_states
        hidden_states, present_kv = self.self_attn(
            self.input_layernorm(hidden_states), attention_mask, position_ids, past_key_value
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.block_sparse_moe(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states, present_kv


class MiniMaxM2PreTrainedModel(PreTrainedModel):
    config_class = MiniMaxM2Config
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


class MiniMaxM2Model(MiniMaxM2PreTrainedModel):
    """
    Core Transformer decoder model for MiniMax-M2.
    """
    def __init__(self, config: MiniMaxM2Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MiniMaxM2Block(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=None, **kwargs):
        if use_cache is None:
            use_cache = getattr(self.config, "use_cache", False)

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
            if use_cache:
                next_decoder_cache.append(present_kv)

        return self.norm(hidden_states), next_decoder_cache


class MiniMaxM2MTPModule(nn.Module):
    """
    Multi-Token Prediction (MTP) Module (Section 2.3).
    Predicts the k-th future token during training/speculative decoding draft paths.
    """
    def __init__(self, config: MiniMaxM2Config, depth_k: int):
        super().__init__()
        self.depth_k = depth_k
        self.proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer = MiniMaxM2Block(config, layer_idx=depth_k)
        self.post_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, main_hidden_states, token_embeds):
        # Concatenate main model hidden states with embeddings of current tokens
        concat_states = torch.cat([main_hidden_states, token_embeds], dim=-1)
        hidden_states = self.pre_norm(self.proj(concat_states))
        hidden_states, _ = self.layer(hidden_states)
        return self.post_norm(hidden_states)


class MiniMaxM2ForCausalLM(MiniMaxM2PreTrainedModel, GenerationMixin):
    """
    Causal Language Model with MiniMax-M2 backbone and Multi-Token Prediction (MTP) support.
    """
    def __init__(self, config: MiniMaxM2Config):
        super().__init__(config)
        self.model = MiniMaxM2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Multi-Token Prediction (MTP) modules (Section 2.3)
        if config.num_mtp_modules > 0:
            self.mtp_modules = nn.ModuleList([
                MiniMaxM2MTPModule(config, k + 1) for k in range(config.num_mtp_modules)
            ])
        else:
            self.mtp_modules = None

        self.post_init()

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, labels=None, use_cache=None, **kwargs):
        outputs, past_kv = self.model(input_ids, attention_mask, position_ids, past_key_values, use_cache, **kwargs)
        logits = self.lm_head(outputs)

        loss = None
        if labels is not None:
            # 1. Main Next-Token Prediction Cross-Entropy Loss
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

            # 2. Multi-Token Prediction (MTP) Loss (Section 2.3)
            if self.mtp_modules is not None and self.training:
                total_mtp_loss = 0.0
                for k, mtp_module in enumerate(self.mtp_modules):
                    step_k = k + 1
                    if shift_labels.shape[1] > step_k:
                        h_mtp = outputs[:, :-step_k, :]
                        mtp_token_embeds = self.model.embed_tokens(input_ids[:, step_k:-1])
                        mtp_features = mtp_module(h_mtp, mtp_token_embeds)
                        mtp_logits = self.lm_head(mtp_features).float()
                        mtp_target_labels = labels[:, (step_k + 1):].contiguous()

                        mtp_loss = loss_fct(
                            mtp_logits.view(-1, self.config.vocab_size), 
                            mtp_target_labels.view(-1)
                        )
                        total_mtp_loss = total_mtp_loss + mtp_loss

                loss = loss + self.config.mtp_loss_factor * total_mtp_loss

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