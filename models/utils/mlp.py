import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .normalization import RMSNorm


class SwiGLUMLP(nn.Module):
    """SwiGLU Multi-Layer Perceptron with optional numerical clamping."""
    def __init__(self, hidden_size: int, intermediate_size: int, clamp_val: float = 10.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.clamp_val = clamp_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.clamp_val is not None:
            gate = torch.clamp(gate, max=self.clamp_val)
            up = torch.clamp(up, min=-self.clamp_val, max=self.clamp_val)
        return self.down_proj(self.act_fn(gate) * up)


class SiTUGLU(nn.Module):
    """
    Sigmoid Tanh Unit GLU (SiTU-GLU) as introduced in Kimi K3 (Eq. 12).
    Smoothly caps activations using scaled tanh on gate and up projections
    to suppress activation explosion while preserving SwiGLU behavior near the origin.
    """
    def __init__(self, hidden_size: int, intermediate_size: int, beta1: float = 4.0, beta2: float = 25.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.beta1 = beta1
        self.beta2 = beta2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        gate_capped = self.beta1 * torch.tanh(gate / self.beta1) * torch.sigmoid(gate)
        up_capped = self.beta2 * torch.tanh(up / self.beta2)

        return self.down_proj(gate_capped * up_capped)


class DeepSeekMoE(nn.Module):
    """
    DeepSeekMoE module with shared experts and fine-grained routed experts.
    Uses Sqrt(Softplus(.)) activation for affinity scores and hash routing for initial layers.
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_routed = config.num_routed_experts
        self.num_active = config.num_active_experts
        self.is_hash_routing = layer_idx < config.hash_routing_layers

        # Shared Expert
        self.shared_expert = SwiGLUMLP(config.hidden_size, config.intermediate_size * config.num_shared_experts)

        # Routed Experts
        self.experts = nn.ModuleList([
            SwiGLUMLP(config.hidden_size, config.intermediate_size)
            for _ in range(self.num_routed)
        ])

        if not self.is_hash_routing:
            self.gate = nn.Linear(config.hidden_size, self.num_routed, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d = hidden_states.size()
        x_flat = hidden_states.view(-1, d)

        # 1. Compute Shared Expert output
        shared_out = self.shared_expert(x_flat)

        # 2. Compute Routed Experts output
        if self.is_hash_routing:
            token_ids = torch.arange(x_flat.size(0), device=hidden_states.device)
            top_indices = (token_ids.unsqueeze(-1) + torch.arange(self.num_active, device=hidden_states.device)) % self.num_routed
            top_weights = torch.full_like(top_indices, 1.0 / self.num_active, dtype=hidden_states.dtype)
        else:
            logits = self.gate(x_flat)
            scores = torch.sqrt(F.softplus(logits))
            top_weights, top_indices = torch.topk(scores, k=self.num_active, dim=-1)
            top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-8)

        routed_out = torch.zeros_like(x_flat)
        for i in range(self.num_active):
            idx = top_indices[:, i]
            weight = top_weights[:, i].unsqueeze(-1)
            for e_idx in range(self.num_routed):
                mask = (idx == e_idx)
                if mask.any():
                    routed_out[mask] += weight[mask] * self.experts[e_idx](x_flat[mask])

        return (shared_out + routed_out).view(bsz, seq_len, d)


class StableLatentMoE(nn.Module):
    """
    Stable LatentMoE module as specified in Kimi K3 (Section 2.3).
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.latent_dim = getattr(config, "latent_dim", config.hidden_size // 2)
        self.num_routed = getattr(config, "num_routed_experts", 64)
        self.num_active = getattr(config, "num_active_experts", 4)
        self.num_shared = getattr(config, "num_shared_experts", 1)
        self.moe_intermediate = getattr(config, "moe_intermediate_size", 1024)
        self.shared_intermediate = getattr(config, "shared_intermediate_size", 2048)

        self.w_down = nn.Linear(self.hidden_size, self.latent_dim, bias=False)

        self.shared_experts = nn.ModuleList([
            SiTUGLU(self.hidden_size, self.shared_intermediate)
            for _ in range(self.num_shared)
        ])

        self.routed_experts = nn.ModuleList([
            SiTUGLU(self.latent_dim, self.moe_intermediate)
            for _ in range(self.num_routed)
        ])

        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.u_norm = RMSNorm(self.latent_dim, eps=eps)
        self.w_up = nn.Linear(self.latent_dim, self.hidden_size, bias=False)

        self.gate = nn.Linear(self.hidden_size, self.num_routed, bias=False)
        self.register_buffer("expert_bias", torch.zeros(self.num_routed))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d = hidden_states.size()
        x_flat = hidden_states.view(-1, d)

        shared_out = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(x_flat)

        z_flat = self.w_down(x_flat)

        logits = self.gate(x_flat)
        scores = torch.sigmoid(logits)
        biased_scores = scores + self.expert_bias

        top_weights, top_indices = torch.topk(biased_scores, k=self.num_active, dim=-1)

        selected_scores = torch.gather(scores, 1, top_indices)
        routing_weights = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-8)

        u_flat = torch.zeros_like(z_flat)
        for i in range(self.num_active):
            idx = top_indices[:, i]
            weight = routing_weights[:, i].unsqueeze(-1)
            for e_idx in range(self.num_routed):
                mask = (idx == e_idx)
                if mask.any():
                    u_flat[mask] += weight[mask] * self.routed_experts[e_idx](z_flat[mask])

        routed_out = self.w_up(self.u_norm(u_flat))

        return (shared_out + routed_out).view(bsz, seq_len, d)


class TopKMoE(nn.Module):
    """
    Top-K Routed Mixture of Experts (MoE) block with optional shared expert.
    """
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = getattr(config, "num_local_experts", 8)
        self.top_k = getattr(config, "num_experts_per_tok", 2)
        self.moe_intermediate_size = getattr(config, "moe_intermediate_size", config.intermediate_size)
        self.num_shared_experts = getattr(config, "num_shared_experts", 0)

        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        self.experts = nn.ModuleList([
            SwiGLUMLP(self.hidden_size, self.moe_intermediate_size)
            for _ in range(self.num_experts)
        ])

        if self.num_shared_experts > 0:
            shared_dim = self.moe_intermediate_size * self.num_shared_experts
            self.shared_expert = SwiGLUMLP(self.hidden_size, shared_dim)
        else:
            self.shared_expert = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_dim = hidden_states.size()
        x_flat = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        top_weights, top_indices = torch.topk(routing_weights, self.top_k, dim=-1)

        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-8)
        top_weights = top_weights.to(hidden_states.dtype)

        routed_out = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = top_indices[:, i]
            weight = top_weights[:, i].unsqueeze(-1)
            for e_idx in range(self.num_experts):
                mask = (expert_idx == e_idx)
                if mask.any():
                    routed_out[mask] += weight[mask] * self.experts[e_idx](x_flat[mask])

        if self.shared_expert is not None:
            shared_out = self.shared_expert(x_flat)
            routed_out = routed_out + shared_out

        return routed_out.view(bsz, seq_len, hidden_dim)


class FineGrainedSigmoidMoE(nn.Module):
    """
    Fine-Grained Mixture of Experts with Sigmoid Gating and Learnable Expert Bias
    as specified in MiniMax-M2 (Section 2.2.1).

    Features:
    - Fine-grained experts (larger number of smaller FFN experts).
    - Sigmoid gating replacing zero-sum softmax gating for smoother dynamics.
    - Learnable per-expert routing bias terms for implicit load regulation.
    - Optional shared expert running in parallel.
    """
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_experts = getattr(config, "num_experts", 64)
        self.num_experts_per_tok = getattr(config, "num_experts_per_tok", 8)
        self.expert_intermediate_size = getattr(config, "intermediate_size", 512)
        self.num_shared_experts = getattr(config, "num_shared_experts", 1)
        self.shared_intermediate_size = getattr(config, "shared_expert_intermediate_size", 1024)

        # Router Gate Projection & Learnable Expert Bias (Section 2.2.1)
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.expert_bias = nn.Parameter(torch.zeros(self.num_experts))

        # Fine-Grained Routed Experts
        self.experts = nn.ModuleList([
            SwiGLUMLP(self.hidden_size, self.expert_intermediate_size)
            for _ in range(self.num_experts)
        ])

        # Optional Shared Expert
        if self.num_shared_experts > 0 and self.shared_intermediate_size > 0:
            self.shared_expert = SwiGLUMLP(
                self.hidden_size, 
                self.shared_intermediate_size * self.num_shared_experts
            )
        else:
            self.shared_expert = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d = hidden_states.size()
        x_flat = hidden_states.view(-1, d)

        # 1. Compute Shared Expert output
        shared_out = self.shared_expert(x_flat) if self.shared_expert is not None else None

        # 2. Sigmoid Gating with Expert Bias (Section 2.2.1)
        raw_logits = self.gate(x_flat)
        scores = torch.sigmoid(raw_logits)
        biased_scores = scores + self.expert_bias

        # 3. Top-K Expert Selection
        _, top_indices = torch.topk(biased_scores, k=self.num_experts_per_tok, dim=-1)

        # Normalize selected top-k weights using original unbiased sigmoid scores
        selected_scores = torch.gather(scores, 1, top_indices)
        top_weights = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-8)
        top_weights = top_weights.to(hidden_states.dtype)

        # 4. Route tokens to assigned fine-grained experts
        routed_out = torch.zeros_like(x_flat)
        for i in range(self.num_experts_per_tok):
            expert_idx = top_indices[:, i]
            weight = top_weights[:, i].unsqueeze(-1)
            for e_idx in range(self.num_experts):
                mask = (expert_idx == e_idx)
                if mask.any():
                    routed_out[mask] += weight[mask] * self.experts[e_idx](x_flat[mask])

        if shared_out is not None:
            routed_out = routed_out + shared_out

        return routed_out.view(bsz, seq_len, d)


class LatentMoE(nn.Module):
    """
    LatentMoE Architecture (Nemotron 3, Section 2.2).
    Tokens are projected from hidden dimension d into a latent dimension \ell = d / 4 for
    expert routing and FFN computation. Reduces memory bandwidth costs and all-to-all communication
    payloads by 4x, allowing expert capacity scaling without sacrificing inference throughput.
    """
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.latent_dim = getattr(config, "latent_dim", config.hidden_size // 4)
        self.num_experts = getattr(config, "num_routed_experts", 32)
        self.top_k = getattr(config, "num_active_experts", 4)
        self.intermediate_size = getattr(config, "intermediate_size", config.hidden_size * 2)

        # Latent projection layers
        self.latent_down_proj = nn.Linear(self.hidden_size, self.latent_dim, bias=False)
        self.latent_up_proj = nn.Linear(self.latent_dim, self.hidden_size, bias=False)
        self.latent_norm = RMSNorm(self.latent_dim)

        # Router Gating Network operates in hidden dimension d for gating precision
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        # Fine-grained routed experts operate entirely in latent dimension \ell
        self.experts = nn.ModuleList([
            SwiGLUMLP(self.latent_dim, self.intermediate_size // 2)
            for _ in range(self.num_experts)
        ])

        # Shared expert operates in original hidden dimension d
        self.shared_expert = SwiGLUMLP(self.hidden_size, self.intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d = hidden_states.size()
        x_flat = hidden_states.view(-1, d)

        # 1. Non-routed shared expert computation
        shared_out = self.shared_expert(x_flat)

        # 2. Project token embedding to latent dimension \ell
        z_flat = self.latent_down_proj(x_flat)

        # 3. Compute expert routing probabilities
        router_logits = self.gate(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        top_weights, top_indices = torch.topk(routing_weights, self.top_k, dim=-1)

        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-8)
        top_weights = top_weights.to(hidden_states.dtype)

        # 4. Expert computation in latent space
        latent_routed_out = torch.zeros_like(z_flat)
        for i in range(self.top_k):
            expert_idx = top_indices[:, i]
            weight = top_weights[:, i].unsqueeze(-1)
            for e_idx in range(self.num_experts):
                mask = (expert_idx == e_idx)
                if mask.any():
                    latent_routed_out[mask] += weight[mask] * self.experts[e_idx](z_flat[mask])

        # 5. Project back to original hidden dimension d
        routed_out = self.latent_up_proj(self.latent_norm(latent_routed_out))

        return (shared_out + routed_out).view(bsz, seq_len, d)


class FineGrainedMoE(nn.Module):
    """
    Fine-Grained Mixture of Experts (MoE) block without shared experts,
    specifically designed for the Qwen3 MoE architecture series.
    
    Key Qwen3 MoE Architectural Features:
    - Fine-grained expert segmentation (Dai et al., 2024).
    - 128 total experts with Top-8 activated experts per token.
    - Excludes shared experts (unlike Qwen2.5-MoE).
    - Incorporates global-batch load balancing loss (Qiu et al., 2025).
    """
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_experts = getattr(config, "num_experts", 128)
        self.num_experts_per_tok = getattr(config, "num_experts_per_tok", 8)
        self.moe_intermediate_size = getattr(config, "moe_intermediate_size", 704)
        self.router_aux_loss_coef = getattr(config, "router_aux_loss_coef", 0.01)

        # Router Gate Projection (Linear routing without bias)
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        # Fine-Grained Routed Experts (No shared experts as per Qwen3 report)
        self.experts = nn.ModuleList([
            SwiGLUMLP(self.hidden_size, self.moe_intermediate_size)
            for _ in range(self.num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seq_len, d = hidden_states.size()
        x_flat = hidden_states.view(-1, d)

        # 1. Compute router logits & softmax routing probabilities
        router_logits = self.gate(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        # 2. Select Top-K fine-grained experts
        top_weights, top_indices = torch.topk(routing_weights, k=self.num_experts_per_tok, dim=-1)

        # 3. Renormalize top-k probabilities
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-8)
        top_weights = top_weights.to(hidden_states.dtype)

        # 4. Global-batch load balancing auxiliary loss (Qiu et al., 2025)
        aux_loss = None
        if self.training and self.router_aux_loss_coef > 0:
            tokens_per_expert = torch.zeros(self.num_experts, device=hidden_states.device)
            for i in range(self.num_experts_per_tok):
                tokens_per_expert.scatter_add_(0, top_indices[:, i], torch.ones_like(top_indices[:, i], dtype=torch.float))
            density = tokens_per_expert / (x_flat.size(0) * self.num_experts_per_tok)
            mean_prob = routing_weights.mean(dim=0)
            aux_loss = self.router_aux_loss_coef * self.num_experts * torch.sum(density * mean_prob)

        # 5. Route tokens to assigned experts
        routed_out = torch.zeros_like(x_flat)
        for i in range(self.num_experts_per_tok):
            expert_idx = top_indices[:, i]
            weight = top_weights[:, i].unsqueeze(-1)
            for e_idx in range(self.num_experts):
                mask = (expert_idx == e_idx)
                if mask.any():
                    routed_out[mask] += weight[mask] * self.experts[e_idx](x_flat[mask])

        return routed_out.view(bsz, seq_len, d), aux_loss