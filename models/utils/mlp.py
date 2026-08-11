import torch
import torch.nn as nn
import torch.nn.functional as F
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
            # Deterministic Hash Routing based on token positions
            token_ids = torch.arange(x_flat.size(0), device=hidden_states.device)
            top_indices = (token_ids.unsqueeze(-1) + torch.arange(self.num_active, device=hidden_states.device)) % self.num_routed
            top_weights = torch.full_like(top_indices, 1.0 / self.num_active, dtype=hidden_states.dtype)
        else:
            logits = self.gate(x_flat)
            # Sqrt(Softplus(.)) affinity scoring activation
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
    
    Features:
    - Separates model width d from routed expert width \ell (Eq. 11).
    - Inserted RMSNorm on the aggregated routed representation before up-projection.
    - SiTU-GLU expert activations to prevent numerical instability.
    - Quantile Balancing (QB) router bias for loss-free load balancing (Eq. 13-14).
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

        # Down-projection to latent space W_\downarrow
        self.w_down = nn.Linear(self.hidden_size, self.latent_dim, bias=False)

        # Shared Experts (operating in full hidden_size space)
        self.shared_experts = nn.ModuleList([
            SiTUGLU(self.hidden_size, self.shared_intermediate)
            for _ in range(self.num_shared)
        ])

        # Routed Experts (operating in compact latent_dim space)
        self.routed_experts = nn.ModuleList([
            SiTUGLU(self.latent_dim, self.moe_intermediate)
            for _ in range(self.num_routed)
        ])

        # Normalization before up-projection (Eq. 11)
        eps = getattr(config, "rms_norm_eps", 1e-6)
        self.u_norm = RMSNorm(self.latent_dim, eps=eps)

        # Up-projection W_\uparrow back to full hidden_size space
        self.w_up = nn.Linear(self.latent_dim, self.hidden_size, bias=False)

        # Router Gate & Quantile Balancing (QB) Bias
        self.gate = nn.Linear(self.hidden_size, self.num_routed, bias=False)
        self.register_buffer("expert_bias", torch.zeros(self.num_routed))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d = hidden_states.size()
        x_flat = hidden_states.view(-1, d)

        # 1. Shared Experts Computation
        shared_out = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(x_flat)

        # 2. Down-project to Latent Space: z = W_\downarrow x
        z_flat = self.w_down(x_flat)

        # 3. Router Scoring with Quantile Balancing Bias (Eq. 13)
        logits = self.gate(x_flat)
        scores = torch.sigmoid(logits)
        biased_scores = scores + self.expert_bias

        top_weights, top_indices = torch.topk(biased_scores, k=self.num_active, dim=-1)

        # Unbiased Routing Weights p_{i,j}
        selected_scores = torch.gather(scores, 1, top_indices)
        routing_weights = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-8)

        # 4. Routed Experts Aggregation in Latent Space
        u_flat = torch.zeros_like(z_flat)
        for i in range(self.num_active):
            idx = top_indices[:, i]
            weight = routing_weights[:, i].unsqueeze(-1)
            for e_idx in range(self.num_routed):
                mask = (idx == e_idx)
                if mask.any():
                    u_flat[mask] += weight[mask] * self.routed_experts[e_idx](z_flat[mask])

        # 5. Normalized LatentMoE Up-Projection: y = y_shared + W_\uparrow RMSNorm(u) (Eq. 11)
        routed_out = self.w_up(self.u_norm(u_flat))

        return (shared_out + routed_out).view(bsz, seq_len, d)


class TopKMoE(nn.Module):
    """
    Top-K Routed Mixture of Experts (MoE) block with optional shared expert.
    Standard MoE component used in Mistral / Mixtral / Magistral MoE architectures.
    """
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = getattr(config, "num_local_experts", 8)
        self.top_k = getattr(config, "num_experts_per_tok", 2)
        self.moe_intermediate_size = getattr(config, "moe_intermediate_size", config.intermediate_size)
        self.num_shared_experts = getattr(config, "num_shared_experts", 0)

        # Router Gate Projection
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        # Routed Experts
        self.experts = nn.ModuleList([
            SwiGLUMLP(self.hidden_size, self.moe_intermediate_size)
            for _ in range(self.num_experts)
        ])

        # Optional Shared Expert
        if self.num_shared_experts > 0:
            shared_dim = self.moe_intermediate_size * self.num_shared_experts
            self.shared_expert = SwiGLUMLP(self.hidden_size, shared_dim)
        else:
            self.shared_expert = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden_dim = hidden_states.size()
        x_flat = hidden_states.view(-1, hidden_dim)

        # Compute routing logits and top-k expert selection
        router_logits = self.gate(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        top_weights, top_indices = torch.topk(routing_weights, self.top_k, dim=-1)

        # Normalize routing weights over selected top-k experts
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-8)
        top_weights = top_weights.to(hidden_states.dtype)

        # Aggregate outputs across assigned experts
        routed_out = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = top_indices[:, i]
            weight = top_weights[:, i].unsqueeze(-1)
            for e_idx in range(self.num_experts):
                mask = (expert_idx == e_idx)
                if mask.any():
                    routed_out[mask] += weight[mask] * self.experts[e_idx](x_flat[mask])

        # Add shared expert output if configured
        if self.shared_expert is not None:
            shared_out = self.shared_expert(x_flat)
            routed_out = routed_out + shared_out

        return routed_out.view(bsz, seq_len, hidden_dim)