import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUMLP(nn.Module):
    """SwiGLU Multi-Layer Perceptron"""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class ClampedSwiGLUMLP(nn.Module):
    """
    SwiGLU MLP featuring DeepSeek-V4 numerical clamping for training stability.
    Clamps linear component in [-10, 10] and gate upper bound to 10.
    """
    def __init__(
        self, 
        hidden_size: int, 
        intermediate_size: int, 
        clamp_min: float = -10.0, 
        clamp_max: float = 10.0, 
        gate_max: float = 10.0
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.gate_max = gate_max

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        if self.clamp_min is not None and self.clamp_max is not None:
            up = torch.clamp(up, self.clamp_min, self.clamp_max)
            gate = torch.clamp(gate, self.clamp_min, min(self.clamp_max, self.gate_max))

        return self.down_proj(self.act_fn(gate) * up)


class DeepSeekMoE(nn.Module):
    """
    DeepSeekMoE Module incorporating shared experts, fine-grained routed experts,
    Sqrt(Softplus(.)) affinity routing, and optional Hash Routing for initial layers.
    """
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_routed_experts: int = 32,
        num_active_experts: int = 4,
        num_shared_experts: int = 1,
        use_hash_routing: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_routed_experts = num_routed_experts
        self.num_active_experts = num_active_experts
        self.num_shared_experts = num_shared_experts
        self.use_hash_routing = use_hash_routing

        # Shared Experts
        if num_shared_experts > 0:
            self.shared_experts = ClampedSwiGLUMLP(hidden_size, intermediate_size * num_shared_experts)
        else:
            self.shared_experts = None

        # Fine-grained Routed Experts
        self.routed_experts = nn.ModuleList([
            ClampedSwiGLUMLP(hidden_size, intermediate_size) for _ in range(num_routed_experts)
        ])

        # Routing Gate Projection (Sqrt(Softplus(.)) activated)
        if not use_hash_routing:
            self.gate = nn.Linear(hidden_size, num_routed_experts, bias=False)

    def _compute_routing_scores(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x)
        return torch.sqrt(F.softplus(logits))

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor = None) -> torch.Tensor:
        bsz, seq_len, d = x.shape
        x_flat = x.view(-1, d)

        # 1. Compute Shared Expert Output
        shared_output = self.shared_experts(x) if self.shared_experts is not None else 0.0

        # 2. Compute Routed Expert Output
        if self.use_hash_routing and input_ids is not None:
            flat_ids = input_ids.view(-1)
            base_idx = (flat_ids % self.num_routed_experts).unsqueeze(-1)
            offsets = torch.arange(self.num_active_experts, device=x.device).unsqueeze(0)
            topk_indices = (base_idx + offsets) % self.num_routed_experts
            topk_weights = torch.full_like(topk_indices, 1.0 / self.num_active_experts, dtype=x.dtype)
        else:
            scores = self._compute_routing_scores(x_flat)
            topk_weights, topk_indices = torch.topk(scores, self.num_active_experts, dim=-1)
            topk_weights = F.softmax(topk_weights, dim=-1)

        routed_output = torch.zeros_like(x_flat)
        for i in range(self.num_active_experts):
            expert_idx = topk_indices[:, i]
            weights = topk_weights[:, i].unsqueeze(-1)

            for e_idx in range(self.num_routed_experts):
                mask = (expert_idx == e_idx)
                if mask.any():
                    expert_in = x_flat[mask]
                    expert_out = self.routed_experts[e_idx](expert_in)
                    routed_output[mask] += expert_out * weights[mask]

        return shared_output + routed_output.view(bsz, seq_len, d)