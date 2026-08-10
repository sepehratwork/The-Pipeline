import torch
import torch.nn as nn
import torch.nn.functional as F
from .normalization import RMSNorm


class ManifoldConstrainedHyperConnections(nn.Module):
    """
    Manifold-Constrained Hyper-Connections (mHC) as introduced in DeepSeek-V4.
    
    Expands residual stream width by n_hc and constrains the residual mapping matrix 
    to the Birkhoff polytope (doubly stochastic matrices) using the Sinkhorn-Knopp algorithm
    to preserve numerical stability across ultra-deep networks.
    """
    def __init__(self, hidden_size: int, n_hc: int = 4, t_max: int = 20):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_hc = n_hc
        self.t_max = t_max

        self.norm = RMSNorm(n_hc * hidden_size)

        # Projections for dynamic parameter generation
        self.w_pre = nn.Linear(n_hc * hidden_size, n_hc, bias=False)
        self.w_res = nn.Linear(n_hc * hidden_size, n_hc * n_hc, bias=False)
        self.w_post = nn.Linear(n_hc * hidden_size, n_hc, bias=False)

        # Static learnable biases
        self.s_pre = nn.Parameter(torch.zeros(1, n_hc))
        self.s_res = nn.Parameter(torch.zeros(n_hc, n_hc))
        self.s_post = nn.Parameter(torch.zeros(n_hc, 1))

        # Learnable gating factors initialized to small values
        self.alpha_pre = nn.Parameter(torch.tensor(0.01))
        self.alpha_res = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))

    def _sinkhorn_knopp(self, b_raw: torch.Tensor) -> torch.Tensor:
        """
        Projects unconstrained raw matrix onto the manifold of doubly stochastic matrices.
        """
        m = torch.exp(b_raw)
        for _ in range(self.t_max):
            m = m / (m.sum(dim=-1, keepdim=True) + 1e-8)
            m = m / (m.sum(dim=-2, keepdim=True) + 1e-8)
        return m

    def forward(self, residual_state: torch.Tensor, block_fn, *args, **kwargs):
        """
        Args:
            residual_state: Shape [bsz, seq_len, n_hc, hidden_size]
            block_fn: Callable layer module (Attention or MoE)
        Returns:
            Updated residual_state: Shape [bsz, seq_len, n_hc, hidden_size]
        """
        bsz, seq_len, n_hc, d = residual_state.shape
        x_flat = residual_state.view(bsz, seq_len, n_hc * d)
        x_norm = self.norm(x_flat)

        # Dynamic parameter generation
        a_raw = self.alpha_pre * self.w_pre(x_norm) + self.s_pre
        b_raw = self.alpha_res * self.w_res(x_norm).view(bsz, seq_len, n_hc, n_hc) + self.s_res
        c_raw = self.alpha_post * self.w_post(x_norm).unsqueeze(-1) + self.s_post

        # Parameter constraints (Sigmoid for A & C, Sinkhorn-Knopp for B)
        a = torch.sigmoid(a_raw)                       # [bsz, seq_len, n_hc]
        c = 2.0 * torch.sigmoid(c_raw)                 # [bsz, seq_len, n_hc, 1]
        b = self._sinkhorn_knopp(b_raw)               # [bsz, seq_len, n_hc, n_hc]

        # Inner layer input: A_l * X_l -> [bsz, seq_len, d]
        layer_input = torch.matmul(a.unsqueeze(-2), residual_state).squeeze(-2)

        # Execute block computation (Attention / MoE)
        block_out = block_fn(layer_input, *args, **kwargs)
        if isinstance(block_out, tuple):
            layer_output, aux_info = block_out[0], block_out[1:]
        else:
            layer_output, aux_info = block_out, ()

        # Update equation: X_{l+1} = B_l * X_l + C_l * F_l(A_l * X_l)
        bx = torch.matmul(b, residual_state)
        cy = c * layer_output.unsqueeze(-2)
        next_residual_state = bx + cy

        if aux_info:
            return next_residual_state, *aux_info
        return next_residual_state