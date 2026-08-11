import torch
import torch.nn as nn
import torch.nn.functional as F
from .normalization import RMSNorm


class ManifoldConstrainedHyperConnections(nn.Module):
    """
    Manifold-Constrained Hyper-Connections (mHC) module.
    
    Expands the residual stream width by a factor of n_hc, using dynamic parameterization
    and projecting the residual mapping matrix onto the Birkhoff polytope (doubly stochastic
    matrices) using the Sinkhorn-Knopp algorithm.
    """
    def __init__(self, hidden_size: int, n_hc: int = 4, t_max: int = 20):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_hc = n_hc
        self.t_max = t_max

        # Normalization over flattened residual state (1, n_hc * d)
        self.norm = RMSNorm(n_hc * hidden_size)

        # Dynamic weight projection matrices
        self.w_pre = nn.Linear(n_hc * hidden_size, n_hc, bias=False)
        self.w_res = nn.Linear(n_hc * hidden_size, n_hc * n_hc, bias=False)
        self.w_post = nn.Linear(n_hc * hidden_size, n_hc, bias=False)

        # Static biases
        self.s_pre = nn.Parameter(torch.zeros(1, n_hc))
        self.s_res = nn.Parameter(torch.zeros(n_hc, n_hc))
        self.s_post = nn.Parameter(torch.zeros(n_hc, 1))

        # Learnable gating factors initialized to small values
        self.alpha_pre = nn.Parameter(torch.full((1,), 1e-2))
        self.alpha_res = nn.Parameter(torch.full((1,), 1e-2))
        self.alpha_post = nn.Parameter(torch.full((1,), 1e-2))

    def _sinkhorn_knopp(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Projects matrix onto the manifold of doubly stochastic matrices using Sinkhorn-Knopp algorithm.
        matrix shape: (batch_size, n_hc, n_hc)
        """
        # Ensure positivity via exponentiation
        m = torch.exp(matrix)
        for _ in range(self.t_max):
            # Column normalization
            m = m / (m.sum(dim=-2, keepdim=True) + 1e-8)
            # Row normalization
            m = m / (m.sum(dim=-1, keepdim=True) + 1e-8)
        return m

    def forward(self, x_l: torch.Tensor):
        """
        Args:
            x_l: Residual state tensor of shape (batch_size, seq_len, n_hc, hidden_size)
        Returns:
            a_l: Input mapping (batch_size, seq_len, 1, n_hc)
            b_l: Residual mapping (batch_size, seq_len, n_hc, n_hc)
            c_l: Output mapping (batch_size, seq_len, n_hc, 1)
        """
        bsz, seq_len, n_hc, d = x_l.size()
        
        # Flatten and normalize input hidden state
        x_flat = x_l.view(bsz, seq_len, n_hc * d)
        x_norm = self.norm(x_flat)

        # Generate unconstrained dynamic components
        a_tilde = self.alpha_pre * self.w_pre(x_norm) + self.s_pre
        b_tilde = self.alpha_res * self.w_res(x_norm).view(bsz, seq_len, n_hc, n_hc) + self.s_res
        c_tilde = self.alpha_post * self.w_post(x_norm).unsqueeze(-1) + self.s_post

        # Apply constraints
        a_l = torch.sigmoid(a_tilde).unsqueeze(-2)  # (bsz, seq_len, 1, n_hc)
        c_l = 2.0 * torch.sigmoid(c_tilde)          # (bsz, seq_len, n_hc, 1)
        b_l = self._sinkhorn_knopp(b_tilde)         # (bsz, seq_len, n_hc, n_hc)

        return a_l, b_l, c_l