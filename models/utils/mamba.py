import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .normalization import RMSNorm


class Mamba2Layer(nn.Module):
    """
    Pure PyTorch implementation of the Mamba-2 (State Space Dual / SSD) layer.
    Used as the primary sequence modeling block in Nemotron 3 to maintain fixed
    inference memory overhead and high processing throughput.
    """
    def __init__(
        self, 
        d_model: int, 
        d_state: int = 64, 
        d_conv: int = 4, 
        expand: int = 2, 
        headdim: int = 64, 
        ngroups: int = 1
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.headdim = headdim
        self.nheads = self.d_inner // headdim
        self.ngroups = ngroups

        # Input projections for gate (z), ssm input (x), dt, B, C
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        # 1D Depthwise Convolution over sequence length
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=True,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
        )

        # Learnable SSM parameters (Log-spaced decay rates & bias for dt)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.nheads + 1, dtype=torch.float32)))
        self.dt_bias = nn.Parameter(torch.ones(self.nheads))

        # Output normalization and projection
        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, hidden_states: torch.Tensor, past_key_value=None):
        bsz, seq_len, _ = hidden_states.shape

        # 1. Linear input projection
        projected = self.in_proj(hidden_states)
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        z, x_bc, dt = torch.split(projected, [self.d_inner, conv_dim, self.nheads], dim=-1)

        # 2. Short 1D convolution with SiLU activation
        x_bc_conv = self.conv1d(x_bc.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_bc_conv = F.silu(x_bc_conv)

        x_ssm, B, C = torch.split(
            x_bc_conv, 
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], 
            dim=-1
        )

        # 3. Discretization step for dt and A matrix
        dt = F.softplus(dt + self.dt_bias)
        A = -torch.exp(self.A_log)

        # Reshape for multi-head state space dual execution
        x_ssm = x_ssm.view(bsz, seq_len, self.nheads, self.headdim)
        B = B.view(bsz, seq_len, self.ngroups, self.d_state)
        C = C.view(bsz, seq_len, self.ngroups, self.d_state)

        if self.ngroups < self.nheads:
            heads_per_group = self.nheads // self.ngroups
            B = B.repeat_interleave(heads_per_group, dim=2)
            C = C.repeat_interleave(heads_per_group, dim=2)

        # 4. State Space Recurrence over sequence tokens
        decay = torch.exp(A.view(1, 1, self.nheads, 1) * dt.unsqueeze(-1))
        
        y_states = []
        h = torch.zeros(
            bsz, self.nheads, self.headdim, self.d_state, 
            device=hidden_states.device, dtype=hidden_states.dtype
        )
        
        for t in range(seq_len):
            dt_t = dt[:, t, :, None, None]
            x_t = x_ssm[:, t, :, :, None]
            B_t = B[:, t, :, None, :]
            C_t = C[:, t, :, None, :]
            decay_t = decay[:, t, :, None]

            # Recurrent hidden state update: h_t = h_{t-1} * decay_t + dt_t * (x_t @ B_t)
            h = h * decay_t + dt_t * (x_t @ B_t)
            
            # Compute token state output: y_t = h_t @ C_t^T
            y_t = (h @ C_t.transpose(-1, -2)).squeeze(-1)
            y_states.append(y_t)

        y = torch.stack(y_states, dim=1).view(bsz, seq_len, self.d_inner)

        # 5. Gating and output linear projection
        y = self.norm(y * F.silu(z))
        output = self.out_proj(y)

        return output, (h,)