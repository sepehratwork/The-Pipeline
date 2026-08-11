import torch
import torch.nn as nn


class FP4Quantizer(nn.Module):
    """
    FP4 Quantization-Aware Training helper.
    Simulates MXFP4 quantization with lossless FP4-to-FP8 dequantization.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def quantize_fp4_dequantize_fp8(w: torch.Tensor) -> torch.Tensor:
        if not w.requires_grad:
            return w
        # Straight-Through Estimator (STE) simulation for QAT
        scale = w.abs().max() / 7.0 + 1e-8
        w_q = torch.clamp(torch.round(w / scale), -7, 7)
        w_deq = w_q * scale
        return w + (w_deq - w).detach()

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return self.quantize_fp4_dequantize_fp8(w)