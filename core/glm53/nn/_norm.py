"""AMP-safe RMSNorm used by the GLM-5.3-Flash mini components.

The native `torch.nn.RMSNorm` keeps its `weight` parameter in FP32
and emits a `Mismatch dtype between input and weight` warning
when the input is FP16 (e.g. under autocast). For our use this
warning is harmless but noisy in training logs.

The project's `core.model.RMSNorm` already handles this by
multiplying in the input dtype (`x * rms * self.weight`). We
re-implement the same idea locally to avoid a circular import
from `core.model` into `core.glm53.nn`.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the normalization in the input dtype; only the
        # final multiplication with `weight` may promote. This
        # matches the existing project's `RMSNorm` behavior and
        # avoids the AMP dtype-mismatch warning.
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight
