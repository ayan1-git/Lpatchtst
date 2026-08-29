"""Sinkhorn-Knopp projection onto the Birkhoff polytope.

Projects a positive matrix onto the set of doubly-stochastic matrices
(rows and columns sum to 1, entries >= 0) by iterating row-normalize /
column-normalize. Used by the mHC residual stream to constrain
H_res to the doubly-stochastic manifold.

Reference: Xie et al. arXiv 2512.24880 (Manifold-Constrained
Hyper-Connections), Eq. 8 and 9.
"""
from __future__ import annotations
import torch


def sinkhorn_knopp(M: torch.Tensor, iters: int = 20, eps: float = 1e-6) -> torch.Tensor:
    """Project a positive matrix to doubly-stochastic via Sinkhorn-Knopp.

    Args:
        M: tensor with last two dims being the matrix dims to normalize,
           e.g. [B, T, n, n] or [n, n]. Values are expected to be
           already-positive (we will not exponentiate; that is done
           before calling this function in the mHC module).
        iters: number of row/column normalization rounds.
        eps: small constant for numerical stability in the divisor.

    Returns:
        Tensor of the same shape as M, doubly-stochastic along the
        last two dims (rows and columns each sum to 1 within `eps`).
    """
    # Row-max subtraction for numerical stability, then exp to ensure positivity.
    M = M - M.amax(dim=-1, keepdim=True)
    M = M.exp()
    for _ in range(iters):
        # Row normalize: each row sums to 1
        M = M / (M.sum(dim=-1, keepdim=True) + eps)
        # Column normalize: each column sums to 1
        M = M / (M.sum(dim=-2, keepdim=True) + eps)
    return M
