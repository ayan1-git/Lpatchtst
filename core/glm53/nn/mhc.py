"""Manifold-Constrained Hyper-Connections (mHC) residual stream.

Implements the 2-stream variant of the mHC residual path from
Xie et al. arXiv 2512.24880, Eq. 5/7/8/9.

Per sublayer:
    x_n : [B, T, n, H]    (n-streams of width H)
    x_flat = x_n.flatten(-2)                            # [B, T, n*H]
    x_n = RMSNorm(x_n)                                  # per-stream
    proj = Linear(n*H, n^2 + 2n)                        # one linear
    h_pre  = sigmoid(proj[..., 0:n]   * alpha_pre  + bias[0:n])
    h_post = 2*sigmoid(proj[..., n:2n] * alpha_post + bias[n:2n])
    h_res  = sinkhorn_knopp(exp(proj[..., 2n:].view(B,T,n,n) * alpha_res + bias[2n:].view(n,n)))
    z = aggregate n -> 1 streams:  z = sum_i h_pre[..., i] * x_n[..., i, :]
    y = Sublayer(z)
    y_n = expand 1 -> n streams:  y_n[..., i, :] = h_post[..., i] * y
    x_out = einsum("btsc, btsr -> btrc", y_n, h_res) + x_n
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .sinkhorn import sinkhorn_knopp


class MHCSublayer(nn.Module):
    """Wraps a 1-stream sublayer with a 2-stream mHC residual.

    Input/output shape: [B, T, n_streams, d_model] (we default to n=2).
    The 1-stream sublayer must accept and return [B, T, d_model].

    The forward applies:
        x_n = mhc_pre_norm(x_n)            # per-stream RMSNorm
        z   = aggregate(x_n, h_pre)
        y   = sublayer(z)
        y_n = expand(y, h_post)
        x_out = mix(y_n, h_res) + x_n
    """

    def __init__(self, d_model: int, n_streams: int = 2, sinkhorn_iters: int = 8,
                 eps: float = 1e-6, pre_norm_eps: float = 1e-6):
        super().__init__()
        if n_streams < 2:
            raise ValueError("mHC needs n_streams >= 2")
        self.d_model = d_model
        self.n_streams = n_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps

        n2_plus_2n = n_streams * n_streams + 2 * n_streams
        self.pre_norm = nn.RMSNorm(d_model, eps=pre_norm_eps)
        # The combined mapping projection: n*H -> n^2 + 2n.
        self.mapping_proj = nn.Linear(n_streams * d_model, n2_plus_2n, bias=False)
        # Learnable scaling factors (alpha) and bias, init to 0 so the
        # initial mapping is the constant: h_pre = 0.5, h_post = 1.0,
        # h_res = uniform 1/n.
        self.alpha = nn.Parameter(torch.zeros(3))
        self.bias = nn.Parameter(torch.zeros(n2_plus_2n))
        # Initialise mapping_proj small so the alpha/bias-dominated
        # projection is close to the bias vector at init.
        nn.init.normal_(self.mapping_proj.weight, std=0.02)

    def _compute_mappings(self, x_flat: torch.Tensor):
        """Compute (h_pre, h_post, h_res) from flattened streams.

        Args:
            x_flat: [B, T, n*H]
        Returns:
            h_pre:  [B, T, n]
            h_post: [B, T, n]
            h_res:  [B, T, n, n]   (doubly stochastic)
        """
        B, T, _ = x_flat.shape
        n = self.n_streams
        proj = self.mapping_proj(x_flat)                   # [B, T, n^2+2n]
        # Apply learnable per-coefficient scale + bias.
        # proj has (n^2 + 2n) channels; we split into pre, post, res.
        pre = proj[..., 0:n]
        post = proj[..., n:2 * n]
        res = proj[..., 2 * n:].view(B, T, n, n)
        pre = self.alpha[0] * pre + self.bias[0:n]
        post = self.alpha[1] * post + self.bias[n:2 * n]
        res = self.alpha[2] * res + self.bias[2 * n:].view(1, 1, n, n)

        h_pre = torch.sigmoid(pre)                         # [B, T, n] in [0,1]
        h_post = 2.0 * torch.sigmoid(post)                 # [B, T, n] in [0,2]
        # H_res: doubly stochastic via Sinkhorn.
        h_res = sinkhorn_knopp(res, iters=self.sinkhorn_iters, eps=self.eps)
        return h_pre, h_post, h_res

    def forward(self, x_n: torch.Tensor, sublayer) -> torch.Tensor:
        """Apply mHC-wrapped sublayer to n-stream input.

        Args:
            x_n: [B, T, n, d_model]
            sublayer: a *bound method or callable* mapping
                [B, T, d_model] -> [B, T, d_model]. We accept both
                nn.Module (which would normally intercept __call__)
                and a plain function. The mHCSublayer only invokes
                `sublayer(z)` directly, not `sublayer.__call__(z)`.
        Returns:
            x_out: [B, T, n, d_model]
        """
        B, T, n, H = x_n.shape
        assert n == self.n_streams, f"got n={n}, expected {self.n_streams}"

        # Per-stream RMSNorm along the last dim.
        x_normed = self.pre_norm(x_n.reshape(B * T * n, H)).reshape(B, T, n, H)
        # Flatten streams for the mapping projection.
        x_flat = x_normed.reshape(B, T, n * H)
        h_pre, h_post, h_res = self._compute_mappings(x_flat)

        # Aggregate n -> 1 stream:  z[b,t] = sum_i h_pre[b,t,i] * x_n[b,t,i,:]
        z = (h_pre.unsqueeze(-1) * x_normed).sum(dim=2)    # [B, T, H]

        # Run the wrapped sublayer on the aggregated 1-stream.
        # We support both nn.Module and plain functions. nn.Module's
        # __call__ is what gets invoked when you do `m(z)`, but here
        # we want to bypass any hooks the Module might have, so we
        # call `m.forward` if it is a Module, else the function.
        if isinstance(sublayer, nn.Module):
            y = sublayer.forward(z)
        else:
            y = sublayer(z)                                # [B, T, H]

        # Expand 1 -> n streams:  y_n[b,t,i] = h_post[b,t,i] * y[b,t]
        y_n = h_post.unsqueeze(-1) * y.unsqueeze(2)        # [B, T, n, H]

        # Residual mix:  x_out = einsum("btsc, btsr -> btrc", y_n, h_res) + x_n
        # We don't need the input x_n here for the mix (the residual is
        # added in mHC's outside addition; the wrapped sublayer already
        # returns y_n mixed via h_res). The "x_n" is added at the call site.
        mixed = torch.einsum("btsc,btsr->btrc", y_n, h_res)  # [B, T, n, H]
        return mixed
