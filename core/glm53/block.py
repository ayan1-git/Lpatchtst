"""Combined GLM-style block: mHC-wrapped attention + mHC-wrapped MoE.

Layer layout (mirrors the GLM 3:1 KDA:Sparse ratio at small scale):
    [KDA, KDA, KDA, Sparse-MLA] repeated.

Each block runs the pattern:
    x = mhc_attn(attn, norm_attn, x) + x   # mHC-wrapped sublayer, then identity add
    x = mhc_mlp(moe,  norm_moe,  x) + x
where the "x + x" outside is the mHC convention of treating the
output of MHCSublayer as the additive delta.

If `use_mhc=False`, we fall back to a standard residual:
    x = x + attn(norm_attn(x))
    x = x + moe(norm_moe(x))
"""
from __future__ import annotations
import torch
import torch.nn as nn
from .nn.mhc import MHCSublayer
from .nn._norm import RMSNorm


class GlmBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int,
                 n_experts: int, topk: int, expert_dim: int,
                 use_sparse_mla: bool = False,
                 # KDA params
                 kda_head_dim: int | None = None,
                 kda_short_conv: int = 4,
                 kda_gate_lower_bound: float = -5.0,
                 # Sparse MLA params
                 q_lora_rank: int = 64, kv_lora_rank: int = 32,
                 qk_nope_dim: int = 64, v_head_dim: int = 64,
                 sparse_topk: int = 64,
                 indexer_n_heads: int = 4, indexer_head_dim: int = 32,
                 # mHC params
                 use_mhc: bool = True,
                 mhc_n_streams: int = 2,
                 sinkhorn_iters: int = 8,
                 # MoE params
                 n_shared_experts: int = 1,
                 routed_scaling_factor: float = 2.5,
                 swiglu_limit: float = 10.0,
                 dropout: float = 0.0,
                 max_len: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.use_sparse_mla = use_sparse_mla
        self.use_mhc = use_mhc
        self.mhc_n_streams = mhc_n_streams

        # Pre-norms.
        self.norm_attn = RMSNorm(d_model, eps=1e-6)
        self.norm_moe = RMSNorm(d_model, eps=1e-6)

        # Attention sublayer.
        if use_sparse_mla:
            from .nn.sparse_mla import SparseMLA
            self.attn = SparseMLA(
                d_model=d_model, n_heads=n_heads,
                q_lora_rank=q_lora_rank, kv_lora_rank=kv_lora_rank,
                qk_nope_dim=qk_nope_dim, v_head_dim=v_head_dim,
                topk=sparse_topk,
                indexer_n_heads=indexer_n_heads,
                indexer_head_dim=indexer_head_dim,
            )
        else:
            from .nn.kda import KDALinearAttention
            self.attn = KDALinearAttention(
                d_model=d_model, n_heads=n_heads,
                head_dim=kda_head_dim,
                short_conv_kernel=kda_short_conv,
                gate_lower_bound=kda_gate_lower_bound,
                max_len=max_len,
            )

        # MoE sublayer.
        from .nn.moe import NoAuxMoE
        self.moe = NoAuxMoE(
            d_model=d_model, n_experts=n_experts, topk=topk,
            expert_dim=expert_dim, n_shared=n_shared_experts,
            routed_scaling_factor=routed_scaling_factor,
            swiglu_limit=swiglu_limit,
        )

        # mHC wrappers.
        if use_mhc:
            self.hc_attn = MHCSublayer(
                d_model=d_model, n_streams=mhc_n_streams,
                sinkhorn_iters=sinkhorn_iters,
            )
            self.hc_moe = MHCSublayer(
                d_model=d_model, n_streams=mhc_n_streams,
                sinkhorn_iters=sinkhorn_iters,
            )
        else:
            self.hc_attn = None
            self.hc_moe = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, x_n: torch.Tensor) -> torch.Tensor:
        """x_n: [B, T, n_streams, d_model] when mHC is enabled, else
        [B, T, d_model]. The mhc_expand/mhc_collapse ops at the
        PatchTST level take care of the 1<->2 stream conversion.
        """
        if self.use_mhc:
            assert x_n.dim() == 4, "mHC expects [B, T, n_streams, d_model]"
            # mHC-wrapped sublayer returns the *delta* (mixed y_n).
            delta_attn = self.hc_attn(x_n, self._attn_sublayer)
            x_n = x_n + self.dropout(delta_attn)
            delta_moe = self.hc_moe(x_n, self._moe_sublayer)
            x_n = x_n + self.dropout(delta_moe)
            return x_n
        else:
            assert x_n.dim() == 3
            h = self.norm_attn(x_n)
            x_n = x_n + self.dropout(self.attn(h))
            h = self.norm_moe(x_n)
            x_n = x_n + self.dropout(self.moe(h))
            return x_n

    def _attn_sublayer(self, z: torch.Tensor) -> torch.Tensor:
        return self.attn(self.norm_attn(z))

    def _moe_sublayer(self, z: torch.Tensor) -> torch.Tensor:
        return self.moe(self.norm_moe(z))
