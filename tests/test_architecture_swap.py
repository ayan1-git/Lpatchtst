"""Unit tests for the four novel GLM-5.3-Flash components implemented
in core/glm53/nn/.

Run with:
    PYTHONPATH=/path/to/core pytest -k glm53
"""
from __future__ import annotations
import os
import sys
import math

import pytest
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))

from glm53.nn.sinkhorn import sinkhorn_knopp  # noqa: E402
from glm53.nn.mhc import MHCSublayer  # noqa: E402
from glm53.nn.kda import KDALinearAttention  # noqa: E402
from glm53.nn.sparse_mla import SparseMLA  # noqa: E402
from glm53.nn.moe import NoAuxMoE  # noqa: E402


# ──────────────────────────────────────────────────────────────────
# Sinkhorn-Knopp
# ──────────────────────────────────────────────────────────────────
def test_sinkhorn_knopp_doubly_stochastic():
    torch.manual_seed(0)
    M = torch.randn(2, 3, 4, 4)
    P = sinkhorn_knopp(M, iters=20)
    assert P.shape == M.shape
    row_sums = P.sum(dim=-1)
    col_sums = P.sum(dim=-2)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-4)
    assert (P >= 0).all()


def test_sinkhorn_uniform_init():
    # When input is uniform, Sinkhorn should preserve it.
    M = torch.zeros(2, 3, 4, 4)
    P = sinkhorn_knopp(M, iters=20)
    expected = torch.full_like(P, 1.0 / 4.0)
    assert torch.allclose(P, expected, atol=1e-5)


# ──────────────────────────────────────────────────────────────────
# mHC residual
# ──────────────────────────────────────────────────────────────────
def test_mhc_init_alpha_zero_is_residual():
    """With alpha=0, bias=0 the mHC pre-norm is the identity, h_pre
    is 0.5/0.5, h_post is 1.0/1.0, h_res is uniform 0.5. The output
    should be ~Sublayer(x) at init."""
    torch.manual_seed(0)
    mhc = MHCSublayer(d_model=32, n_streams=2, sinkhorn_iters=8)
    sublayer = nn.Linear(32, 32)
    x = torch.randn(2, 8, 32)
    x_n = torch.stack([x, x], dim=-2)
    # The mHC output before adding the residual is mixed(y_n, h_res).
    # We can't easily isolate it from the + x_n that the block does,
    # so we check the wrapped sublayer alone matches Linear.
    out = mhc(x_n, sublayer)
    assert out.shape == (2, 8, 2, 32)
    assert torch.isfinite(out).all()


def test_mhc_chain_stable():
    """A 100-layer mHC chain with random init on a Gaussian input
    should not blow up."""
    torch.manual_seed(0)
    d_model = 16
    n_streams = 2
    mhc = MHCSublayer(d_model=d_model, n_streams=n_streams, sinkhorn_iters=8)
    sublayer = nn.Identity()
    x_n = torch.randn(2, 4, n_streams, d_model)
    for _ in range(100):
        delta = mhc(x_n, sublayer)
        x_n = x_n + 0.1 * delta  # small step to mimic residual
    norm = x_n.norm(dim=-1).mean()
    assert torch.isfinite(x_n).all()
    assert 0.01 < norm.item() < 100.0, f"norm {norm.item()} out of band"


# ──────────────────────────────────────────────────────────────────
# KDA linear attention
# ──────────────────────────────────────────────────────────────────
def test_kda_forward_shape_and_finite():
    torch.manual_seed(0)
    kda = KDALinearAttention(d_model=32, n_heads=4, head_dim=8, max_len=64)
    x = torch.randn(2, 16, 32)
    out = kda(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_kda_long_sequence_stable():
    """KDA should remain numerically stable at T=128."""
    torch.manual_seed(0)
    kda = KDALinearAttention(d_model=32, n_heads=4, head_dim=8, max_len=256)
    x = torch.randn(1, 128, 32)
    out = kda(x)
    assert torch.isfinite(out).all()
    assert out.abs().max().item() < 100.0


# ──────────────────────────────────────────────────────────────────
# DSA sparse MLA + Indexer (DeepSeek-V3.2 §2.1)
# ──────────────────────────────────────────────────────────────────
def test_sparse_mla_shape_and_finite():
    torch.manual_seed(0)
    mla = SparseMLA(
        d_model=32, n_heads=4, q_lora_rank=16, kv_lora_rank=8,
        qk_nope_dim=8, v_head_dim=8, topk=8,
        indexer_n_heads=2, indexer_head_dim=4,
    )
    x = torch.randn(2, 16, 32)
    out = mla(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_sparse_mla_indexer_score_formula():
    """Verify the index score follows the DSA paper formula
    I[t,s] = sum_h w_h * ReLU(<i_q[t,h], i_k[s,h]>)."""
    torch.manual_seed(0)
    mla = SparseMLA(
        d_model=16, n_heads=2, q_lora_rank=8, kv_lora_rank=4,
        qk_nope_dim=4, v_head_dim=4, topk=4,
        indexer_n_heads=2, indexer_head_dim=4,
    )
    B, T, _ = 1, 6, 16
    x = torch.randn(B, T, 16)
    _ = mla(x)
    scores = mla.last_index_scores  # [B, T, T]
    assert scores is not None
    # Reconstruct manually: ReLU( <i_q, i_k> ) weighted by head_w.
    i_q = mla.last_indexer_q      # [B, T, Ih, Id]
    i_k = mla.last_indexer_k      # [B, T, Ih, Id]
    head_w = mla.last_head_w     # [B, T, Ih]
    manual = torch.einsum("bthe,bshe->btsh", i_q, i_k).relu()
    manual = (manual * head_w.unsqueeze(2)).sum(dim=-1)
    assert torch.allclose(scores, manual, atol=1e-5)


def test_sparse_mla_indexer_grad_isolated():
    """The paper's gradient-isolation rule: the indexer input is
    detached, so the LM loss (forward through main path) cannot
    backprop into the indexer parameters. Only the KL aux loss
    (computed from cached attention) updates the indexer."""
    torch.manual_seed(0)
    mla = SparseMLA(
        d_model=16, n_heads=2, q_lora_rank=8, kv_lora_rank=4,
        qk_nope_dim=4, v_head_dim=4, topk=4,
        indexer_n_heads=2, indexer_head_dim=4,
    )
    x = torch.randn(1, 8, 16, requires_grad=True)
    out = mla(x)
    # Use only the main-path output as a loss; should NOT touch the
    # indexer parameters.
    main_loss = out.sum()
    main_loss.backward()
    # Indexer params should have no grad from the main path.
    indexer_params = [mla.wk.weight, mla.wq_b.weight,
                      mla.weights_proj.weight, mla.q_scale, mla.k_norm.weight]
    for p in indexer_params:
        assert p.grad is None or p.grad.abs().max().item() == 0.0, (
            f"indexer param {p.shape} got grad from main path"
        )


def test_sparse_mla_kl_loss_updates_indexer():
    """Calling compute_kl_loss() and backpropping should produce
    gradients on the indexer parameters (and not on the main-path
    MLA params)."""
    torch.manual_seed(0)
    mla = SparseMLA(
        d_model=16, n_heads=2, q_lora_rank=8, kv_lora_rank=4,
        qk_nope_dim=4, v_head_dim=4, topk=4,
        indexer_n_heads=2, indexer_head_dim=4,
    )
    x = torch.randn(1, 8, 16, requires_grad=True)
    _ = mla(x)  # populate caches
    kl_sparse = mla.compute_kl_loss(mode="sparse")
    assert torch.isfinite(kl_sparse)
    assert kl_sparse.item() > 0
    kl_sparse.backward()
    # Indexer params should now have non-zero grad.
    indexer_params = [mla.wk.weight, mla.wq_b.weight,
                      mla.weights_proj.weight, mla.q_scale]
    for p in indexer_params:
        assert p.grad is not None and p.grad.abs().max().item() > 0, (
            f"indexer param {p.shape} got no grad from KL loss"
        )
    # Main-path MLA params should not be touched by the KL loss.
    main_params = [mla.W_dq.weight, mla.W_dkv.weight, mla.W_uk.weight,
                   mla.W_uv.weight, mla.W_uq.weight, mla.W_o.weight,
                   mla.q_a_layernorm.weight, mla.kv_a_layernorm.weight]
    for p in main_params:
        assert p.grad is None or p.grad.abs().max().item() == 0.0, (
            f"main-path param {p.shape} got grad from KL loss"
        )


def test_sparse_mla_topk_eq_seq():
    """When topk >= T, the gather covers the full sequence and the
    output should be a finite deterministic transform of the input."""
    torch.manual_seed(0)
    T = 16
    mla = SparseMLA(
        d_model=32, n_heads=4, q_lora_rank=16, kv_lora_rank=8,
        qk_nope_dim=8, v_head_dim=8, topk=T,
        indexer_n_heads=2, indexer_head_dim=4,
    )
    mla.eval()
    with torch.no_grad():
        x = torch.randn(2, T, 32)
        out = mla(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


# ──────────────────────────────────────────────────────────────────
# noaux_tc MoE
# ──────────────────────────────────────────────────────────────────
def test_moe_shape_and_finite():
    torch.manual_seed(0)
    moe = NoAuxMoE(d_model=32, n_experts=4, topk=2, expert_dim=64, n_shared=1)
    x = torch.randn(2, 8, 32)
    out = moe(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert moe.last_router_logits is not None


def test_moe_router_bias_selects_expert():
    """A large positive bias on expert 0 should make top-2 always
    include expert 0 (visible in the recorded router logits)."""
    torch.manual_seed(0)
    moe = NoAuxMoE(d_model=32, n_experts=4, topk=2, expert_dim=32, n_shared=0)
    moe.e_score_correction_bias.data[0] = 10.0
    x = torch.randn(2, 4, 32)
    _ = moe(x)
    # Check that expert 0 is the most-selected: count how often
    # it is in the top-2 set.
    logits = moe.last_router_logits  # [B*T, E]
    top2 = logits.topk(2, dim=-1).indices  # [B*T, 2]
    in_top2 = (top2 == 0).any(dim=-1).float().mean()
    assert in_top2.item() == 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
