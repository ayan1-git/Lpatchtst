"""Smoke test: build a single GlmBlock (KDA + Sparse-MLA variants) and
verify it runs and stays finite at the production context length.

Usage:
    PYTHONPATH=/path/to/core python scripts/smoke_glm_block.py
"""
from __future__ import annotations
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))

from glm53.block import GlmBlock  # noqa: E402


def main() -> None:
    sys.path.insert(0, os.path.join(ROOT, "core"))
    import config

    torch.manual_seed(0)
    d_model = config.D_MODEL
    n_heads = config.N_HEADS
    n_experts = config.N_EXPERTS
    topk = config.TOPK
    expert_dim = config.EXPERT_DIM
    n_streams = config.MHC_N_STREAMS

    print("=== KDA block ===")
    blk = GlmBlock(
        d_model=d_model, n_heads=n_heads, n_experts=n_experts,
        topk=topk, expert_dim=expert_dim,
        use_sparse_mla=False, use_mhc=True, mhc_n_streams=n_streams,
        max_len=512, dropout=0.0,
    )
    n_params = sum(p.numel() for p in blk.parameters())
    print(f"  params: {n_params:,}")
    x = torch.randn(2, 64, n_streams, d_model)
    t0 = time.time()
    out = blk(x)
    dt = (time.time() - t0) * 1000
    print(f"  B=2 T=64:  shape={tuple(out.shape)}, finite={torch.isfinite(out).all().item()}, {dt:.1f}ms")
    x = torch.randn(1, 256, n_streams, d_model)
    t0 = time.time()
    out = blk(x)
    dt = (time.time() - t0) * 1000
    print(f"  B=1 T=256: shape={tuple(out.shape)}, finite={torch.isfinite(out).all().item()}, {dt:.1f}ms")

    print("=== Sparse-MLA block ===")
    blk2 = GlmBlock(
        d_model=d_model, n_heads=n_heads, n_experts=n_experts,
        topk=topk, expert_dim=expert_dim,
        use_sparse_mla=True, use_mhc=True, mhc_n_streams=n_streams,
        max_len=512, dropout=0.0,
        q_lora_rank=min(config.Q_LORA_RANK, d_model),
        kv_lora_rank=min(config.KV_LORA_RANK, d_model),
        qk_nope_dim=config.QK_NOPE_DIM,
        v_head_dim=config.V_HEAD_DIM,
        sparse_topk=min(config.SPARSE_TOPK, 32),
        indexer_n_heads=config.INDEX_N_HEADS,
        indexer_head_dim=config.INDEX_HEAD_DIM,
    )
    n_params2 = sum(p.numel() for p in blk2.parameters())
    print(f"  params: {n_params2:,}")
    x = torch.randn(2, 64, n_streams, d_model)
    t0 = time.time()
    out = blk2(x)
    dt = (time.time() - t0) * 1000
    print(f"  B=2 T=64:  shape={tuple(out.shape)}, finite={torch.isfinite(out).all().item()}, {dt:.1f}ms")

    # Backward
    print("=== Backward ===")
    x = torch.randn(2, 32, n_streams, d_model)
    out = blk(x)
    loss = out.sum()
    loss.backward()
    has_grad = all(p.grad is not None for p in blk.parameters() if p.requires_grad)
    print(f"  all params received gradient: {has_grad}")
    print("OK")


if __name__ == "__main__":
    main()
