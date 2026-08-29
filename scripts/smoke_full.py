"""End-to-end smoke test: build the full LPatchTST with the new GLM
encoder body and verify forward and backward on a small batch.

Usage:
    PYTHONPATH=/path/to/core python scripts/smoke_full.py
"""
from __future__ import annotations
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))

import config  # noqa: E402
from model import LPatchTST  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    print(f"USE_MHC={config.USE_MHC} LSTM_LAYERS={config.LSTM_LAYERS} "
          f"USE_SPARSE_LAYER={config.USE_SPARSE_LAYER} N_EXPERTS={config.N_EXPERTS} "
          f"TOPK={config.TOPK}")

    m = LPatchTST(
        input_mode="features_only",
        d_model=config.D_MODEL, n_layers=config.N_LAYERS, n_heads=config.N_HEADS,
        n_features=9, n_queries=4, n_dec_layers=1,
        dropout=0.0,
    )
    n = sum(p.numel() for p in m.parameters())
    print(f"LPatchTST total params: {n:,}")
    print(f"  use_glm_encoder: {m.use_glm_encoder}")
    print(f"  mhc_n_streams:   {m.mhc_n_streams}")

    # Forward
    B, T, F = 2, 64, 9
    x = torch.randn(B, T, F)
    out = m(features=x)
    print(f"forward features-only B={B} T={T}: shape={tuple(out.shape)} "
          f"finite={torch.isfinite(out).all().item()}")

    # Backward
    loss = out.sum()
    loss.backward()
    bad = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    print(f"params without grad after backward: {len(bad)} (expected 0)")

    # Combined mode
    m2 = LPatchTST(
        input_mode="combined", d_model=config.D_MODEL,
        n_layers=config.N_LAYERS, n_heads=config.N_HEADS,
        n_features=9, vocab_size=2 ** 20, s1_bits=10, s2_bits=10,
        n_queries=4, n_dec_layers=1, dropout=0.0,
    )
    n2 = sum(p.numel() for p in m2.parameters())
    print(f"\ncombined-mode params: {n2:,}")
    idx_c = torch.randint(0, 2 ** 10, (B, 64))
    idx_f = torch.randint(0, 2 ** 10, (B, 64))
    feat = torch.randn(B, 64, 9)
    out = m2(tokens=(idx_c, idx_f), features=feat)
    print(f"forward combined B={B} T=64: shape={tuple(out.shape)} "
          f"finite={torch.isfinite(out).all().item()}")
    print("OK")


if __name__ == "__main__":
    main()
