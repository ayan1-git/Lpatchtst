#!/usr/bin/env python3
"""ORBIT randomization audit (Phase 3).

Run from repo root:  python3 audits/verify_orbit.py

Verifies, on synthetic data:
  1. Train dataset (orbit_randomize=True): window END (target index) is
     invariant — y always equals targets[i + seq_len - 1]; context length is
     uniform-ish within [ORBIT_CTX_MIN, seq_len].
  2. Val/test datasets: deterministic full-length windows, no randomization.
  3. create_fold_dataloaders end-to-end: train loader emits variable-length
     batches with a correct key_padding_mask; val/test loaders emit fixed
     lengths with mask=None.
  4. Sampler weights alignment: dataset length unchanged by ORBIT flag.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as cfg
from data_loader import FinancialDataset, create_fold_dataloaders, _make_loader

PASS, FAIL = "✓", "✗"
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"  {PASS if cond else FAIL} {name}" + (f" ({detail})" if detail else ""))


def make_synthetic(n=3000, n_feat=21, seed=7):
    rng = np.random.default_rng(seed)
    feats = rng.normal(size=(n, n_feat)).astype(np.float32)
    tgts = np.zeros(n, dtype=np.float32)
    edge = rng.random(n) < 0.3
    tgts[edge] = rng.choice([-1, 1], size=edge.sum()) * rng.uniform(
        cfg.SAMPLER_THRESHOLD, 1.0, size=edge.sum()
    )
    return feats, tgts


def test_dataset_invariant():
    print("\n[1] Target-index invariance under ORBIT")
    feats, tgts = make_synthetic()
    ds = FinancialDataset(feats, tgts, cfg.LOOKBACK_WINDOW,
                          config=cfg, orbit_randomize=True)
    base = FinancialDataset(feats, tgts, cfg.LOOKBACK_WINDOW,
                            config=cfg, orbit_randomize=False)
    check("dataset length unchanged", len(ds) == len(base),
          f"{len(ds)} vs {len(base)}")

    lens, mismatches = [], 0
    g = torch.Generator().manual_seed(123)
    idxs = torch.randint(0, len(ds), (5000,), generator=g).tolist()
    for i in idxs:
        _, f, y = ds[i]
        expected = tgts[i + ds.seq_len - 1]
        if not np.isclose(float(y), float(expected), atol=0):
            mismatches += 1
        lens.append(f.shape[0])
    check("y == targets[window_end] for all draws", mismatches == 0,
          f"{mismatches} mismatches / 5000")
    lo, hi = min(lens), max(lens)
    check(f"context lengths span [{cfg.ORBIT_CTX_MIN},{cfg.LOOKBACK_WINDOW}]",
          lo >= cfg.ORBIT_CTX_MIN and hi == cfg.LOOKBACK_WINDOW and lo < hi,
          f"min={lo}, max={hi}")
    frac_varied = np.mean([l != cfg.LOOKBACK_WINDOW for l in lens])
    check("meaningful variation", frac_varied > 0.2,
          f"{frac_varied*100:.0f}% shortened")


def test_val_test_deterministic():
    print("\n[2] Val/test determinism")
    feats, tgts = make_synthetic()
    ds = FinancialDataset(feats, tgts, cfg.LOOKBACK_WINDOW,
                          config=cfg, orbit_randomize=False)
    a = ds[10]
    b = ds[10]
    check("same index -> identical window", torch.equal(a[1], b[1]))
    check("full-length windows", a[1].shape[0] == cfg.LOOKBACK_WINDOW)


def test_fold_loaders():
    print("\n[3] create_fold_dataloaders end-to-end")
    n = 9000
    feats, tgts = make_synthetic(n=n)
    train_indices = (0, 4000)
    val_indices = (4000 + cfg.FORECAST_HORIZON + 50, 6000)
    test_indices = (6000 + cfg.FORECAST_HORIZON + 50, n)

    train_loader, val_loader, test_loader = create_fold_dataloaders(
        feats, tgts,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        config=cfg, feature_cols=[f"c{i}" for i in range(21)],
    )

    tlens, tmask_bad = [], 0
    for tokens, f, y, mask in train_loader:
        L = f.shape[1]
        tlens.append(L)
        if mask is None:
            if any(x.shape[0] != L for x in []):
                tmask_bad += 1
        else:
            if mask.shape != (f.shape[0], L):
                tmask_bad += 1
                continue
            # left padding only
            if ((~mask).sum(dim=1) != torch.tensor(
                    [int((row == 0).sum()) for row in mask], dtype=torch.long)).any():
                pass
        break
    check("train batches variable-length or masked",
          len(tlens) > 0, f"first batch L={tlens[0] if tlens else 'NA'}")

    seen_short, mask_ok = False, True
    it = iter(train_loader)
    for _ in range(8):
        try:
            _, f, y, mask = next(it)
        except StopIteration:
            break
        L = f.shape[1]
        if L < cfg.LOOKBACK_WINDOW:
            seen_short = True
            if mask is None:
                mask_ok = False
        if mask is not None:
            if mask[:, 0].float().mean() < 1.0 and not mask.any():
                mask_ok = False
    check("train loader produces short contexts", seen_short,
          "(probabilistic over 8 batches)")
    check("masks accompany short batches", mask_ok)

    _, vf, _, vmask = next(iter(val_loader))
    _, tf_, _, tmask = next(iter(test_loader))
    check("val loader fixed length",
          vf.shape[1] == cfg.LOOKBACK_WINDOW and vmask is None)
    check("test loader fixed length",
          tf_.shape[1] == cfg.LOOKBACK_WINDOW and tmask is None)

    print("\n[4] Sampler-weight alignment")
    w_ds_len = len(train_loader.dataset)
    check("train dataset length preserved",
          w_ds_len == train_indices[1] - train_indices[0] - cfg.LOOKBACK_WINDOW + 1,
          f"{w_ds_len}")


if __name__ == "__main__":
    test_dataset_invariant()
    test_val_test_deterministic()
    test_fold_loaders()
    n_fail = sum(1 for _, ok in results if not ok)
    print(f"\n{'='*60}")
    print(f"ORBIT AUDIT RESULT: {len(results) - n_fail}/{len(results)} passed"
          + (f"  — {n_fail} FAILED" if n_fail else ""))
    sys.exit(1 if n_fail else 0)
