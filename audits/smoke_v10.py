#!/usr/bin/env python3
"""Smoke tests for the v10 quantile-head upgrade (Phases 1, 4, 5).

Run from repo root:  python3 audits/smoke_v10.py
Covers:
  1. QuantileHead monotonicity + median identity + shape
  2. Pinball gradient bound |dL/dŷ| <= max(q, 1-q)
  3. Full PatchTST forward/backward under all flag states
     (USE_MODERN_NORM x USE_ROPE), including masked variable-length input
  4. collate_with_none padding/mask correctness
  5. Warm-start compatibility: legacy (scalar-head) checkpoint loads
     non-strictly into the quantile architecture, transferring everything
     except head weights
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))

import torch
import torch.nn.functional as F

import config as cfg
from model import PatchTST, QuantileHead, RotarySelfAttention
from loss import pinball_loss, v10_total_loss, median_index
from data_loader import collate_with_none
from train import _load_compatible

PASS, FAIL = "✓", "✗"
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"  {PASS if cond else FAIL} {name}" + (f"  ({detail})" if detail else ""))


def test_head_monotone():
    print("\n[1] QuantileHead monotonicity")
    torch.manual_seed(0)
    levels = list(cfg.QUANTILE_LEVELS)
    head = QuantileHead(d_model=64, levels=levels, dropout=0.0, pool="mixing")
    dec_out = torch.randn(8, 4, 64)
    q = head(dec_out)
    check("output shape (B,Q)", tuple(q.shape) == (8, len(levels)), f"{tuple(q.shape)}")
    diffs = q[:, 1:] - q[:, :-1]
    check("strictly ascending", bool((diffs > 0).all()),
          f"min diff={diffs.min().item():.6f}")
    med_col = q[:, head.m]
    pooled = dec_out[:, 0] if head.pool == "cls" else dec_out.mean(1)
    raw_med = head.net(pooled).float()[:, head.m]
    check("median column is raw anchor", bool(torch.equal(med_col, raw_med)))
    big = head(dec_out * 50.0)
    check("unbounded tails (no tanh)", bool(big.abs().max() > SAT_BOUND),
          f"max|q|={big.abs().max().item():.2f}")


SAT_BOUND = 2.5


def test_pinball_grad():
    print("\n[2] Pinball gradient bound")
    for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
        pred = torch.tensor([0.3], requires_grad=True)
        tgt = torch.tensor([0.1])
        levels = [q]
        saved = cfg.QUANTILE_LEVELS
        cfg.QUANTILE_LEVELS = levels
        try:
            loss = pinball_loss(pred.unsqueeze(0), tgt)
            loss.backward()
            g = pred.grad.abs().item()
            bound = max(q, 1 - q)
            check(f"q={q}: |grad| {g:.3f} <= {bound:.3f}", g <= bound + 1e-6)
        finally:
            cfg.QUANTILE_LEVELS = saved


def _build_and_run(use_norm: bool, use_rope: bool):
    cfg.USE_MODERN_NORM = use_norm
    cfg.USE_ROPE = use_rope
    cfg.INPUT_MODE = "features_only"
    torch.manual_seed(1)
    net = PatchTST(
        seq_len=cfg.LOOKBACK_WINDOW, num_features=21,
        patch_len=16, stride=12, d_model=64, n_heads=8,
        n_layers=2, n_dec_layers=1, n_queries=4,
        lstm_layers=1, dropout=0.1, aggregation="mixing",
        input_mode="features_only",
    )
    B, L, Fdim = 4, cfg.LOOKBACK_WINDOW, 21
    feats_full = torch.randn(B, L, Fdim)
    tgts = torch.zeros(B)
    tgts[0] = 0.4
    tgts[1] = -0.3

    out = net(features=feats_full)
    ok_shape = out.shape[-1] == len(cfg.QUANTILE_LEVELS) and out.shape[0] == B
    loss = v10_total_loss(out, tgts)
    loss.backward()
    grads_finite = all(
        p.grad is None or torch.isfinite(p.grad).all()
        for p in net.parameters()
    )

    Lshort = 300
    feats_var = torch.randn(B, Lshort, Fdim)
    pad = torch.zeros(B, Lshort, dtype=torch.bool)
    pad[1, :100] = True
    pad[3, :250] = True
    out_m = net(features=feats_var, key_padding_mask=pad)
    loss_m = v10_total_loss(out_m, tgts)
    loss_m.backward()
    masked_ok = torch.isfinite(out_m).all().item()

    check(f"[norm={'RMS' if use_norm else 'LN'}, rope={use_rope}] "
          f"shape+loss+bwd", ok_shape and grads_finite and torch.isfinite(loss))
    check(f"[norm={'RMS' if use_norm else 'LN'}, rope={use_rope}] "
          f"masked var-len forward", masked_ok)
    return net


def test_flag_states():
    print("\n[3] Flag-state matrix (forward/backward, incl. masked)")
    saved = (cfg.USE_MODERN_NORM, cfg.USE_ROPE, cfg.QUANTILE_HEAD)
    try:
        for norm in (False, True):
            for rope in (False, True):
                _build_and_run(norm, rope)
        rot = RotarySelfAttention(64, 8, max_len=512)
        x = torch.randn(2, 128, 64)
        o, _ = rot(x, x, x)
        check("RotarySelfAttention standalone", o.shape == x.shape and torch.isfinite(o).all())
    finally:
        cfg.USE_MODERN_NORM, cfg.USE_ROPE, cfg.QUANTILE_HEAD = saved


def test_collate():
    print("\n[4] collate_with_none padding")
    batch = [
        (None, torch.randn(200, 21), torch.tensor(0.1)),
        (None, torch.randn(120, 21), torch.tensor(0.0)),
        (None, torch.randn(200, 21), torch.tensor(-0.2)),
    ]
    tokens, feats, tgts, mask = collate_with_none(batch)
    check("padded to batch max", feats.shape == (3, 200, 21), f"{tuple(feats.shape)}")
    check("mask shape/True=padded", mask.shape == (3, 200)
          and mask[1, :80].all() and not mask[1, 80:].any()
          and not mask[0].any())
    check("targets intact", torch.allclose(tgts, torch.tensor([0.1, 0.0, -0.2])))

    batch_fixed = [
        (None, torch.randn(200, 21), torch.tensor(0.1)),
        (None, torch.randn(200, 21), torch.tensor(0.0)),
    ]
    _, _, _, mask_fixed = collate_with_none(batch_fixed)
    check("fixed-length -> mask None", mask_fixed is None)


def test_warm_start_compat():
    print("\n[5] Legacy checkpoint warm-start (strict=False)")
    cfg.USE_MODERN_NORM = False
    cfg.USE_ROPE = False
    torch.manual_seed(2)
    legacy_kwargs = dict(
        seq_len=256, num_features=21, patch_len=16, stride=12,
        d_model=64, n_heads=8, n_layers=2, n_dec_layers=1,
        n_queries=4, lstm_layers=1, dropout=0.1,
        aggregation="mixing", input_mode="features_only",
    )
    saved_qh = cfg.QUANTILE_HEAD
    try:
        cfg.QUANTILE_HEAD = False
        legacy = PatchTST(**legacy_kwargs)
        sd = {k: v.clone() for k, v in legacy.state_dict().items()}

        cfg.QUANTILE_HEAD = True
        new_net = PatchTST(**legacy_kwargs)
        missing, skipped = _load_compatible(new_net, sd)

        transferred = len(sd) - len(skipped)
        only_head_missing = all(k.startswith(("feature_head", "head")) for k in missing)
        only_head_skipped = all(k.startswith(("feature_head", "head")) for k in skipped)
        check("only shape-mismatched head tensors skipped",
              only_head_skipped and len(skipped) == 2,
              f"skipped={sorted(skipped)}")
        check("only head keys missing", only_head_missing,
              f"missing={sorted(missing)}")
        check(f"transferred {transferred}/{len(sd)} tensors",
              transferred >= int(0.9 * len(sd)))
        check("transferred weights actually applied",
              torch.equal(new_net.input_stem.feature_proj.weight,
                          legacy.input_stem.feature_proj.weight))
    finally:
        cfg.QUANTILE_HEAD = saved_qh


if __name__ == "__main__":
    test_head_monotone()
    test_pinball_grad()
    test_flag_states()
    test_collate()
    test_warm_start_compat()
    n_fail = sum(1 for _, ok in results if not ok)
    print(f"\n{'='*60}")
    print(f"SMOKE RESULT: {len(results) - n_fail}/{len(results)} passed"
          + (f"  — {n_fail} FAILED" if n_fail else ""))
    sys.exit(1 if n_fail else 0)
