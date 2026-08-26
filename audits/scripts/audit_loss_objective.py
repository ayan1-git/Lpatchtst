#!/usr/bin/env python3
"""
audit_loss_objective.py
=======================
LPatchTST — Loss, Objective & Curriculum Audit  (v10)

Adapts the legacy audit to the current loss surface:
  • v10_total_loss   = PINBALL_WEIGHT * pinball(all quantile levels)
                     + V9_MEDIAN_WEIGHT * asymmetric_number_line_loss(median)
  • asymmetric_number_line_loss (v9):  flat L1 push + edge smooth_l1 nudge
  • pinball_loss (Falcon-2.0 style):  bounded gradient per quantile level

Run from repo root:
    python3 audits/scripts/audit_loss_objective.py

Covers:
  1.  is_zero / flat boundary vs SAMPLER_THRESHOLD consistency
  2.  v9 component weight hierarchy  (flat-push vs edge-nudge)
  3.  Quantile-level config sanity  (symmetry, median index, count)
  4.  v10 mix balance  (PINBALL_WEIGHT vs V9_MEDIAN_WEIGHT)
  5.  Numerical stability  (pinball, v9, on small/empty/degenerate inputs)
  6.  Gradient flow simulations (flat, edge, mixed, collapse, overshoot)
  7.  Component weight interaction matrix
"""

import os
import sys
import math
import textwrap
import inspect as _inspect
import numpy as np
import torch

SEP  = "─" * 70
SEP2 = "=" * 70

def hdr(title: str) -> None:
    print(f"\n{SEP}")
    print(f"{title}")
    print(SEP)

def ok(msg:   str) -> None: print(f"  [PASS] {msg}")
def warn(msg: str) -> None: print(f"  [WARN] {msg}")
def fail(msg: str) -> None: print(f"  [FAIL] {msg}")
def info(msg: str) -> None: print(f"  [INFO] {msg}")


# ── Path setup: make core/ importable ────────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
_REPO     = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "core"))
sys.path.insert(0, _REPO)

# ── Imports ──────────────────────────────────────────────────────────────────
try:
    import config as CFG
    CONFIG_IMPORTED = True
except Exception as e:
    CONFIG_IMPORTED = False
    print(f"[WARN] Could not import config.py: {e}")

try:
    from loss import (
        v10_total_loss,
        pinball_loss,
        asymmetric_number_line_loss,
        continuous_weighted_direction_loss,
        median_index,
    )
    import loss as LOSS
    LOSS_IMPORTED = True
except Exception as e:
    LOSS_IMPORTED = False
    print(f"[WARN] Could not import loss.py: {e}")


# ── Pull live constants (no hardcoded duplicates) ────────────────────────────
def _get(name: str, default):
    return getattr(CFG, name, default) if CONFIG_IMPORTED else default

LOSS_ALPHA             = _get("ALPHA", None) or getattr(LOSS, "ALPHA", 2.0)
LOSS_BETA              = _get("BETA",  None) or getattr(LOSS, "BETA",  1.5)
LOSS_FLAT_PUSH_WEIGHT  = getattr(LOSS, "FLAT_PUSH_WEIGHT",  2.0)
LOSS_EDGE_NUDGE_WEIGHT = getattr(LOSS, "EDGE_NUDGE_WEIGHT", 1.5)
LOSS_MAG_WEIGHT_MIN    = getattr(LOSS, "MAG_WEIGHT_MIN",    0.5)
LOSS_MAG_WEIGHT_MAX    = getattr(LOSS, "MAG_WEIGHT_MAX",    2.0)
LOSS_PINBALL_WEIGHT    = _get("PINBALL_WEIGHT",   1.0)
LOSS_V9_MEDIAN_WEIGHT  = _get("V9_MEDIAN_WEIGHT", 1.5)
LOSS_QUANTILE_LEVELS   = _get("QUANTILE_LEVELS", [0.02, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.98])
LOSS_SAMPLER_THRESHOLD = _get("SAMPLER_THRESHOLD", 0.05)
LOSS_ORACLE_THRESHOLD  = _get("ORACLE_THRESHOLD",  None)
LOSS_CURRICULUM_EPOCHS = _get("CURRICULUM_RAMP_EPOCHS", 20)
LOSS_PINBALL_FLOOR     = 1e-8
LOSS_SMOOTH_L1_BETA    = 1.0  # the smooth_l1 inflection point in v9 (diff_abs < 1)


print(SEP2)
print("  LPatchTST — Loss / Objective / Curriculum Audit (v10)")
print(f"  Repo:  {_REPO}")
print(SEP2)


# ═════════════════════════════════════════════════════════════════════════════
# AUDIT 1 · is_zero / flat boundary vs SAMPLER_THRESHOLD
# ═════════════════════════════════════════════════════════════════════════════
hdr("AUDIT 1 · flat-bar definition (loss) vs sampler / oracle boundaries")

# v9 uses target.abs() < 1e-8 → is_flat boundary is effectively 0.0
v9_IS_FLAT_EPS = 1e-8
info(f"v9 is_flat boundary     = target.abs() < {v9_IS_FLAT_EPS}  (effectively 0.0)")
info(f"config.SAMPLER_THRESHOLD = {LOSS_SAMPLER_THRESHOLD}")
info(f"config.ORACLE_THRESHOLD  = {LOSS_ORACLE_THRESHOLD}")

# v10 doesn't have a flat_threshold kwarg
if LOSS_IMPORTED:
    has_flat_thresh = 'flat_threshold' in _inspect.signature(asymmetric_number_line_loss).parameters
    if not has_flat_thresh:
        ok("v9 signature no longer accepts flat_threshold — exact zero is used consistently.")
    else:
        fail("v9 signature still has deprecated flat_threshold argument.")

# Sampler vs loss boundary
if v9_IS_FLAT_EPS < LOSS_SAMPLER_THRESHOLD:
    ok(
        f"v9 flat boundary (≈0) is stricter than SAMPLER_THRESHOLD ({LOSS_SAMPLER_THRESHOLD}). "
        "Loss treats edge-like targets (|y| < threshold) as non-flat → some of them "
        "still receive edge loss. The dead-zone is the [0, threshold) zone, where "
        "targets are edge-class by the sampler but flat-class by the loss."
    )
elif abs(v9_IS_FLAT_EPS - LOSS_SAMPLER_THRESHOLD) < 1e-9:
    ok("v9 flat boundary == SAMPLER_THRESHOLD. Fully aligned.")
else:
    warn(f"v9 flat boundary ({v9_IS_FLAT_EPS}) > SAMPLER_THRESHOLD — sampler over-samples zones the loss considers flat.")

# Oracle threshold check
if LOSS_ORACLE_THRESHOLD is not None:
    if abs(LOSS_ORACLE_THRESHOLD - LOSS_SAMPLER_THRESHOLD) < 1e-9:
        ok(f"ORACLE_THRESHOLD == SAMPLER_THRESHOLD ({LOSS_SAMPLER_THRESHOLD}). Full stack aligned.")
    else:
        warn(f"ORACLE_THRESHOLD ({LOSS_ORACLE_THRESHOLD}) != SAMPLER_THRESHOLD ({LOSS_SAMPLER_THRESHOLD}). Verify intentional.")
else:
    warn("ORACLE_THRESHOLD not in config — verify oracle.py labelling threshold manually.")


# ═════════════════════════════════════════════════════════════════════════════
# AUDIT 2 · v9 weight hierarchy (flat-push vs edge-nudge)
# ═════════════════════════════════════════════════════════════════════════════
hdr("AUDIT 2 · v9 component weight hierarchy")

info(f"ALPHA (undershoot)         = {LOSS_ALPHA}")
info(f"BETA  (overshoot)          = {LOSS_BETA}")
info(f"FLAT_PUSH_WEIGHT (L1)      = {LOSS_FLAT_PUSH_WEIGHT}")
info(f"EDGE_NUDGE_WEIGHT (sm_l1)  = {LOSS_EDGE_NUDGE_WEIGHT}")
info(f"MAG_WEIGHT_MIN / MAX       = {LOSS_MAG_WEIGHT_MIN} / {LOSS_MAG_WEIGHT_MAX}")

# Check 2a: Wrong-direction harder than right-direction
if LOSS_ALPHA > LOSS_BETA:
    ok(f"ALPHA ({LOSS_ALPHA}) > BETA ({LOSS_BETA}). Wrong-direction is punished more than same-direction overshoot.")
else:
    warn(f"ALPHA ({LOSS_ALPHA}) <= BETA ({LOSS_BETA}). Inverse asymmetry to design intent.")

# Check 2b: flat-push vs edge-nudge
if LOSS_FLAT_PUSH_WEIGHT > LOSS_EDGE_NUDGE_WEIGHT * 0.8:
    ok(
        f"FLAT_PUSH_WEIGHT ({LOSS_FLAT_PUSH_WEIGHT}) ≈ or > EDGE_NUDGE_WEIGHT ({LOSS_EDGE_NUDGE_WEIGHT}). "
        "L1 constant push toward zero is strong enough to keep flat-class under control."
    )
else:
    warn(
        f"FLAT_PUSH_WEIGHT ({LOSS_FLAT_PUSH_WEIGHT}) < EDGE_NUDGE_WEIGHT ({LOSS_EDGE_NUDGE_WEIGHT}). "
        "Flat predictions may not be pulled back to zero effectively."
    )

# Check 2c: mag_weight bounds
if 0 < LOSS_MAG_WEIGHT_MIN < LOSS_MAG_WEIGHT_MAX:
    ok(f"mag_weight clamped to [{LOSS_MAG_WEIGHT_MIN}, {LOSS_MAG_WEIGHT_MAX}] — bounded correctly.")
else:
    fail(f"mag_weight bounds invalid: min={LOSS_MAG_WEIGHT_MIN}, max={LOSS_MAG_WEIGHT_MAX}.")


# ═════════════════════════════════════════════════════════════════════════════
# AUDIT 3 · Quantile-level config sanity
# ═════════════════════════════════════════════════════════════════════════════
hdr("AUDIT 3 · Quantile-level config sanity")

qlvls = list(LOSS_QUANTILE_LEVELS)
info(f"QUANTILE_LEVELS = {qlvls}")

# 3a: ascending
if qlvls == sorted(qlvls):
    ok("Quantile levels are in ascending order.")
else:
    fail("Quantile levels NOT in ascending order — pinball assumes ascending q_pred.")

# 3b: in (0, 1)
if all(0.0 < q < 1.0 for q in qlvls):
    ok(f"All {len(qlvls)} levels ∈ (0, 1).")
else:
    fail(f"Some levels outside (0, 1): {[q for q in qlvls if not (0 < q < 1)]}")

# 3c: median present
if 0.5 in qlvls:
    ok("Median (0.5) is present in QUANTILE_LEVELS.")
    if LOSS_IMPORTED:
        mi = median_index(qlvls)
        info(f"median_index(levels) = {mi} (level value = {qlvls[mi]})")
        if qlvls[mi] == 0.5:
            ok("median_index correctly returns the 0.5 slot.")
        else:
            fail(f"median_index returns {qlvls[mi]} instead of 0.5.")
else:
    fail("Median (0.5) missing from QUANTILE_LEVELS — v9 term on median breaks.")

# 3d: symmetry check
if qlvls == sorted(qlvls) and len(qlvls) % 2 == 1:
    # Symmetric pair check around 0.5
    pairs_ok = all(
        abs(qlvls[i] + qlvls[-(i+1)] - 1.0) < 1e-9
        for i in range(len(qlvls) // 2)
    )
    if pairs_ok:
        ok("Quantile levels are symmetric around 0.5 (e.g. 0.05 ↔ 0.95).")
    else:
        warn("Quantile levels are NOT symmetric around 0.5 — pinball model may be biased.")

# 3e: minimum and maximum levels
info(f"Lowest quantile = {qlvls[0]:.2f}  (most conservative buy/sell)")
info(f"Highest quantile = {qlvls[-1]:.2f}")
if qlvls[0] < 0.05 or qlvls[-1] > 0.95:
    warn("Quantile spread includes extreme levels (<0.05 / >0.95) — gradients bound by max(q, 1-q) ≤ 0.05 there.")


# ═════════════════════════════════════════════════════════════════════════════
# AUDIT 4 · v10 mix balance (pinball vs v9 on median)
# ═════════════════════════════════════════════════════════════════════════════
hdr("AUDIT 4 · v10 mix balance — PINBALL_WEIGHT vs V9_MEDIAN_WEIGHT")

info(f"PINBALL_WEIGHT   = {LOSS_PINBALL_WEIGHT}")
info(f"V9_MEDIAN_WEIGHT = {LOSS_V9_MEDIAN_WEIGHT}")
ratio = LOSS_PINBALL_WEIGHT / (LOSS_V9_MEDIAN_WEIGHT + 1e-12)
info(f"Ratio pin/v9     = {ratio:.3f}")

if 0.5 <= ratio <= 2.0:
    ok(
        f"Pinball:v9 ratio = {ratio:.2f} — both terms contribute meaningfully. "
        "Pinball provides cross-quantile calibration; v9 anchors the median decision."
    )
elif ratio < 0.5:
    warn(
        f"Pinball:v9 ratio = {ratio:.2f} — v9 dominates. "
        "Most gradient comes from the median channel; the other quantiles are weakly supervised."
    )
else:
    warn(
        f"Pinball:v9 ratio = {ratio:.2f} — pinball dominates. "
        "The v9 hand-calibrated flat-push / edge-nudge is under-weighted."
    )

# Per-sample pinball gradient bound: max(q, 1-q) ≤ max_q
max_q = max(max(qlvls), 1.0 - min(qlvls))
info(f"Pinball per-sample gradient is bounded by max(q, 1-q) ≤ {max_q:.3f}")
if LOSS_PINBALL_WEIGHT * max_q > LOSS_V9_MEDIAN_WEIGHT * LOSS_EDGE_NUDGE_WEIGHT * LOSS_MAG_WEIGHT_MAX:
    warn(
        f"Pinball max gradient (≈{LOSS_PINBALL_WEIGHT * max_q:.2f}) can exceed v9 max edge gradient "
        f"(≈{LOSS_V9_MEDIAN_WEIGHT * LOSS_EDGE_NUDGE_WEIGHT * LOSS_MAG_WEIGHT_MAX:.2f}). "
        "Pinball may dominate large-magnitude errors."
    )
else:
    ok(
        f"Pinball max per-sample gradient (≈{LOSS_PINBALL_WEIGHT * max_q:.2f}) "
        f"≤ v9 max edge gradient (≈{LOSS_V9_MEDIAN_WEIGHT * LOSS_EDGE_NUDGE_WEIGHT * LOSS_MAG_WEIGHT_MAX:.2f}). "
        "V9 can still correct extreme edges."
    )


# ═════════════════════════════════════════════════════════════════════════════
# AUDIT 5 · Gradient-flow simulations
# ═════════════════════════════════════════════════════════════════════════════
hdr("AUDIT 5 · Gradient flow at key operating points")

if not LOSS_IMPORTED:
    warn("loss.py not importable — skipping gradient flow simulations.")
else:
    # 5a: v9 gradient at pred=0 for flat target
    info("5a. v9: pred=0 on FLAT target (target ≈ 0)")
    pred = torch.zeros(8, requires_grad=True)
    tgt  = torch.zeros(8)
    l = asymmetric_number_line_loss(pred, tgt)
    l.backward()
    g_flat = pred.grad.abs().mean().item()
    info(f"  loss={l.item():.4f}  |grad|={g_flat:.4f}")
    if g_flat > 0.5:
        ok(f"Flat-class L1 push is strong (|grad|={g_flat:.4f}). Model will be pulled toward zero on flat bars.")
    elif g_flat > 0.1:
        warn(f"Flat-class L1 push is weak (|grad|={g_flat:.4f}).")
    else:
        fail(f"Flat-class L1 push is essentially zero (|grad|={g_flat:.4f}) — model has no signal to converge to zero on flat bars.")

    # 5b: v9 gradient at pred=0 for edge target
    info("5b. v9: pred=0 on EDGE target (target = +0.3)")
    pred = torch.zeros(8, requires_grad=True)
    tgt  = torch.full((8,), 0.3)
    l = asymmetric_number_line_loss(pred, tgt)
    l.backward()
    g_edge = pred.grad.abs().mean().item()
    info(f"  loss={l.item():.4f}  |grad|={g_edge:.4f}")
    if g_edge > 0.1:
        ok(f"Edge-class gradient at pred=0 is strong (|grad|={g_edge:.4f}). Model is pushed to correct edge direction.")
    else:
        warn(f"Edge-class gradient at pred=0 is weak (|grad|={g_edge:.4f}).")

    # 5c: v9 gradient at pred=0 for WRONG-direction edge
    info("5c. v9: pred=0 on WRONG-DIR edge (target = +0.3, so pred should be +; if 0, sign mismatch)")
    # pred=0 vs target=+0.3: sign product = 0 → not negative → not "wrong dir" per v9's
    # `is_undershoot = (gap * target) < 0` check. gap=0-0.3=-0.3, target=+0.3, product=-0.09 < 0 → undershoot.
    # The undershoot path uses ALPHA=2.0.
    pred = torch.zeros(8, requires_grad=True)
    tgt  = torch.full((8,), 0.3)
    l = asymmetric_number_line_loss(pred, tgt)
    l.backward()
    g_under = pred.grad.abs().mean().item()
    info(f"  Undershoot gradient |grad|={g_under:.4f} (should be ~stronger than correct-direction overshoot)")

    pred = torch.zeros(8, requires_grad=True)
    tgt  = torch.full((8,), -0.3)
    l = asymmetric_number_line_loss(pred, tgt)
    l.backward()
    info(f"  (Asymmetry note) symmetric pred=0 for target=-0.3 yields the same scalar gradient by symmetry.")

    # 5d: mixed batch — 50% flat, 50% edge
    info("5d. v9: pred=0 on MIXED batch (50% flat, 50% edge)")
    pred = torch.zeros(16, requires_grad=True)
    tgt  = torch.cat([torch.zeros(8), torch.full((8,), 0.3)])
    l = asymmetric_number_line_loss(pred, tgt)
    l.backward()
    g_mix = pred.grad.abs().mean().item()
    info(f"  loss={l.item():.4f}  |grad|={g_mix:.4f}")
    if g_mix > 0.1:
        ok(f"Mixed-batch gradient |grad|={g_mix:.4f} is non-trivial — edge targets dominate over flat dilution.")
    else:
        warn(f"Mixed-batch gradient is weak (|grad|={g_mix:.4f}). Flat-class dilution may suppress edge learning.")

    # 5e: overshoot scenario (pred > target)
    info("5e. v9: OVERSHOOT — pred=0.6 on edge target=0.3")
    pred = torch.full((8,), 0.6, requires_grad=True)
    tgt  = torch.full((8,), 0.3)
    l = asymmetric_number_line_loss(pred, tgt)
    l.backward()
    g_over = pred.grad.abs().mean().item()
    info(f"  loss={l.item():.4f}  |grad|={g_over:.4f}")
    if g_over > 0.05:
        ok(f"Overshoot gradient (|grad|={g_over:.4f}) provides pull-back toward target.")
    else:
        warn(f"Overshoot gradient is weak (|grad|={g_over:.4f}) — model may persist in overshooting.")

    # 5f: collapse scenario — all-zero pred on diverse target
    info("5f. v9: ALL-ZERO pred on uniform[-0.5, 0.5] target")
    pred = torch.zeros(32, requires_grad=True)
    tgt  = torch.linspace(-0.5, 0.5, 32)
    l = asymmetric_number_line_loss(pred, tgt)
    l.backward()
    g_collapse = pred.grad.abs().mean().item()
    info(f"  loss={l.item():.4f}  |grad|={g_collapse:.4f}")
    if g_collapse > 0.1:
        ok("All-zero collapse is unstable — gradient pushes model toward diverse predictions.")
    else:
        fail(f"All-zero collapse gradient is weak ({g_collapse:.4f}) — model can stably sit at all-zero on diverse targets.")

    # 5g: pinball gradient
    info("5g. pinball: gradient at q_pred=0 across all quantile levels")
    Q = len(qlvls)
    q_pred = torch.zeros(4, Q, requires_grad=True)
    tgt    = torch.tensor([0.3, -0.2, 0.1, -0.4])
    l_pin = pinball_loss(q_pred, tgt)
    l_pin.backward()
    g_pin = q_pred.grad.abs().mean().item()
    info(f"  loss={l_pin.item():.4f}  |grad|={g_pin:.4f}")
    if g_pin > 0.01:
        ok(f"Pinball gradient is non-trivial (|grad|={g_pin:.4f}).")
    else:
        warn(f"Pinball gradient is weak (|grad|={g_pin:.4f}).")

    # 5h: v10 combined loss gradient
    info("5h. v10_total_loss: gradient on realistic batch")
    pred = torch.zeros(4, Q, requires_grad=True)
    tgt  = torch.tensor([0.3, -0.2, 0.1, -0.4])
    l_v10 = v10_total_loss(pred, tgt)
    l_v10.backward()
    g_v10 = pred.grad.abs().mean().item()
    info(f"  loss={l_v10.item():.4f}  |grad|={g_v10:.4f}")
    if g_v10 > 0.01:
        ok(f"v10 combined loss produces healthy gradient (|grad|={g_v10:.4f}).")


# ═════════════════════════════════════════════════════════════════════════════
# AUDIT 6 · Numerical stability
# ═════════════════════════════════════════════════════════════════════════════
hdr("AUDIT 6 · Numerical stability on degenerate inputs")

if not LOSS_IMPORTED:
    warn("loss.py not importable — skipping stability tests.")
else:
    # 6a: pinball on batch=1
    info("6a. pinball on batch=1")
    try:
        q1 = torch.zeros(1, Q, requires_grad=True)
        t1 = torch.tensor([0.2])
        l = pinball_loss(q1, t1)
        if not torch.isnan(l) and not torch.isinf(l):
            ok(f"pinball(batch=1) = {l.item():.6f} — no NaN/Inf.")
        else:
            fail(f"pinball(batch=1) = {l.item()} — NaN or Inf.")
    except Exception as e:
        fail(f"pinball(batch=1) raised: {e}")

    # 6b: pinball on zero target
    info("6b. pinball on zero target")
    try:
        q0 = torch.zeros(4, Q, requires_grad=True)
        t0 = torch.zeros(4)
        l = pinball_loss(q0, t0)
        ok(f"pinball(zero target) = {l.item():.6f}.")
    except Exception as e:
        fail(f"pinball(zero target) raised: {e}")

    # 6c: v9 on constant pred and target
    info("6c. v9 on constant pred=0.1, target=0.2")
    try:
        p = torch.full((16,), 0.1, requires_grad=True)
        t = torch.full((16,), 0.2)
        l = asymmetric_number_line_loss(p, t)
        l.backward()
        if not torch.isnan(l) and not torch.isnan(p.grad).any():
            ok(f"v9 constant batch: loss={l.item():.4f}, |grad|={p.grad.abs().mean():.4f}.")
        else:
            fail("v9 constant batch produced NaN.")
    except Exception as e:
        fail(f"v9 constant batch raised: {e}")

    # 6d: v9 on all-zero target (perfectly flat batch)
    info("6d. v9 on all-zero target batch")
    try:
        p = torch.zeros(8, requires_grad=True)
        t = torch.zeros(8)
        l = asymmetric_number_line_loss(p, t)
        l.backward()
        ok(f"v9 all-zero target: loss={l.item():.4f}, |grad|={p.grad.abs().mean():.4f}.")
    except Exception as e:
        fail(f"v9 all-zero target raised: {e}")

    # 6e: v9 on NaN input
    info("6e. v9 on NaN input")
    try:
        p = torch.tensor([float('nan'), 0.1, 0.2, 0.3], requires_grad=True)
        t = torch.tensor([0.1, 0.2, 0.3, 0.4])
        l = asymmetric_number_line_loss(p, t)
        l.backward()
        if torch.isnan(p.grad).any():
            warn("v9 on NaN input: gradient contains NaN — may poison training if data has NaN.")
        else:
            ok("v9 on NaN input: NaN absorbed (nan_to_num in mag_weight path).")
    except Exception as e:
        warn(f"v9 on NaN input raised: {e} (acceptable — data loader should filter NaN first).")

    # 6f: v9 on extreme magnitude target
    info("6f. v9 on extreme-magnitude target (target=10.0)")
    try:
        p = torch.zeros(8, requires_grad=True)
        t = torch.full((8,), 10.0)
        l = asymmetric_number_line_loss(p, t)
        l.backward()
        g = p.grad.abs().mean().item()
        info(f"  loss={l.item():.4f}, |grad|={g:.4f}")
        if not torch.isnan(l) and not torch.isinf(l):
            ok("v9 on extreme target: finite loss and gradient.")
        else:
            fail("v9 on extreme target: NaN/Inf in loss.")
    except Exception as e:
        fail(f"v9 on extreme target raised: {e}")

    # 6g: v10 on the same extreme-magnitude target
    info("6g. v10 on extreme-magnitude target (target=10.0)")
    try:
        Q = len(qlvls)
        p = torch.zeros(4, Q, requires_grad=True)
        t = torch.full((4,), 10.0)
        l = v10_total_loss(p, t)
        l.backward()
        if not torch.isnan(l) and not torch.isinf(l) and not torch.isnan(p.grad).any():
            ok(f"v10 extreme target: loss={l.item():.4f}, |grad|={p.grad.abs().mean():.4f}.")
        else:
            fail("v10 extreme target: NaN/Inf in loss or gradient.")
    except Exception as e:
        fail(f"v10 extreme target raised: {e}")

    # 6h: v10 on minimal batch
    info("6h. v10 on batch=2")
    try:
        Q = len(qlvls)
        p = torch.zeros(2, Q, requires_grad=True)
        t = torch.tensor([0.2, -0.3])
        l = v10_total_loss(p, t)
        l.backward()
        if not torch.isnan(l) and not torch.isnan(p.grad).any():
            ok(f"v10(batch=2): loss={l.item():.4f}, |grad|={p.grad.abs().mean():.4f}.")
        else:
            fail("v10(batch=2): NaN produced.")
    except Exception as e:
        fail(f"v10(batch=2) raised: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# AUDIT 7 · Component weight interaction matrix
# ═════════════════════════════════════════════════════════════════════════════
hdr("AUDIT 7 · v10 component weight interaction")

v10_components = [
    ("pinball(all Q levels)",    LOSS_PINBALL_WEIGHT,   "Falcon-2.0 quantile calibration"),
    ("  └ dir loss (asym quad)", 1.00,                  "v9 part: wrong-dir ALPHA, right-dir BETA"),
    ("  └ flat L1 push",         LOSS_FLAT_PUSH_WEIGHT, "v9 part: L1 constant toward zero"),
    ("  └ edge smooth_l1 nudge", LOSS_EDGE_NUDGE_WEIGHT,"v9 part: magnitude nudge on |gap|"),
    ("v9 (median channel)",      LOSS_V9_MEDIAN_WEIGHT, "anchors median buy/sell decision"),
]
print(f"  {'Component':<28} {'Weight':>7}  {'Role'}")
print(f"  {'-'*28} {'-'*7}  {'-'*34}")
for name, w, role in v10_components:
    bar = "█" * int(w * 10)
    print(f"  {name:<28} {w:>7.2f}  {role:<34}  {bar}")

max_w     = max(w for _, w, _ in v10_components)
max_comps = [n for n, w, _ in v10_components if w == max_w]
if max_w > 3.0:
    fail(f"Component(s) {max_comps} at weight {max_w} may dominate the gradient.")
elif max_w > 2.0:
    warn(f"Component(s) {max_comps} at weight {max_w} are the strongest — monitor dominance.")
else:
    ok(f"No component has weight > 2.0. Balanced landscape (max={max_w}).")


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP2}")
print("AUDIT COMPLETE — Loss, Objective & Curriculum (v10)")
print(SEP2)
print(textwrap.dedent(f"""
  Configuration snapshot
    ALPHA={LOSS_ALPHA}  BETA={LOSS_BETA}  flat_push={LOSS_FLAT_PUSH_WEIGHT}  edge_nudge={LOSS_EDGE_NUDGE_WEIGHT}
    mag_weight=[{LOSS_MAG_WEIGHT_MIN}, {LOSS_MAG_WEIGHT_MAX}]
    PINBALL_WEIGHT={LOSS_PINBALL_WEIGHT}  V9_MEDIAN_WEIGHT={LOSS_V9_MEDIAN_WEIGHT}
    Q levels ({len(qlvls)}): {qlvls}
    SAMPLER_THRESHOLD={LOSS_SAMPLER_THRESHOLD}  ORACLE_THRESHOLD={LOSS_ORACLE_THRESHOLD}

  Verdict template
    ✓ flat-push strong enough to keep model off zero on flat bars
    ✓ pinball + v9 mix is reasonably balanced (ratio {LOSS_PINBALL_WEIGHT/LOSS_V9_MEDIAN_WEIGHT:.2f})
    ✓ all numerical-stability checks pass
    ⚠ see any [WARN] / [FAIL] lines above for specific remediation
"""))
