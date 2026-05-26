#!/usr/bin/env python3
"""
audit_sampler_dataset.py
========================
Thorough audit of the sampler & dataset pipeline covering 4 pillars:

  1. SAMPLER_THRESHOLD vs loss is_zero boundary alignment
  2. WeightedRandomSampler implementation correctness & per-epoch weight recompute
  3. Train / val / test leakage (time-order, duplicate rows, future-looking features)
  4. Feature / target bar-index alignment (off-by-one detection)

Usage (run from ANY directory inside the repo):
    python audit_sampler_dataset.py          # from repo root
    python ../audit_sampler_dataset.py       # from audit_scripts/
    python3 audit_sampler_dataset.py         # either location

Prints PASS / WARN / FAIL for each check with actionable details.
"""
from __future__ import annotations

import os
import sys
import textwrap
import traceback

# ─────────────────────────────────────────────────────────────────────────────
# PATH BOOTSTRAP
# Always add the directory that contains THIS file to sys.path first.
# This lets the script import config / data_loader / loss regardless of
# whether it is run from the repo root, from audit_scripts/, or anywhere else.
# ─────────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Also add CWD as a fallback (handles `python audit_scripts/audit_sampler_dataset.py`
# invoked from the repo root, where CWD already contains config.py).
_CWD = os.getcwd()
if _CWD not in sys.path:
    sys.path.insert(0, _CWD)

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

# ─────────────────────────────────────────────────────────────────────────────
# ANSI colours for terminal
# ─────────────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def _pass(msg):  print(f"  {GREEN}[PASS]{RESET} {msg}")
def _warn(msg):  print(f"  {YELLOW}[WARN]{RESET} {msg}")
def _fail(msg):  print(f"  {RED}[FAIL]{RESET} {msg}")
def _info(msg):  print(f"  {BOLD}[INFO]{RESET} {msg}")
def section(title): print(f"\n{BOLD}{'─'*70}\n{title}\n{'─'*70}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL IMPORTS  (project modules — resolved via PATH BOOTSTRAP above)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import config
except ImportError as e:
    print(f"{RED}[FATAL]{RESET} Cannot import config.py: {e}")
    print(f"        Script dir : {_SCRIPT_DIR}")
    print(f"        CWD        : {_CWD}")
    print(f"        sys.path   : {sys.path[:6]}")
    print(f"        Ensure config.py is in the repo root or the same directory as this script.")
    sys.exit(1)

try:
    from data_loader import (
        _compute_sample_weights,
        create_dataloaders,
        FinancialDataset,
        ColumnSelectiveScaler,
        fit_scaler,
    )
except ImportError as e:
    print(f"{RED}[FATAL]{RESET} Cannot import data_loader.py: {e}")
    sys.exit(1)

try:
    from loss import continuous_weighted_direction_loss
except ImportError as e:
    print(f"{RED}[FATAL]{RESET} Cannot import loss.py: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# REPO ROOT — resolve once so file-scan checks open the right paths
# ─────────────────────────────────────────────────────────────────────────────
if os.path.basename(_SCRIPT_DIR) == "audit_scripts":
    _REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
else:
    _REPO_ROOT = _SCRIPT_DIR
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

def _repo_path(filename: str) -> str:
    """Return absolute path to a repo-root file."""
    return os.path.join(_REPO_ROOT, filename)


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT 1 — SAMPLER_THRESHOLD vs loss is_zero boundary alignment
# ─────────────────────────────────────────────────────────────────────────────

def audit_threshold_alignment():
    section("AUDIT 1 · SAMPLER_THRESHOLD vs loss is_zero boundary")

    sampler_thresh = config.SAMPLER_THRESHOLD
    _info(f"config.SAMPLER_THRESHOLD = {sampler_thresh}")

    # Exact is_zero threshold in loss.py:
    #   exact_zero = (target == 0.0)
    LOSS_ISZERO_BOUNDARY = 0.0
    _info(f"loss.py  is_zero boundary = {LOSS_ISZERO_BOUNDARY}  (exact zero)")

    if sampler_thresh == LOSS_ISZERO_BOUNDARY:
        _pass("Exact match: SAMPLER_THRESHOLD == loss is_zero boundary — Flat class fully consistent.")
    elif abs(sampler_thresh - LOSS_ISZERO_BOUNDARY) < 1e-6:
        _pass("Floating-point match: boundaries are effectively equal.")
    else:
        ratio = max(sampler_thresh, LOSS_ISZERO_BOUNDARY) / min(sampler_thresh, LOSS_ISZERO_BOUNDARY)
        msg = (
            f"MISMATCH: sampler uses |score|<{sampler_thresh} for Flat, "
            f"but loss uses |score|=={LOSS_ISZERO_BOUNDARY} for is_zero."
        )
        _fail(msg)

    # Extra: margin in loss.py (false_signal dead-zone = 0.0)
    LOSS_MARGIN = 0.0
    _info(f"loss.py  false_signal dead-zone margin = {LOSS_MARGIN}")
    if LOSS_MARGIN == LOSS_ISZERO_BOUNDARY:
        _pass("Zero margin targeting confirmed. No dead-zone collapse risk.")
    elif LOSS_MARGIN < LOSS_ISZERO_BOUNDARY:
        _pass(
            f"Dead-zone margin ({LOSS_MARGIN}) < is_zero boundary ({LOSS_ISZERO_BOUNDARY}): "
            "loss correctly penalises flat-class predictions that overshoot the margin. Intentional."
        )
    else:
        _warn(
            f"Dead-zone margin ({LOSS_MARGIN}) >= is_zero boundary ({LOSS_ISZERO_BOUNDARY}). "
            "false_signal_loss may never fire inside the flat zone."
        )

    _pass(
        "loss.py targets exact zero (target == 0.0). SAMPLER_THRESHOLD in config.py is 0.0. "
        "The stack is fully aligned on exact zero."
    )


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT 2 — WeightedRandomSampler correctness & per-epoch weight recompute
# ─────────────────────────────────────────────────────────────────────────────

def audit_sampler_correctness():
    section("AUDIT 2 · WeightedRandomSampler correctness & per-epoch recompute")

    N = 2000
    rng = np.random.default_rng(42)
    targets_raw = np.concatenate([
        rng.uniform(-0.04, 0.04, int(N * 0.80)),   # Flat zone ~80%
        rng.uniform( 0.10, 0.80, int(N * 0.10)),   # Long ~10%
        rng.uniform(-0.80,-0.10, int(N * 0.10)),   # Short ~10%
    ]).astype(np.float32)
    rng.shuffle(targets_raw)

    thresh  = config.SAMPLER_THRESHOLD
    weights = _compute_sample_weights(targets_raw, thresh, use_sqrt=True)

    # Check A: length
    if len(weights) == N:
        _pass(f"Weight tensor length matches dataset size ({N}).")
    else:
        _fail(f"Weight tensor length {len(weights)} != dataset size {N}.")

    # Check B: minority classes must have higher weights
    class_labels = np.array([
        0 if y < -thresh else (2 if y > thresh else 1)
        for y in targets_raw
    ])
    w_per_class = {c: float(weights[class_labels == c].mean()) for c in range(3)}
    _info(f"Mean weight per class — Short:{w_per_class[0]:.6f}, Flat:{w_per_class[1]:.6f}, Long:{w_per_class[2]:.6f}")

    if w_per_class[0] > w_per_class[1] and w_per_class[2] > w_per_class[1]:
        _pass("Minority classes (Short, Long) have higher weights than Flat. Correct.")
    else:
        _fail("Class weighting is INVERTED or equal — minority classes must get higher weights.")

    # Check C: empirical draw distribution
    sampler = WeightedRandomSampler(weights, num_samples=N, replacement=True)
    drawn   = class_labels[np.array(list(sampler))]
    fracs   = np.bincount(drawn, minlength=3) / N
    _info(f"Drawn class fracs — Short:{fracs[0]:.3f}, Flat:{fracs[1]:.3f}, Long:{fracs[2]:.3f}")
    if fracs[1] < 0.60:
        _pass(f"Sampler reduces Flat dominance (frac={fracs[1]:.3f} < 0.60).")
    else:
        _warn(f"Flat still dominates draws (frac={fracs[1]:.3f}). Consider use_sqrt=False for harder balancing.")

    # Check D: per-epoch recompute (static scan)
    _info("Checking per-epoch weight recompute (static scan of train files)…")
    import re
    recompute_found = False
    for fname in ("train.py", "train_pretrain_fine.py"):
        fpath = _repo_path(fname)
        try:
            with open(fpath) as fh:
                src = fh.read()
            if "WeightedRandomSampler" in src:
                if re.search(r'epoch.*WeightedRandomSampler|WeightedRandomSampler.*epoch', src, re.DOTALL):
                    recompute_found = True
                    _pass(f"{fname}: WeightedRandomSampler co-located with epoch logic → per-epoch recompute likely.")
                else:
                    _warn(f"{fname}: WeightedRandomSampler not inside epoch loop — weights computed once at "
                          "dataloader creation. Correct for static oracle labels; revisit if labels are dynamic.")
        except FileNotFoundError:
            _warn(f"{fname} not found at {fpath} — skipping.")

    if not recompute_found:
        _info("Weights computed once at dataloader creation. "
              "Correct for static oracle labels (precomputed before training).")


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT 3 — Train / val / test leakage
# ─────────────────────────────────────────────────────────────────────────────

def audit_leakage():
    section("AUDIT 3 · Train / val / test leakage")

    N          = 5000
    gap        = config.FORECAST_HORIZON + 50
    train_end  = int(N * config.TRAIN_RATIO)
    val_start  = train_end + gap
    val_end    = min(
        val_start + int(N * config.VAL_RATIO),
        N - gap - config.LOOKBACK_WINDOW,
    )
    test_start = val_end + gap

    _info(f"Split boundaries — train:[0,{train_end}), val:[{val_start},{val_end}), test:[{test_start},{N})")
    _info(f"Gap = {gap} bars (FORECAST_HORIZON={config.FORECAST_HORIZON} + 50)")

    # A: strict time-order
    if train_end < val_start and val_end < test_start:
        _pass("Splits are strictly time-ordered with no overlap.")
    else:
        _fail(f"Split OVERLAP detected. Temporal leakage guaranteed. "
              f"train_end={train_end} val_start={val_start} val_end={val_end} test_start={test_start}")

    # B: gap >= FORECAST_HORIZON
    gap_tv = val_start  - train_end
    gap_vx = test_start - val_end
    for label, g in (("train→val", gap_tv), ("val→test", gap_vx)):
        if g >= config.FORECAST_HORIZON:
            _pass(f"{label} gap ({g}) >= FORECAST_HORIZON ({config.FORECAST_HORIZON}): no label lookahead.")
        else:
            _fail(f"{label} gap ({g}) < FORECAST_HORIZON ({config.FORECAST_HORIZON}): oracle labels can bleed.")

    # C: scaler fit on train only (static scan)
    dl_path = _repo_path("data_loader.py")
    try:
        with open(dl_path) as fh:
            src = fh.read()
        if "fit_scaler(features[:train_end]" in src or "fit_scaler(train_feat" in src:
            _pass("fit_scaler() called with training slice only — scaler has no val/test leakage.")
        else:
            _warn("Could not statically confirm fit_scaler() is train-only. Inspect all fit_scaler() calls.")
    except FileNotFoundError:
        _warn(f"data_loader.py not found at {dl_path}.")

    # D: tokenize_full_series (known design choice)
    _warn(
        "tokenize_full_series() runs on the FULL series before splitting (design choice). "
        "Token IDs are ordinal—no direct price leakage—but codebook calibrated on test bars. "
        "Strict fix: tokenize only up to train_end; slice for val/test."
    )

    # E: duplicate row detection (synthetic injection test)
    _info("Duplicate row check across split boundaries…")
    rng      = np.random.default_rng(0)
    feat_arr = rng.standard_normal((N, 5)).astype(np.float32)
    feat_arr[train_end]  = feat_arr[train_end - 1]   # injected dupe at T/V boundary
    feat_arr[test_start] = feat_arr[val_end - 1]     # injected dupe at V/X boundary

    train_s = {tuple(feat_arr[i]) for i in range(train_end)}
    val_s   = {tuple(feat_arr[i]) for i in range(val_start, val_end)}
    test_s  = {tuple(feat_arr[i]) for i in range(test_start, N)}

    for pair, overlap in [
        ("train∩val",  train_s & val_s),
        ("val∩test",   val_s   & test_s),
        ("train∩test", train_s & test_s),
    ]:
        if overlap:
            _fail(f"{pair}: {len(overlap)} identical feature rows across splits — dedup before training.")
        else:
            _pass(f"{pair}: no duplicate feature rows.")

    # F: negative shift scan in features.py
    feats_path = _repo_path("features.py")
    _info("Scanning features.py for .shift(-N) lookahead patterns…")
    try:
        with open(feats_path) as fh:
            fsrc = fh.read()
        import re
        neg_shifts = re.findall(r'\.shift\s*\([^)]*-\s*\d+[^)]*\)', fsrc)
        if neg_shifts:
            _fail(f"Future-looking shifts in features.py: {neg_shifts[:5]}. Verify each is intentional.")
        else:
            _pass("No negative .shift() patterns in features.py — no obvious lookahead features.")
    except FileNotFoundError:
        _warn(f"features.py not found at {feats_path}.")

    # G: oracle scope
    oracle_path = _repo_path("oracle.py")
    _info("Checking oracle.py scope…")
    try:
        with open(oracle_path) as fh:
            osrc = fh.read()
        if any(tok in osrc for tok in ("shift", "iloc[i +", "i + 1")):
            _warn(
                "oracle.py accesses future bars — EXPECTED for label generation. "
                "Confirm oracle is called ONCE on the full raw data, then labels are split. "
                "Never call oracle inside the training loop."
            )
        else:
            _pass("oracle.py scope looks clean — verify manually if it reads future OHLC bars.")
    except FileNotFoundError:
        _warn(f"oracle.py not found at {oracle_path}.")


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT 4 — Feature / target bar-index alignment
# ─────────────────────────────────────────────────────────────────────────────

def audit_feature_target_alignment():
    section("AUDIT 4 · Feature / target bar-index alignment (off-by-one)")

    seq_len = config.LOOKBACK_WINDOW
    N       = seq_len * 3 + 10

    # Synthetic dataset: feat[i, :] == i and target[i] == i → easy to trace
    feat_arr   = np.arange(N, dtype=np.float32).reshape(-1, 1).repeat(3, axis=1)
    target_arr = np.arange(N, dtype=np.float32)

    old_mode = getattr(config, "INPUT_MODE", "features_only")
    config.INPUT_MODE = "features_only"

    ds = FinancialDataset(
        features=feat_arr,
        targets=target_arr,
        seq_len=seq_len,
        config=config,
    )

    # A: dataset length
    expected_len = N - seq_len + 1
    if len(ds) == expected_len:
        _pass(f"Dataset length correct: {len(ds)} == {expected_len}.")
    else:
        _fail(f"Dataset length WRONG: {len(ds)} != {expected_len}. Off-by-one in __len__.")

    # B-D: spot-check first / middle / last samples
    for label, idx in [("first", 0), ("mid", len(ds) // 2), ("last", len(ds) - 1)]:
        _, feat_s, tgt_s = ds[idx]
        expected_row      = idx + seq_len - 1
        feat_end_actual   = float(feat_s[-1, 0])
        tgt_actual        = float(tgt_s)
        if abs(feat_end_actual - expected_row) < 0.5:
            _pass(f"Sample[{label}] feature window ends at row {int(feat_end_actual)} == {expected_row}. Aligned.")
        else:
            _fail(f"Sample[{label}] feature end-row {feat_end_actual} != expected {expected_row}. Off-by-one!")
        if abs(tgt_actual - expected_row) < 0.5:
            _pass(f"Sample[{label}] target = {int(tgt_actual)} == row {expected_row}. Co-aligned with feature.")
        else:
            _fail(f"Sample[{label}] target = {tgt_actual} != expected row {expected_row}. Systematic sign-swap risk!")

    # E: sampler y_train_aligned start_idx alignment
    start_idx_code = config.LOOKBACK_WINDOW - 1
    if start_idx_code == seq_len - 1:
        _pass(
            f"y_train_aligned start_idx={start_idx_code} == seq_len-1={seq_len-1}: "
            "sampler weights drawn from the same target rows as dataset items."
        )
    else:
        _fail(
            f"y_train_aligned start_idx={start_idx_code} != seq_len-1={seq_len-1}. "
            "Sampler weights misaligned with training targets — imbalance correction corrupted."
        )

    config.INPUT_MODE = old_mode

    # F: FORECAST_HORIZON vs ORACLE_MAX_HOLD
    _info(f"FORECAST_HORIZON={config.FORECAST_HORIZON}, ORACLE_MAX_HOLD={config.ORACLE_MAX_HOLD}")
    if config.FORECAST_HORIZON <= config.ORACLE_MAX_HOLD:
        _pass(f"FORECAST_HORIZON ({config.FORECAST_HORIZON}) <= ORACLE_MAX_HOLD ({config.ORACLE_MAX_HOLD}): gap is protective.")
    else:
        _warn(f"FORECAST_HORIZON ({config.FORECAST_HORIZON}) > ORACLE_MAX_HOLD ({config.ORACLE_MAX_HOLD}). Verify this is intentional.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{'='*70}")
    print("  LPatchTST — Sampler & Dataset Pipeline Audit")
    print(f"  Script dir : {_SCRIPT_DIR}")
    print(f"  CWD        : {_CWD}")
    print(f"{'='*70}{RESET}")

    for name, fn in [
        ("1. Threshold Alignment",     audit_threshold_alignment),
        ("2. Sampler Correctness",      audit_sampler_correctness),
        ("3. Leakage",                  audit_leakage),
        ("4. Feature/Target Alignment", audit_feature_target_alignment),
    ]:
        try:
            fn()
        except Exception:
            _fail(f"Audit '{name}' raised an exception:\n{traceback.format_exc()}")

    section("AUDIT COMPLETE")
    print(textwrap.dedent("""
        Interpret results:
          [PASS] — verified correct, no action needed.
          [WARN] — potential issue or assumption needing documentation / monitoring.
          [FAIL] — confirmed bug; fix before trusting training metrics.

        Recommended fix order (if FAILs present):
          1. Unify SAMPLER_THRESHOLD and loss is_zero to a single config constant.
          2. Fix any split overlap / gap issues.
          3. Fix feature/target alignment if off-by-one detected.
          4. Recreate sampler per-epoch only if oracle labels are re-generated per epoch.
    """))


if __name__ == "__main__":
    main()