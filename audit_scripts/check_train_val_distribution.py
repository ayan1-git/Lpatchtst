#!/usr/bin/env python3
"""
check_train_val_distribution.py
═══════════════════════════════════════════════════════════════════════════════
Replicates the EXACT train/val split from train.py and checks target distribution
on both sides. Uses the real oracle, real features, and real date boundaries.

This tells you:
  1. What % of train vs val targets are flat (zero) vs directional
  2. Whether the sampler sees a drastically different class balance
  3. The magnitude distribution of non-zero targets in each split
  4. Whether the gap between train and val is sufficient

Run from repo root:
  python audit_scripts/check_train_val_distribution.py
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import math
import glob
import numpy as np
import pandas as pd

# ── Bootstrap path ────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config
from features import FeatureEngineer
from oracle import generate_targets
from tokenizer import KronosTokenizer, prepare_ohlc_features


def log(msg=""):
    print(msg)


def describe_target_dist(targets, label, sampler_threshold):
    """Print detailed target distribution statistics."""
    n = len(targets)
    abs_t = np.abs(targets)

    exact_zero = (targets == 0.0).sum()
    sampler_flat = (abs_t <= sampler_threshold).sum()
    positive = (targets > sampler_threshold).sum()
    negative = (targets < -sampler_threshold).sum()

    # Non-zero magnitude buckets
    nz_mask = abs_t > sampler_threshold
    nz_vals = abs_t[nz_mask]

    log(f"\n{'='*60}")
    log(f"  {label}  (n={n:,})")
    log(f"{'='*60}")
    log(f"  Exact zeros:        {exact_zero:>8,}  ({exact_zero/n*100:5.1f}%)")
    log(f"  Sampler flat (≤{sampler_threshold}): {sampler_flat:>8,}  ({sampler_flat/n*100:5.1f}%)")
    log(f"  Positive (> {sampler_threshold}):   {positive:>8,}  ({positive/n*100:5.1f}%)")
    log(f"  Negative (< -{sampler_threshold}):  {negative:>8,}  ({negative/n*100:5.1f}%)")

    if len(nz_vals) > 0:
        log(f"\n  Non-zero magnitude distribution:")
        log(f"    mean   : {nz_vals.mean():.4f}")
        log(f"    median : {np.median(nz_vals):.4f}")
        log(f"    std    : {nz_vals.std():.4f}")
        log(f"    min    : {nz_vals.min():.4f}")
        log(f"    max    : {nz_vals.max():.4f}")
        log(f"    p25    : {np.percentile(nz_vals, 25):.4f}")
        log(f"    p75    : {np.percentile(nz_vals, 75):.4f}")
        log(f"    p90    : {np.percentile(nz_vals, 90):.4f}")
        log(f"    p95    : {np.percentile(nz_vals, 95):.4f}")

        # Bucket breakdown
        buckets = [
            (0.00, 0.05, "0.00-0.05"),
            (0.05, 0.10, "0.05-0.10"),
            (0.10, 0.20, "0.10-0.20"),
            (0.20, 0.50, "0.20-0.50"),
            (0.50, 1.00, "0.50-1.00"),
            (1.00, 9.99, "1.00+"),
        ]
        for lo, hi, name in buckets:
            cnt = ((abs_t > lo) & (abs_t <= hi)).sum()
            log(f"    {name:>12s}: {cnt:>6,}  ({cnt/n*100:4.1f}%)")

    return {
        "n": n,
        "exact_zero_pct": exact_zero / n,
        "sampler_flat_pct": sampler_flat / n,
        "positive_pct": positive / n,
        "negative_pct": negative / n,
        "pos_neg_ratio": positive / max(negative, 1),
    }


def main():
    log(f"\n{'═'*60}")
    log("  TRAIN / VAL TARGET DISTRIBUTION CHECK")
    log(f"{'═'*60}")

    # ── Load data ──────────────────────────────────────────────────────────────
    data_dir = config.DATA_DIR
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        log(f"  ✗ No CSV files found in {data_dir}")
        sys.exit(1)

    log(f"\n  Data dir : {data_dir}")
    log(f"  Files    : {len(files)}")
    for f in files:
        log(f"    {os.path.basename(f)}")

    # ── Config ─────────────────────────────────────────────────────────────────
    sampler_threshold = getattr(config, "SAMPLER_THRESHOLD", 0.05)
    lookback = config.LOOKBACK_WINDOW
    max_hold = config.ORACLE_MAX_HOLD
    log(f"\n  Config:")
    log(f"    SAMPLER_THRESHOLD  = {sampler_threshold}")
    log(f"    LOOKBACK_WINDOW    = {lookback}")
    log(f"    ORACLE_MAX_HOLD    = {max_hold}")

    # ── Process each file ─────────────────────────────────────────────────────
    fe = FeatureEngineer()

    all_train_stats = []
    all_val_stats = []

    for fpath in files:
        fname = os.path.basename(fpath)
        log(f"\n{'─'*60}")
        log(f"  Processing: {fname}")
        log(f"{'─'*60}")

        df = pd.read_csv(fpath, index_col=0, parse_dates=True)
        if len(df) < lookback * 3:
            log(f"  ⚠ Skipping: only {len(df)} rows (need {lookback*3})")
            continue

        # Match column names case-insensitively (same as train.py process_dataset)
        cols = {c.lower(): c for c in df.columns}
        o_col = cols.get('open', 'Open')
        h_col = cols.get('high', 'High')
        l_col = cols.get('low', 'Low')
        c_col = cols.get('close', 'Close')

        # Build features
        close_full = df[c_col].values.astype(np.float32)
        feat_df = fe.build(df[c_col], ohlc=df, include_target=False, dropna=False)
        feat_df['open']  = df[o_col].astype(np.float32)
        feat_df['high']  = df[h_col].astype(np.float32)
        feat_df['low']   = df[l_col].astype(np.float32)
        feat_df['close'] = df[c_col].astype(np.float32)

        # Generate targets — must match train.py process_dataset exactly:
        # 1. Compute ATR the same way (pd.concat max + rolling mean)
        # 2. Call generate_targets with positional args
        # 3. Zero out targets below SAMPLER_THRESHOLD
        feat_df = feat_df.ffill().bfill()
        close_s = feat_df['close']
        high_s  = feat_df['high']
        low_s   = feat_df['low']
        open_s  = feat_df['open']

        hl = high_s - low_s
        hc = (high_s - close_s.shift()).abs()
        lc = (low_s  - close_s.shift()).abs()
        atr = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(config.ATR_PERIOD).mean()

        targets = generate_targets(
            open_s.values,
            high_s.values,
            low_s.values,
            close_s.values,
            atr.values,
        )
        # Apply the same zeroing as train.py line 245
        targets[np.abs(targets) < sampler_threshold] = 0.0

        # Apply warmup offset (same as train.py process_dataset line 249)
        dates = feat_df.index
        warmup = 3536
        if len(targets) > warmup:
            targets = targets[warmup:]
            dates   = dates[warmup:]
        n = len(targets)
        log(f"  Total rows (post-warmup): {n:,}")
        log(f"  Date range: {dates[0]} → {dates[-1]}")

        # ── Replicate train.py split ───────────────────────────────────────────
        # From train.py lines 1458-1469:
        #   single_train_start = pd.Timestamp("2021-01-01")
        #   single_train_end   = pd.Timestamp("2025-07-01")
        #   gap = computed from LOOKBACK_WINDOW and avg_bars_per_day
        #   single_val_start   = single_train_end + gap
        #   single_val_end     = global_end

        train_start_ts = pd.Timestamp("2021-01-01")
        train_end_ts   = pd.Timestamp("2025-07-01")

        # Compute gap the same way train.py does
        # We need avg_bars_per_day from this file
        trading_days = len(pd.DatetimeIndex(dates).normalize().unique())
        avg_bars_per_day = len(dates) / max(trading_days, 1)
        safety_factor = getattr(config, "GAP_DENSITY_SAFETY", 0.70)
        eff_bars_per_day = max(avg_bars_per_day * safety_factor, 1e-6)
        gap_days_raw = math.ceil(lookback / eff_bars_per_day)
        margin_days  = getattr(config, "GAP_MARGIN_DAYS", 5)
        min_gap_days = getattr(config, "MIN_GAP_DAYS", 7)
        gap_days = max(gap_days_raw + margin_days, min_gap_days)
        gap = pd.Timedelta(days=gap_days)

        val_start_ts = train_end_ts + gap
        val_end_ts   = dates[-1]  # global_end

        log(f"\n  Split boundaries:")
        log(f"    Train: {train_start_ts.date()} → {train_end_ts.date()}")
        log(f"    Gap:   {gap_days} days")
        log(f"    Val:   {val_start_ts.date()} → {val_end_ts.date()}")
        log(f"    Avg bars/day: {avg_bars_per_day:.1f}")

        # Convert to indices
        train_start_idx = int(dates.searchsorted(train_start_ts, side="left"))
        train_end_idx   = int(dates.searchsorted(train_end_ts, side="right"))
        # train_end_safe accounts for ORACLE_MAX_HOLD (from train.py line 980)
        train_end_safe  = train_end_idx - (max_hold - 1)
        val_start_idx   = int(dates.searchsorted(val_start_ts, side="left"))
        val_end_idx     = int(dates.searchsorted(val_end_ts, side="right"))

        # Also account for LOOKBACK_WINDOW offset (targets align with features[lookback-1:])
        # The first valid target index is lookback_window - 1
        first_valid = lookback - 1

        train_start_idx = max(train_start_idx, first_valid)

        log(f"  Bar indices:")
        log(f"    Train: [{train_start_idx} : {train_end_safe}] = {max(0, train_end_safe - train_start_idx):,} bars")
        log(f"    Val:   [{val_start_idx} : {val_end_idx}] = {max(0, val_end_idx - val_start_idx):,} bars")

        if train_end_safe <= train_start_idx:
            log(f"  ⚠ No training bars after alignment!")
            continue
        if val_end_idx <= val_start_idx:
            log(f"  ⚠ No validation bars after alignment!")
            continue

        # Slice targets
        train_targets = targets[train_start_idx:train_end_safe]
        val_targets   = targets[val_start_idx:val_end_idx]

        # Remove NaN targets (from ATR warmup)
        train_valid = np.isfinite(train_targets)
        val_valid   = np.isfinite(val_targets)
        train_targets = train_targets[train_valid]
        val_targets   = val_targets[val_valid]

        log(f"  After removing NaN:")
        log(f"    Train: {len(train_targets):,} valid targets")
        log(f"    Val:   {len(val_targets):,} valid targets")

        # ── Distribution analysis ──────────────────────────────────────────────
        train_stats = describe_target_dist(train_targets, "TRAIN", sampler_threshold)
        val_stats   = describe_target_dist(val_targets,   "VAL",   sampler_threshold)

        # ── Comparison ──────────────────────────────────────────────────────────
        log(f"\n{'─'*60}")
        log(f"  COMPARISON")
        log(f"{'─'*60}")
        log(f"  {'Metric':<25s}  {'Train':>10s}  {'Val':>10s}  {'Δ':>10s}")
        log(f"  {'─'*55}")

        metrics = [
            ("Flat %",      "sampler_flat_pct", 100),
            ("Positive %",  "positive_pct",     100),
            ("Negative %",  "negative_pct",     100),
            ("Pos/Neg Ratio", "pos_neg_ratio",  1),
        ]
        for label, key, scale in metrics:
            tv = train_stats[key] * scale
            vv = val_stats[key] * scale
            delta = tv - vv
            flag = ""
            if abs(delta) > 5 and key.endswith("_pct"):
                flag = " ⚠ DRIFT"
            elif abs(delta) > 0.2 and key == "pos_neg_ratio":
                flag = " ⚠ IMBALANCE"
            log(f"  {label:<25s}  {tv:>9.2f}%  {vv:>9.2f}%  {delta:>+9.2f}pp{flag}")

        all_train_stats.append(train_stats)
        all_val_stats.append(val_stats)

    # ── Summary ────────────────────────────────────────────────────────────────
    if not all_train_stats:
        log("\n  ✗ No valid data processed.")
        sys.exit(1)

    log(f"\n{'═'*60}")
    log(f"  OVERALL SUMMARY")
    log(f"{'═'*60}")

    avg_train_flat = np.mean([s["sampler_flat_pct"] for s in all_train_stats]) * 100
    avg_val_flat   = np.mean([s["sampler_flat_pct"] for s in all_val_stats]) * 100
    avg_train_pos  = np.mean([s["positive_pct"] for s in all_train_stats]) * 100
    avg_val_pos    = np.mean([s["positive_pct"] for s in all_val_stats]) * 100
    avg_train_neg  = np.mean([s["negative_pct"] for s in all_train_stats]) * 100
    avg_val_neg    = np.mean([s["negative_pct"] for s in all_val_stats]) * 100

    log(f"\n  Average across {len(all_train_stats)} asset(s):")
    log(f"    Train flat: {avg_train_flat:.1f}%  |  Val flat: {avg_val_flat:.1f}%  |  Δ: {avg_train_flat - avg_val_flat:+.1f}pp")
    log(f"    Train pos:  {avg_train_pos:.1f}%   |  Val pos:  {avg_val_pos:.1f}%   |  Δ: {avg_train_pos - avg_val_pos:+.1f}pp")
    log(f"    Train neg:  {avg_train_neg:.1f}%   |  Val neg:  {avg_val_neg:.1f}%   |  Δ: {avg_train_neg - avg_val_neg:+.1f}pp")

    drift = abs(avg_train_flat - avg_val_flat)
    if drift > 10:
        log(f"\n  ✗ CRITICAL: {drift:.1f}pp flat-rate drift between train and val!")
        log(f"    The model sees a fundamentally different class distribution.")
        log(f"    This explains val loss degradation — the model optimizes for")
        log(f"    train's flat rate but val has a different base rate.")
    elif drift > 5:
        log(f"\n  ⚠ WARNING: {drift:.1f}pp flat-rate drift — moderate distribution shift.")
    else:
        log(f"\n  ✓ Flat-rate drift is small ({drift:.1f}pp) — distributions are similar.")

    log(f"\n{'═'*60}\n")


if __name__ == "__main__":
    main()
