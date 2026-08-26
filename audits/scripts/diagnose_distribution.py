"""
diagnose_distribution.py
═══════════════════════════════════════════════════════════════════════════════
Train/val distribution audit for the v2 close-only feature pipeline.

Adapted from the legacy tokenizer-focused diagnose_distribution.py — that
script audited the BSQ token space; this pipeline runs INPUT_MODE="features_only"
with the Kronos tokenizer excluded. The audit is now anchored to what the
PatchTST model actually consumes: the 9 engineered features produced by
FeatureEngineer (close-only).

What this checks (in order):
  STAGE 0 — Raw engineered feature statistics (mean, std, skew) train vs val
  STAGE 1 — Per-feature symmetric KL divergence (train ‖ val)
  STAGE 2 — Per-feature variance ratio (val/train)  — detects regime drift
  STAGE 3 — Tail/quantile drift (p1 / p99 train vs val) — catches tail mismatch
            that KL on dense bins can miss
  STAGE 4 — Linear probe: per-feature univariate correlation with next-bar
            close log-return, train vs val
  STAGE 5 — NaN propagation footprint (rows lost to warm-up in train vs val)

All outputs are printed AND written to diagnose_output.txt.

Usage
-----
    python diagnose_distribution.py
    python diagnose_distribution.py --data_dir data/ --train_frac 0.7
"""

from __future__ import annotations
import argparse
import os
import sys
import warnings
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Repo-aware path setup so the script runs from anywhere ────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CORE = os.path.join(_REPO_ROOT, "core")
sys.path.insert(0, _CORE)
sys.path.insert(0, _REPO_ROOT)

# ── Logging helper ────────────────────────────────────────────────────────────
LOG_LINES: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    LOG_LINES.append(str(msg))


def save_log(path: str = "diagnose_output.txt") -> None:
    with open(path, "w") as f:
        f.write("\n".join(LOG_LINES))
    print(f"\n✓ Full output saved → {path}")


# ── Statistical helpers ───────────────────────────────────────────────────────

def describe(arr: np.ndarray, name: str, indent: str = "  ") -> None:
    """Print mean/std/min/max/skew for a 2-D array (samples × features)."""
    if arr.ndim == 1:
        arr = arr[:, None]
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    mn   = arr.min(axis=0)
    mx   = arr.max(axis=0)
    skew = ((arr - mean) ** 3).mean(axis=0) / (std ** 3 + 1e-9)
    log(f"{indent}{name}")
    log(f"{indent}  mean : {np.round(mean, 4)}")
    log(f"{indent}  std  : {np.round(std,  4)}")
    log(f"{indent}  min  : {np.round(mn,   4)}")
    log(f"{indent}  max  : {np.round(mx,   4)}")
    log(f"{indent}  skew : {np.round(skew, 4)}")


def kl_div_bins(a: np.ndarray, b: np.ndarray, n_bins: int = 50) -> float:
    """Symmetric KL divergence between two 1-D arrays using histogram bins."""
    lo = float(min(np.nanmin(a), np.nanmin(b)))
    hi = float(max(np.nanmax(a), np.nanmax(b)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    pa, _ = np.histogram(a[~np.isnan(a)], bins=bins, density=True)
    pb, _ = np.histogram(b[~np.isnan(b)], bins=bins, density=True)
    pa = pa / (pa.sum() + 1e-12)
    pb = pb / (pb.sum() + 1e-12)
    eps = 1e-10
    kl_ab = np.sum(np.where(pa > eps, pa * np.log((pa + eps) / (pb + eps)), 0))
    kl_ba = np.sum(np.where(pb > eps, pb * np.log((pb + eps) / (pa + eps)), 0))
    return float(0.5 * (kl_ab + kl_ba))


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - np.nanmean(a)
    b = b - np.nanmean(b)
    denom = (np.nanstd(a) * np.nanstd(b) + 1e-12)
    return float(np.nanmean(a * b) / denom)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_csv_files(data_dir: str) -> list[pd.DataFrame]:
    """Return a list of DataFrames (lowercase OHLCV columns)."""
    dfs: list[pd.DataFrame] = []
    for p in sorted(Path(data_dir).glob("**/*.csv")):
        try:
            df = pd.read_csv(p)
            df.columns = [c.strip().lower() for c in df.columns]
            required = {"open", "high", "low", "close"}
            if not required.issubset(df.columns):
                log(f"  ⚠ Skipping {p.name}: missing OHLC columns")
                continue
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["open", "high", "low", "close"])
            if len(df) < 600:
                log(f"  ⚠ Skipping {p.name}: only {len(df)} rows (<600)")
                continue
            dfs.append(df.reset_index(drop=True))
            log(f"  ✓ Loaded {p.name} → {len(df)} rows")
        except Exception as e:
            log(f"  ✗ Error loading {p.name}: {e}")
    return dfs


# ── Feature engineering via the project's FeatureEngineer ────────────────────

def build_features(dfs: list[pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    """
    Run the project's FeatureEngineer over each file and concatenate the
    results. Returns (combined_feature_df, feature_cols) — chronological
    concatenation respects the within-file ordering but does NOT re-sort
    across files. Use --per_file_split for honest train/val cuts.
    """
    from train import _make_feature_config
    from features import FeatureEngineer
    import config  # local core/config.py

    fe = FeatureEngineer(_make_feature_config())
    parts: list[pd.DataFrame] = []
    feature_cols: list[str] | None = None
    for df in dfs:
        df = df.sort_index() if df.index.name else df
        feats = fe.build(df["close"], ohlc=df, include_target=False, dropna=True)
        parts.append(feats)
        if feature_cols is None:
            feature_cols = list(feats.columns)
    combined = pd.concat(parts, axis=0, ignore_index=True)
    assert feature_cols is not None
    return combined, feature_cols


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def run_diagnostics(args: argparse.Namespace) -> None:
    log(f"Args: {vars(args)}\n")

    # 1. Load data
    log("=" * 70)
    log("LOADING DATA")
    log("=" * 70)
    if not os.path.isdir(args.data_dir):
        log(f"✗ --data_dir {args.data_dir!r} not found.")
        return
    dfs = load_csv_files(args.data_dir)
    if not dfs:
        log("✗ No valid CSV files found. Check --data_dir")
        return

    # 2. Build v2 features across all loaded data (chronological concat)
    log("\n" + "=" * 70)
    log("BUILDING v2 FEATURES (FeatureEngineer, close-only, 9 columns)")
    log("=" * 70)
    feats_df, feature_cols = build_features(dfs)
    log(f"  Combined feature matrix: {feats_df.shape}")
    log(f"  Feature columns: {feature_cols}")

    N = len(feats_df)
    split_idx = int(N * args.train_frac)
    f_train = feats_df.iloc[:split_idx].values
    f_val   = feats_df.iloc[split_idx:].values

    log(f"\nTotal rows after warm-up drop: {N}")
    log(f"Train      : {len(f_train)} rows (0 → {split_idx})")
    log(f"Val        : {len(f_val)} rows ({split_idx} → {N})")

    # 3. Build the target series from raw close log-returns (independent of features)
    raw_close = pd.concat([d["close"] for d in dfs], axis=0, ignore_index=True)
    raw_close = raw_close.dropna().reset_index(drop=True)
    close_train = raw_close.iloc[: split_idx + 1]
    close_val   = raw_close.iloc[split_idx + 1 :]
    r_next_train = np.log(close_train.values[1:] / close_train.values[:-1])
    r_next_val   = np.log(close_val.values[1:]   / close_val.values[:-1])

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 0 — Feature statistics
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("STAGE 0 — Feature statistics (train vs val)")
    log("=" * 70)
    describe(f_train, "TRAIN features", indent="  ")
    log()
    describe(f_val,   "VAL   features", indent="  ")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Per-feature symmetric KL
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("STAGE 1 — Per-feature symmetric KL (train ‖ val)")
    log("=" * 70)
    log(f"  {'Feature':<18} | {'KL':>7} | {'flag':<8}")
    log("  " + "-" * 42)
    kls: list[float] = []
    for i, col in enumerate(feature_cols):
        kl = kl_div_bins(f_train[:, i], f_val[:, i])
        kls.append(kl)
        flag = "✗ HIGH" if kl > 0.3 else ("⚠ MED" if kl > 0.1 else "✓ OK")
        log(f"  {col:<18} | {kl:7.4f} | {flag}")
    log("  " + "-" * 42)
    log(f"  Mean KL across features: {np.mean(kls):.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2 — Variance ratio (val / train) — detects regime drift
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("STAGE 2 — Variance ratio (val / train) per feature")
    log("  Ratio >1 = val more volatile, <1 = calmer than train")
    log("=" * 70)
    log(f"  {'Feature':<18} | {'std train':>10} | {'std val':>10} | {'ratio':>7} | {'flag':<8}")
    log("  " + "-" * 60)
    ratios: list[float] = []
    for i, col in enumerate(feature_cols):
        s_tr = float(np.nanstd(f_train[:, i]))
        s_va = float(np.nanstd(f_val[:, i]))
        r = s_va / (s_tr + 1e-12)
        ratios.append(r)
        flag = "✗" if (r > 2.0 or r < 0.5) else ("⚠" if (r > 1.5 or r < 0.66) else "✓")
        log(f"  {col:<18} | {s_tr:10.4f} | {s_va:10.4f} | {r:7.3f} | {flag:<8}")
    log("  " + "-" * 60)
    log(f"  Median variance ratio: {np.median(ratios):.3f}")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 3 — Tail/quantile drift (p1 / p99)
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("STAGE 3 — Tail/quantile drift (p1 / p99 train vs val)")
    log("  Surfaces distributional tail shift that dense-bin KL can miss")
    log("=" * 70)
    log(f"  {'Feature':<18} | {'p1 tr':>8} | {'p1 va':>8} | {'Δp1':>8} | "
        f"{'p99 tr':>8} | {'p99 va':>8} | {'Δp99':>8} | {'flag':<6}")
    log("  " + "-" * 84)
    for i, col in enumerate(feature_cols):
        p1_tr, p99_tr = np.nanpercentile(f_train[:, i], [1, 99])
        p1_va, p99_va = np.nanpercentile(f_val[:, i],   [1, 99])
        d1  = float(p1_va - p1_tr)
        d99 = float(p99_va - p99_tr)
        # scale-invariant flag: relative shift
        scale = max(abs(p1_tr), abs(p99_tr), 1e-9)
        rel   = max(abs(d1), abs(d99)) / scale
        flag  = "✗" if rel > 0.5 else ("⚠" if rel > 0.2 else "✓")
        log(f"  {col:<18} | {p1_tr:8.3f} | {p1_va:8.3f} | {d1:+8.3f} | "
            f"{p99_tr:8.3f} | {p99_va:8.3f} | {d99:+8.3f} | {flag:<6}")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 4 — Linear probe: per-feature correlation with next-bar log return
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("STAGE 4 — Per-feature linear probe (correlation with r_{t+1})")
    log("  Negative correlation with the target is expected for short-horizon")
    log("  features (predictor = ret_norm_1; IC is usually positive when")
    log("  correlating predictor_t with r_{t+1}).")
    log("=" * 70)
    log(f"  {'Feature':<18} | {'Corr train':>11} | {'Corr val':>11} | {'flag':<8}")
    log("  " + "-" * 56)
    for i, col in enumerate(feature_cols):
        n_tr = min(len(f_train) - 1, len(r_next_train))
        n_va = min(len(f_val)   - 1, len(r_next_val))
        x_tr = f_train[:n_tr, i]
        x_va = f_val[:n_va,   i]
        c_tr = pearson_corr(x_tr, r_next_train[:n_tr])
        c_va = pearson_corr(x_va, r_next_val[:n_va])
        # Sign-consistency flag (model relies on consistent polarity)
        flag = "✗" if (np.sign(c_tr) != np.sign(c_va) and
                        min(abs(c_tr), abs(c_va)) > 0.01) else "✓"
        log(f"  {col:<18} | {c_tr:+11.5f} | {c_va:+11.5f} | {flag:<8}")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 5 — Warm-up footprint
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("STAGE 5 — NaN / warm-up footprint")
    log("=" * 70)
    # Re-build WITHOUT dropna to count lost rows per file
    from train import _make_feature_config
    from features import FeatureEngineer
    fe = FeatureEngineer(_make_feature_config())
    total_in = 0
    total_out = 0
    for df in dfs:
        raw = fe.build(df["close"], ohlc=df, include_target=False, dropna=False)
        n_in  = len(raw)
        n_out = int(raw.dropna().shape[0])
        total_in  += n_in
        total_out += n_out
        log(f"  {len(df)} raw bars → {n_in} feature rows ({n_in - n_out} NaN/warmup dropped)")
    if total_in:
        lost_pct = 100 * (total_in - total_out) / total_in
        log(f"\n  Combined: {total_in} bars → {total_out} clean rows "
            f"({total_in - total_out} lost, {lost_pct:.2f}%)")

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY & VERDICT
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("SUMMARY & VERDICT")
    log("=" * 70)
    mean_kl = float(np.mean(kls))
    med_ratio = float(np.median(ratios))
    log(f"""
  Mean per-feature KL (train ‖ val)  : {mean_kl:.4f}
  Median variance ratio (val/train)  : {med_ratio:.3f}

  Interpretation guide:
    • KL < 0.1            ✓ distributions match closely
    • 0.1 < KL < 0.3      ⚠ moderate shift — usually safe, monitor
    • KL > 0.3            ✗ significant shift — consider WalkForward CV

    • Variance ratio ≈ 1.0          ✓ similar regime
    • 0.66 ≤ ratio ≤ 1.5            ✓ acceptable drift
    • Outside [0.5, 2.0]            ✗ val period is structurally different

  Common causes of high KL on this v2 feature set:
    1. val period straddles a vol regime change (σ60 jumps) — log_ewma_vol_60
       and vol_ratio_* drift in lockstep, with ret_norm_*/mom_norm_* drifting
       only mildly. Check STAGE 2 ratios; if vol_ratio_* >> 1, that's the cause.
    2. crash/gap event in val (e.g. single bar with |ret_norm_1| > 10) that
       never appeared in train — STAGE 3 tail Δ will flag it.
    3. asset added late in val period with shorter history — features are
       NaN at the file boundary and were dropped silently. Check STAGE 5.

  Next steps if shift is concerning:
    a. Re-run with --train_frac 0.5 / 0.6 / 0.8 to check KL sensitivity.
    b. Enable walk-forward validation in train.py to see if the model still
       generalises.
    c. If vol_ratio_* ratio >> 1: consider rolling-normalizing σ60 in the
       build (out of scope for v2; the absolute-level feature is
       intentionally preserved).
""")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train/val distribution audit for the v2 feature pipeline"
    )
    parser.add_argument(
        "--data_dir", default="data/",
        help="Directory with CSV files (relative to repo root or absolute)",
    )
    parser.add_argument(
        "--train_frac", type=float, default=0.7,
        help="Chronological train fraction",
    )
    parser.add_argument(
        "--output", default="diagnose_output.txt",
        help="Output log file path",
    )
    args = parser.parse_args()

    # Resolve --data_dir against the repo root if it isn't absolute
    if not os.path.isabs(args.data_dir):
        cand1 = args.data_dir
        cand2 = os.path.join(_REPO_ROOT, args.data_dir)
        if os.path.isdir(cand1):
            pass
        elif os.path.isdir(cand2):
            args.data_dir = cand2
        else:
            # final fallback
            args.data_dir = cand2

    try:
        run_diagnostics(args)
    finally:
        save_log(os.path.join(_REPO_ROOT, args.output))
