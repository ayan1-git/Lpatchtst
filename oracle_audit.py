"""
oracle_audit.py  —  Oracle 4.1 Target Diagnostics
===================================================
Standalone script.  Does NOT modify oracle.py, config.py, or any pipeline file.

Reports
-------
Per data file and aggregate:
  • Total bars / valid signal bars / flat bars
  • Long signals vs Short signals (count + %)
  • Winning / losing breakdown for each side
  • Average target magnitude for winning long, losing long, winning short, losing short
  • Raw PnL R-multiples (pre-tanh) for deeper insight
  • Holding-period histogram (how many trades exit at stop vs time-exit)

Usage
-----
    python oracle_audit.py                       # uses config.DATA_FILE list
    python oracle_audit.py /path/to/custom.csv   # override with one CSV
"""

from __future__ import annotations
import sys, os, textwrap
import numpy as np
import pandas as pd
import numba

# ── import project modules without modification ──────────────────────────────
import config
from oracle import generate_targets


# ─────────────────────────────────────────────────────────────────────────────
# Extended oracle: returns per-bar diagnostics alongside the final target
# ─────────────────────────────────────────────────────────────────────────────

@numba.jit(nopython=True, cache=True, fastmath=True)
def _generate_targets_diagnostic(
    open_arr,
    high_arr,
    low_arr,
    close_arr,
    atr_arr,
    max_hold,
    fee_per_side=0.001,
    slippage=0.0005,
    sl_atr_mult=1.6,
    tp_atr_mult=3.7,
    enable_trailing= True,
    trail_atr_mult=3.3,
    saturation_factor=2.5,
    mae_penalty=0.20,
):
    """
    Same logic as oracle.generate_targets, but returns extra diagnostic arrays:
      - direction : +1 long, -1 short, 0 flat
      - raw_r     : the net R-multiple *before* tanh squash (signed)
      - hold_bars : bars held until exit (long or short, whichever won)
      - exit_type : 0 = flat, 1 = stop-loss, 2 = take-profit, 3 = trailing-stop, 4 = time-exit
      - gross_r   : realized pre-cost, pre-MAE PnL in R-multiple
      - cost_r_out: cost buffer R-multiple (costs / risk)
      - capture_ratio: realized net R-multiple divided by Maximum Favorable Excursion (MFE)
    """
    n = len(close_arr)
    targets    = np.zeros(n, dtype=np.float32)
    direction  = np.zeros(n, dtype=np.int8)
    raw_r      = np.zeros(n, dtype=np.float64)
    hold_bars  = np.zeros(n, dtype=np.int32)
    exit_type  = np.zeros(n, dtype=np.int8)
    gross_r    = np.zeros(n, dtype=np.float32)
    cost_r_out = np.zeros(n, dtype=np.float32)
    capture_ratio = np.zeros(n, dtype=np.float32)

    total_cost_pct = (fee_per_side + slippage) * 2.0

    sl_distances = atr_arr * sl_atr_mult
    tp_distances = atr_arr * tp_atr_mult
    trail_distances = atr_arr * trail_atr_mult

    for i in range(n - max_hold):
        entry_price = close_arr[i]
        sl_dist = sl_distances[i]
        tp_dist = tp_distances[i]
        trail_dist = trail_distances[i]

        if sl_dist <= 0.0 or tp_dist <= 0.0 or entry_price <= 0.0:
            continue

        risk_pct = sl_dist / entry_price
        if risk_pct <= 0.0:
            continue

        cost_r = total_cost_pct / risk_pct
        if cost_r > 1.0:
            continue

        # ---------------- LONG LOGIC ----------------
        initial_stop = entry_price - sl_dist
        long_stop = initial_stop
        long_tp = entry_price + tp_dist
        peak_price = entry_price
        max_risk_consumed_long = 0.0
        long_pnl_pct = 0.0
        long_hold = max_hold - 1
        long_exit = 4
        long_mfe_price = entry_price

        for k in range(1, max_hold):
            idx = i + k
            c_open = open_arr[idx]
            c_high = high_arr[idx]
            c_low = low_arr[idx]
            c_close = close_arr[idx]

            # Gap handling
            if c_open <= long_stop:
                exit_price = c_open
                long_pnl_pct = (exit_price - entry_price) / entry_price
                max_risk_consumed_long = 1.0
                long_hold = k
                long_exit = 3 if (long_stop > initial_stop) else 1
                long_mfe_price = max(long_mfe_price, c_open)
                break
            if c_open >= long_tp:
                exit_price = c_open
                long_pnl_pct = (exit_price - entry_price) / entry_price
                long_hold = k
                long_exit = 2
                long_mfe_price = max(long_mfe_price, c_open)
                break

            long_mfe_price = max(long_mfe_price, c_high)

            # Optional trailing update only after favorable movement
            if enable_trailing and c_high > peak_price:
                peak_price = c_high
                new_stop = peak_price - trail_dist
                if new_stop > long_stop:
                    long_stop = new_stop

            # Intrabar resolution: conservative tie-breaker
            hit_sl = c_low <= long_stop
            hit_tp = c_high >= long_tp

            if hit_sl and hit_tp:
                exit_price = long_stop
                long_pnl_pct = (exit_price - entry_price) / entry_price
                max_risk_consumed_long = 1.0
                long_hold = k
                long_exit = 3 if (long_stop > initial_stop) else 1
                break
            elif hit_sl:
                exit_price = long_stop
                long_pnl_pct = (exit_price - entry_price) / entry_price
                max_risk_consumed_long = 1.0
                long_hold = k
                long_exit = 3 if (long_stop > initial_stop) else 1
                break
            elif hit_tp:
                exit_price = long_tp
                long_pnl_pct = (exit_price - entry_price) / entry_price
                long_hold = k
                long_exit = 2
                break

            current_risk_consumed = (entry_price - c_low) / sl_dist
            if current_risk_consumed < 0.0:
                current_risk_consumed = 0.0
            elif current_risk_consumed > 1.0:
                current_risk_consumed = 1.0
            if current_risk_consumed > max_risk_consumed_long:
                max_risk_consumed_long = current_risk_consumed

            if k == max_hold - 1:
                long_pnl_pct = (c_close - entry_price) / entry_price
                long_hold = k
                long_exit = 4

        # ---------------- SHORT LOGIC ----------------
        initial_stop_short = entry_price + sl_dist
        short_stop = initial_stop_short
        short_tp = entry_price - tp_dist
        trough_price = entry_price
        max_risk_consumed_short = 0.0
        short_pnl_pct = 0.0
        short_hold = max_hold - 1
        short_exit = 4
        short_mfe_price = entry_price

        for k in range(1, max_hold):
            idx = i + k
            c_open = open_arr[idx]
            c_high = high_arr[idx]
            c_low = low_arr[idx]
            c_close = close_arr[idx]

            # Gap handling
            if c_open >= short_stop:
                exit_price = c_open
                short_pnl_pct = (entry_price - exit_price) / entry_price
                max_risk_consumed_short = 1.0
                short_hold = k
                short_exit = 3 if (short_stop < initial_stop_short) else 1
                short_mfe_price = min(short_mfe_price, c_open)
                break
            if c_open <= short_tp:
                exit_price = c_open
                short_pnl_pct = (entry_price - exit_price) / entry_price
                short_hold = k
                short_exit = 2
                short_mfe_price = min(short_mfe_price, c_open)
                break

            short_mfe_price = min(short_mfe_price, c_low)

            # Optional trailing update only after favorable movement
            if enable_trailing and c_low < trough_price:
                trough_price = c_low
                new_stop = trough_price + trail_dist
                if new_stop < short_stop:
                    short_stop = new_stop

            # Intrabar resolution: conservative tie-breaker
            hit_sl = c_high >= short_stop
            hit_tp = c_low <= short_tp

            if hit_sl and hit_tp:
                exit_price = short_stop
                short_pnl_pct = (entry_price - exit_price) / entry_price
                max_risk_consumed_short = 1.0
                short_hold = k
                short_exit = 3 if (short_stop < initial_stop_short) else 1
                break
            elif hit_sl:
                exit_price = short_stop
                short_pnl_pct = (entry_price - exit_price) / entry_price
                max_risk_consumed_short = 1.0
                short_hold = k
                short_exit = 3 if (short_stop < initial_stop_short) else 1
                break
            elif hit_tp:
                exit_price = short_tp
                short_pnl_pct = (entry_price - exit_price) / entry_price
                short_hold = k
                short_exit = 2
                break

            current_risk_consumed = (c_high - entry_price) / sl_dist
            if current_risk_consumed < 0.0:
                current_risk_consumed = 0.0
            elif current_risk_consumed > 1.0:
                current_risk_consumed = 1.0
            if current_risk_consumed > max_risk_consumed_short:
                max_risk_consumed_short = current_risk_consumed

            if k == max_hold - 1:
                short_pnl_pct = (entry_price - c_close) / entry_price
                short_hold = k
                short_exit = 4

        # ---------------- SCORING ----------------
        long_r = long_pnl_pct / risk_pct
        short_r = short_pnl_pct / risk_pct

        if long_r > 0.0:
            long_r *= (1.0 - (mae_penalty * max_risk_consumed_long))
        if short_r > 0.0:
            short_r *= (1.0 - (mae_penalty * max_risk_consumed_short))

        long_r_net = long_r - cost_r
        short_r_net = short_r - cost_r

        if long_r_net > 0.0 and long_r_net > short_r_net:
            targets[i] = np.tanh(long_r_net / saturation_factor)
            direction[i] = 1
            raw_r[i] = long_r_net
            hold_bars[i] = long_hold
            exit_type[i] = long_exit
            gross_r[i] = long_r
            cost_r_out[i] = cost_r
            long_mfe_r = (long_mfe_price - entry_price) / sl_dist
            capture_ratio[i] = long_r_net / max(long_mfe_r, 1e-8)
        elif short_r_net > 0.0 and short_r_net > long_r_net:
            targets[i] = -np.tanh(short_r_net / saturation_factor)
            direction[i] = -1
            raw_r[i] = -short_r_net
            hold_bars[i] = short_hold
            exit_type[i] = short_exit
            gross_r[i] = short_r
            cost_r_out[i] = cost_r
            short_mfe_r = (entry_price - short_mfe_price) / sl_dist
            capture_ratio[i] = short_r_net / max(short_mfe_r, 1e-8)

    return targets, direction, raw_r, hold_bars, exit_type, gross_r, cost_r_out, capture_ratio


# ─────────────────────────────────────────────────────────────────────────────
# Regime and Stability Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_run_lengths(direction: np.ndarray) -> np.ndarray:
    sig = direction[direction != 0]
    if len(sig) == 0:
        return np.array([], dtype=np.int32)

    runs = []
    run_len = 1
    for i in range(1, len(sig)):
        if sig[i] == sig[i - 1]:
            run_len += 1
        else:
            runs.append(run_len)
            run_len = 1
    runs.append(run_len)
    return np.asarray(runs, dtype=np.int32)


def _sign_flip_rate(direction: np.ndarray) -> float:
    sig = direction[direction != 0]
    if len(sig) < 2:
        return np.nan
    return float(np.mean(sig[1:] != sig[:-1]))


def _neighbor_agreement(direction: np.ndarray) -> float:
    sig = direction[direction != 0]
    if len(sig) < 2:
        return np.nan
    return float(np.mean(sig[1:] == sig[:-1]))


def _target_bins(targets_abs: np.ndarray) -> dict:
    if len(targets_abs) == 0:
        return {
            "pct_005_015": np.nan,
            "pct_015_030": np.nan,
            "pct_030_050": np.nan,
            "pct_gt_050": np.nan,
        }

    return {
        "pct_005_015": float(np.mean((targets_abs >= 0.05) & (targets_abs < 0.15))),
        "pct_015_030": float(np.mean((targets_abs >= 0.15) & (targets_abs < 0.30))),
        "pct_030_050": float(np.mean((targets_abs >= 0.30) & (targets_abs < 0.50))),
        "pct_gt_050": float(np.mean(targets_abs >= 0.50)),
    }


def _vol_regime_bucket(atr_arr: np.ndarray, n_buckets: int = 5) -> np.ndarray:
    q = np.linspace(0, 1, n_buckets + 1)
    edges = np.quantile(atr_arr, q)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.digitize(atr_arr, edges[1:-1], right=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (minimal — just OHLC + ATR, no feature engineering needed)
# ─────────────────────────────────────────────────────────────────────────────

def _load_ohlc_with_atr(filepath: str) -> pd.DataFrame:
    """Load CSV, sort by time, and compute ATR."""
    df = pd.read_csv(filepath)

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Time sorting (crucial for multi-asset consistency)
    time_col = next((c for c in df.columns if c.lower() in ("date", "datetime", "timestamp")), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
    
    df = df.sort_index()
    
    for required in ("open", "high", "low", "close"):
        if required not in df.columns:
            raise ValueError(f"CSV '{filepath}' missing required column: {required}")

    high_low   = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close  = (df["low"]  - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"]  = true_range.rolling(config.ATR_PERIOD).mean()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "  N/A"
    return f"{100 * num / denom:5.1f}%"


def _fmt(val: float, decimals: int = 4) -> str:
    if np.isnan(val):
        return "  N/A"
    return f"{val:+.{decimals}f}"


def audit_one_file(filepath: str) -> dict:
    """Run oracle diagnostics on a single CSV. Returns summary dict."""
    df = _load_ohlc_with_atr(filepath)

    targets, direction, raw_r, hold_bars, exit_type, gross_r, cost_r_arr, capture_ratio = _generate_targets_diagnostic(
        df["open"].values,
        df["high"].values,
        df["low"].values,
        df["close"].values,
        df["atr"].values,
        max_hold=config.ORACLE_MAX_HOLD,
        fee_per_side=config.FEE_PER_SIDE,
        slippage=config.SLIPPAGE,
        sl_atr_mult=config.ORACLE_SL_ATR_MULT,
        tp_atr_mult=config.ORACLE_TP_ATR_MULT,
        enable_trailing=config.ORACLE_ENABLE_TRAILING,
        trail_atr_mult=config.ORACLE_TRAIL_ATR_MULT,
        saturation_factor=config.SATURATION_FACTOR,
        mae_penalty=config.MAE_PENALTY,
    )

    # ── verify targets match the production oracle exactly ───────────────
    prod_targets = generate_targets(
        df["open"].values,
        df["high"].values,
        df["low"].values,
        df["close"].values,
        df["atr"].values,
        max_hold=config.ORACLE_MAX_HOLD,
        fee_per_side=config.FEE_PER_SIDE,
        slippage=config.SLIPPAGE,
        sl_atr_mult=config.ORACLE_SL_ATR_MULT,
        tp_atr_mult=config.ORACLE_TP_ATR_MULT,
        enable_trailing=config.ORACLE_ENABLE_TRAILING,
        trail_atr_mult=config.ORACLE_TRAIL_ATR_MULT,
        saturation_factor=config.SATURATION_FACTOR,
        mae_penalty=config.MAE_PENALTY,
    )
    max_diff = np.max(np.abs(targets - prod_targets))
    if max_diff > 1e-6:
        print(f"  ⚠️  WARNING: diagnostic vs production max diff = {max_diff:.2e}")
    else:
        print(f"  ✓ Diagnostic targets match production oracle (max diff = {max_diff:.2e})")

    # ── decompose ────────────────────────────────────────────────────────
    valid_len = len(targets) - config.ORACLE_MAX_HOLD
    targets   = targets[:valid_len]
    direction = direction[:valid_len]
    raw_r     = raw_r[:valid_len]
    hold_bars = hold_bars[:valid_len]
    exit_type = exit_type[:valid_len]
    gross_r   = gross_r[:valid_len]
    cost_r_arr = cost_r_arr[:valid_len]
    capture_ratio = capture_ratio[:valid_len]

    n_total = len(targets)
    is_long  = direction == 1
    is_short = direction == -1
    is_flat  = direction == 0
    is_signal = direction != 0

    n_long  = int(is_long.sum())
    n_short = int(is_short.sum())
    n_flat  = int(is_flat.sum())

    # Winning = the target has beneficial direction (positive for long, negative for short)
    long_targets  = targets[is_long]
    short_targets = targets[is_short]

    long_raw_r    = raw_r[is_long]
    short_raw_r   = np.abs(raw_r[is_short])   # make positive for "winning" comparison

    long_hold     = hold_bars[is_long]
    short_hold    = hold_bars[is_short]
    long_exit     = exit_type[is_long]
    short_exit    = exit_type[is_short]

    # Pre-training quality & stability stats
    signal_targets_abs = np.abs(targets[is_signal])
    signal_raw_r_abs = np.abs(raw_r[is_signal])

    run_lengths = _compute_run_lengths(direction)
    flip_rate = _sign_flip_rate(direction)
    neighbor_agree = _neighbor_agreement(direction)
    target_bin_stats = _target_bins(signal_targets_abs)

    weak_signal_share = float(np.mean(signal_targets_abs < 0.15)) if len(signal_targets_abs) else np.nan
    strong_signal_share = float(np.mean(signal_targets_abs >= 0.30)) if len(signal_targets_abs) else np.nan
    very_strong_signal_share = float(np.mean(signal_targets_abs >= 0.50)) if len(signal_targets_abs) else np.nan

    avg_run_length = float(np.mean(run_lengths)) if len(run_lengths) else np.nan
    median_run_length = float(np.median(run_lengths)) if len(run_lengths) else np.nan
    p90_run_length = float(np.percentile(run_lengths, 90)) if len(run_lengths) else np.nan

    signal_cost_r = cost_r_arr[is_signal]
    cost_buffer_ratio = signal_raw_r_abs / np.maximum(signal_cost_r, 1e-8)

    avg_cost_buffer = float(np.mean(cost_buffer_ratio)) if len(cost_buffer_ratio) else np.nan
    p25_cost_buffer = float(np.percentile(cost_buffer_ratio, 25)) if len(cost_buffer_ratio) else np.nan

    signal_capture = capture_ratio[is_signal]
    avg_capture_ratio = float(np.mean(signal_capture)) if len(signal_capture) else np.nan

    # Volatility-bucket diagnostics
    atr_trim = df["atr"].values[:valid_len]
    bucket_id = _vol_regime_bucket(atr_trim, n_buckets=5)
    bucket_stats = {}

    for b in range(5):
        m = bucket_id == b
        sig_m = m & is_signal
        long_m = m & is_long
        short_m = m & is_short

        bucket_stats[f"bucket_{b+1}_signal_pct"] = float(np.mean(sig_m)) if np.sum(m) else np.nan
        bucket_stats[f"bucket_{b+1}_long_pct"] = float(np.mean(long_m)) if np.sum(m) else np.nan
        bucket_stats[f"bucket_{b+1}_short_pct"] = float(np.mean(short_m)) if np.sum(m) else np.nan
        bucket_stats[f"bucket_{b+1}_avg_abs_r"] = float(np.mean(np.abs(raw_r[sig_m]))) if np.sum(sig_m) else np.nan
        bucket_stats[f"bucket_{b+1}_avg_abs_target"] = float(np.mean(np.abs(targets[sig_m]))) if np.sum(sig_m) else np.nan

    # R-multiple thresholds for quality buckets
    r_thresholds = [0.0, 0.5, 1.0, 2.0, 5.0]

    stats = {
        "file": os.path.basename(filepath),
        "n_total": n_total,
        "n_long": n_long,
        "n_short": n_short,
        "n_flat": n_flat,
        "long_pct": n_long / max(1, n_total),
        "short_pct": n_short / max(1, n_total),
        "flat_pct": n_flat / max(1, n_total),
        "signal_pct": (n_long + n_short) / max(1, n_total),
        # Target magnitudes
        "avg_long_target": float(np.mean(long_targets)) if n_long > 0 else np.nan,
        "avg_short_target": float(np.mean(np.abs(short_targets))) if n_short > 0 else np.nan,
        "med_long_target": float(np.median(long_targets)) if n_long > 0 else np.nan,
        "med_short_target": float(np.median(np.abs(short_targets))) if n_short > 0 else np.nan,
        # Raw R-multiples (pre-tanh, pre-cost)
        "avg_long_r": float(np.mean(long_raw_r)) if n_long > 0 else np.nan,
        "avg_short_r": float(np.mean(short_raw_r)) if n_short > 0 else np.nan,
        "med_long_r": float(np.median(long_raw_r)) if n_long > 0 else np.nan,
        "med_short_r": float(np.median(short_raw_r)) if n_short > 0 else np.nan,
        "p75_long_r": float(np.percentile(long_raw_r, 75)) if n_long > 0 else np.nan,
        "p75_short_r": float(np.percentile(short_raw_r, 75)) if n_short > 0 else np.nan,
        "p95_long_r": float(np.percentile(long_raw_r, 95)) if n_long > 0 else np.nan,
        "p95_short_r": float(np.percentile(short_raw_r, 95)) if n_short > 0 else np.nan,
        # Holding periods
        "avg_long_hold": float(np.mean(long_hold)) if n_long > 0 else np.nan,
        "avg_short_hold": float(np.mean(short_hold)) if n_short > 0 else np.nan,
        # Exit type breakdown
        "long_sl_pct": float(np.mean(long_exit == 1)) if n_long > 0 else np.nan,
        "long_tp_pct": float(np.mean(long_exit == 2)) if n_long > 0 else np.nan,
        "long_trail_pct": float(np.mean(long_exit == 3)) if n_long > 0 else np.nan,
        "long_time_pct": float(np.mean(long_exit == 4)) if n_long > 0 else np.nan,
        "short_sl_pct": float(np.mean(short_exit == 1)) if n_short > 0 else np.nan,
        "short_tp_pct": float(np.mean(short_exit == 2)) if n_short > 0 else np.nan,
        "short_trail_pct": float(np.mean(short_exit == 3)) if n_short > 0 else np.nan,
        "short_time_pct": float(np.mean(short_exit == 4)) if n_short > 0 else np.nan,
        # Pre-training confidence & quality stats
        "flip_rate": flip_rate,
        "neighbor_agreement": neighbor_agree,
        "avg_run_length": avg_run_length,
        "median_run_length": median_run_length,
        "p90_run_length": p90_run_length,
        "weak_signal_share": weak_signal_share,
        "strong_signal_share": strong_signal_share,
        "very_strong_signal_share": very_strong_signal_share,
        "avg_cost_buffer": avg_cost_buffer,
        "p25_cost_buffer": p25_cost_buffer,
        "avg_capture_ratio": avg_capture_ratio,
    }
    stats.update(target_bin_stats)
    stats.update(bucket_stats)
    return stats


def _print_report(stats: dict) -> None:
    """Pretty-print one file's diagnostic report."""
    s = stats
    w = 60  # report width

    print("=" * w)
    print(f"  FILE: {s['file']}")
    print("=" * w)
    print(f"  Total bars (after ATR warmup + oracle trim) : {s['n_total']:,}")
    print(f"  Signals (long + short)                      : {s['n_long'] + s['n_short']:,}  ({_pct(s['n_long'] + s['n_short'], s['n_total'])})")
    print(f"    Long  signals                             : {s['n_long']:,}  ({_pct(s['n_long'], s['n_total'])})")
    print(f"    Short signals                             : {s['n_short']:,}  ({_pct(s['n_short'], s['n_total'])})")
    print(f"  Flat (no trade)                             : {s['n_flat']:,}  ({_pct(s['n_flat'], s['n_total'])})")
    print()

    print("  ── Target Magnitudes (post-tanh, what the model sees) ──")
    print(f"    Avg  long  target    : {_fmt(s['avg_long_target'])}")
    print(f"    Med  long  target    : {_fmt(s['med_long_target'])}")
    print(f"    Avg  short target    : {_fmt(s['avg_short_target'])}")
    print(f"    Med  short target    : {_fmt(s['med_short_target'])}")
    print()

    print("  ── Raw R-Multiples (net of costs, pre-tanh) ──")
    print(f"    Avg  long  R    : {_fmt(s['avg_long_r'])}")
    print(f"    Med  long  R    : {_fmt(s['med_long_r'])}")
    print(f"    P75  long  R    : {_fmt(s['p75_long_r'])}")
    print(f"    P95  long  R    : {_fmt(s['p95_long_r'])}")
    print(f"    Avg  short R    : {_fmt(s['avg_short_r'])}")
    print(f"    Med  short R    : {_fmt(s['med_short_r'])}")
    print(f"    P75  short R    : {_fmt(s['p75_short_r'])}")
    print(f"    P95  short R    : {_fmt(s['p95_short_r'])}")
    print()

    print("  ── Holding Period (bars) ──")
    print(f"    Avg  long  hold : {_fmt(s['avg_long_hold'], 1)} bars")
    print(f"    Avg  short hold : {_fmt(s['avg_short_hold'], 1)} bars")
    print()

    print("  ── Exit Type Breakdown ──")
    print(f"    LONG  — stop-loss : {s['long_sl_pct']*100:5.1f}%"
          f"  | take-profit : {s['long_tp_pct']*100:5.1f}%"
          f"  | trailing-stop : {s['long_trail_pct']*100:5.1f}%"
          f"  | time-exit : {s['long_time_pct']*100:5.1f}%")
    print(f"    SHORT — stop-loss : {s['short_sl_pct']*100:5.1f}%"
          f"  | take-profit : {s['short_tp_pct']*100:5.1f}%"
          f"  | trailing-stop : {s['short_trail_pct']*100:5.1f}%"
          f"  | time-exit : {s['short_time_pct']*100:5.1f}%")
    print()

    print("Pre-training quality checks")
    print(f"  Flip rate               : {s['flip_rate']:.3f}")
    print(f"  Neighbor agreement      : {s['neighbor_agreement']:.3f}")
    print(f"  Avg / Med run length    : {s['avg_run_length']:.2f} / {s['median_run_length']:.2f}")
    print(f"  P90 run length          : {s['p90_run_length']:.2f}")
    print(f"  Weak signal share       : {s['weak_signal_share']*100:5.1f}%")
    print(f"  Strong signal share     : {s['strong_signal_share']*100:5.1f}%")
    print(f"  Very strong share       : {s['very_strong_signal_share']*100:5.1f}%")
    print(f"  Avg cost buffer         : {s['avg_cost_buffer']:.2f}x")
    print(f"  P25 cost buffer         : {s['p25_cost_buffer']:.2f}x")
    print(f"  |t| bins 0.05-0.15      : {s['pct_005_015']*100:5.1f}%")
    print(f"  |t| bins 0.15-0.30      : {s['pct_015_030']*100:5.1f}%")
    print(f"  |t| bins 0.30-0.50      : {s['pct_030_050']*100:5.1f}%")
    print(f"  |t| bins >0.50          : {s['pct_gt_050']*100:5.1f}%")
    print()

    print("  ── Volatility Bucket Diagnostics (ATR regimes 1-5) ──")
    for b in range(5):
        print(f"    Bucket {b+1} — Sig: {s[f'bucket_{b+1}_signal_pct']*100:4.1f}%"
              f" | L: {s[f'bucket_{b+1}_long_pct']*100:4.1f}%"
              f" | S: {s[f'bucket_{b+1}_short_pct']*100:4.1f}%"
              f" | Abs R: {_fmt(s[f'bucket_{b+1}_avg_abs_r'], 2)}"
              f" | Abs Target: {_fmt(s[f'bucket_{b+1}_avg_abs_target'], 3)}")
    print()


def _print_aggregate(all_stats: list[dict]) -> None:
    """Aggregate and print cross-file summary."""
    total_bars  = sum(s["n_total"] for s in all_stats)
    total_long  = sum(s["n_long"]  for s in all_stats)
    total_short = sum(s["n_short"] for s in all_stats)
    total_flat  = sum(s["n_flat"]  for s in all_stats)

    print("\n" + "█" * 60)
    print("  AGGREGATE SUMMARY")
    print("█" * 60)
    print(f"  Files processed  : {len(all_stats)}")
    print(f"  Total bars       : {total_bars:,}")
    print(f"  Total signals    : {total_long + total_short:,}  ({_pct(total_long + total_short, total_bars)})")
    print(f"    Long  signals  : {total_long:,}   ({_pct(total_long, total_bars)})")
    print(f"    Short signals  : {total_short:,}   ({_pct(total_short, total_bars)})")
    print(f"  Flat bars        : {total_flat:,}   ({_pct(total_flat, total_bars)})")
    print()

    # weighted averages
    if total_long > 0:
        wavg_long_r = sum(s["avg_long_r"] * s["n_long"] for s in all_stats if not np.isnan(s["avg_long_r"])) / total_long
        wavg_long_t = sum(s["avg_long_target"] * s["n_long"] for s in all_stats if not np.isnan(s["avg_long_target"])) / total_long
        wavg_long_h = sum(s["avg_long_hold"] * s["n_long"] for s in all_stats if not np.isnan(s["avg_long_hold"])) / total_long
        wavg_long_sl = sum(s["long_sl_pct"] * s["n_long"] for s in all_stats if not np.isnan(s["long_sl_pct"])) / total_long
        wavg_long_tp = sum(s["long_tp_pct"] * s["n_long"] for s in all_stats if not np.isnan(s["long_tp_pct"])) / total_long
        wavg_long_ts = sum(s["long_trail_pct"] * s["n_long"] for s in all_stats if not np.isnan(s["long_trail_pct"])) / total_long
        wavg_long_time = sum(s["long_time_pct"] * s["n_long"] for s in all_stats if not np.isnan(s["long_time_pct"])) / total_long

        print(f"  Weighted avg long  R-mult    : {_fmt(wavg_long_r)}")
        print(f"  Weighted avg long  target    : {_fmt(wavg_long_t)}")
        print(f"  Weighted avg long  hold      : {_fmt(wavg_long_h, 1)} bars")
        print(f"  Weighted avg long exits      : SL={wavg_long_sl*100:.1f}% | TP={wavg_long_tp*100:.1f}% | TS={wavg_long_ts*100:.1f}% | Time={wavg_long_time*100:.1f}%")

    if total_short > 0:
        wavg_short_r = sum(s["avg_short_r"] * s["n_short"] for s in all_stats if not np.isnan(s["avg_short_r"])) / total_short
        wavg_short_t = sum(s["avg_short_target"] * s["n_short"] for s in all_stats if not np.isnan(s["avg_short_target"])) / total_short
        wavg_short_h = sum(s["avg_short_hold"] * s["n_short"] for s in all_stats if not np.isnan(s["avg_short_hold"])) / total_short
        wavg_short_sl = sum(s["short_sl_pct"] * s["n_short"] for s in all_stats if not np.isnan(s["short_sl_pct"])) / total_short
        wavg_short_tp = sum(s["short_tp_pct"] * s["n_short"] for s in all_stats if not np.isnan(s["short_tp_pct"])) / total_short
        wavg_short_ts = sum(s["short_trail_pct"] * s["n_short"] for s in all_stats if not np.isnan(s["short_trail_pct"])) / total_short
        wavg_short_time = sum(s["short_time_pct"] * s["n_short"] for s in all_stats if not np.isnan(s["short_time_pct"])) / total_short

        print(f"  Weighted avg short R-mult    : {_fmt(wavg_short_r)}")
        print(f"  Weighted avg short target    : {_fmt(wavg_short_t)}")
        print(f"  Weighted avg short hold      : {_fmt(wavg_short_h, 1)} bars")
        print(f"  Weighted avg short exits     : SL={wavg_short_sl*100:.1f}% | TP={wavg_short_tp*100:.1f}% | TS={wavg_short_ts*100:.1f}% | Time={wavg_short_time*100:.1f}%")

    total_signals = total_long + total_short
    if total_signals > 0:
        wavg_run_length = sum(s["avg_run_length"] * (s["n_long"] + s["n_short"]) for s in all_stats if not np.isnan(s["avg_run_length"])) / total_signals
        wavg_flip_rate = sum(s["flip_rate"] * (s["n_long"] + s["n_short"]) for s in all_stats if not np.isnan(s["flip_rate"])) / total_signals
        wavg_neighbor_agreement = sum(s["neighbor_agreement"] * (s["n_long"] + s["n_short"]) for s in all_stats if not np.isnan(s["neighbor_agreement"])) / total_signals
        wavg_cost_buffer = sum(s["avg_cost_buffer"] * (s["n_long"] + s["n_short"]) for s in all_stats if not np.isnan(s["avg_cost_buffer"])) / total_signals
        wavg_capture_ratio = sum(s["avg_capture_ratio"] * (s["n_long"] + s["n_short"]) for s in all_stats if not np.isnan(s["avg_capture_ratio"])) / total_signals
        wavg_weak_signal = sum(s["weak_signal_share"] * (s["n_long"] + s["n_short"]) for s in all_stats if not np.isnan(s["weak_signal_share"])) / total_signals
        wavg_strong_signal = sum(s["strong_signal_share"] * (s["n_long"] + s["n_short"]) for s in all_stats if not np.isnan(s["strong_signal_share"])) / total_signals
        wavg_very_strong_signal = sum(s["very_strong_signal_share"] * (s["n_long"] + s["n_short"]) for s in all_stats if not np.isnan(s["very_strong_signal_share"])) / total_signals

        print()
        print(f"  Weighted avg run length      : {_fmt(wavg_run_length, 1)} bars")
        print(f"  Weighted avg flip rate       : {_fmt(wavg_flip_rate, 3)}")
        print(f"  Weighted avg neighbor agree  : {_fmt(wavg_neighbor_agreement, 3)}")
        print(f"  Weighted avg cost buffer     : {_fmt(wavg_cost_buffer, 2)}x")
        print(f"  Weighted avg capture ratio   : {_fmt(wavg_capture_ratio, 3)}")
        print(f"  Weighted avg weak / strong   : Weak={wavg_weak_signal*100:.1f}% | Strong={wavg_strong_signal*100:.1f}% | V_Strong={wavg_very_strong_signal*100:.1f}%")

    # ── Per-volatility-bucket aggregate ─────────────────────────────────
    if total_signals > 0:
        print()
        print("  ── Aggregate Volatility Bucket Diagnostics (ATR regimes 1-5) ──")
        for b in range(5):
            key_sig = f"bucket_{b+1}_signal_pct"
            key_r   = f"bucket_{b+1}_avg_abs_r"
            key_t   = f"bucket_{b+1}_avg_abs_target"
            wavg_sig = sum(
                s[key_sig] * (s["n_long"] + s["n_short"])
                for s in all_stats if not np.isnan(s.get(key_sig, np.nan))
            ) / total_signals
            valid_r   = [s for s in all_stats if not np.isnan(s.get(key_r, np.nan)) and (s["n_long"] + s["n_short"]) > 0]
            wavg_r    = sum(s[key_r] * (s["n_long"] + s["n_short"]) for s in valid_r) / max(1, sum(s["n_long"] + s["n_short"] for s in valid_r)) if valid_r else np.nan
            valid_t   = [s for s in all_stats if not np.isnan(s.get(key_t, np.nan)) and (s["n_long"] + s["n_short"]) > 0]
            wavg_t    = sum(s[key_t] * (s["n_long"] + s["n_short"]) for s in valid_t) / max(1, sum(s["n_long"] + s["n_short"] for s in valid_t)) if valid_t else np.nan
            print(f"    Bucket {b+1} — Signal%: {wavg_sig*100:5.1f}%"
                  f" | Avg |R|: {_fmt(wavg_r, 2)}"
                  f" | Avg |Target|: {_fmt(wavg_t, 3)}")

    print()
    print("  Long / Short ratio : ", end="")
    if total_short > 0:
        print(f"{total_long / total_short:.2f}")
    else:
        print("∞ (no short signals)")
    print("█" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # allow CLI override, otherwise use config.DATA_FILE
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = config.DATA_FILE if isinstance(config.DATA_FILE, list) else [config.DATA_FILE]

    print("\n" + "─" * 60)
    print("  Oracle 5.0 Target Audit (Multi-Asset)")
    print("─" * 60)
    print(f"  Assets to process : {len(files)}")
    print(f"  ATR_PERIOD       = {config.ATR_PERIOD}")
    print(f"  ORACLE_MAX_HOLD  = {config.ORACLE_MAX_HOLD}")
    print(f"  FEE_PER_SIDE     = {config.FEE_PER_SIDE}")
    print(f"  SLIPPAGE         = {config.SLIPPAGE}")
    print(f"  SL_ATR_MULT      = {config.ORACLE_SL_ATR_MULT}")
    print(f"  TP_ATR_MULT      = {config.ORACLE_TP_ATR_MULT}")
    print(f"  TRAIL_ATR_MULT   = {config.ORACLE_TRAIL_ATR_MULT}")
    print(f"  ENABLE_TRAILING  = {config.ORACLE_ENABLE_TRAILING}")
    print("─" * 60 + "\n")

    all_stats = []
    # If there are many files, suppress the detailed per-file print to keep output readable
    verbose = len(files) <= 3

    for f in files:
        if not os.path.exists(f):
            print(f"  ⚠️  File not found: {f}  — skipping.")
            continue
        
        if verbose:
            print(f"  Processing: {f}")
        else:
            print(f"  Processing: {os.path.basename(f):<35s} ... ", end="", flush=True)

        try:
            stats = audit_one_file(f)
            all_stats.append(stats)
            if verbose:
                _print_report(stats)
            else:
                sig_pct = (stats['n_long'] + stats['n_short']) / max(1, stats['n_total'])
                print(f"OK ({stats['n_total']:,} bars, {sig_pct*100:.1f}% signals)")
        except Exception as e:
            print(f"FAILED: {e}")

    if len(all_stats) > 1:
        _print_aggregate(all_stats)
    elif len(all_stats) == 1:
        if not verbose:
            _print_report(all_stats[0])
        print("  (Single file — no aggregate needed.)")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
