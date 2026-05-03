#!/usr/bin/env python3
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
    open_arr, high_arr, low_arr, close_arr, atr_arr,
    max_hold,
    fee_per_side=0.001,
    slippage=0.0005,
    atr_mult=3.5,
    saturation_factor=2.5,
    mae_penalty=0.20,
):
    """
    Same logic as oracle.generate_targets, but returns extra diagnostic arrays:
      - direction : +1 long, -1 short, 0 flat
      - raw_r     : the net R-multiple *before* tanh squash (signed)
      - hold_bars : bars held until exit (long or short, whichever won)
      - exit_type : 0 = flat, 1 = stop-hit, 2 = trail-stop, 3 = time-exit
    """
    n = len(close_arr)
    targets    = np.zeros(n, dtype=np.float32)
    direction  = np.zeros(n, dtype=np.int8)
    raw_r      = np.zeros(n, dtype=np.float64)
    hold_bars  = np.zeros(n, dtype=np.int32)
    exit_type  = np.zeros(n, dtype=np.int8)     # 0=flat, 1=init-stop, 2=trail, 3=time

    total_cost_pct = (fee_per_side + slippage) * 2.0
    stop_distances = atr_arr * atr_mult

    for i in range(n - max_hold):
        entry_price = close_arr[i]
        vol_dist = stop_distances[i]

        if vol_dist <= 0 or entry_price <= 0:
            continue

        vol_pct = max(vol_dist / entry_price, 0.0001)  # safety floor only — prevents /0

        # Skip structurally untradeable regimes (costs > 1R means no positive EV possible)
        if total_cost_pct / vol_pct > 1.0:
            continue

        # ──────── LONG ────────
        stop_level = entry_price - vol_dist
        peak_price = entry_price
        max_risk_consumed = 0.0
        long_pnl_pct = 0.0
        long_hold = max_hold - 1
        long_exit_type = 3  # default: time-exit

        for k in range(1, max_hold):
            idx = i + k
            c_open  = open_arr[idx]
            c_low   = low_arr[idx]
            c_high  = high_arr[idx]
            c_close = close_arr[idx]

            if c_low <= stop_level:
                exit_price = min(c_open, stop_level)
                long_pnl_pct = (exit_price - entry_price) / entry_price
                max_risk_consumed = 1.0
                long_hold = k
                long_exit_type = 1
                break

            # 2. Trail Stop + intrabar re-check
            if c_high > peak_price:
                peak_price = c_high
                new_stop = peak_price - vol_dist
                if new_stop > stop_level:
                    stop_level = new_stop
                    if c_low <= stop_level:
                        exit_price = stop_level
                        long_pnl_pct = (exit_price - entry_price) / entry_price
                        max_risk_consumed = 1.0
                        long_hold = k
                        long_exit_type = 2
                        break

            # 3. Update Risk Consumption (proximity to current trailing stop)
            current_risk_consumed = min(1.0, max(0.0, (vol_dist - (c_low - stop_level)) / vol_dist))
            if current_risk_consumed > max_risk_consumed:
                max_risk_consumed = current_risk_consumed

            if k == max_hold - 1:
                long_pnl_pct = (c_close - entry_price) / entry_price

        # ──────── SHORT ────────
        stop_level_short = entry_price + vol_dist
        trough_price = entry_price
        max_risk_consumed_short = 0.0
        short_pnl_pct = 0.0
        short_hold = max_hold - 1
        short_exit_type = 3

        for k in range(1, max_hold):
            idx = i + k
            c_open  = open_arr[idx]
            c_high  = high_arr[idx]
            c_low   = low_arr[idx]
            c_close = close_arr[idx]

            if c_high >= stop_level_short:
                exit_price = max(c_open, stop_level_short)
                short_pnl_pct = (entry_price - exit_price) / entry_price
                max_risk_consumed_short = 1.0
                short_hold = k
                short_exit_type = 1
                break

            # 2. Trail Stop + intrabar re-check
            if c_low < trough_price:
                trough_price = c_low
                new_stop = trough_price + vol_dist
                if new_stop < stop_level_short:
                    stop_level_short = new_stop
                    if c_high >= stop_level_short:
                        exit_price = stop_level_short
                        short_pnl_pct = (entry_price - exit_price) / entry_price
                        max_risk_consumed_short = 1.0
                        short_hold = k
                        short_exit_type = 2
                        break

            # 3. Update Risk Consumption (proximity to current trailing stop)
            current_risk_consumed = min(1.0, max(0.0, (vol_dist - (stop_level_short - c_high)) / vol_dist))
            if current_risk_consumed > max_risk_consumed_short:
                max_risk_consumed_short = current_risk_consumed

            if k == max_hold - 1:
                short_pnl_pct = (entry_price - c_close) / entry_price

        # ──────── SCORING (identical to oracle.py) ────────
        long_r  = long_pnl_pct  / vol_pct
        short_r = short_pnl_pct / vol_pct

        max_risk_consumed       = min(max(max_risk_consumed, 0.0), 1.0)
        max_risk_consumed_short = min(max(max_risk_consumed_short, 0.0), 1.0)

        if long_r > 0:
            long_r *= (1.0 - (mae_penalty * max_risk_consumed))
        if short_r > 0:
            short_r *= (1.0 - (mae_penalty * max_risk_consumed_short))

        cost_r = total_cost_pct / vol_pct
        long_r_net  = long_r  - cost_r
        short_r_net = short_r - cost_r

        if long_r_net > 0 and long_r_net > short_r_net:
            targets[i]   = np.tanh(long_r_net / saturation_factor)
            direction[i] = 1
            raw_r[i]     = long_r_net
            hold_bars[i] = long_hold
            exit_type[i] = long_exit_type
        elif short_r_net > 0 and short_r_net > long_r_net:
            targets[i]   = -np.tanh(short_r_net / saturation_factor)
            direction[i] = -1
            raw_r[i]     = -short_r_net    # keep sign convention: negative = short
            hold_bars[i] = short_hold
            exit_type[i] = short_exit_type

    return targets, direction, raw_r, hold_bars, exit_type


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (minimal — just OHLC + ATR, no feature engineering needed)
# ─────────────────────────────────────────────────────────────────────────────

def _load_ohlc_with_atr(filepath: str) -> pd.DataFrame:
    """Load CSV and compute ATR — identical to train._build_features ATR block."""
    df = pd.read_csv(filepath)

    # normalise column names
    df.columns = [c.strip().lower() for c in df.columns]
    for required in ("open", "high", "low", "close"):
        if required not in df.columns:
            raise ValueError(f"CSV missing required column: {required}")

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

    targets, direction, raw_r, hold_bars, exit_type = _generate_targets_diagnostic(
        df["open"].values,
        df["high"].values,
        df["low"].values,
        df["close"].values,
        df["atr"].values,
        max_hold=config.ORACLE_MAX_HOLD,
        fee_per_side=config.FEE_PER_SIDE,
        slippage=config.SLIPPAGE,
        atr_mult=config.ATR_MULT,
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
        atr_mult=config.ATR_MULT,
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

    n_total = len(targets)
    is_long  = direction == 1
    is_short = direction == -1
    is_flat  = direction == 0

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

    # For oracle targets, all long signals have target > 0 (winning by definition —
    # the oracle only labels a direction if net R > 0). But we can still decompose
    # by magnitude to see quality distribution.

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
        "long_stop_pct": float(np.mean(long_exit == 1)) if n_long > 0 else np.nan,
        "long_trail_pct": float(np.mean(long_exit == 2)) if n_long > 0 else np.nan,
        "long_time_pct": float(np.mean(long_exit == 3)) if n_long > 0 else np.nan,
        "short_stop_pct": float(np.mean(short_exit == 1)) if n_short > 0 else np.nan,
        "short_trail_pct": float(np.mean(short_exit == 2)) if n_short > 0 else np.nan,
        "short_time_pct": float(np.mean(short_exit == 3)) if n_short > 0 else np.nan,
    }
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
    print(f"    LONG  — initial stop : {_pct(int(s.get('long_stop_pct', 0) * s['n_long']), s['n_long'])}"
          f"  | trail stop : {_pct(int(s.get('long_trail_pct', 0) * s['n_long']), s['n_long'])}"
          f"  | time exit : {_pct(int(s.get('long_time_pct', 0) * s['n_long']), s['n_long'])}")
    print(f"    SHORT — initial stop : {_pct(int(s.get('short_stop_pct', 0) * s['n_short']), s['n_short'])}"
          f"  | trail stop : {_pct(int(s.get('short_trail_pct', 0) * s['n_short']), s['n_short'])}"
          f"  | time exit : {_pct(int(s.get('short_time_pct', 0) * s['n_short']), s['n_short'])}")
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
        print(f"  Weighted avg long  R-mult    : {_fmt(wavg_long_r)}")
        print(f"  Weighted avg long  target    : {_fmt(wavg_long_t)}")
        print(f"  Weighted avg long  hold      : {_fmt(wavg_long_h, 1)} bars")

    if total_short > 0:
        wavg_short_r = sum(s["avg_short_r"] * s["n_short"] for s in all_stats if not np.isnan(s["avg_short_r"])) / total_short
        wavg_short_t = sum(s["avg_short_target"] * s["n_short"] for s in all_stats if not np.isnan(s["avg_short_target"])) / total_short
        wavg_short_h = sum(s["avg_short_hold"] * s["n_short"] for s in all_stats if not np.isnan(s["avg_short_hold"])) / total_short
        print(f"  Weighted avg short R-mult    : {_fmt(wavg_short_r)}")
        print(f"  Weighted avg short target    : {_fmt(wavg_short_t)}")
        print(f"  Weighted avg short hold      : {_fmt(wavg_short_h, 1)} bars")

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
    print("  Oracle 4.1 Target Audit")
    print("─" * 60)
    print(f"  ATR_PERIOD       = {config.ATR_PERIOD}")
    print(f"  ORACLE_MAX_HOLD  = {config.ORACLE_MAX_HOLD}")
    print(f"  FEE_PER_SIDE     = {config.FEE_PER_SIDE}")
    print(f"  SLIPPAGE         = {config.SLIPPAGE}")
    print(f"  ATR_MULT         = {config.ATR_MULT}")
    print(f"  SATURATION_FACTOR= {config.SATURATION_FACTOR}")
    print(f"  MAE_PENALTY      = {config.MAE_PENALTY}")
    print("─" * 60 + "\n")

    all_stats = []
    for f in files:
        if not os.path.exists(f):
            print(f"  ⚠️  File not found: {f}  — skipping.")
            continue
        print(f"  Processing: {f}")
        stats = audit_one_file(f)
        _print_report(stats)
        all_stats.append(stats)

    if len(all_stats) > 1:
        _print_aggregate(all_stats)
    elif len(all_stats) == 1:
        print("  (Single file — no aggregate needed.)")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
