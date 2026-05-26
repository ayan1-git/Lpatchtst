#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from config import *

def load_config(data_dir_override=None):
    import importlib
    cfg = importlib.import_module('config')
    if data_dir_override is not None:
        cfg.DATA_DIR = data_dir_override
        files = sorted(glob.glob(os.path.join(cfg.DATA_DIR, '*.csv')))
        if files:
            cfg.DATA_FILE = files
    return cfg


def find_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    for c in df.columns:
        cl = c.lower().strip()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


def load_ohlc_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts_col = find_col(df, ['datetime', 'date', 'time', 'timestamp'])
    open_col = find_col(df, ['open'])
    high_col = find_col(df, ['high'])
    low_col = find_col(df, ['low'])
    close_col = find_col(df, ['close'])
    vol_col = find_col(df, ['volume', 'vol'])

    missing = [name for name, col in [('open', open_col), ('high', high_col), ('low', low_col), ('close', close_col)] if col is None]
    if missing:
        raise ValueError(f'{path}: missing required columns {missing}; columns={list(df.columns)}')

    out = pd.DataFrame()
    if ts_col is not None:
        out['timestamp'] = pd.to_datetime(df[ts_col], errors='coerce')
    else:
        out['timestamp'] = pd.RangeIndex(len(df))
    out['open'] = pd.to_numeric(df[open_col], errors='coerce')
    out['high'] = pd.to_numeric(df[high_col], errors='coerce')
    out['low'] = pd.to_numeric(df[low_col], errors='coerce')
    out['close'] = pd.to_numeric(df[close_col], errors='coerce')
    if vol_col is not None:
        out['volume'] = pd.to_numeric(df[vol_col], errors='coerce')
    return out


def atr_wilder(df: pd.DataFrame, period: int) -> np.ndarray:
    high = df['high'].to_numpy(dtype=np.float64)
    low = df['low'].to_numpy(dtype=np.float64)
    close = df['close'].to_numpy(dtype=np.float64)
    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = np.full_like(tr, np.nan)
    if len(tr) >= period:
        atr[period - 1] = np.nanmean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def generate_targets_py(open_arr, high_arr, low_arr, close_arr, atr_arr, max_hold, fee_per_side=0.001, slippage=0.0005,
                        sl_atr_mult=1.5, tp_atr_mult=3.0, enable_trailing=False, trail_atr_mult=1.5,
                        saturation_factor=2.5, mae_penalty=0.20):
    n = len(close_arr)
    targets = np.zeros(n, dtype=np.float32)
    total_cost_pct = (fee_per_side + slippage) * 2.0
    sl_distances = atr_arr * sl_atr_mult
    tp_distances = atr_arr * tp_atr_mult
    trail_distances = atr_arr * trail_atr_mult

    audit_rows = []
    for i in range(n - max_hold):
        entry_price = close_arr[i]
        sl_dist = sl_distances[i]
        tp_dist = tp_distances[i]
        trail_dist = trail_distances[i]
        reason = 'zero_default'
        long_r_net = np.nan
        short_r_net = np.nan
        if not (sl_dist > 0.0 and tp_dist > 0.0 and entry_price > 0.0) or np.isnan(sl_dist) or np.isnan(tp_dist):
            reason = 'invalid_atr_or_price'
            audit_rows.append((i, targets[i], long_r_net, short_r_net, reason))
            continue

        risk_pct = sl_dist / entry_price
        if not (risk_pct > 0.0) or np.isnan(risk_pct):
            reason = 'invalid_risk_pct'
            audit_rows.append((i, targets[i], long_r_net, short_r_net, reason))
            continue

        cost_r = total_cost_pct / risk_pct
        if cost_r > 1.0:
            reason = 'cost_too_high'
            audit_rows.append((i, targets[i], long_r_net, short_r_net, reason))
            continue

        long_stop = entry_price - sl_dist
        long_tp = entry_price + tp_dist
        peak_price = entry_price
        max_risk_consumed_long = 0.0
        long_pnl_pct = 0.0
        for k in range(1, max_hold):
            idx = i + k
            c_open = open_arr[idx]
            c_high = high_arr[idx]
            c_low = low_arr[idx]
            c_close = close_arr[idx]
            if c_open <= long_stop:
                exit_price = c_open
                long_pnl_pct = (exit_price - entry_price) / entry_price
                max_risk_consumed_long = 1.0
                break
            if c_open >= long_tp:
                exit_price = c_open
                long_pnl_pct = (exit_price - entry_price) / entry_price
                break
            if enable_trailing and c_high > peak_price:
                peak_price = c_high
                new_stop = peak_price - trail_dist
                if new_stop > long_stop:
                    long_stop = new_stop
            hit_sl = c_low <= long_stop
            hit_tp = c_high >= long_tp
            if hit_sl and hit_tp:
                exit_price = long_stop
                long_pnl_pct = (exit_price - entry_price) / entry_price
                max_risk_consumed_long = 1.0
                break
            elif hit_sl:
                exit_price = long_stop
                long_pnl_pct = (exit_price - entry_price) / entry_price
                max_risk_consumed_long = 1.0
                break
            elif hit_tp:
                exit_price = long_tp
                long_pnl_pct = (exit_price - entry_price) / entry_price
                break
            current_risk_consumed = (entry_price - c_low) / sl_dist
            current_risk_consumed = min(1.0, max(0.0, current_risk_consumed))
            max_risk_consumed_long = max(max_risk_consumed_long, current_risk_consumed)
            if k == max_hold - 1:
                long_pnl_pct = (c_close - entry_price) / entry_price

        short_stop = entry_price + sl_dist
        short_tp = entry_price - tp_dist
        trough_price = entry_price
        max_risk_consumed_short = 0.0
        short_pnl_pct = 0.0
        for k in range(1, max_hold):
            idx = i + k
            c_open = open_arr[idx]
            c_high = high_arr[idx]
            c_low = low_arr[idx]
            c_close = close_arr[idx]
            if c_open >= short_stop:
                exit_price = c_open
                short_pnl_pct = (entry_price - exit_price) / entry_price
                max_risk_consumed_short = 1.0
                break
            if c_open <= short_tp:
                exit_price = c_open
                short_pnl_pct = (entry_price - exit_price) / entry_price
                break
            if enable_trailing and c_low < trough_price:
                trough_price = c_low
                new_stop = trough_price + trail_dist
                if new_stop < short_stop:
                    short_stop = new_stop
            hit_sl = c_high >= short_stop
            hit_tp = c_low <= short_tp
            if hit_sl and hit_tp:
                exit_price = short_stop
                short_pnl_pct = (entry_price - exit_price) / entry_price
                max_risk_consumed_short = 1.0
                break
            elif hit_sl:
                exit_price = short_stop
                short_pnl_pct = (entry_price - exit_price) / entry_price
                max_risk_consumed_short = 1.0
                break
            elif hit_tp:
                exit_price = short_tp
                short_pnl_pct = (entry_price - exit_price) / entry_price
                break
            current_risk_consumed = (c_high - entry_price) / sl_dist
            current_risk_consumed = min(1.0, max(0.0, current_risk_consumed))
            max_risk_consumed_short = max(max_risk_consumed_short, current_risk_consumed)
            if k == max_hold - 1:
                short_pnl_pct = (entry_price - c_close) / entry_price

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
            reason = 'long_selected'
        elif short_r_net > 0.0 and short_r_net > long_r_net:
            targets[i] = -np.tanh(short_r_net / saturation_factor)
            reason = 'short_selected'
        else:
            reason = 'no_positive_edge_after_cost'
        audit_rows.append((i, targets[i], long_r_net, short_r_net, reason))
    audit = pd.DataFrame(audit_rows, columns=['idx', 'target', 'long_r_net', 'short_r_net', 'reason'])
    return targets, audit


def basic_ohlc_checks(df: pd.DataFrame) -> Dict:
    out = {}
    out['rows'] = int(len(df))
    out['null_counts'] = {c: int(df[c].isna().sum()) for c in df.columns}
    out['duplicated_rows'] = int(df.duplicated().sum())
    out['duplicated_timestamps'] = int(df['timestamp'].duplicated().sum()) if 'timestamp' in df.columns else None
    bad = {}
    bad['high_lt_low'] = int((df['high'] < df['low']).sum())
    bad['open_outside_hl'] = int(((df['open'] > df['high']) | (df['open'] < df['low'])).sum())
    bad['close_outside_hl'] = int(((df['close'] > df['high']) | (df['close'] < df['low'])).sum())
    bad['nonpositive_prices'] = int(((df[['open','high','low','close']] <= 0).any(axis=1)).sum())
    out['ohlc_violations'] = bad
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        d = df['timestamp'].diff().dropna()
        out['non_monotonic_timestamps'] = int((d <= pd.Timedelta(0)).sum())
        vc = d.value_counts().head(5)
        out['top_time_deltas'] = {str(k): int(v) for k, v in vc.items()}
    else:
        out['non_monotonic_timestamps'] = None
        out['top_time_deltas'] = {}
    return out


def target_distribution(targets: np.ndarray, sampler_threshold: float) -> Dict:
    a = np.abs(targets)
    exact_zero = targets == 0.0
    eps_zero = np.isclose(targets, 0.0, atol=1e-12)
    out = {
        'n': int(len(targets)),
        'mean': float(np.nanmean(targets)),
        'std': float(np.nanstd(targets)),
        'min': float(np.nanmin(targets)),
        'max': float(np.nanmax(targets)),
        'exact_zero_count': int(exact_zero.sum()),
        'exact_zero_pct': float(exact_zero.mean()),
        'eps_zero_count': int(eps_zero.sum()),
        'eps_zero_pct': float(eps_zero.mean()),
        'tiny_abs_lt_1e8_count': int((a < 1e-8).sum()),
        'tiny_abs_lt_1e6_count': int((a < 1e-6).sum()),
        'abs_0_to_0p01': int(((a > 0) & (a < 0.01)).sum()),
        'abs_0p01_to_0p03': int(((a >= 0.01) & (a < 0.03)).sum()),
        'abs_0p03_to_0p05': int(((a >= 0.03) & (a < 0.05)).sum()),
        'abs_0p05_to_0p10': int(((a >= 0.05) & (a < 0.10)).sum()),
        'abs_ge_0p10': int((a >= 0.10).sum()),
        'sampler_flat_count': int((a < sampler_threshold).sum()),
        'sampler_flat_pct': float((a < sampler_threshold).mean()),
        'positive_count': int((targets > 0).sum()),
        'negative_count': int((targets < 0).sum()),
    }
    return out


def sample_rows(df: pd.DataFrame, targets: np.ndarray, mask: np.ndarray, n: int = 10) -> List[Dict]:
    idxs = np.where(mask)[0][:n]
    rows = []
    for i in idxs:
        rows.append({
            'idx': int(i),
            'timestamp': None if 'timestamp' not in df.columns else str(df.iloc[i]['timestamp']),
            'open': float(df.iloc[i]['open']),
            'high': float(df.iloc[i]['high']),
            'low': float(df.iloc[i]['low']),
            'close': float(df.iloc[i]['close']),
            'target': float(targets[i]),
        })
    return rows


def leakage_surrogates(df: pd.DataFrame, targets: np.ndarray, max_lag: int = 3) -> Dict:
    close = df['close'].to_numpy(dtype=np.float64)
    ret1 = np.full_like(close, np.nan)
    ret1[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-12)
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x = ret1[-lag:]
            y = targets[:len(targets)+lag]
        elif lag > 0:
            x = ret1[:-lag]
            y = targets[lag:]
        else:
            x = ret1
            y = targets
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() > 10:
            out[f'ret1_vs_target_lag_{lag}'] = float(np.corrcoef(x[m], y[m])[0,1])
        else:
            out[f'ret1_vs_target_lag_{lag}'] = None
    return out


def main():
    ap = argparse.ArgumentParser(description='Audit data and labels before training.')
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--out', default='output/data_label_audit')
    ap.add_argument('--limit-files', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = load_config(args.data_dir)

    files = list(cfg.DATA_FILE)
    if args.limit_files and args.limit_files > 0:
        files = files[:args.limit_files]

    all_summaries = []
    for path in files:
        df = load_ohlc_csv(path)
        ohlc = basic_ohlc_checks(df)
        atr = atr_wilder(df, getattr(cfg, 'ATR_PERIOD', 14))
        targets, oracle_audit = generate_targets_py(
            df['open'].to_numpy(dtype=np.float64),
            df['high'].to_numpy(dtype=np.float64),
            df['low'].to_numpy(dtype=np.float64),
            df['close'].to_numpy(dtype=np.float64),
            atr,
            max_hold=getattr(cfg, 'ORACLE_MAX_HOLD', 15),
            fee_per_side=getattr(cfg, 'FEE_PER_SIDE', 0.001),
            slippage=getattr(cfg, 'SLIPPAGE', 0.0005),
            sl_atr_mult=getattr(cfg, 'ORACLE_SL_ATR_MULT', 1.5),
            tp_atr_mult=getattr(cfg, 'ORACLE_TP_ATR_MULT', 3.0),
            enable_trailing=getattr(cfg, 'ORACLE_ENABLE_TRAILING', False),
            trail_atr_mult=getattr(cfg, 'ORACLE_TRAIL_ATR_MULT', 1.5),
            saturation_factor=getattr(cfg, 'SATURATION_FACTOR', 2.5),
            mae_penalty=getattr(cfg, 'MAE_PENALTY', 0.20),
        )
        dist = target_distribution(targets, getattr(cfg, 'SAMPLER_THRESHOLD', 0.1))
        reasons = oracle_audit['reason'].value_counts(dropna=False).to_dict()
        zero_examples = sample_rows(df, targets, targets == 0.0, 12)
        tiny_examples = sample_rows(df, targets, (np.abs(targets) > 0) & (np.abs(targets) < 0.05), 12)
        small_examples = sample_rows(df, targets, (np.abs(targets) >= 0.05) & (np.abs(targets) < 0.10), 12)
        pos_tail_examples = sample_rows(df, targets, targets > 0.5, 12)
        neg_tail_examples = sample_rows(df, targets, targets < -0.5, 12)
        oracle_window_examples = []
        for i in list(oracle_audit.index[:8]):
            idx = int(oracle_audit.iloc[i]['idx'])
            end = min(idx + getattr(cfg, 'ORACLE_MAX_HOLD', 15), len(df)-1)
            window = df.iloc[idx:end+1][['timestamp','open','high','low','close']].copy()
            window['target_at_entry'] = np.nan
            window.iloc[0, window.columns.get_loc('target_at_entry')] = float(targets[idx])
            oracle_window_examples.append({
                'idx': idx,
                'reason': oracle_audit.iloc[i]['reason'],
                'target': float(targets[idx]),
                'rows': window.astype(str).to_dict(orient='records')
            })
        leak = leakage_surrogates(df, targets, max_lag=3)

        base = os.path.splitext(os.path.basename(path))[0]
        oracle_audit.to_csv(os.path.join(args.out, f'{base}_oracle_audit.csv'), index=False)
        pd.DataFrame(zero_examples).to_csv(os.path.join(args.out, f'{base}_exact_zero_examples.csv'), index=False)
        pd.DataFrame(tiny_examples).to_csv(os.path.join(args.out, f'{base}_tiny_signal_examples.csv'), index=False)
        pd.DataFrame(small_examples).to_csv(os.path.join(args.out, f'{base}_small_signal_examples.csv'), index=False)
        pd.DataFrame(pos_tail_examples).to_csv(os.path.join(args.out, f'{base}_positive_tail_examples.csv'), index=False)
        pd.DataFrame(neg_tail_examples).to_csv(os.path.join(args.out, f'{base}_negative_tail_examples.csv'), index=False)

        summary = {
            'file': path,
            'ohlc_checks': ohlc,
            'target_distribution': dist,
            'oracle_reason_counts': {k: int(v) for k, v in reasons.items()},
            'zero_examples_preview': zero_examples[:5],
            'tiny_signal_examples_preview': tiny_examples[:5],
            'small_signal_examples_preview': small_examples[:5],
            'leakage_surrogates': leak,
            'notes': [
                'Exact zero semantics should ideally correspond to oracle outputs equal to 0.0.',
                'If abs_0_to_0p01 or abs_0p01_to_0p03 are non-zero, investigate float noise or label post-processing.',
                'Large non-zero ret1_vs_target correlations at negative lags can indicate suspicious alignment or leakage proxies.',
                'Review oracle_audit.csv for reason breakdown: many cost_too_high or invalid_atr_or_price rows can starve labels.',
            ],
            'oracle_window_examples_preview': oracle_window_examples,
        }
        with open(os.path.join(args.out, f'{base}_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        all_summaries.append(summary)

    aggregate = {
        'config_snapshot': {
            'DATA_FILE': files,
            'ATR_PERIOD': getattr(cfg, 'ATR_PERIOD', None),
            'ORACLE_MAX_HOLD': getattr(cfg, 'ORACLE_MAX_HOLD', None),
            'FEE_PER_SIDE': getattr(cfg, 'FEE_PER_SIDE', None),
            'SLIPPAGE': getattr(cfg, 'SLIPPAGE', None),
            'ORACLE_SL_ATR_MULT': getattr(cfg, 'ORACLE_SL_ATR_MULT', None),
            'ORACLE_TP_ATR_MULT': getattr(cfg, 'ORACLE_TP_ATR_MULT', None),
            'ORACLE_ENABLE_TRAILING': getattr(cfg, 'ORACLE_ENABLE_TRAILING', None),
            'ORACLE_TRAIL_ATR_MULT': getattr(cfg, 'ORACLE_TRAIL_ATR_MULT', None),
            'SATURATION_FACTOR': getattr(cfg, 'SATURATION_FACTOR', None),
            'MAE_PENALTY': getattr(cfg, 'MAE_PENALTY', None),
            'SAMPLER_THRESHOLD': getattr(cfg, 'SAMPLER_THRESHOLD', None),
        },
        'files': all_summaries,
        'checklist_covered': [
            'Oracle correctness and exact-zero semantics',
            'Target distribution sanity including exact zero / tiny / small / tails',
            'Intrabar logic effect via audit windows and reason codes',
            'Target float precision / zero exactness',
            'Basic data integrity: OHLC violations, nulls, duplicate timestamps',
            'Simple leakage surrogate correlations across lags',
        ]
    }
    with open(os.path.join(args.out, 'aggregate_summary.json'), 'w') as f:
        json.dump(aggregate, f, indent=2)

    md = ['# Data and Label Audit', '']
    md.append('This audit covers data integrity, oracle label generation, zero semantics, target distribution, float-noise around zero, and simple leakage surrogates.')
    md.append('')
    for s in all_summaries:
        md.append(f"## {os.path.basename(s['file'])}")
        td = s['target_distribution']
        md.append(f"- Rows: {s['ohlc_checks']['rows']}")
        md.append(f"- Exact zero count: {td['exact_zero_count']} ({td['exact_zero_pct']:.2%})")
        md.append(f"- Sampler-flat count (< threshold): {td['sampler_flat_count']} ({td['sampler_flat_pct']:.2%})")
        md.append(f"- Tiny non-zero counts: 0-0.01={td['abs_0_to_0p01']}, 0.01-0.03={td['abs_0p01_to_0p03']}, 0.03-0.05={td['abs_0p03_to_0p05']}")
        md.append(f"- Small counts: 0.05-0.10={td['abs_0p05_to_0p10']}, >=0.10={td['abs_ge_0p10']}")
        md.append(f"- Positive / negative counts: {td['positive_count']} / {td['negative_count']}")
        md.append(f"- OHLC violations: {json.dumps(s['ohlc_checks']['ohlc_violations'])}")
        md.append(f"- Top oracle reasons: {json.dumps(s['oracle_reason_counts'])}")
        md.append(f"- Leakage surrogates: {json.dumps(s['leakage_surrogates'])}")
        md.append('')
    with open(os.path.join(args.out, 'README.md'), 'w') as f:
        f.write('\n'.join(md))


if __name__ == '__main__':
    main()