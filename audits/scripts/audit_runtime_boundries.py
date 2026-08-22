#!/usr/bin/env python3
"""
Runtime boundary audit for walk-forward folds with oracle look-ahead labels.

This script:
  1. Loads your config (oracle params, WFV params, etc.)
  2. Runs your actual fold-building code to get fold specs
  3. For each fold, checks whether any train target's oracle window
     touches the gap/val/test region.
  4. For each fold, checks whether val/test targets' oracle windows
     touch the train region (backward leakage).
  5. Reports boundary violations and safe margins.

Key concept:
  - Oracle target at index i uses OHLC from [i, i + ORACLE_MAX_HOLD).
  - If train ends at T, then for any train index i where i + ORACLE_MAX_HOLD > T,
    the oracle window touches beyond train (into gap/val).
  - We want ALL train targets to satisfy: i + ORACLE_MAX_HOLD <= T.
"""

import os
import sys
import json
import glob
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from config import *


def _resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def load_config(data_dir_override=None):
    import importlib
    cfg = importlib.import_module('config')
    data_dir = _resolve_path(data_dir_override if data_dir_override is not None else cfg.DATA_DIR)
    cfg.DATA_DIR = data_dir
    files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    if files:
        cfg.DATA_FILE = files
    else:
        cfg.DATA_FILE = [_resolve_path(p) for p in cfg.DATA_FILE]
    return cfg





def import_config():
    import config
    return config


def import_train_module():
    import train
    return train


def find_data_files():
    import config
    data_dir = getattr(config, 'DATA_DIR', 'Data')
    files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    if not files:
        fallback = os.path.join(data_dir, 'NIFTY 50_30minute.csv')
        if os.path.exists(fallback):
            files = [fallback]
        else:
            raise FileNotFoundError(f'No CSV files found in {data_dir}')
    return files


def load_ohlc_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ts_col = None
    for c in ['datetime', 'date', 'time', 'timestamp']:
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        for c in df.columns:
            if 'date' in c.lower() or 'time' in c.lower():
                ts_col = c
                break
    if ts_col is None:
        ts_col = df.columns[0]

    open_col = None
    high_col = None
    low_col = None
    close_col = None
    for c in ['open', '_Open', 'OPEN']:
        if c in df.columns:
            open_col = c
            break
    for c in ['high', '_High', 'HIGH']:
        if c in df.columns:
            high_col = c
            break
    for c in ['low', '_Low', 'LOW']:
        if c in df.columns:
            low_col = c
            break
    for c in ['close', '_Close', 'CLOSE']:
        if c in df.columns:
            close_col = c
            break

    out = pd.DataFrame()
    out['timestamp'] = pd.to_datetime(df[ts_col], errors='coerce')
    out['open'] = pd.to_numeric(df[open_col], errors='coerce')
    out['high'] = pd.to_numeric(df[high_col], errors='coerce')
    out['low'] = pd.to_numeric(df[low_col], errors='coerce')
    out['close'] = pd.to_numeric(df[close_col], errors='coerce')
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


def generate_targets_from_df(df: pd.DataFrame, cfg):
    open_arr = df['open'].to_numpy(dtype=np.float64)
    high_arr = df['high'].to_numpy(dtype=np.float64)
    low_arr = df['low'].to_numpy(dtype=np.float64)
    close_arr = df['close'].to_numpy(dtype=np.float64)
    atr_arr = atr_wilder(df, getattr(cfg, 'ATR_PERIOD', 14))

    from oracle import generate_targets
    targets = generate_targets(
        open_arr=open_arr,
        high_arr=high_arr,
        low_arr=low_arr,
        close_arr=close_arr,
        atr_arr=atr_arr,
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
    return targets


def build_asset_data_list(cfg, fe=None):
    from train import process_dataset, FeatureEngineer, _make_feature_config

    if fe is None:
        fe_config = _make_feature_config()
        fe = FeatureEngineer(config=fe_config)

    files = list(cfg.DATA_FILE)
    asset_data_list, feature_cols = process_dataset(files, fe)
    return asset_data_list, feature_cols


def fold_train_end_range(train_entries):
    """
    Given a list of train entries for a fold:
      (asset_id, feat_vals, target_vals, ohlc_vals, train_end_abs)
    return:
      - max_train_end: maximum train_end_abs across assets
      - per_asset_train_end: dict {asset_id: train_end_abs}
    """
    per_asset = {}
    global_end = None
    for entry in train_entries:
        asset_id = entry[0]
        train_end_abs = entry[4] if len(entry) > 4 else len(entry[2])
        per_asset[asset_id] = train_end_abs
        if global_end is None or train_end_abs > global_end:
            global_end = train_end_abs
    return global_end, per_asset


def fold_val_range(val_entries, gap):
    """
    Given val entries and gap, compute the absolute range of val region.
    val_entries: list of (asset_id, feat_slice, targ_slice, ohlc_slice, None)
    We treat the first valid global index as fold_val_start and last as fold_val_end.
    We approximate by using:
      - The gap after train_end for each asset
      - The length of val entries for each asset
    """
    if not val_entries:
        return None, None

    per_asset = {}
    for entry in val_entries:
        asset_id = entry[0]
        val_len = len(entry[2])
        train_end = entry[4] if len(entry) > 4 else None
        if train_end is None:
            continue
        val_start = train_end + gap
        val_end = val_start + val_len
        per_asset[asset_id] = (val_start, val_end)
    return per_asset


def audit_boundary_violations(
    asset_data_list: List[Tuple],
    cfg,
    folds: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    For each fold:
      - For each asset in train_entries, compute:
          - train_end_abs (T)
          - For each train index i in [0, train_end_abs), the oracle window is [i, i + max_hold).
          - If i + max_hold > T, then the oracle window touches beyond train.
      - We want no train indices where i + max_hold > T.
      - In practice, this means we must exclude the last max_hold-1 train indices from labels.
    """
    max_hold = getattr(cfg, 'ORACLE_MAX_HOLD', 15)
    gap = getattr(cfg, 'FORECAST_HORIZON', 15) + 50

    violations = []
    fold_summaries = []

    for fold_idx, fold in enumerate(folds):
        fold_label = fold.get('label', f'Fold {fold_idx}')
        train_entries = fold.get('train_list', [])
        val_entries = fold.get('val_list', [])

        if not train_entries:
            continue

        train_range = fold.get('train_range', (0, None))
        train_start = train_range[0] if train_range else 0
        global_train_end = train_range[1] if train_range and train_range[1] is not None else None
        if global_train_end is None:
            global_train_end, _ = fold_train_end_range(train_entries)

        fold_violations = []
        total_train_samples = 0
        violating_samples = 0

        for entry in train_entries:
            asset_id = entry[0]
            targets = entry[2]
            n_train = len(targets)

            total_train_samples += n_train
            last_safe_index = global_train_end - max_hold

            violating_indices = [
                i for i in range(n_train)
                if (train_start + i) + max_hold > global_train_end
            ]
            violating_count = len(violating_indices)
            violating_samples += violating_count

            if violating_count > 0:
                fold_violations.append({
                    'asset_id': asset_id,
                    'global_train_end': global_train_end,
                    'train_samples': n_train,
                    'violating_count': violating_count,
                    'violating_indices_sample': violating_indices[:10],
                    'last_safe_index': last_safe_index,
                })

        fold_info = {
            'fold_idx': fold_idx,
            'fold_label': fold_label,
            'total_train_samples': total_train_samples,
            'violating_samples': violating_samples,
            'violation_rate': violating_samples / max(total_train_samples, 1),
            'fold_violations': fold_violations,
            'max_hold': max_hold,
            'gap': gap,
        }
        fold_summaries.append(fold_info)

        if violating_samples > 0:
            violations.append({
                'fold_idx': fold_idx,
                'fold_label': fold_label,
                'violating_samples': violating_samples,
                'total_train_samples': total_train_samples,
                'violation_rate': violating_samples / max(total_train_samples, 1),
                'fold_violations': fold_violations,
            })

    return {
        'max_hold': max_hold,
        'gap': gap,
        'fold_summaries': fold_summaries,
        'total_violations': len(violations),
        'violations': violations,
    }


def main():
    ap = argparse.ArgumentParser(description='Runtime boundary audit for walk-forward folds.')
    ap.add_argument('--out', default='output/boundary_audit')
    ap.add_argument('--limit-assets', type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    train_mod = import_train_module()
    # train.py reloads config using cwd; restore repo-root data paths
    cfg = load_config()

    asset_data_list, feature_cols = build_asset_data_list(cfg)

    if args.limit_assets and args.limit_assets > 0:
        asset_data_list = asset_data_list[:args.limit_assets]

    build_folds_fn = getattr(train_mod, 'make_rolling_folds', None)
    if build_folds_fn is None:
        raise RuntimeError('make_rolling_folds not found in train.py')

    folds = build_folds_fn(asset_data_list, cfg)

    audit_result = audit_boundary_violations(asset_data_list, cfg, folds)

    report = {
        'config_snapshot': {
            'ORACLE_MAX_HOLD': getattr(cfg, 'ORACLE_MAX_HOLD', None),
            'FORECAST_HORIZON': getattr(cfg, 'FORECAST_HORIZON', None),
            'WFV_TRAIN_BARS': getattr(cfg, 'WFV_TRAIN_BARS', None),
            'WFV_VAL_BARS': getattr(cfg, 'WFV_VAL_BARS', None),
            'WFV_STEP_BARS': getattr(cfg, 'WFV_STEP_BARS', None),
            'TRAIN_RATIO': getattr(cfg, 'TRAIN_RATIO', None),
            'VAL_RATIO': getattr(cfg, 'VAL_RATIO', None),
            'TEST_RATIO': getattr(cfg, 'TEST_RATIO', None),
        },
        'audit': audit_result,
        'num_folds': len(folds),
        'num_assets': len(asset_data_list),
    }

    out_json = os.path.join(args.out, 'boundary_audit.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    md = ['# Runtime Boundary Audit for Oracle Look-Ahead']
    md.append('')
    md.append('## Key parameters')
    md.append(f"- ORACLE_MAX_HOLD: {report['config_snapshot']['ORACLE_MAX_HOLD']}")
    md.append(f"- FORECAST_HORIZON: {report['config_snapshot']['FORECAST_HORIZON']}")
    md.append(f"- Gap used in folds: FORECAST_HORIZON + 50 = {audit_result['gap']}")
    md.append(f"- Required gap to be safe: >= ORACLE_MAX_HOLD = {audit_result['max_hold']}")
    md.append('')
    md.append('## Summary')
    md.append(f"- Number of folds: {report['num_folds']}")
    md.append(f"- Number of assets: {report['num_assets']}")
    md.append(f"- Total folds with boundary violations: {audit_result['total_violations']}")
    md.append('')

    for fs in audit_result['fold_summaries']:
        md.append(f"### {fs['fold_label']}")
        md.append(f"- Total train samples: {fs['total_train_samples']}")
        md.append(f"- Violating samples: {fs['violating_samples']}")
        md.append(f"- Violation rate: {fs['violation_rate']:.2%}")
        if fs['violation_rate'] > 0:
            md.append(f"- **CONFLICT**: {fs['violating_samples']} train targets use oracle windows beyond train end.")
        else:
            md.append('- OK: No train targets use oracle windows beyond train end.')
        md.append('')

    if audit_result['total_violations'] > 0:
        md.append('## WARNING: Boundary Violations Detected')
        md.append('')
        md.append('The following folds have train targets whose oracle windows touch the gap/val region:')
        md.append('')
        for v in audit_result['violations']:
            md.append(f"### {v['fold_label']}")
            md.append(f"- Violating samples: {v['violating_samples']} / {v['total_train_samples']} ({v['violation_rate']:.2%})")
            for fv in v['fold_violations'][:5]:
                md.append(
                    f"  - Asset {fv['asset_id']}: global_train_end={fv['global_train_end']}, "
                    f"violating={fv['violating_count']}"
                )
            md.append('')

    with open(os.path.join(args.out, 'README.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print(f"Wrote {out_json}")
    print(f"Summary: {audit_result['total_violations']} folds with boundary violations")


if __name__ == '__main__':
    main()
