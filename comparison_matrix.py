"""
comparison_matrix.py — Complete comparison
═══════════════════════════════════════════════════════════════════════════════

Compare ALL combinations:
  Feature sets:  raw (unnormalized) vs pre-normalized (features.py)
  Normalizations:  none, global-zscore, pw-zscore(90), pw-zscore(512)

This gives the definitive answer to: what's the best pipeline?
"""

import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import config
from features import FeatureEngineer, FeatureConfig
from oracle import generate_targets

# ── Load data ────────────────────────────────────────────────────────────────
DATA_PATH = "Data/NIFTY 50_30minute.csv"
df_full = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
col_map = {c.lower(): c for c in df_full.columns}
o_col, h_col, l_col, c_col = col_map.get('open','Open'), col_map.get('high','High'), col_map.get('low','Low'), col_map.get('close','Close')
v_col = col_map.get('volume', 'Volume')

close = df_full[c_col].values.astype(np.float64)
high  = df_full[h_col].values.astype(np.float64)
low   = df_full[l_col].values.astype(np.float64)
opn   = df_full[o_col].values.astype(np.float64)

hl = df_full[h_col] - df_full[l_col]
hc = (df_full[h_col] - df_full[c_col].shift()).abs()
lc = (df_full[l_col]  - df_full[c_col].shift()).abs()
atr_full = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(config.ATR_PERIOD).mean()
target_full = generate_targets(df_full[o_col].values, df_full[h_col].values, df_full[l_col].values, df_full[c_col].values, atr_full.values)
target_full[np.abs(target_full) < config.SAMPLER_THRESHOLD] = 0.0

warmup = 3536
close = close[warmup:]; high = high[warmup:]; low = low[warmup:]; opn = opn[warmup:]
target = target_full[:len(close)]
min_len = min(len(close), len(target))
close = close[:min_len]; high = high[:min_len]; low = low[:min_len]; opn = opn[:min_len]; target = target[:min_len]

SUBSET = 6000
close_s = close[:SUBSET]; high_s = high[:SUBSET]; low_s = low[:SUBSET]; opn_s = opn[:SUBSET]; target_s = target[:min(SUBSET, len(target))]

print("=" * 80)
print("  COMPARISON MATRIX — All feature sets × all normalizations")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════════════
# Feature Set A: Pre-normalized (from features.py)
# ═══════════════════════════════════════════════════════════════════════════════

fe = FeatureEngineer(config=FeatureConfig())
feat_df = fe.build(df_full[c_col].iloc[warmup:warmup+min_len], ohlc=df_full.iloc[warmup:warmup+min_len], include_target=False, dropna=False)
feat_df['open']  = df_full[o_col].iloc[warmup:warmup+min_len].values.astype(np.float64)
feat_df['high']  = df_full[h_col].iloc[warmup:warmup+min_len].values.astype(np.float64)
feat_df['low']   = df_full[l_col].iloc[warmup:warmup+min_len].values.astype(np.float64)
feat_df['close'] = df_full[c_col].iloc[warmup:warmup+min_len].values.astype(np.float64)

FEATURE_COLS = [
    'open', 'high', 'low', 'close',
    'ret_norm_1d', 'ret_norm_3d', 'ret_norm_6d', 'ret_norm_13d',
    'ret_norm_26d', 'ret_norm_65d', 'ret_norm_130d', 'ret_norm_260d',
    'macd_8_24', 'macd_26_78', 'macd_52_156',
    'feat_efficiency', 'feat_icp', 'feat_momentum_rsi',
    'feat_vol_asymmetry', 'feat_local_structure',
    'feat_session_sin', 'feat_session_cos', 'feat_vol_squeeze'
]
for col in FEATURE_COLS:
    if col not in feat_df.columns: feat_df[col] = 0.0
feat_prenorm = np.nan_to_num(feat_df[FEATURE_COLS].values[:SUBSET].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

# ═══════════════════════════════════════════════════════════════════════════════
# Feature Set B: Raw (unnormalized)
# ═══════════════════════════════════════════════════════════════════════════════

def raw_returns(c, h):
    ret = np.full_like(c, np.nan)
    ret[h:] = np.log(c[h:] / c[:-h])
    return ret

def raw_ema(arr, span):
    alpha = 2.0 / (span + 1.0)
    out = np.zeros_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out

feat_raw = np.column_stack([
    opn_s, high_s, low_s, close_s,
    raw_returns(close_s, 1), raw_returns(close_s, 3), raw_returns(close_s, 6), raw_returns(close_s, 13),
    raw_returns(close_s, 26), raw_returns(close_s, 65), raw_returns(close_s, 130), raw_returns(close_s, 260),
    raw_ema(close_s, 8) - raw_ema(close_s, 24),
    raw_ema(close_s, 26) - raw_ema(close_s, 78),
    raw_ema(close_s, 52) - raw_ema(close_s, 156),
    np.zeros(SUBSET),  # efficiency placeholder
    np.zeros(SUBSET),  # icp placeholder
    np.zeros(SUBSET),  # rsi placeholder
    np.zeros(SUBSET),  # vol_asym placeholder
    np.zeros(SUBSET),  # local_struct placeholder
    np.zeros(SUBSET),  # session_sin placeholder
    np.zeros(SUBSET),  # session_cos placeholder
    np.zeros(SUBSET),  # vol_squeeze placeholder
])
feat_raw = np.nan_to_num(feat_raw, nan=0.0, posinf=0.0, neginf=0.0)

# ═══════════════════════════════════════════════════════════════════════════════
# Normalization functions
# ═══════════════════════════════════════════════════════════════════════════════

def pw_zscore(feat, lookback):
    n = len(feat) - lookback + 1
    out = np.zeros((n, feat.shape[1]))
    for w in range(n):
        window = feat[w:w+lookback]
        mean = np.mean(window, axis=0)
        std = np.std(window, axis=0)
        min_std = 0.005 * np.abs(mean).clip(min=0.01)
        const = std < min_std
        std[const] = 1.0; mean[const] = 0.0
        norm = (window[-1] - mean) / (std + 1e-5)
        norm[const] = 0.0
        out[w] = np.clip(norm, -5, 5)
    return out

def global_zscore(feat):
    mean = np.mean(feat, axis=0)
    std = np.std(feat, axis=0) + 1e-8
    return np.clip((feat - mean) / std, -5, 5)

# ═══════════════════════════════════════════════════════════════════════════════
# Compute total MI for all combinations
# ═══════════════════════════════════════════════════════════════════════════════

lookbacks = [90, 512]
split = 0.7

results = {}

for feat_name, feat_mat in [("prenorm", feat_prenorm), ("raw", feat_raw)]:
    for norm_name, norm_fn in [
        ("none", lambda f: f),
        ("global", lambda f: global_zscore(f)),
        ("pw90", lambda f: pw_zscore(f, 90)),
        ("pw512", lambda f: pw_zscore(f, 512)),
    ]:
        normalized = norm_fn(feat_mat)
        
        # Align targets
        if norm_name.startswith("pw"):
            aligned_target = target_s[normalized.shape[0]-1:] if normalized.shape[0] < len(target_s) else target_s[:normalized.shape[0]]
            # For pw zscore, the output has fewer rows
            aligned_target = target_s[len(target_s)-normalized.shape[0]:]
        else:
            aligned_target = target_s[:normalized.shape[0]]
        
        n = min(normalized.shape[0], len(aligned_target))
        normalized = normalized[:n]
        aligned_target = aligned_target[:n]
        
        # Compute total MI
        total_mi = 0
        for i in range(normalized.shape[1]):
            mi = mutual_info_regression(
                normalized[:, i:i+1], aligned_target,
                n_neighbors=5, random_state=42, n_jobs=1
            )[0]
            total_mi += mi
        
        key = f"{feat_name}+{norm_name}"
        results[key] = total_mi

# ═══════════════════════════════════════════════════════════════════════════════
# Print comparison matrix
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n  {'Normalization':<20} {'Pre-normalized':>15} {'Raw':>15} {'Best':>10}")
print(f"  {'─'*20} {'─'*15} {'─'*15} {'─'*10}")

for norm in ["none", "global", "pw90", "pw512"]:
    pre = results[f"prenorm+{norm}"]
    raw = results[f"raw+{norm}"]
    best = "prenorm" if pre > raw else "raw"
    print(f"  {norm:<20} {pre:>15.6f} {raw:>15.6f} {best:>10}")

print(f"\n  {'Feature Set':<20} {'Best Norm':>15} {'Max MI':>15}")
print(f"  {'─'*20} {'─'*15} {'─'*15}")

for feat_name in ["prenorm", "raw"]:
    best_norm = max(["none", "global", "pw90", "pw512"], key=lambda n: results[f"{feat_name}+{n}"])
    best_mi = results[f"{feat_name}+{best_norm}"]
    print(f"  {feat_name:<20} {best_norm:>15} {best_mi:>15.6f}")

# Best overall
best_overall = max(results, key=results.get)
print(f"\n  BEST OVERALL: {best_overall} = {results[best_overall]:.6f}")

# Worst overall
worst_overall = min(results, key=results.get)
print(f"  WORST OVERALL: {worst_overall} = {results[worst_overall]:.6f}")

# Signal destruction from z-score
print(f"\n  SIGNAL DESTRUCTION ANALYSIS:")
print(f"  {'─'*60}")
for feat_name, feat_label in [("prenorm", "Pre-normalized"), ("raw", "Raw")]:
    baseline = results[f"{feat_name}+none"]
    for norm in ["global", "pw90", "pw512"]:
        mi = results[f"{feat_name}+{norm}"]
        pct = mi / max(baseline, 1e-10) * 100
        print(f"  {feat_label}+{norm}: {mi:.6f} ({pct:.1f}% of no-norm baseline)")

print("\n" + "=" * 80)
print("  AUDIT COMPLETE")
print("=" * 80)
