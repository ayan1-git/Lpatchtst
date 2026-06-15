"""
clean_slatest_audit.py — The Definitive Test
═══════════════════════════════════════════════════════════════════════════════

TEST: What if we use RAW, UNNORMALIZED features (no vol-scaling, no MACD
normalization, no bounded indicators) and apply ONLY per-window z-score
with lookback=90?

PIPELINE:
  1. Compute raw features (no normalization at all)
  2. Apply per-window z-score with lookback=90
  3. Measure MI(feature_after_zscore, target) for each feature
  4. Compare against: raw features with no normalization at all

If MI_after_zscore ≈ MI_raw → z-score preserves signal (good)
If MI_after_zscore << MI_raw → z-score destroys signal (bad)

We test with lookback=90 (tokenizer window) since the user asked about that.

═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import config
from oracle import generate_targets

# ── Load data ────────────────────────────────────────────────────────────────
DATA_PATH = "Data/NIFTY 50_30minute.csv"
df_full = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
col_map = {c.lower(): c for c in df_full.columns}
o_col = col_map.get('open', 'Open')
h_col = col_map.get('high', 'High')
l_col = col_map.get('low', 'Low')
c_col = col_map.get('close', 'Close')
v_col = col_map.get('volume', 'Volume')

close = df_full[c_col].values.astype(np.float64)
high  = df_full[h_col].values.astype(np.float64)
low   = df_full[l_col].values.astype(np.float64)
opn   = df_full[o_col].values.astype(np.float64)
vol   = df_full[v_col].values.astype(np.float64) if v_col in df_full.columns else np.zeros_like(close)

# ATR + oracle
hl = df_full[h_col] - df_full[l_col]
hc = (df_full[h_col] - df_full[c_col].shift()).abs()
lc = (df_full[l_col]  - df_full[c_col].shift()).abs()
atr_full = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(config.ATR_PERIOD).mean()
target_full = generate_targets(df_full[o_col].values, df_full[h_col].values, df_full[l_col].values, df_full[c_col].values, atr_full.values)
target_full[np.abs(target_full) < config.SAMPLER_THRESHOLD] = 0.0

# Cut warmup
warmup = 3536
close = close[warmup:]; high = high[warmup:]; low = low[warmup:]
opn  = opn[warmup:];  vol  = vol[warmup:]
target = target_full[:len(close)]

min_len = min(len(close), len(target))
close = close[:min_len]; high = high[:min_len]; low = low[:min_len]
opn  = opn[:min_len];  vol  = vol[:min_len]; target = target[:min_len]

print("=" * 80)
print("  CLEAN SLATE AUDIT — Raw features + ONLY per-window z-score(90)")
print("=" * 80)
print(f"  Data: {min_len:,} bars")

# ═══════════════════════════════════════════════════════════════════════════════
# Compute RAW (completely unnormalized) features
# ═══════════════════════════════════════════════════════════════════════════════

print("\n  Computing raw (unnormalized) features...")

# 1. Raw prices (no normalization)
raw_open  = opn.copy()
raw_high  = high.copy()
raw_low   = low.copy()
raw_close = close.copy()

# 2. Raw returns (NOT vol-scaled) — just simple log returns at various horizons
def raw_returns(close, h):
    """Simple log return over h bars — NO vol scaling."""
    ret = np.full_like(close, np.nan)
    ret[h:] = np.log(close[h:] / close[:-h])
    return ret

raw_ret_1d   = raw_returns(close, 1)
raw_ret_3d   = raw_returns(close, 3)
raw_ret_6d   = raw_returns(close, 6)
raw_ret_13d  = raw_returns(close, 13)
raw_ret_26d  = raw_returns(close, 26)
raw_ret_65d  = raw_returns(close, 65)
raw_ret_130d = raw_returns(close, 130)
raw_ret_260d = raw_returns(close, 260)

# 3. Raw MACD (NOT normalized) — just EMA(short) - EMA(long)
def raw_ema(arr, span):
    """Simple EWMA — no normalization."""
    alpha = 2.0 / (span + 1.0)
    out = np.zeros_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out

raw_macd_8_24   = raw_ema(close, 8)  - raw_ema(close, 24)
raw_macd_26_78  = raw_ema(close, 26) - raw_ema(close, 78)
raw_macd_52_156 = raw_ema(close, 52) - raw_ema(close, 156)

# 4. Raw Kaufman Efficiency Ratio (not clipped to [-1,1])
def raw_ker(close, period):
    """KER without clipping."""
    net_move = np.abs(close - np.roll(close, period))
    path_len = np.zeros_like(close)
    for i in range(1, len(close)):
        path_len[i] = path_len[i-1] + abs(close[i] - close[i-1])
    path_len_period = path_len - np.roll(path_len, period)
    path_len_period[:period] = np.nan
    ker = net_move / np.maximum(path_len_period, 1e-10)
    sign = np.sign(close - np.roll(close, period))
    return ker * sign

raw_efficiency = raw_ker(close, 26)

# 5. Raw ICP (not clipped)
def raw_icp(high, low, close, period):
    """ICP without clipping to [-1,1]."""
    hl_range = high - low
    raw = (close - low) / np.maximum(hl_range, 1e-10)
    # Rolling mean
    out = np.full_like(close, np.nan)
    for i in range(period-1, len(close)):
        out[i] = np.mean(raw[i-period+1:i+1])
    return out * 2.0 - 1.0  # scale to [-1,1] but NO clipping

raw_icp = raw_icp(high, low, close, 13)

# 6. Raw RSI (not centered, not clipped)
def raw_rsi(close, period):
    """RSI without centering — raw [0, 100] values."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    alpha = 1.0 / period
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, len(close)):
        avg_gain[i] = alpha * gain[i] + (1-alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1-alpha) * avg_loss[i-1]
    rs = avg_gain / np.maximum(avg_loss, 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))

raw_rsi = raw_rsi(close, 14)

# 7. Raw directional vol asymmetry (not clipped)
def raw_vol_asym(close, window):
    """Vol asymmetry without clipping."""
    log_ret = np.diff(np.log(close), prepend=np.log(close[0]))
    out = np.full_like(close, np.nan)
    for i in range(window, len(close)):
        r = log_ret[i-window:i]
        up_vol = np.std(r[r > 0]) if (r > 0).sum() > 1 else 0
        dn_vol = np.std(r[r < 0]) if (r < 0).sum() > 1 else 0
        out[i] = (up_vol - dn_vol) / max(up_vol + dn_vol, 1e-10)
    return out

raw_vol_asym = raw_vol_asym(close, 65)

# 8. Raw local structure (not clipped)
def raw_local_structure(high, low, close, window):
    """Donchian position without clipping."""
    out = np.full_like(close, np.nan)
    for i in range(window-1, len(close)):
        roll_high = np.max(high[i-window+1:i+1])
        roll_low  = np.min(low[i-window+1:i+1])
        rng = roll_high - roll_low
        out[i] = (close[i] - roll_low) / max(rng, 1e-10) * 2.0 - 1.0
    return out

raw_local_struct = raw_local_structure(high, low, close, 65)

# 9. Session features (these are inherently bounded, keep as-is)
# Parse timestamps for session encoding
dates = df_full.index[warmup:warmup+min_len]
minutes = dates.hour * 60 + dates.minute
open_min = 9*60 + 15
close_min = 15*60 + 30
session_len = close_min - open_min
session_pos = np.clip((minutes.values - open_min) / float(session_len), 0, 1)
raw_session_sin = np.sin(np.pi * session_pos)
raw_session_cos = np.cos(np.pi * session_pos)

# 10. Raw vol squeeze (ATR ratio, not clipped)
def raw_vol_squeeze(high, low, close, fast=5, slow=26):
    """ATR fast / ATR slow — no clipping."""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr_fast = np.zeros_like(close)
    atr_slow = np.zeros_like(close)
    for i in range(1, len(close)):
        atr_fast[i] = atr_fast[i-1] * (fast-1)/fast + tr[i] / fast
        atr_slow[i] = atr_slow[i-1] * (slow-1)/slow + tr[i] / slow
    return atr_fast / np.maximum(atr_slow, 1e-10)

raw_vol_squeeze = raw_vol_squeeze(high, low, close, 5, 26)

# Stack all raw features
RAW_FEATURE_NAMES = [
    'open', 'high', 'low', 'close',
    'ret_1d', 'ret_3d', 'ret_6d', 'ret_13d',
    'ret_26d', 'ret_65d', 'ret_130d', 'ret_260d',
    'macd_8_24', 'macd_26_78', 'macd_52_156',
    'efficiency', 'icp', 'rsi', 'vol_asym',
    'local_struct', 'session_sin', 'session_cos', 'vol_squeeze'
]

raw_features = np.column_stack([
    raw_open, raw_high, raw_low, raw_close,
    raw_ret_1d, raw_ret_3d, raw_ret_6d, raw_ret_13d,
    raw_ret_26d, raw_ret_65d, raw_ret_130d, raw_ret_260d,
    raw_macd_8_24, raw_macd_26_78, raw_macd_52_156,
    raw_efficiency, raw_icp, raw_rsi, raw_vol_asym,
    raw_local_struct, raw_session_sin, raw_session_cos, raw_vol_squeeze
])

# Clean NaN/Inf
raw_features = np.nan_to_num(raw_features, nan=0.0, posinf=0.0, neginf=0.0)

print(f"  Raw feature matrix: {raw_features.shape}")
print(f"\n  {'Feature':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")
for i, name in enumerate(RAW_FEATURE_NAMES):
    f = raw_features[:, i]
    print(f"  {name:<20} {np.mean(f):>+12.4f} {np.std(f):>12.4f} {np.min(f):>+12.2f} {np.max(f):>+12.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Apply per-window z-score with lookback=90
# ═══════════════════════════════════════════════════════════════════════════════

LOOKBACK = 90  # tokenizer window

print(f"\n  Applying per-window z-score with lookback={LOOKBACK}...")

SUBSET = 8000
feat_sub = raw_features[:SUBSET]
target_sub = target[:SUBSET]

n_windows = SUBSET - LOOKBACK + 1
target_aligned = target_sub[LOOKBACK-1:]

# Apply per-window z-score (same logic as FinancialDataset.__getitem__)
def apply_pw_zscore(feat_array, lookback):
    """Apply per-window z-score, return last element of each window."""
    n, d = feat_array.shape
    out = np.zeros((n - lookback + 1, d))
    
    for w in range(n - lookback + 1):
        window = feat_array[w:w+lookback]
        w_mean = np.mean(window, axis=0)
        w_std  = np.std(window, axis=0)
        
        # Same guard as FinancialDataset
        min_std = 0.005 * np.abs(w_mean).clip(min=0.01)
        const_mask = (w_std < min_std)
        w_std[const_mask] = 1.0
        w_mean[const_mask] = 0.0
        
        norm = (window[-1] - w_mean) / (w_std + 1e-5)
        norm[const_mask] = 0.0
        out[w] = np.clip(norm, -5.0, 5.0)
    
    return out

feat_zscore = apply_pw_zscore(feat_sub, LOOKBACK)
print(f"  Z-scored feature matrix: {feat_zscore.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# Measure MI at three stages
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("  MI COMPARISON: Raw vs Per-window Z-scored (lookback=90)")
print("─" * 80)

# Stage A: Raw features (no normalization at all)
feat_raw_aligned = raw_features[LOOKBACK-1:SUBSET]

# Stage B: Per-window z-scored (what model would see)
# feat_zscore is already computed

# Stage C: Global z-score only (for comparison)
global_mean = feat_raw_aligned[:int(0.7*n_windows)].mean(axis=0)
global_std  = feat_raw_aligned[:int(0.7*n_windows)].std(axis=0) + 1e-8
feat_global_zscore = (feat_raw_aligned - global_mean) / global_std
feat_global_zscore = np.clip(feat_global_zscore, -5.0, 5.0)

print(f"\n  Computing MI for {n_windows} windows...")

mi_raw = np.zeros(len(RAW_FEATURE_NAMES))
mi_zscore90 = np.zeros(len(RAW_FEATURE_NAMES))
mi_global = np.zeros(len(RAW_FEATURE_NAMES))

for i, name in enumerate(RAW_FEATURE_NAMES):
    mi_raw[i] = mutual_info_regression(
        feat_raw_aligned[:, i:i+1], target_aligned,
        n_neighbors=5, random_state=42, n_jobs=1
    )[0]
    mi_zscore90[i] = mutual_info_regression(
        feat_zscore[:, i:i+1], target_aligned,
        n_neighbors=5, random_state=42, n_jobs=1
    )[0]
    mi_global[i] = mutual_info_regression(
        feat_global_zscore[:, i:i+1], target_aligned,
        n_neighbors=5, random_state=42, n_jobs=1
    )[0]

# Print results
print(f"\n  {'Feature':<20} {'MI_raw':>10} {'MI_pw90':>10} {'MI_global':>10} {'pw90/raw':>10} {'global/raw':>10}")
print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

order = np.argsort(-mi_raw)
for idx in order:
    name = RAW_FEATURE_NAMES[idx]
    raw = mi_raw[idx]
    pw90 = mi_zscore90[idx]
    glb = mi_global[idx]
    ratio_pw90 = pw90 / max(raw, 1e-10)
    ratio_glb = glb / max(raw, 1e-10)
    
    flag = "✓" if ratio_pw90 > 0.8 else ("~" if ratio_pw90 > 0.5 else "✗")
    print(f"  {name:<20} {raw:>10.6f} {pw90:>10.6f} {glb:>10.6f} {ratio_pw90:>9.1%} {ratio_glb:>9.1f} {flag}")

# ═══════════════════════════════════════════════════════════════════════════════
# DC/AC decomposition for raw features
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("  DC/AC DECOMPOSITION for raw features")
print("─" * 80)
print("  DC = window mean (destroyed by z-score)")
print("  AC = residual from window mean (preserved by z-score)\n")

dc_mis = np.zeros(len(RAW_FEATURE_NAMES))
ac_mis = np.zeros(len(RAW_FEATURE_NAMES))

for i, name in enumerate(RAW_FEATURE_NAMES):
    window_means = np.zeros(n_windows)
    window_residuals = np.zeros(n_windows)
    
    for w in range(n_windows):
        window = feat_sub[w:w+LOOKBACK, i]
        w_mean = np.mean(window)
        window_means[w] = w_mean
        window_residuals[w] = window[-1] - w_mean
    
    if np.std(window_means) > 1e-10:
        dc_mis[i] = mutual_info_regression(
            window_means.reshape(-1, 1), target_aligned,
            n_neighbors=5, random_state=42, n_jobs=1
        )[0]
    if np.std(window_residuals) > 1e-10:
        ac_mis[i] = mutual_info_regression(
            window_residuals.reshape(-1, 1), target_aligned,
            n_neighbors=5, random_state=42, n_jobs=1
        )[0]

print(f"  {'Feature':<20} {'MI_raw':>10} {'MI_DC':>10} {'MI_AC':>10} {'DC%':>8} {'AC%':>8}")
print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")

for idx in order:
    name = RAW_FEATURE_NAMES[idx]
    raw = mi_raw[idx]
    dc = dc_mis[idx]
    ac = ac_mis[idx]
    dc_pct = (dc / raw * 100) if raw > 0 else 0
    ac_pct = (ac / raw * 100) if raw > 0 else 0
    flag = "⚠ DC" if dc_pct > 50 else "✓ AC"
    print(f"  {name:<20} {raw:>10.6f} {dc:>10.6f} {ac:>10.6f} {dc_pct:>7.1f}% {ac_pct:>7.1f}% {flag}")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary statistics
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  SUMMARY")
print("=" * 80)

n_preserved = sum(1 for i in range(len(RAW_FEATURE_NAMES)) if mi_zscore90[i] > 0.8 * mi_raw[i])
n_partial   = sum(1 for i in range(len(RAW_FEATURE_NAMES)) if 0.5 * mi_raw[i] < mi_zscore90[i] <= 0.8 * mi_raw[i])
n_destroyed = sum(1 for i in range(len(RAW_FEATURE_NAMES)) if mi_zscore90[i] <= 0.5 * mi_raw[i])
n_dc_dominant = sum(1 for i in range(len(RAW_FEATURE_NAMES)) if dc_mis[i] > ac_mis[i] and mi_raw[i] > 0.001)

total_raw_mi = np.sum(mi_raw)
total_pw90_mi = np.sum(mi_zscore90)
total_global_mi = np.sum(mi_global)

print(f"""
  Features with >80% signal preserved by pw-zscore(90): {n_preserved}/{len(RAW_FEATURE_NAMES)}
  Features with 50-80% signal preserved:                {n_partial}/{len(RAW_FEATURE_NAMES)}
  Features with <50% signal preserved (destroyed):      {n_destroyed}/{len(RAW_FEATURE_NAMES)}
  Features where DC > AC (signal in window mean):       {n_dc_dominant}/{len(RAW_FEATURE_NAMES)}

  Total MI across all features:
    Raw (no normalization):     {total_raw_mi:.6f}
    Per-window z-score (90):    {total_pw90_mi:.6f}  ({total_pw90_mi/total_raw_mi*100:.1f}% of raw)
    Global z-score only:        {total_global_mi:.6f}  ({total_global_mi/total_raw_mi*100:.1f}% of raw)
""")

if total_pw90_mi > total_raw_mi * 0.9:
    print("  🟢 VERDICT: Per-window z-score(90) PRESERVES signal well!")
    print("     Raw features + pw-zscore(90) is a viable pipeline.")
elif total_pw90_mi > total_raw_mi * 0.5:
    print("  🟡 VERDICT: Per-window z-score(90) preserves signal partially.")
    print("     Some features lose signal, but overall information survives.")
else:
    print("  🔴 VERDICT: Per-window z-score(90) DESTROYS most signal!")
    print("     Even with raw features, the z-score removes too much information.")

if total_global_mi > total_pw90_mi:
    print(f"\n  Global z-score ({total_global_mi:.6f}) > Per-window z-score ({total_pw90_mi:.6f})")
    print("  → Global normalization preserves MORE signal than per-window")
else:
    print(f"\n  Per-window z-score ({total_pw90_mi:.6f}) >= Global z-score ({total_global_mi:.6f})")
    print("  → Per-window normalization is competitive with global")

print("\n" + "=" * 80)
print("  AUDIT COMPLETE")
print("=" * 80)
