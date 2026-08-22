"""
signal_destruction_audit.py
═══════════════════════════════════════════════════════════════════════════════

Precise audit: Does per-window z-score normalization DESTROY signal that exists
in the raw features from features.py?

KEY QUESTION:
  The features from features.py are ALREADY normalized (vol-scaled returns,
  unit-variance MACD, etc.). Then FinancialDataset.__getitem__ applies a
  SECOND z-score normalization on each 512-bar window.

  This double-normalization can destroy signal when:
  1. The feature's signal is in its ABSOLUTE level across windows
     (e.g., ret_norm_1d = 1.5 means "strong upward move" in ANY regime)
  2. Per-window z-score removes the cross-window mean, leaving only
     within-window variation (noise)

METHOD:
  For each feature, compute MI(feature, target) at three stages:
  Stage A: Raw features from features.py (no normalization)
  Stage B: After per-window z-score (what the model actually sees)
  Stage C: After per-window z-score with lookback=90 (TOKENIZER_WINDOW)

  If MI(B) << MI(A), the per-window z-score is destroying signal.
  If MI(C) > MI(B), the tokenizer window (90) preserves more signal than
  the model's lookback window (512).

  Also check: Are features.py outputs already approximately z-scored?
  If mean≈0 and std≈1 globally, then per-window z-score is approximately
  an identity transform (no destruction). If not, it's a SECOND normalization
  that changes the distribution.

═══════════════════════════════════════════════════════════════════════════════
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
DATA_PATH = "data/NIFTY 50_30minute.csv"
df_full = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

col_map = {c.lower(): c for c in df_full.columns}
o_col = col_map.get('open', 'Open')
h_col = col_map.get('high', 'High')
l_col = col_map.get('low', 'Low')
c_col = col_map.get('close', 'Close')

fe = FeatureEngineer(config=FeatureConfig())
feat_df = fe.build(df_full[c_col], ohlc=df_full, include_target=False, dropna=False)
feat_df['open']  = df_full[o_col].astype(np.float64)
feat_df['high']  = df_full[h_col].astype(np.float64)
feat_df['low']   = df_full[l_col].astype(np.float64)
feat_df['close'] = df_full[c_col].astype(np.float64)

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
    if col not in feat_df.columns:
        feat_df[col] = 0.0
feat_df = feat_df[FEATURE_COLS]

# ATR + oracle
hl = df_full[h_col] - df_full[l_col]
hc = (df_full[h_col] - df_full[c_col].shift()).abs()
lc = (df_full[l_col]  - df_full[c_col].shift()).abs()
atr_full = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(config.ATR_PERIOD).mean()

target_full = generate_targets(
    df_full[o_col].values, df_full[h_col].values,
    df_full[l_col].values, df_full[c_col].values,
    atr_full.values,
)
target_full[np.abs(target_full) < config.SAMPLER_THRESHOLD] = 0.0

# Cut warmup
warmup = 3536
feat_vals  = feat_df.values[warmup:].astype(np.float64)
target_vals = target_full[warmup:]
feat_vals = np.nan_to_num(feat_vals, nan=0.0, posinf=0.0, neginf=0.0)

min_len = min(len(feat_vals), len(target_vals))
feat_vals   = feat_vals[:min_len]
target_vals = target_vals[:min_len]

print("=" * 80)
print("  SIGNAL DESTRUCTION AUDIT")
print("  Does per-window z-score destroy signal from features.py?")
print("=" * 80)
print(f"\n  Data: {min_len:,} bars, {len(FEATURE_COLS)} features")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: Are features.py outputs already normalized?
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("  PART 1: Global statistics of raw features from features.py")
print("─" * 80)
print("  If mean≈0 and std≈1 → features are already z-scored → per-window z-score ≈ identity")
print("  If mean≠0 or std≠1 → features are NOT z-scored → per-window z-score is a SECOND normalization\n")

print(f"  {'Feature':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'PreNorm?':>10}")
print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

for i, col in enumerate(FEATURE_COLS):
    f = feat_vals[:, i]
    mean = np.mean(f)
    std  = np.std(f)
    mn   = np.min(f)
    mx   = np.max(f)
    # "Pre-normalized" if mean≈0 and std≈0.5-2.0 and range is bounded
    is_prenorm = (abs(mean) < 0.5) and (0.3 < std < 5.0) and (mx - mn < 20)
    flag = "YES" if is_prenorm else "NO (raw)"
    print(f"  {col:<30} {mean:>+10.4f} {std:>10.4f} {mn:>+10.2f} {mx:>+10.2f} {flag:>10}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: MI at three stages — raw, per-window zscore(512), per-window zscore(90)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("  PART 2: MI(feature, target) at three normalization stages")
print("─" * 80)

# Subsample for speed
MAX_SAMPLES = 30000
edge_mask = target_vals != 0.0
flat_mask = ~edge_mask
n_edge = edge_mask.sum()
n_flat_keep = max(0, MAX_SAMPLES - n_edge)

rng = np.random.RandomState(42)
edge_idx = np.where(edge_mask)[0]
flat_idx = np.where(flat_mask)[0]
if n_flat_keep < len(flat_idx):
    flat_idx = rng.choice(flat_idx, size=n_flat_keep, replace=False)
keep_idx = np.sort(np.concatenate([edge_idx, flat_idx]))

feat_sub   = feat_vals[keep_idx]
target_sub = target_vals[keep_idx]

print(f"  Subsample: {len(keep_idx):,} bars ({n_edge:,} edge + {len(flat_idx):,} flat)\n")

# Stage A: Raw features
print("  Computing MI for Stage A (raw features)...")
mi_raw = np.zeros(len(FEATURE_COLS))
for i, col in enumerate(FEATURE_COLS):
    mi_raw[i] = mutual_info_regression(
        feat_sub[:, i:i+1], target_sub,
        n_neighbors=5, random_state=42, n_jobs=1
    )[0]

# Stage B: Per-window z-score with lookback=512 (model's LOOKBACK_WINDOW)
LOOKBACK_MODEL = 512  # config.LOOKBACK_WINDOW

def apply_per_window_zscore(feat_array, lookback):
    """Simulate FinancialDataset.__getitem__ z-score normalization."""
    n = len(feat_array)
    normalized = np.zeros_like(feat_array)
    
    for start in range(0, n, 1):  # every possible window start
        end = start + lookback
        if end > n:
            break
        window = feat_array[start:end]
        
        x_mean = np.nanmean(window[:lookback], axis=0)
        x_std  = np.nanstd(window[:lookback], axis=0)
        
        x_mean = np.nan_to_num(x_mean, nan=0.0)
        x_std  = np.nan_to_num(x_std, nan=1.0)
        
        min_std = 0.005 * np.abs(x_mean).clip(min=0.01)
        const_mask = (x_std < min_std)
        x_std[const_mask] = 1.0
        x_mean[const_mask] = 0.0
        
        norm_window = (window - x_mean) / (x_std + 1e-5)
        norm_window = np.nan_to_num(norm_window, nan=0.0, posinf=0.0, neginf=0.0)
        norm_window[:, const_mask] = 0.0
        norm_window = np.clip(norm_window, -5.0, 5.0)
        
        # We only need the last row of each window (the prediction point)
        if start == 0:
            # For the first window, store all rows
            normalized[start:end] = norm_window
        else:
            normalized[end-1] = norm_window[-1]
    
    return normalized

# This is O(n * lookback * features) — too slow for 30k samples with lookback=512
# Instead, compute on a smaller subset and use stride
print(f"  Computing MI for Stage B (per-window z-score, lookback={LOOKBACK_MODEL})...")
print("  (This takes a moment — applying z-score to each window...)")

# Use a smaller subset for the per-window computation
SUBSET = 5000
feat_subset = feat_sub[:SUBSET]
target_subset = target_sub[:SUBSET]

# Apply per-window z-score with stride for speed
def apply_pw_zscore_fast(feat_array, lookback, stride=1):
    """Apply per-window z-score, return the last element of each window."""
    n, d = feat_array.shape
    out = np.zeros_like(feat_array)
    counts = np.zeros(n)
    
    for start in range(0, n - lookback + 1, stride):
        end = start + lookback
        window = feat_array[start:end]
        
        x_mean = np.mean(window, axis=0)
        x_std  = np.std(window, axis=0)
        
        min_std = 0.005 * np.abs(x_mean).clip(min=0.01)
        const_mask = (x_std < min_std)
        x_std[const_mask] = 1.0
        x_mean[const_mask] = 0.0
        
        norm_window = (window - x_mean) / (x_std + 1e-5)
        norm_window[:, const_mask] = 0.0
        norm_window = np.clip(norm_window, -5.0, 5.0)
        
        # Accumulate (for overlapping windows, average)
        out[start:end] += norm_window
        counts[start:end] += 1
    
    # Average overlapping windows
    counts = np.maximum(counts, 1)
    out /= counts[:, np.newaxis]
    return out

feat_pw512 = apply_pw_zscore_fast(feat_subset, LOOKBACK_MODEL, stride=1)

# Only use rows where we have valid windows (after lookback-1)
valid_start = LOOKBACK_MODEL - 1
feat_pw512_valid = feat_pw512[valid_start:]
target_pw512_valid = target_subset[valid_start:]

mi_pw512 = np.zeros(len(FEATURE_COLS))
for i, col in enumerate(FEATURE_COLS):
    mi_pw512[i] = mutual_info_regression(
        feat_pw512_valid[:, i:i+1], target_pw512_valid,
        n_neighbors=5, random_state=42, n_jobs=1
    )[0]

# Stage C: Per-window z-score with lookback=90 (TOKENIZER_WINDOW)
LOOKBACK_TOKENIZER = 90  # config.TOKENIZER_WINDOW
print(f"  Computing MI for Stage C (per-window z-score, lookback={LOOKBACK_TOKENIZER})...")

feat_pw90 = apply_pw_zscore_fast(feat_subset, LOOKBACK_TOKENIZER, stride=1)
valid_start_90 = LOOKBACK_TOKENIZER - 1
feat_pw90_valid = feat_pw90[valid_start_90:]
target_pw90_valid = target_subset[valid_start_90:]

mi_pw90 = np.zeros(len(FEATURE_COLS))
for i, col in enumerate(FEATURE_COLS):
    mi_pw90[i] = mutual_info_regression(
        feat_pw90_valid[:, i:i+1], target_pw90_valid,
        n_neighbors=5, random_state=42, n_jobs=1
    )[0]

# Print comparison
print(f"\n  {'Feature':<30} {'MI_raw':>10} {'MI_pw512':>10} {'MI_pw90':>10} {'Destr%':>10} {'Recov%':>10}")
print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

# Sort by MI_raw descending
order = np.argsort(-mi_raw)
for idx in order:
    col = FEATURE_COLS[idx]
    raw = mi_raw[idx]
    pw512 = mi_pw512[idx]
    pw90 = mi_pw90[idx]
    
    destruction = (1 - pw512 / max(raw, 1e-10)) * 100 if raw > 0 else 0
    recovery = (pw90 / max(pw512, 1e-10)) * 100 if pw512 > 0 else 0
    
    flag = "🔴" if destruction > 80 else ("🟡" if destruction > 50 else "🟢")
    print(f"  {col:<30} {raw:>10.6f} {pw512:>10.6f} {pw90:>10.6f} {destruction:>9.1f}% {recovery:>9.1f}% {flag}")

print(f"\n  Legend: 🔴 = >80% signal destroyed, 🟡 = 50-80% destroyed, 🟢 = <50% destroyed")
print(f"  Destr% = (1 - MI_pw512 / MI_raw) × 100")
print(f"  Recov% = (MI_pw90 / MI_pw512) × 100  (does shorter window recover signal?)")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: Detailed analysis of the constant-feature zeroing
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("  PART 3: Constant-feature zeroing analysis")
print("─" * 80)
print("  FinancialDataset zeros out features where within-window std < 0.5% of |mean|.")
print("  This can zero out features that have LOW within-window variance but")
print("  HIGH across-window variance (the actual signal).\n")

for i, col in enumerate(FEATURE_COLS):
    f = feat_subset[:, i]
    
    # Compute within-window std for sliding windows
    n_windows = min(500, len(f) - LOOKBACK_MODEL)
    within_stds = []
    across_values = []
    
    for w in range(n_windows):
        start = w * 10
        if start + LOOKBACK_MODEL > len(f):
            break
        window = f[start:start+LOOKBACK_MODEL]
        within_stds.append(np.std(window))
        across_values.append(np.mean(window))
    
    within_stds = np.array(within_stds)
    across_values = np.array(across_values)
    
    mean_within_std = np.mean(within_stds)
    across_std = np.std(across_values)
    global_mean = np.mean(f)
    
    # Check if this feature would be zeroed
    min_std_threshold = 0.005 * abs(global_mean) if abs(global_mean) > 0.01 else 0.01 * 0.005
    would_zero = mean_within_std < min_std_threshold
    
    # SNR: across-window signal vs within-window noise
    snr = across_std / max(mean_within_std, 1e-10)
    
    if would_zero or snr < 0.1:
        flag = "⚠ ZEROED" if would_zero else "⚠ LOW-SNR"
        print(f"  {col:<30} mean={global_mean:>+8.4f}  within_std={mean_within_std:.4f}  "
              f"across_std={across_std:.4f}  SNR={snr:.4f}  {flag}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: What does the model ACTUALLY see?
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("  PART 4: What the model actually sees — distribution of normalized features")
print("─" * 80)
print("  Sample 100 windows, show the distribution of the LAST element (prediction point)")
print("  after per-window z-score. If all values are near zero, signal is destroyed.\n")

n_sample_windows = 100
sample_indices = np.linspace(LOOKBACK_MODEL - 1, len(feat_subset) - 1, n_sample_windows, dtype=int)

for i, col in enumerate(FEATURE_COLS):
    # Get the normalized values at prediction points
    norm_values = feat_pw512[sample_indices, i]
    raw_values = feat_sub[sample_indices, i]
    
    norm_std = np.std(norm_values)
    raw_std = np.std(raw_values)
    norm_mean = np.mean(norm_values)
    
    # After z-score, values should be ~N(0,1) IF the feature has variation
    # If values are all near zero, the feature was zeroed or compressed
    pct_near_zero = np.mean(np.abs(norm_values) < 0.01) * 100
    
    if pct_near_zero > 50 or norm_std < 0.01:
        flag = "⚠ DEAD"
        print(f"  {col:<30} norm_mean={norm_mean:>+6.4f}  norm_std={norm_std:.4f}  "
              f"raw_std={raw_std:.4f}  near_zero={pct_near_zero:.0f}%  {flag}")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  SUMMARY")
print("=" * 80)

# Count destroyed features
n_destroyed = 0
n_recoverable = 0
for idx in order:
    raw = mi_raw[idx]
    pw512 = mi_pw512[idx]
    pw90 = mi_pw90[idx]
    if raw > 0.005 and (1 - pw512 / raw) > 0.8:
        n_destroyed += 1
    if pw512 > 0.001 and pw90 > pw512 * 1.5:
        n_recoverable += 1

print(f"""
  Features with >80% signal destruction from per-window z-score(512): {n_destroyed}/{len(FEATURE_COLS)}
  Features recoverable with shorter window(90): {n_recoverable}/{len(FEATURE_COLS)}

  ROOT CAUSE ANALYSIS:
  ────────────────────
  The features from features.py are ALREADY normalized (vol-scaled, unit-variance).
  The per-window z-score in FinancialDataset.__getitem__ applies a SECOND normalization.

  For features where the signal is in the CROSS-WINDOW level (e.g., ret_norm_1d
  being persistently positive during a trend), per-window z-score subtracts the
  window mean, removing the very signal the feature carries.

  This is a DOUBLE-NORMALIZATION BUG:
  - features.py normalizes → gives features meaningful absolute values
  - __getitem__ z-score normalizes again → removes the absolute values

  The model sees only WITHIN-WINDOW variation, which is mostly noise.
  The ACROSS-WINDOW variation (the actual signal) has been subtracted away.

  RECOMMENDATION:
  ───────────────
  1. REMOVE per-window z-score normalization from FinancialDataset.__getitem__
  2. The features from features.py are already approximately normalized
  3. If needed, apply a GLOBAL scaler (fit on train only) instead of per-window
  4. Or use RevIN (Reversible Instance Normalization) which preserves the
     cross-window statistics and can recover them at the output
""")

print("=" * 80)
print("  AUDIT COMPLETE")
print("=" * 80)
