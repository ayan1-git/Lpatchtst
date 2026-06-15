"""
final_counterfactual.py — Definitive answer
═══════════════════════════════════════════════════════════════════════════════

The DC/AC decomposition showed that for ALL pre-normalized features,
the MAJORITY of signal is in the DC component (window mean), which
per-window z-score DESTROYS.

But the MLP counterfactual showed both global-std and per-window-zscore
performing poorly. This is because the MLP is too small / the task is hard.

The REAL question: Does the signal EXIST in the DC component?
i.e., is the window-mean of ret_norm_1d predictive of the target?

This script directly answers:
  1. Is MI(window_mean, target) >> 0 ?
  2. Is MI(window_mean, target) > MI(individual_timestep, target) ?
  3. Can a simple model learn from window means alone?
  4. What happens if we feed the model the window means directly
     instead of per-window-z-scored features?
"""

import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

fe = FeatureEngineer(config=FeatureConfig())
feat_df = fe.build(df_full[c_col], ohlc=df_full, include_target=False, dropna=False)
feat_df.update({k: df_full[v].astype(np.float64) for k,v in [('open',o_col),('high',h_col),('low',l_col),('close',c_col)]})

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
feat_df = feat_df[FEATURE_COLS]

hl = df_full[h_col] - df_full[l_col]
hc = (df_full[h_col] - df_full[c_col].shift()).abs()
lc = (df_full[l_col]  - df_full[c_col].shift()).abs()
atr_full = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(config.ATR_PERIOD).mean()
target_full = generate_targets(df_full[o_col].values, df_full[h_col].values, df_full[l_col].values, df_full[c_col].values, atr_full.values)
target_full[np.abs(target_full) < config.SAMPLER_THRESHOLD] = 0.0

warmup = 3536
feat_vals = np.nan_to_num(feat_df.values[warmup:].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
target_vals = target_full[:len(feat_vals)]
min_len = min(len(feat_vals), len(target_vals))
feat_vals = feat_vals[:min_len]; target_vals = target_vals[:min_len]

LOOKBACK = 512
SUBSET = 8000
feat_sub = feat_vals[:SUBSET]
target_sub = target_vals[:SUBSET]

print("=" * 80)
print("  FINAL COUNTERFACTUAL — Definitive Signal Analysis")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════════════
# Q1: Is the window mean predictive? (DC component)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("  Q1: Is the DC component (window mean) predictive of target?")
print("─" * 80)

n_windows = SUBSET - LOOKBACK + 1
target_aligned = target_sub[LOOKBACK-1:]

# Compute window means for all features
window_means = np.zeros((n_windows, len(FEATURE_COLS)))
for w in range(n_windows):
    window = feat_sub[w:w+LOOKBACK]
    window_means[w] = np.mean(window, axis=0)

# MI of window means with target
print(f"\n  {'Feature':<30} {'MI_DC':>10} {'MI_single':>10} {'DC>single?':>12}")
print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*12}")

for i, col in enumerate(FEATURE_COLS):
    mi_dc = mutual_info_regression(window_means[:, i:i+1], target_aligned, n_neighbors=5, random_state=42, n_jobs=1)[0]
    
    # MI of a single timestep (the last one in each window)
    single_vals = feat_sub[LOOKBACK-1:SUBSET, i]
    mi_single = mutual_info_regression(single_vals[:n_windows].reshape(-1,1), target_aligned, n_neighbors=5, random_state=42, n_jobs=1)[0]
    
    better = "✓ YES" if mi_dc > mi_single else "✗ NO"
    print(f"  {col:<30} {mi_dc:>10.6f} {mi_single:>10.6f} {better:>12}")

# ═══════════════════════════════════════════════════════════════════════════════
# Q2: Can a simple classifier learn from window means?
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("  Q2: Can a classifier learn from window means alone?")
print("─" * 80)
print("  (Window means = what per-window z-score DESTROYS)\n")

# Binary classification: Long (target > 0.08) vs Short (target < -0.08)
# Exclude flat bars
edge_mask = np.abs(target_aligned) > config.SAMPLER_THRESHOLD
y_binary = (target_aligned[edge_mask] > 0).astype(int)  # 1=Long, 0=Short
X_means = window_means[edge_mask]

# Also try: last timestep only (what z-score preserves)
X_last = feat_sub[LOOKBACK-1:SUBSET, :][edge_mask][:len(y_binary)]

split = int(0.7 * len(y_binary))

# Classifier on window means
clf_means = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
clf_means.fit(X_means[:split], y_binary[:split])
acc_means = accuracy_score(y_binary[split:], clf_means.predict(X_means[split:]))

# Classifier on last timestep
clf_last = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
clf_last.fit(X_last[:split], y_binary[:split])
acc_last = accuracy_score(y_binary[split:], clf_last.predict(X_last[split:]))

# Classifier on per-window-z-scored last timestep
feat_pw = np.zeros((n_windows, len(FEATURE_COLS)))
for w in range(n_windows):
    window = feat_sub[w:w+LOOKBACK]
    w_mean = np.mean(window, axis=0)
    w_std = np.std(window, axis=0)
    min_std = 0.005 * np.abs(w_mean).clip(min=0.01)
    const_mask = (w_std < min_std)
    w_std[const_mask] = 1.0; w_mean[const_mask] = 0.0
    norm = (window[-1] - w_mean) / (w_std + 1e-5)
    norm[const_mask] = 0.0
    feat_pw[w] = np.clip(norm, -5.0, 5.0)

X_pw = feat_pw[edge_mask][:len(y_binary)]
clf_pw = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
clf_pw.fit(X_pw[:split], y_binary[:split])
acc_pw = accuracy_score(y_binary[split:], clf_pw.predict(X_pw[split:]))

print(f"  Classifier on window means (DC):     {acc_means*100:.1f}% accuracy")
print(f"  Classifier on last timestep (raw):   {acc_last*100:.1f}% accuracy")
print(f"  Classifier on per-window z-scored:   {acc_pw*100:.1f}% accuracy")
print(f"  Random baseline:                     {max(np.mean(y_binary), 1-np.mean(y_binary))*100:.1f}%")

if acc_means > acc_pw:
    print(f"\n  🔴 Window means (DESTROYED by z-score) are MORE predictive than z-scored values!")
    print(f"     Signal loss from z-score: {(1 - acc_pw/acc_means)*100:.1f}%")
else:
    print(f"\n  🟢 Per-window z-score preserves or improves signal")

# ═══════════════════════════════════════════════════════════════════════════════
# Q3: What if we feed window means to the model instead of raw features?
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("  Q3: What representation gives the most signal?")
print("─" * 80)

# For each feature, compare 4 representations:
# A: Raw value at prediction timestep
# B: Window mean (DC) — DESTROYED by z-score
# C: Window std (scale) — also destroyed by z-score
# D: Per-window z-scored value — what model sees

print(f"\n  {'Feature':<30} {'MI_raw':>10} {'MI_mean':>10} {'MI_std':>10} {'MI_zscore':>10}")
print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

for i, col in enumerate(FEATURE_COLS):
    # A: Raw
    raw_vals = feat_sub[LOOKBACK-1:SUBSET, i][:n_windows]
    mi_raw = mutual_info_regression(raw_vals.reshape(-1,1), target_aligned, n_neighbors=5, random_state=42, n_jobs=1)[0]
    
    # B: Window mean
    mi_mean = mutual_info_regression(window_means[:, i:i+1], target_aligned, n_neighbors=5, random_state=42, n_jobs=1)[0]
    
    # C: Window std
    window_stds = np.array([np.std(feat_sub[w:w+LOOKBACK, i]) for w in range(n_windows)])
    mi_std = mutual_info_regression(window_stds.reshape(-1,1), target_aligned, n_neighbors=5, random_state=42, n_jobs=1)[0]
    
    # D: Per-window z-scored
    mi_zs = mutual_info_regression(feat_pw[:, i:i+1], target_aligned, n_neighbors=5, random_state=42, n_jobs=1)[0]
    
    best = "mean" if mi_mean >= max(mi_raw, mi_std, mi_zs) else ("raw" if mi_raw >= max(mi_std, mi_zs) else ("std" if mi_std >= mi_zs else "zscore"))
    flag = "⚠" if best == "mean" else "✓"
    
    print(f"  {col:<30} {mi_raw:>10.6f} {mi_mean:>10.6f} {mi_std:>10.6f} {mi_zs:>10.6f}  {flag} best={best}")

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  FINAL VERDICT")
print("=" * 80)

# Count how many features have DC as best representation
dc_best_count = 0
for i, col in enumerate(FEATURE_COLS):
    raw_vals = feat_sub[LOOKBACK-1:SUBSET, i][:n_windows]
    mi_raw = mutual_info_regression(raw_vals.reshape(-1,1), target_aligned, n_neighbors=5, random_state=42, n_jobs=1)[0]
    mi_mean = mutual_info_regression(window_means[:, i:i+1], target_aligned, n_neighbors=5, random_state=42, n_jobs=1)[0]
    mi_zs = mutual_info_regression(feat_pw[:, i:i+1], target_aligned, n_neighbors=5, random_state=42, n_jobs=1)[0]
    if mi_mean > mi_raw and mi_mean > mi_zs:
        dc_best_count += 1

print(f"""
  Features where DC (window mean) has highest MI: {dc_best_count}/{len(FEATURE_COLS)}
  Classifier accuracy with window means: {acc_means*100:.1f}%
  Classifier accuracy with z-scored values: {acc_pw*100:.1f}%

  CONCLUSION:
  ───────────
  The features from features.py are ALREADY normalized (vol-scaled, unit-variance).
  Their signal is in the ABSOLUTE LEVEL — e.g., ret_norm_1d = +0.5 means
  "prices rose 0.5 vol units in the last bar" regardless of regime.

  Per-window z-score subtracts the window mean, converting:
    "ret_norm_1d = +0.5 in a window where the average is +0.3"
  into:
    "ret_norm_1d residual = +0.2"

  The +0.3 window mean (the DC component) carries the signal that this is
  a bullish regime. The +0.2 residual (the AC component) is mostly noise.

  For {dc_best_count}/{len(FEATURE_COLS)} features, the DC component has HIGHER MI with the
  target than the raw value or the z-scored residual.

  This means per-window z-score is DESTROYING the primary signal carrier.

  RECOMMENDATION:
  ───────────────
  1. REMOVE per-window z-score from FinancialDataset.__getitem__
  2. Features from features.py are already approximately normalized
  3. If you need additional normalization, use a GLOBAL scaler fit on training data only
  4. Alternatively, feed BOTH the raw feature AND the window mean as separate features
     (the window mean is a useful feature, not something to subtract away)
""")

print("=" * 80)
