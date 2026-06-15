"""
double_norm_audit.py
═══════════════════════════════════════════════════════════════════════════════

Focused audit: The features from features.py fall into TWO categories:

  Category A — ALREADY NORMALIZED (mean≈0, std≈1):
    ret_norm_*, macd_*, feat_efficiency, feat_local_structure, feat_momentum_rsi
    → These are volatility-scaled, dimensionless, concentrated in [-2, 2]
    → Per-window z-score is approximately an IDENTITY transform (no destruction)
    → The signal is in the ABSOLUTE VALUE at each timestep

  Category B — RAW / DIFFERENT SCALE:
    open, high, low, close (price levels ~15000)
    feat_session_sin (range [0, 1], mean ~0.61)
    feat_session_cos (range [-1, 1], mean ~0.04)
    feat_vol_squeeze (range [0.2, 3.5], mean ~1.0)
    feat_icp (range [-0.7, 0.8], mean ~0.03)
    feat_vol_asymmetry (range [-0.8, 0.7], mean ~0.02)

  For Category A: per-window z-score should be ~identity (no destruction)
  For Category B: per-window z-score is a SECOND normalization (may help or hurt)

  BUT WAIT: The key insight is that for Category A features, the signal is in
  the PERSISTENCE of values across time. ret_norm_1d = +0.5 for 10 consecutive
  bars means something different than ret_norm_1d oscillating ±0.5.

  Per-window z-score removes the DC component (mean) of each window, which
  DESTROYS the persistence signal.

  This audit measures: for each feature, what fraction of MI comes from
  the DC component (window mean) vs the AC component (within-window variation)?

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
    if col not in feat_df.columns:
        feat_df[col] = 0.0
feat_df = feat_df[FEATURE_COLS]

hl = df_full[h_col] - df_full[l_col]
hc = (df_full[h_col] - df_full[c_col].shift()).abs()
lc = (df_full[l_col]  - df_full[c_col].shift()).abs()
atr_full = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(config.ATR_PERIOD).mean()

target_full = generate_targets(
    df_full[o_col].values, df_full[h_col].values,
    df_full[l_col].values, df_full[c_col].values, atr_full.values,
)
target_full[np.abs(target_full) < config.SAMPLER_THRESHOLD] = 0.0

warmup = 3536
feat_vals   = feat_df.values[warmup:].astype(np.float64)
target_vals = target_full[warmup:]
feat_vals = np.nan_to_num(feat_vals, nan=0.0, posinf=0.0, neginf=0.0)
min_len = min(len(feat_vals), len(target_vals))
feat_vals = feat_vals[:min_len]
target_vals = target_vals[:min_len]

# ═══════════════════════════════════════════════════════════════════════════════
# For each feature, decompose MI into DC (window mean) and AC (residual) parts
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("  DOUBLE-NORMALIZATION AUDIT")
print("  Decomposing MI into DC (window mean) vs AC (within-window) components")
print("=" * 80)

LOOKBACK = 512  # model's LOOKBACK_WINDOW

# Use a manageable subset
SUBSET = 8000
feat_sub = feat_vals[:SUBSET]
target_sub = target_vals[:SUBSET]

# Compute window means and residuals
n_windows = SUBSET - LOOKBACK + 1
print(f"\n  Computing window statistics for {n_windows} windows of size {LOOKBACK}...")

dc_mis = np.zeros(len(FEATURE_COLS))   # MI from window means (DC component)
ac_mis = np.zeros(len(FEATURE_COLS))   # MI from residuals (AC component)
raw_mis = np.zeros(len(FEATURE_COLS))  # MI from raw features

# Compute raw MI on the same window-aligned subset
target_aligned = target_sub[LOOKBACK-1:]
feat_aligned   = feat_sub[LOOKBACK-1:]

for i, col in enumerate(FEATURE_COLS):
    raw_mis[i] = mutual_info_regression(
        feat_aligned[:, i:i+1], target_aligned,
        n_neighbors=5, random_state=42, n_jobs=1
    )[0]

# Compute DC and AC components
for i, col in enumerate(FEATURE_COLS):
    window_means = []
    window_residuals_at_end = []
    
    for w in range(n_windows):
        start = w
        end = start + LOOKBACK
        window = feat_sub[start:end, i]
        
        w_mean = np.mean(window)
        w_std  = np.std(window)
        
        # The residual at the prediction point (last element)
        last_val = window[-1]
        residual = last_val - w_mean  # this is what z-score preserves (divided by std)
        
        window_means.append(w_mean)
        window_residuals_at_end.append(residual)
    
    window_means = np.array(window_means)
    window_residuals = np.array(window_residuals_at_end)
    window_targets = target_sub[LOOKBACK-1:]
    
    # MI from DC component (window means)
    if np.std(window_means) > 1e-10:
        dc_mis[i] = mutual_info_regression(
            window_means.reshape(-1, 1), window_targets,
            n_neighbors=5, random_state=42, n_jobs=1
        )[0]
    
    # MI from AC component (residuals)
    if np.std(window_residuals) > 1e-10:
        ac_mis[i] = mutual_info_regression(
            window_residuals.reshape(-1, 1), window_targets,
            n_neighbors=5, random_state=42, n_jobs=1
        )[0]

# Print results
print(f"\n  {'Feature':<30} {'MI_raw':>10} {'MI_DC':>10} {'MI_AC':>10} {'DC%':>8} {'AC%':>8}")
print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")

order = np.argsort(-raw_mis)
for idx in order:
    col = FEATURE_COLS[idx]
    raw = raw_mis[idx]
    dc  = dc_mis[idx]
    ac  = ac_mis[idx]
    
    dc_pct = (dc / raw * 100) if raw > 0 else 0
    ac_pct = (ac / raw * 100) if raw > 0 else 0
    
    # DC% = fraction of signal in the window mean (DESTROYED by z-score)
    # AC% = fraction of signal in the residuals (PRESERVED by z-score)
    
    flag = "⚠ DC" if dc_pct > 50 else "✓ AC"
    print(f"  {col:<30} {raw:>10.6f} {dc:>10.6f} {ac:>10.6f} {dc_pct:>7.1f}% {ac_pct:>7.1f}% {flag}")

print(f"\n  DC% = signal in window mean (DESTROYED by per-window z-score)")
print(f"  AC% = signal in residuals (PRESERVED by per-window z-score)")
print(f"  ⚠ DC = >50% of signal is in the DC component → z-score destroys it")

# ═══════════════════════════════════════════════════════════════════════════════
# Key question: For pre-normalized features, is the signal in DC or AC?
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("  CATEGORY ANALYSIS: Pre-normalized vs Raw features")
print("─" * 80)

prenorm_cols = ['ret_norm_1d', 'ret_norm_3d', 'ret_norm_6d', 'ret_norm_13d',
                'ret_norm_26d', 'ret_norm_65d', 'ret_norm_130d', 'ret_norm_260d',
                'macd_8_24', 'macd_26_78', 'macd_52_156',
                'feat_efficiency', 'feat_local_structure']

raw_cols = ['open', 'high', 'low', 'close',
            'feat_icp', 'feat_momentum_rsi', 'feat_vol_asymmetry',
            'feat_session_sin', 'feat_session_cos', 'feat_vol_squeeze']

print(f"\n  {'Category':<20} {'Avg MI_raw':>12} {'Avg DC%':>10} {'Avg AC%':>10}")
print(f"  {'─'*20} {'─'*12} {'─'*10} {'─'*10}")

for cat_name, cat_cols in [("Pre-normalized", prenorm_cols), ("Raw/different scale", raw_cols)]:
    indices = [FEATURE_COLS.index(c) for c in cat_cols if c in FEATURE_COLS]
    avg_raw = np.mean([raw_mis[i] for i in indices])
    avg_dc  = np.mean([dc_mis[i] / max(raw_mis[i], 1e-10) * 100 for i in indices])
    avg_ac  = np.mean([ac_mis[i] / max(raw_mis[i], 1e-10) * 100 for i in indices])
    print(f"  {cat_name:<20} {avg_raw:>12.6f} {avg_dc:>9.1f}% {avg_ac:>9.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# The real question: What if we SKIP per-window z-score entirely?
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("  COUNTERFACTUAL: What if we skip per-window z-score?")
print("─" * 80)
print("  Train a tiny MLP on raw features vs per-window-z-scored features")
print("  to measure actual learning difference.\n")

import torch
import torch.nn as nn

class TinyMLP(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# Prepare data
split = int(0.7 * len(feat_aligned))
target_mlp = target_aligned

# Option 1: Raw features (standardized globally)
feat_raw = feat_aligned
raw_mean = feat_raw[:split].mean(axis=0)
raw_std  = feat_raw[:split].std(axis=0) + 1e-8
feat_raw_scaled = (feat_raw - raw_mean) / raw_std

X_raw_train = torch.FloatTensor(feat_raw_scaled[:split])
X_raw_val   = torch.FloatTensor(feat_raw_scaled[split:])
y_train     = torch.FloatTensor(target_mlp[:split])
y_val       = torch.FloatTensor(target_mlp[split:])

# Option 2: Per-window z-scored features (what model sees)
# Compute per-window z-score for the aligned subset
feat_pw = np.zeros_like(feat_aligned)
for w in range(n_windows):
    start = w
    end = start + LOOKBACK
    if end > len(feat_vals):
        break
    window = feat_sub[start:end]
    w_mean = np.mean(window, axis=0)
    w_std  = np.std(window, axis=0)
    min_std = 0.005 * np.abs(w_mean).clip(min=0.01)
    const_mask = (w_std < min_std)
    w_std[const_mask] = 1.0
    w_mean[const_mask] = 0.0
    norm = (window - w_mean) / (w_std + 1e-5)
    norm[:, const_mask] = 0.0
    norm = np.clip(norm, -5.0, 5.0)
    feat_pw[w] = norm[-1]  # last element

X_pw_train = torch.FloatTensor(feat_pw[:split])
X_pw_val   = torch.FloatTensor(feat_pw[split:])

def train_mlp(X_tr, y_tr, X_va, y_va, label):
    mlp = TinyMLP(len(FEATURE_COLS))
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    
    best_val = float("inf")
    best_state = None
    for epoch in range(50):
        mlp.train()
        opt.zero_grad()
        loss = loss_fn(mlp(X_tr), y_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
        opt.step()
        
        mlp.eval()
        with torch.no_grad():
            v_loss = loss_fn(mlp(X_va), y_va)
        if v_loss.item() < best_val:
            best_val = v_loss.item()
            best_state = {k: v.clone() for k, v in mlp.state_dict().items()}
    
    mlp.load_state_dict(best_state)
    mlp.eval()
    with torch.no_grad():
        pred = mlp(X_va).numpy()
    true = y_va.numpy()
    
    mse = np.mean((pred - true)**2)
    corr = np.corrcoef(pred, true)[0, 1]
    edge = true != 0
    dir_acc = np.mean((pred[edge] * true[edge]) > 0) if edge.any() else 0
    baseline = np.var(true)
    
    print(f"  {label}:")
    print(f"    Val MSE: {mse:.6f}  (baseline: {baseline:.6f}, improvement: {(1-mse/baseline)*100:.1f}%)")
    print(f"    Correlation: {corr:.4f}  Direction accuracy: {dir_acc*100:.1f}%")
    return mse, corr, dir_acc

print("  Training MLP on globally-standardized raw features...")
mse_raw, corr_raw, da_raw = train_mlp(X_raw_train, y_train, X_raw_val, y_val, "  Global standardization")

print("\n  Training MLP on per-window z-scored features...")
mse_pw, corr_pw, da_pw = train_mlp(X_pw_train, y_train, X_pw_val, y_val, "  Per-window z-score")

print(f"\n  {'─'*60}")
print(f"  Comparison:")
print(f"    Global std → corr={corr_raw:.4f}, dir_acc={da_raw*100:.1f}%")
print(f"    Per-window  → corr={corr_pw:.4f}, dir_acc={da_pw*100:.1f}%")
print(f"    Signal loss: {(1 - max(corr_pw,0)/max(corr_raw,0.001))*100:.1f}%")

if corr_pw < corr_raw * 0.5:
    print(f"\n  🔴 Per-window z-score DESTROYS >50% of learnable signal!")
elif corr_pw < corr_raw * 0.8:
    print(f"\n  🟡 Per-window z-score destroys some signal ({(1-corr_pw/max(corr_raw,0.001))*100:.0f}%)")
else:
    print(f"\n  🟢 Per-window z-score preserves most signal")

print("\n" + "=" * 80)
print("  AUDIT COMPLETE")
print("=" * 80)
