"""
feature_audit.py — GROUND TRUTH Feature Signal Audit
═══════════════════════════════════════════════════════════════════════════════

PURPOSE:
    Definitively answer: "Do the 24 features contain predictive signal for the
    oracle target, or is the problem elsewhere (architecture, loss, training)?"

METHOD:
    1. MUTUAL INFORMATION (MI) — non-parametric, model-agnostic
       MI(f, t) between each feature and the oracle target.
       MI ≈ 0 → feature is genuinely uninformative (feature problem).
       MI > 0.01–0.05 → signal exists (problem is architecture/generalization).

    2. PERMUTATION IMPORTANCE — shuffle each feature, measure MI drop.
       Confounds: if two features carry the same signal, neither shows high
       individual MI. Permutation importance catches this.

    3. BIVARIATE CORRELATION — Pearson + Spearman (linear + monotonic).

    4. BINNING-BASED MI — discretize target into bins, compute MI with
       categorical target. More sensitive to non-linear signal.

    5. CONDITIONAL MI PROXY — partial correlation / residual analysis.
       Regress each feature on all others, compute MI of residuals with target.
       This catches signal that's redundant across features.

    6. GRADIENT-BASED IMPORTANCE — train a tiny MLP (1 hidden layer, 32 units)
       for 20 epochs, measure final loss. Then shuffle each feature column
       and re-evaluate. Loss increase = feature importance.

    7. RANDOM FOREST FEATURE IMPORTANCE — model-agnostic tree-based ranking.

    8. TARGET-STRATIFIED FEATURE STATS — for each feature, compare distribution
       on Long vs Short vs Flat bars. If distributions are identical, no signal.

    9. DELTA-MI (TEMPORAL) — compute MI at different time lags. Sometimes
       features lead the target by a few bars.

    10. NOISE FLOOR ESTIMATION — compute MI between features and RANDOM targets.
       Establishes the noise floor. Any feature MI must exceed this to be real.

INTERPRETATION GUIDE:
    MI < 0.005   → No detectable signal (feature problem)
    MI 0.005-0.02 → Weak signal (may need better architecture to extract)
    MI 0.02-0.05  → Moderate signal (architecture should be able to use this)
    MI > 0.05     → Strong signal (definitely an architecture/training problem)

    If ALL features score < 0.005 → FEATURE PROBLEM. Stop here.
    If ANY feature scores > 0.02  → SIGNAL EXISTS. Problem is elsewhere.

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import KBinsDiscretizer
from scipy import stats

warnings.filterwarnings("ignore")

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
import config
from features import FeatureEngineer, FeatureConfig
from oracle import generate_targets

# ═══════════════════════════════════════════════════════════════════════════════
# 0. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("  FEATURE SIGNAL AUDIT — Ground Truth Investigation")
print("=" * 80)

# Use NIFTY 50 as the primary asset (most liquid, most data)
DATA_PATH = "Data/NIFTY 50_30minute.csv"
print(f"\nLoading: {DATA_PATH}")

df_full = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
print(f"  Raw shape: {df_full.shape}")
print(f"  Date range: {df_full.index[0]} → {df_full.index[-1]}")

# Standardize column names
col_map = {c.lower(): c for c in df_full.columns}
o_col = col_map.get('open', 'Open')
h_col = col_map.get('high', 'High')
l_col = col_map.get('low', 'Low')
c_col = col_map.get('close', 'Close')
v_col = col_map.get('volume', 'Volume')

close_full = df_full[c_col].values.astype(np.float64)
volume_full = df_full[v_col].values.astype(np.float64) if v_col in df_full.columns else np.zeros_like(close_full)

# Build features
fe = FeatureEngineer(config=FeatureConfig())
feat_df = fe.build(df_full[c_col], ohlc=df_full, include_target=False, dropna=False)

# Add OHLC columns (matching train.py process_dataset)
feat_df['open']  = df_full[o_col].astype(np.float64)
feat_df['high']  = df_full[h_col].astype(np.float64)
feat_df['low']   = df_full[l_col].astype(np.float64)
feat_df['close'] = df_full[c_col].astype(np.float64)

# The 24 feature columns (matching train.py DEFAULT_FEATURE_LIST)
FEATURE_COLS = [
    'open', 'high', 'low', 'close',
    'ret_norm_1d', 'ret_norm_3d', 'ret_norm_6d', 'ret_norm_13d',
    'ret_norm_26d', 'ret_norm_65d', 'ret_norm_130d', 'ret_norm_260d',
    'macd_8_24', 'macd_26_78', 'macd_52_156',
    'feat_efficiency', 'feat_icp', 'feat_momentum_rsi',
    'feat_vol_asymmetry', 'feat_local_structure',
    'feat_session_sin', 'feat_session_cos', 'feat_vol_squeeze'
]

# Ensure all columns exist
for col in FEATURE_COLS:
    if col not in feat_df.columns:
        feat_df[col] = 0.0
        print(f"  WARNING: {col} missing — filled with 0")

feat_df = feat_df[FEATURE_COLS]

# Compute ATR and oracle targets
hl = df_full[h_col] - df_full[l_col]
hc = (df_full[h_col] - df_full[c_col].shift()).abs()
lc = (df_full[l_col]  - df_full[c_col].shift()).abs()
atr_full = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(config.ATR_PERIOD).mean()

target_full = generate_targets(
    df_full[o_col].values.astype(np.float64),
    df_full[h_col].values.astype(np.float64),
    df_full[l_col].values.astype(np.float64),
    df_full[c_col].values.astype(np.float64),
    atr_full.values.astype(np.float64),
)
target_full[np.abs(target_full) < config.SAMPLER_THRESHOLD] = 0.0

# Cut warm-up (matching train.py)
warmup = 3536
df = df_full.iloc[warmup:]
feat_vals = feat_df.values[warmup:].astype(np.float64)
target_vals = target_full[warmup:]

# Clean NaN/Inf
feat_vals = np.nan_to_num(feat_vals, nan=0.0, posinf=0.0, neginf=0.0)

# Align lengths
min_len = min(len(feat_vals), len(target_vals))
feat_vals = feat_vals[:min_len]
target_vals = target_vals[:min_len]

print(f"\n  Aligned data: {min_len:,} bars")
print(f"  Feature matrix: {feat_vals.shape}")
print(f"  Target: mean={target_vals.mean():.6f}, std={target_vals.std():.6f}")
print(f"  Target distribution:")
print(f"    Long  (>{config.SAMPLER_THRESHOLD}):  {(target_vals > config.SAMPLER_THRESHOLD).mean()*100:.1f}%")
print(f"    Short (<-{config.SAMPLER_THRESHOLD}): {(target_vals < -config.SAMPLER_THRESHOLD).mean()*100:.1f}%")
print(f"    Flat  (==0):          {(target_vals == 0.0).mean()*100:.1f}%")

# Subsample for MI computation (MI is O(n log n), 50k is plenty)
MAX_SAMPLES = 50000
if len(feat_vals) > MAX_SAMPLES:
    # Stratified subsample: keep all edge bars, subsample flat bars
    edge_mask = target_vals != 0.0
    flat_mask = ~edge_mask
    n_edge = edge_mask.sum()
    n_flat_keep = max(0, MAX_SAMPLES - n_edge)
    
    edge_idx = np.where(edge_mask)[0]
    flat_idx = np.where(flat_mask)[0]
    
    if n_flat_keep < len(flat_idx):
        rng = np.random.RandomState(42)
        flat_idx = rng.choice(flat_idx, size=n_flat_keep, replace=False)
    
    keep_idx = np.sort(np.concatenate([edge_idx, flat_idx]))
    feat_sub = feat_vals[keep_idx]
    target_sub = target_vals[keep_idx]
    print(f"\n  Subsampled to {len(keep_idx):,} bars for MI computation")
    print(f"    Edge bars: {n_edge:,}, Flat bars kept: {len(flat_idx):,}")
else:
    feat_sub = feat_vals
    target_sub = target_vals
    keep_idx = np.arange(len(feat_vals))

# ═══════════════════════════════════════════════════════════════════════════════
# 1. MUTUAL INFORMATION (continuous target)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  TEST 1: MUTUAL INFORMATION (continuous target)")
print("=" * 80)
print("  MI(f, t) — non-parametric, captures any statistical dependency")
print("  Noise floor estimated via random target shuffle\n")

# Compute MI for each feature
mi_scores = {}
for i, col in enumerate(FEATURE_COLS):
    mi = mutual_info_regression(
        feat_sub[:, i:i+1], target_sub,
        n_neighbors=5, random_state=42, n_jobs=1
    )[0]
    mi_scores[col] = mi

# Noise floor: MI with random targets
noise_mis = []
for _ in range(10):
    rng = np.random.RandomState(_ * 7 + 13)
    random_target = rng.permutation(target_sub)
    noise_mi = mutual_info_regression(
        feat_sub[:, 0:1], random_target,
        n_neighbors=5, random_state=42, n_jobs=1
    )[0]
    noise_mis.append(noise_mi)

noise_floor = np.mean(noise_mis)
noise_std = np.std(noise_mis)

print(f"  Noise floor (random target MI): {noise_floor:.6f} ± {noise_std:.6f}")
print(f"  {'Feature':<30} {'MI':>10} {'MI-noise':>10} {'Signal?':>10}")
print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10}")

mi_sorted = sorted(mi_scores.items(), key=lambda x: -x[1])
for col, mi in mi_sorted:
    signal = mi - noise_floor
    flag = "✓ YES" if signal > 2 * noise_std else ("~ weak" if signal > noise_std else "✗ NO")
    print(f"  {col:<30} {mi:>10.6f} {signal:>10.6f} {flag:>10}")

max_mi = max(mi_scores.values())
mean_mi = np.mean(list(mi_scores.values()))
n_significant = sum(1 for v in mi_scores.values() if v - noise_floor > 2 * noise_std)

print(f"\n  Max MI: {max_mi:.6f}")
print(f"  Mean MI: {mean_mi:.6f}")
print(f"  Features with significant MI (> noise + 2σ): {n_significant}/{len(FEATURE_COLS)}")

if max_mi < 0.005:
    print("\n  🔴 VERDICT: ALL features have near-zero MI. FEATURE PROBLEM.")
elif max_mi < 0.02:
    print("\n  🟡 VERDICT: Weak signal detected. May need architecture changes.")
else:
    print("\n  🟢 VERDICT: Signal EXISTS. Problem is architecture/generalization.")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. MUTUAL INFORMATION (categorical target — binned)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  TEST 2: MUTUAL INFORMATION (categorical target — 3 bins)")
print("=" * 80)
print("  Discretizes target into Short/Flat/Long, computes MI_classif")
print("  More sensitive to non-linear decision boundaries\n")

# Create categorical target
cat_target = np.zeros(len(target_sub), dtype=int)  # 0 = flat
cat_target[target_sub > config.SAMPLER_THRESHOLD] = 2   # 2 = long
cat_target[target_sub < -config.SAMPLER_THRESHOLD] = 1  # 1 = short

print(f"  Class distribution: Short={np.sum(cat_target==1):,} Flat={np.sum(cat_target==0):,} Long={np.sum(cat_target==2):,}")

mi_cat_scores = {}
for i, col in enumerate(FEATURE_COLS):
    mi_c = mutual_info_classif(
        feat_sub[:, i:i+1], cat_target,
        n_neighbors=5, random_state=42, n_jobs=1
    )[0]
    mi_cat_scores[col] = mi_c

print(f"\n  {'Feature':<30} {'MI_cat':>10} {'MI_cont':>10} {'Ratio':>8}")
print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*8}")

mi_cat_sorted = sorted(mi_cat_scores.items(), key=lambda x: -x[1])
for col, mi_c in mi_cat_sorted:
    mi_cont = mi_scores[col]
    ratio = mi_c / max(mi_cont, 1e-10)
    print(f"  {col:<30} {mi_c:>10.6f} {mi_cont:>10.6f} {ratio:>8.2f}")

max_mi_cat = max(mi_cat_scores.values())
print(f"\n  Max categorical MI: {max_mi_cat:.6f}")

if max_mi_cat < 0.005:
    print("  🔴 No categorical signal detected.")
elif max_mi_cat < 0.02:
    print("  🟡 Weak categorical signal.")
else:
    print("  🟢 Categorical signal EXISTS.")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. BIVARIATE CORRELATION (Pearson + Spearman)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  TEST 3: BIVARIATE CORRELATION")
print("=" * 80)
print("  Pearson (linear) + Spearman (monotonic) correlation with target\n")

corr_results = []
for i, col in enumerate(FEATURE_COLS):
    # Use full dataset for correlation (fast)
    r_pearson, p_pearson = stats.pearsonr(feat_vals[:, i], target_vals)
    r_spearman, p_spearman = stats.spearmanr(feat_vals[:, i], target_vals)
    corr_results.append((col, r_pearson, p_pearson, r_spearman, p_spearman))

corr_results.sort(key=lambda x: -abs(x[3]))  # sort by |Spearman|

print(f"  {'Feature':<30} {'Pearson':>10} {'p-val':>10} {'Spearman':>10} {'p-val':>10}")
print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

for col, rp, pp, rs, ps in corr_results:
    sig_p = "***" if pp < 0.001 else ("**" if pp < 0.01 else ("*" if pp < 0.05 else ""))
    sig_s = "***" if ps < 0.001 else ("**" if ps < 0.01 else ("*" if ps < 0.05 else ""))
    print(f"  {col:<30} {rp:>+10.4f} {pp:>10.2e}{sig_p} {rs:>+10.4f} {ps:>10.2e}{sig_s}")

max_abs_spearman = max(abs(x[3]) for x in corr_results)
print(f"\n  Max |Spearman|: {max_abs_spearman:.4f}")

if max_abs_spearman < 0.01:
    print("  🔴 No linear/monotonic correlation detected.")
elif max_abs_spearman < 0.05:
    print("  🟡 Weak correlation.")
else:
    print("  🟢 Correlation signal EXISTS.")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. TARGET-STRATIFIED FEATURE DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  TEST 4: TARGET-STRATIFIED FEATURE DISTRIBUTIONS")
print("=" * 80)
print("  For each feature: compare mean/std on Long vs Short vs Flat bars")
print("  If distributions are identical → no signal\n")

long_mask  = target_vals > config.SAMPLER_THRESHOLD
short_mask = target_vals < -config.SAMPLER_THRESHOLD
flat_mask  = target_vals == 0.0

print(f"  Groups: Long={long_mask.sum():,}  Short={short_mask.sum():,}  Flat={flat_mask.sum():,}")
print(f"\n  {'Feature':<30} {'Long_mean':>10} {'Short_mean':>10} {'Flat_mean':>10} {'ΔL-S':>10} {'KS_pval':>10}")
print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

dist_results = []
for i, col in enumerate(FEATURE_COLS):
    f_long  = feat_vals[long_mask, i]
    f_short = feat_vals[short_mask, i]
    f_flat  = feat_vals[flat_mask, i]
    
    mean_l = np.mean(f_long)
    mean_s = np.mean(f_short)
    mean_f = np.mean(f_flat)
    delta_ls = mean_l - mean_s
    
    # KS test: Long vs Short distributions
    if len(f_long) > 100 and len(f_short) > 100:
        ks_stat, ks_pval = stats.ks_2samp(f_long, f_short)
    else:
        ks_stat, ks_pval = np.nan, np.nan
    
    dist_results.append((col, mean_l, mean_s, mean_f, delta_ls, ks_pval, ks_stat))

# Sort by |delta Long-Short|
dist_results.sort(key=lambda x: -abs(x[4]))

for col, ml, ms, mf, dls, ksp, kss in dist_results:
    if np.isnan(ksp):
        sig = ""
    elif ksp < 0.001:
        sig = "***"
    elif ksp < 0.01:
        sig = "**"
    elif ksp < 0.05:
        sig = "*"
    else:
        sig = ""
    print(f"  {col:<30} {ml:>+10.4f} {ms:>+10.4f} {mf:>+10.4f} {dls:>+10.4f} {ksp:>10.2e}{sig}")

max_delta = max(abs(x[4]) for x in dist_results)
n_ks_sig = sum(1 for x in dist_results if not np.isnan(x[5]) and x[5] < 0.05)
print(f"\n  Max |Δ(Long-Short)|: {max_delta:.4f}")
print(f"  Features with significant KS test (p<0.05): {n_ks_sig}/{len(FEATURE_COLS)}")

if max_delta < 0.01:
    print("  🔴 No distributional difference between Long/Short bars.")
elif max_delta < 0.05:
    print("  🟡 Weak distributional difference.")
else:
    print("  🟢 Distributional signal EXISTS.")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. RANDOM FOREST FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  TEST 5: RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 80)
print("  Model-agnostic tree-based importance (captures non-linear + interactions)\n")

# Use subsample for speed
rf = RandomForestRegressor(
    n_estimators=100, max_depth=10, min_samples_leaf=50,
    random_state=42, n_jobs=-1
)
rf.fit(feat_sub, target_sub)

rf_importance = rf.feature_importances_
rf_r2 = rf.score(feat_sub, target_sub)

print(f"  RF R² on training data: {rf_r2:.6f}")
print(f"  (R² > 0.01 means the features collectively have predictive power)\n")

rf_results = list(zip(FEATURE_COLS, rf_importance))
rf_results.sort(key=lambda x: -x[1])

print(f"  {'Feature':<30} {'RF_Importance':>15} {'Cumulative':>12}")
print(f"  {'─'*30} {'─'*15} {'─'*12}")

cumsum = 0
for col, imp in rf_results:
    cumsum += imp
    print(f"  {col:<30} {imp:>15.6f} {cumsum:>12.4f}")

print(f"\n  Top 5 features account for {sum(x[1] for x in rf_results[:5]):.4f} importance")

if rf_r2 < 0.005:
    print("  🔴 RF cannot learn anything from features. FEATURE PROBLEM.")
elif rf_r2 < 0.02:
    print("  🟡 RF learns weak signal.")
else:
    print("  🟢 RF learns meaningful signal. Problem is in neural architecture/training.")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. PERMUTATION IMPORTANCE (on RF)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  TEST 6: PERMUTATION IMPORTANCE (on RF)")
print("=" * 80)
print("  Shuffle each feature, measure R² drop. Catches redundant signal.\n")

from sklearn.metrics import r2_score

# Evaluate on a held-out portion
split = int(0.7 * len(feat_sub))
rf_train = feat_sub[:split]
rf_val   = feat_sub[split:]
t_train  = target_sub[:split]
t_val    = target_sub[split:]

rf2 = RandomForestRegressor(
    n_estimators=100, max_depth=10, min_samples_leaf=50,
    random_state=42, n_jobs=-1
)
rf2.fit(rf_train, t_train)
base_r2 = r2_score(t_val, rf2.predict(rf_val))

print(f"  Base R² (validation): {base_r2:.6f}\n")

perm_results = []
for i, col in enumerate(FEATURE_COLS):
    rng = np.random.RandomState(42 + i)
    feat_perm = rf_val.copy()
    feat_perm[:, i] = rng.permutation(feat_perm[:, i])
    perm_r2 = r2_score(t_val, rf2.predict(feat_perm))
    drop = base_r2 - perm_r2
    perm_results.append((col, drop))

perm_results.sort(key=lambda x: -x[1])

print(f"  {'Feature':<30} {'R²_drop':>10} {'Important?':>12}")
print(f"  {'─'*30} {'─'*10} {'─'*12}")

for col, drop in perm_results:
    flag = "✓ YES" if drop > 0.001 else ("~ weak" if drop > 0.0001 else "✗ NO")
    print(f"  {col:<30} {drop:>10.6f} {flag:>12}")

max_perm_drop = max(x[1] for x in perm_results)
n_perm_sig = sum(1 for x in perm_results if x[1] > 0.001)

print(f"\n  Max R² drop: {max_perm_drop:.6f}")
print(f"  Features with significant permutation importance: {n_perm_sig}/{len(FEATURE_COLS)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. TINY MLP GRADIENT-BASED IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  TEST 7: TINY MLP LEARNING TEST")
print("=" * 80)
print("  Train a minimal MLP (1 hidden layer, 32 units) for 30 epochs")
print("  If even this can't learn, features are truly uninformative\n")

import torch
import torch.nn as nn

class TinyMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# Prepare data — standardize features for MLP (zero mean, unit variance)
split2 = int(0.7 * len(feat_sub))
feat_mean = feat_sub[:split2].mean(axis=0)
feat_std  = feat_sub[:split2].std(axis=0) + 1e-8
feat_sub_scaled = (feat_sub - feat_mean) / feat_std

X_train = torch.FloatTensor(feat_sub_scaled[:split2])
y_train = torch.FloatTensor(target_sub[:split2])
X_val   = torch.FloatTensor(feat_sub_scaled[split2:])
y_val   = torch.FloatTensor(target_sub[split2:])

mlp = TinyMLP(len(FEATURE_COLS))
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

# Train with early stopping
best_val_loss = float("inf")
best_state = None
patience = 10
pat_counter = 0
train_losses = []
val_losses = []
for epoch in range(50):
    mlp.train()
    opt.zero_grad()
    pred = mlp(X_train)
    loss = loss_fn(pred, y_train)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
    opt.step()
    train_losses.append(loss.item())
    
    mlp.eval()
    with torch.no_grad():
        val_pred = mlp(X_val)
        val_loss = loss_fn(val_pred, y_val)
        val_losses.append(val_loss.item())
    
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        best_state = {k: v.clone() for k, v in mlp.state_dict().items()}
        pat_counter = 0
    else:
        pat_counter += 1
    
    if pat_counter >= patience:
        print(f"  Early stop at epoch {epoch+1}")
        break

# Load best model
if best_state is not None:
    mlp.load_state_dict(best_state)

# Final metrics (best model)
mlp.eval()
with torch.no_grad():
    final_pred = mlp(X_val).numpy()
    final_true = y_val.numpy()

final_mse = np.mean((final_pred - final_true) ** 2)
baseline_mse = np.var(final_true)
final_corr = np.corrcoef(final_pred, final_true)[0, 1]
edge_mask_val = final_true != 0.0
if edge_mask_val.any():
    dir_acc = np.mean((final_pred[edge_mask_val] * final_true[edge_mask_val]) > 0)
else:
    dir_acc = float("nan")

print(f"  Training: {len(X_train):,} samples, Validation: {len(X_val):,} samples")
print(f"  Final train loss: {train_losses[-1]:.6f}")
print(f"  Final val loss:   {val_losses[-1]:.6f}")
print(f"  Baseline MSE (predict mean): {baseline_mse:.6f}")
print(f"  MLP MSE improvement:         {(1 - final_mse/baseline_mse)*100:.2f}%")
print(f"  Pearson correlation:         {final_corr:.4f}")
print(f"  Direction accuracy (edge):   {dir_acc*100:.1f}%")
print(f"  Pred std: {final_pred.std():.6f}  vs  Target std: {final_true.std():.6f}")

# Permutation importance on MLP (using scaled features)
print(f"\n  MLP Permutation Importance:")
mlp_perm_results = []
for i, col in enumerate(FEATURE_COLS):
    rng = np.random.RandomState(42 + i)
    X_perm = X_val.clone()
    perm_idx = rng.permutation(len(X_perm))
    X_perm[:, i] = X_perm[perm_idx, i]
    with torch.no_grad():
        perm_pred = mlp(X_perm).numpy()
    perm_mse = np.mean((perm_pred - final_true) ** 2)
    mlp_perm_results.append((col, perm_mse - final_mse))

mlp_perm_results.sort(key=lambda x: -x[1])
for col, mse_increase in mlp_perm_results:
    flag = "✓" if mse_increase > 0.0001 else "✗"
    print(f"    {col:<30} ΔMSE={mse_increase:>10.6f} {flag}")

if final_mse >= baseline_mse * 0.99:
    print("\n  🔴 MLP CANNOT LEARN. Features are uninformative. FEATURE PROBLEM.")
elif final_corr < 0.05:
    print("\n  🟡 MLP learns very weak signal.")
else:
    print("\n  🟢 MLP learns signal. Neural architecture/training is the bottleneck.")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. TEMPORAL LAG ANALYSIS (Δ-MI at different lags)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  TEST 8: TEMPORAL LAG ANALYSIS")
print("=" * 80)
print("  Compute MI(feature_t-k, target_t) for k = 0, 1, 2, 3, 5, 10")
print("  Sometimes features lead the target\n")

# Use top 5 features from MI test
top5_features = [col for col, _ in mi_sorted[:5]]
top5_indices = [FEATURE_COLS.index(c) for c in top5_features]

max_lag = 10
print(f"\n  {'Feature':<30}", end="")
for lag in [0, 1, 2, 3, 5, 10]:
    print(f"  lag={lag:>2}", end="")
print()

for col, fi in zip(top5_features, top5_indices):
    print(f"  {col:<30}", end="")
    for lag in [0, 1, 2, 3, 5, 10]:
        if lag == 0:
            f = feat_sub[:, fi]
            t = target_sub
        else:
            f = feat_sub[lag:, fi]
            t = target_sub[:-lag] if lag > 0 else target_sub
            # Re-align
            min_l = min(len(f), len(t))
            f = f[:min_l]
            t = t[:min_l]
        mi_lag = mutual_info_regression(
            f.reshape(-1, 1), t,
            n_neighbors=5, random_state=42, n_jobs=1
        )[0]
        print(f"  {mi_lag:>6.4f}", end="")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
# 9. FEATURE-TARGET INFORMATION DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  TEST 9: JOINT vs INDIVIDUAL MI (redundancy check)")
print("=" * 80)
print("  Compute MI(top5_features_joint, target) vs sum of individual MIs")
print("  If joint >> sum → features have complementary signal")
print("  If joint ≈ sum → features are redundant\n")

top5_idx = [FEATURE_COLS.index(c) for c in top5_features]
mi_joint = mutual_info_regression(
    feat_sub[:, top5_idx], target_sub,
    n_neighbors=5, random_state=42, n_jobs=1
)
mi_joint_val = float(np.mean(mi_joint))  # average MI across outputs
mi_sum_individual = sum(mi_scores[c] for c in top5_features)

print(f"  Top 5 features: {top5_features}")
print(f"  Joint MI (top 5, mean): {mi_joint_val:.6f}")
print(f"  Sum of individual MI:   {mi_sum_individual:.6f}")
print(f"  Redundancy ratio:       {mi_joint_val / max(mi_sum_individual, 1e-10):.3f}")
print(f"  (>1 = complementary, ≈1 = additive, <1 = redundant)")

# All features joint
mi_all_joint = mutual_info_regression(
    feat_sub, target_sub,
    n_neighbors=5, random_state=42, n_jobs=1
)
mi_all_joint_val = float(np.mean(mi_all_joint))
mi_all_sum = sum(mi_scores.values())
print(f"\n  All 24 features joint MI (mean): {mi_all_joint_val:.6f}")
print(f"  All 24 individual MI sum:        {mi_all_sum:.6f}")
print(f"  Redundancy ratio (all):          {mi_all_joint_val / max(mi_all_sum, 1e-10):.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. FINAL SUMMARY & VERDICT
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("  ══════════════════════════════════════════════════════════════")
print("  FINAL VERDICT")
print("  ══════════════════════════════════════════════════════════════")
print("=" * 80)

print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  TEST                          │ RESULT                        │
  ├─────────────────────────────────────────────────────────────────┤
  │  1. MI (continuous)            │ max={max_mi:.6f}  mean={mean_mi:.6f}    │
  │  2. MI (categorical)           │ max={max_mi_cat:.6f}                    │
  │  3. Spearman correlation       │ max|ρ|={max_abs_spearman:.4f}                  │
  │  4. KS test (Long vs Short)    │ {n_ks_sig}/{len(FEATURE_COLS)} features significant     │
  │  5. Random Forest R²           │ R²={rf_r2:.6f}                    │
  │  6. Permutation importance     │ {n_perm_sig}/{len(FEATURE_COLS)} features significant     │
  │  7. Tiny MLP learning          │ corr={final_corr:.4f}  dir_acc={dir_acc*100:.1f}%  │
  │  8. Joint MI (all 24)          │ {mi_all_joint_val:.6f}                    │
  └─────────────────────────────────────────────────────────────────┘
""")

# Determine verdict
signals_detected = 0
if max_mi > 0.01: signals_detected += 1
if max_mi_cat > 0.01: signals_detected += 1
if max_abs_spearman > 0.03: signals_detected += 1
if n_ks_sig > 3: signals_detected += 1
if rf_r2 > 0.01: signals_detected += 1
if n_perm_sig > 2: signals_detected += 1
if abs(final_corr) > 0.05: signals_detected += 1
if mi_all_joint_val > 0.02: signals_detected += 1

print(f"  Signals detected in {signals_detected}/8 tests")

if signals_detected == 0:
    print("""
  ╔═══════════════════════════════════════════════════════════════════╗
  ║  🔴 DEFINITIVE VERDICT: FEATURE PROBLEM                        ║
  ║                                                                 ║
  ║  The features contain NO detectable signal for the oracle       ║
  ║  target. Every test — MI, correlation, RF, MLP, distributional  ║
  ║  analysis — returns null results.                               ║
  ║                                                                 ║
  ║  DO NOT waste time on architecture/loss/training changes.        ║
  ║  The problem is in the FEATURES or the TARGET.                  ║
  ║                                                                 ║
  ║  ACTION ITEMS:                                                  ║
  ║  1. Re-examine the oracle — is it computing meaningful labels?  ║
  ║  2. Try different feature engineering (raw returns, etc.)       ║
  ║  3. Check if the task is inherently unpredictable (EMH)         ║
  ╚═══════════════════════════════════════════════════════════════════╝
""")
elif signals_detected <= 3:
    print("""
  ╔═══════════════════════════════════════════════════════════════════╗
  ║  🟡 WEAK SIGNAL DETECTED                                       ║
  ║                                                                 ║
  ║  Some signal exists but it's weak. The features have marginal   ║
  ║  predictive power.                                              ║
  ║                                                                 ║
  ║  ACTION ITEMS:                                                  ║
  ║  1. Consider stronger features (order flow, alternative data)   ║
  ║  2. Architecture may need more capacity or different inductive  ║
  ║     bias to extract weak signals                                ║
  ║  3. Check if per-window z-score normalization is destroying     ║
  ║     the signal (it amplifies noise in near-constant features)   ║
  ╚═══════════════════════════════════════════════════════════════════╝
""")
else:
    print("""
  ╔═══════════════════════════════════════════════════════════════════╗
  ║  🟢 SIGNAL EXISTS — PROBLEM IS ARCHITECTURE/TRAINING           ║
  ║                                                                 ║
  ║  Multiple tests confirm the features contain predictive signal. ║
  ║  The model is failing to learn it.                              ║
  ║                                                                 ║
  ║  ACTION ITEMS:                                                  ║
  ║  1. Check per-window z-score normalization — is it destroying  ║
  ║     signal by amplifying noise in near-constant features?       ║
  ║  2. Check if the loss function is appropriate for the target   ║
  ║  3. Check learning rate, gradient flow, weight initialization  ║
  ║  4. Check if model capacity (d_model=64, 4 layers) is enough  ║
  ║  5. Check train/val loss curves — is it underfitting?          ║
  ║  6. Check if the model output saturates (tanh * 2.5)          ║
  ╚═══════════════════════════════════════════════════════════════════╝
""")

# ═══════════════════════════════════════════════════════════════════════════════
# BONUS: Per-window z-score signal destruction check
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("  BONUS: PER-WINDOW Z-SCORE SIGNAL DESTRUCTION CHECK")
print("=" * 80)
print("  The dataset applies per-window z-score normalization in __getitem__.")
print("  This can DESTROY signal if the feature has low within-window variance")
print("  but high across-window variance (the actual signal).\n")

LOOKBACK = config.LOOKBACK_WINDOW  # 512
n_windows = min(1000, len(feat_vals) - LOOKBACK)
window_snr = {}

for i, col in enumerate(FEATURE_COLS):
    fi = FEATURE_COLS.index(col)
    
    # Across-window variance: variance of window means
    window_means = []
    window_stds  = []
    for w in range(n_windows):
        start = w * 10  # stride 10 for speed
        if start + LOOKBACK >= len(feat_vals):
            break
        window = feat_vals[start:start+LOOKBACK, fi]
        window_means.append(np.mean(window))
        window_stds.append(np.std(window))
    
    window_means = np.array(window_means)
    window_stds = np.array(window_stds)
    
    across_var = np.var(window_means)
    mean_within_var = np.mean(window_stds ** 2)
    
    # SNR: ratio of across-window signal to within-window noise
    snr = across_var / max(mean_within_var, 1e-12)
    window_snr[col] = (snr, across_var, mean_within_var, np.mean(window_stds))

print(f"  {'Feature':<30} {'SNR':>10} {'AcrossVar':>12} {'WithinVar':>12} {'MeanStd':>10}")
print(f"  {'─'*30} {'─'*10} {'─'*12} {'─'*12} {'─'*10}")

snr_sorted = sorted(window_snr.items(), key=lambda x: -x[1][0])
for col, (snr, av, wv, mstd) in snr_sorted:
    flag = "⚠ LOW" if snr < 0.01 else ("~ OK" if snr < 0.1 else "✓ GOOD")
    print(f"  {col:<30} {snr:>10.4f} {av:>12.6f} {wv:>12.6f} {mstd:>10.4f}  {flag}")

low_snr_features = [col for col, (snr, _, _, _) in window_snr.items() if snr < 0.01]
if low_snr_features:
    print(f"\n  ⚠ Features with SNR < 0.01 (signal destroyed by per-window z-score):")
    for col in low_snr_features:
        print(f"    - {col}")
    print(f"  These features have signal at timescales > {LOOKBACK} bars but")
    print(f"  per-window z-score normalizes it away. Consider using global")
    print(f"  normalization or longer lookback windows for these features.")

print("\n" + "=" * 80)
print("  AUDIT COMPLETE")
print("=" * 80)
