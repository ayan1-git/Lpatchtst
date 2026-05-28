# audit_regime.py
import pandas as pd, numpy as np, sys, os
sys.path.insert(0, os.getcwd())
import config
from oracle import generate_targets
from train import make_date_aligned_folds

# Thresholds for Regime Drift (Professional Guidelines)
THRESH_MEAN_DRIFT = 0.02        # Absolute diff in mean targets
THRESH_STD_DRIFT  = 0.15        # 15% relative change in Std
THRESH_ZERO_DRIFT = 0.10        # 10 percentage point change in Zero%

def get_status(val, thresh, is_relative=False):
    if is_relative:
        return "❌ FAIL" if val > thresh * 1.5 else ("⚠️ WARN" if val > thresh else "✅ OK")
    return "❌ FAIL" if val > thresh * 1.5 else ("⚠️ WARN" if val > thresh else "✅ OK")

# 1. Data Loading
f = config.DATA_FILE[0]
df = pd.read_csv(f)
time_col = next((c for c in df.columns if c.lower() in ("date","datetime")), None)
if time_col:
    df[time_col] = pd.to_datetime(df[time_col]); df = df.set_index(time_col)
df = df.sort_index()

# 2. Oracle Target Generation
hl = df["high"] - df["low"]
hc = (df["high"] - df["close"].shift()).abs()
lc = (df["low"]  - df["close"].shift()).abs()
df["atr"] = pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(config.ATR_PERIOD).mean()
df.dropna(inplace=True)

targets = generate_targets(
    df["open"].values, df["high"].values, df["low"].values,
    df["close"].values, df["atr"].values,
    max_hold=config.ORACLE_MAX_HOLD, fee_per_side=config.FEE_PER_SIDE,
    slippage=config.SLIPPAGE, sl_atr_mult=config.ORACLE_SL_ATR_MULT,
    tp_atr_mult=config.ORACLE_TP_ATR_MULT, saturation_factor=config.SATURATION_FACTOR,
    mae_penalty=config.MAE_PENALTY)

# Target alignment: Zero out ambiguous signals to match training logic
targets[np.abs(targets) < config.SAMPLER_THRESHOLD] = 0.0

# 3. Date-Aligned Fold Slicing
global_start = df.index[0]
global_end = df.index[-1]
folds = make_date_aligned_folds(global_start, global_end, n_folds=5)

print(f"\nOverall Distribution — Long: {(targets > 0).mean():.3f} | Short: {(targets < 0).mean():.3f} | Zero: {(targets == 0.0).mean():.3f}")
print("\n" + "="*180)
print(f"{'Fold':<6} {'TrMean':>8} {'VaMean':>8} {'ΔMean':>8} {'Stat':>8} | {'TrStd':>8} {'VaStd':>8} {'ΔStd%':>8} {'Stat':>8} | {'TrZero':>8} {'VaZero':>8} {'ΔZero':>8} {'Stat':>8} | {'Val Date Range'}")
print("-" * 180)

fold_results = []

for fold in folds:
    fold_id, ts_date, te_date, vs_date, ve_date = fold
    
    ts_idx = df.index.searchsorted(ts_date, side="left")
    te_idx = df.index.searchsorted(te_date, side="right")
    vs_idx = df.index.searchsorted(vs_date, side="left")
    ve_idx = df.index.searchsorted(ve_date, side="right")
    
    te_safe = te_idx - (config.ORACLE_MAX_HOLD - 1)
    
    tr = targets[ts_idx : te_safe]
    va = targets[vs_idx : ve_idx]
    
    # Metrics
    tm, vm = tr.mean(), va.mean()
    ts, vs = tr.std(), va.std()
    tz, vz = (tr == 0.0).mean(), (va == 0.0).mean()
    
    # Drifts
    d_mean = abs(tm - vm)
    d_std  = abs(ts - vs) / (ts + 1e-9)
    d_zero = abs(tz - vz)
    
    s_mean = get_status(d_mean, THRESH_MEAN_DRIFT)
    s_std  = get_status(d_std,  THRESH_STD_DRIFT, is_relative=True)
    s_zero = get_status(d_zero, THRESH_ZERO_DRIFT)
    
    val_dates = f"{vs_date.strftime('%Y-%m')} → {ve_date.strftime('%Y-%m')}"

    print(f"{fold_id:<6} {tm:>8.4f} {vm:>8.4f} {d_mean:>8.4f} {s_mean:>8} | {ts:>8.4f} {vs:>8.4f} {d_std:>8.2%} {s_std:>8} | {tz:>8.1%} {vz:>8.1%} {d_zero:>8.1%} {s_zero:>8} | {val_dates}")
    
    fold_results.append({
        "id": fold_id,
        "mean_drift": d_mean, "std_drift": d_std, "zero_drift": d_zero,
        "s_mean": s_mean, "s_std": s_std, "s_zero": s_zero
    })

print("="*180)

# 4. Final Regime Stability Summary
print("\n" + "█" * 60)
print("  REGIME STABILITY SUMMARY")
print("█" * 60)

avg_m_drift = np.mean([r["mean_drift"] for r in fold_results])
avg_s_drift = np.mean([r["std_drift"] for r in fold_results])
avg_z_drift = np.mean([r["zero_drift"] for r in fold_results])

print(f"  Avg Directional Drift (Mean): {avg_m_drift:.4f} {'(STABLE)' if avg_m_drift < THRESH_MEAN_DRIFT else '(UNSTABLE)'}")
print(f"  Avg Volatility Drift (Std):   {avg_s_drift:.2%} {'(STABLE)' if avg_s_drift < THRESH_STD_DRIFT else '(UNSTABLE)'}")
print(f"  Avg Tradeability Drift (Zero): {avg_z_drift:.2%} {'(STABLE)' if avg_z_drift < THRESH_ZERO_DRIFT else '(UNSTABLE)'}")

# Global Verdict
if any(any("FAIL" in str(v) for v in r.values()) for r in fold_results):
    print("\n  VERDICT: 🔴 HIGH REGIME RISK. Validation sets are structurally different from training.")
elif any(any("WARN" in str(v) for v in r.values()) for r in fold_results):
    print("\n  VERDICT: ⚠️ MODERATE REGIME RISK. Some folds show significant drift. Monitor FSR and DirAcc closely.")
else:
    print("\n  VERDICT: ✅ REGIME STABLE. Distribution is consistent across walk-forward folds.")
print("█" * 60)
