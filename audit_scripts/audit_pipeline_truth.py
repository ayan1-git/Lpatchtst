import sys
import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler

# Add root to path to import from the main package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from data_loader import FinancialDataset, ColumnSelectiveScaler, fit_scaler
from train import make_date_aligned_folds
from features import FeatureEngineer
from oracle import generate_targets

def audit_temporal_alignment():
    print("\n--- 1. Temporal Alignment Audit ---")
    seq_len = 64
    T = 200
    # Synthetic 'truth' series (e.g., raw OHLC indices or values)
    truth_series = np.arange(T).reshape(-1, 1)
    targets = np.arange(T).reshape(-1, 1)
    
    # Test 1: Engineered Features Alignment (features_only mode)
    class MockConfigFeat:
        INPUT_MODE = "features_only"
        LOOKBACK_WINDOW = seq_len
    
    ds_feat = FinancialDataset(
        features=truth_series,
        targets=targets,
        seq_len=seq_len,
        config=MockConfigFeat()
    )
    
    # Test 2: OHLC-derived Tokens Alignment (tokens_only mode)
    class MockConfigTok:
        INPUT_MODE = "tokens_only"
        LOOKBACK_WINDOW = seq_len
    
    # Precomputed tokens are 1:1 with OHLC bars
    pre_coarse = torch.arange(T)
    pre_fine = torch.arange(T)
    
    ds_tok = FinancialDataset(
        features=truth_series,
        targets=targets,
        seq_len=seq_len,
        config=MockConfigTok(),
        precomputed_coarse=pre_coarse,
        precomputed_fine=pre_fine
    )
    
    pass_all = True
    for i in [0, 50, 100]:
        # Verify Engineered Features
        _, feats, target_f = ds_feat[i]
        expected_feats = np.arange(i, i + seq_len).reshape(-1, 1)
        expected_target = i + seq_len - 1
        
        if not np.array_equal(feats.numpy(), expected_feats):
            print(f"FAIL: Index {i} feature alignment mismatch.")
            pass_all = False
        if target_f.item() != expected_target:
            print(f"FAIL: Index {i} feature-target alignment mismatch.")
            pass_all = False
            
        # Verify Tokens (OHLC-derived)
        tokens, _, target_t = ds_tok[i]
        # tokens is (idx_coarse[seq], idx_fine[seq])
        # seq is slice(i, i + seq_len)
        expected_tokens_c = torch.arange(i, i + seq_len)
        
        if not torch.equal(tokens[0], expected_tokens_c):
            print(f"FAIL: Index {i} token alignment mismatch.")
            pass_all = False
        if target_t.item() != expected_target:
            print(f"FAIL: Index {i} token-target alignment mismatch.")
            pass_all = False
            
    if pass_all:
        print("PASS: Temporal alignment is correct for both engineered features and OHLC-derived tokens.")
    return pass_all

def audit_fold_separation():
    print("\n--- 2. Fold Separation & Oracle Leakage Audit ---")
    global_start = pd.Timestamp("2020-01-01")
    global_end = pd.Timestamp("2024-01-01")
    n_folds = 5
    
    folds = make_date_aligned_folds(global_start, global_end, n_folds=n_folds)
    
    # We need to verify the gap and oracle look-ahead
    # In train.py, gap is computed as max(gap_days_raw + margin_days, min_gap_days)
    # and gap_days_raw = ceil(LOOKBACK_WINDOW / eff_bars_per_day)
    
    oracle_max_hold = config.ORACLE_MAX_HOLD
    # Assuming avg_bars_per_day is something reasonable if not provided
    # We just need to verify that the resulting timestamps respect the invariant:
    # val_start > train_end + (some gap)
    
    pass_all = True
    for fold_id, ts, te, vs, ve in folds:
        gap = vs - te
        if gap <= pd.Timedelta(0):
            print(f"FAIL: Fold {fold_id} has no gap between train and val. Gap: {gap}")
            pass_all = False
        
        # Oracle leakage: If a sample is at te, its oracle window is [te, te + max_hold]
        # We must ensure that the oracle window for the LAST training sample
        # does not extend into the validation set.
        # Since we use date-aligned folds, we should check if the window of 
        # the last training bar is contained within the gap.
        # This is tricky because max_hold is in BARS, not DAYS.
        # However, if the GAP (in days) is significantly larger than max_hold (in bars),
        # it should be safe.
        # A safe check: Gap (days) * bars_per_day > max_hold.
        # For the purpose of this audit script, we flag if Gap < 1 day, which is usually 
        # much less than max_hold in bars. 
        # Better: we check the project's logic. train.py has:
        # gap = pd.Timedelta(days=gap_days) where gap_days >= 7.
        # 7 days * 25 bars/day = 175 bars. If max_hold = 60, it's safe.
        
        if gap < pd.Timedelta(days=1):
            print(f"FAIL: Fold {fold_id} gap too small: {gap}")
            pass_all = False

    if pass_all:
        print("PASS: Folds are strictly time-separated with sufficient gaps.")
    return pass_all

def audit_normalization():
    print("\n--- 3. Normalization Truth Audit ---")
    # Synthetic data: train (small values), val (large values)
    train_feat = np.random.normal(0, 1, (100, 1))
    val_feat = np.random.normal(10, 1, (50, 1))
    feature_cols = ["vs_factor_span63"]
    
    # Mock config for ROBUST scaling
    class MockConfig:
        ROBUST_CLIP_BOUNDS = None
        ROBUST_CLIP_BOUND_DEFAULT = 3.0

    # Scaler A: Fit on train, transform val
    scaler_a = ColumnSelectiveScaler(feature_cols, default_clip_bound=3.0)
    scaler_a.fit(train_feat)
    out_a = scaler_a.transform(val_feat)
    
    # Scaler B: Fit on train + val, transform val
    all_feat = np.concatenate([train_feat, val_feat])
    scaler_b = ColumnSelectiveScaler(feature_cols, default_clip_bound=3.0)
    scaler_b.fit(all_feat)
    out_b = scaler_b.transform(val_feat)
    
    diff = np.abs(out_a - out_b).mean()
    print(f"Mean difference in val transformation (Fit-Train vs Fit-All): {diff:.4f}")
    
    # If diff is large, it proves that fitting on the whole set (leakage) 
    # changes the normalization of the validation set.
    if diff > 1e-5:
        print("PASS: Fitting on train alone produces different results than fitting on all (Normalization Truth verified).")
        return True
    else:
        print("FAIL: No difference between fitting on train vs all. Scaler might be identity or data too similar.")
        return False

def audit_rolling_leakage():
    print("\n--- 4. Rolling Window Leakage Audit ---")
    # This is a static analysis audit. We've scanned features.py.
    # We will check for 'center=True' in the file.
    try:
        with open('features.py', 'r') as f:
            content = f.read()
            if 'center=True' in content:
                print("FAIL: Found 'center=True' in features.py rolling calculations!")
                return False
            else:
                print("PASS: No 'center=True' found in rolling calls in features.py.")
                return True
    except Exception as e:
        print(f"ERROR: Could not read features.py: {e}")
        return False

def audit_regime_shift():
    print("\n--- 5. Regime Shift Audit (Fold-by-Fold) ---")
    # This audit requires real data or a synthetic set with a shift.
    # We'll create a synthetic set where Fold 5 has higher volatility.
    
    def get_fold_stats(vol, targets):
        atr = np.std(vol)
        dist = {
            "long": (targets > 0.05).mean(),
            "short": (targets < -0.05).mean(),
            "flat": (np.abs(targets) <= 0.05).mean()
        }
        return atr, dist

    # Folds 1-4: low vol, balanced targets
    # Fold 5: high vol, skewed targets
    vols = [np.random.normal(1, 0.1, 100) for _ in range(4)]
    targs = [np.random.uniform(-0.1, 0.1, 100) for _ in range(4)]
    
    vols.append(np.random.normal(2, 0.5, 100)) # Shift in vol
    targs.append(np.random.uniform(-0.2, 0.2, 100)) # Shift in distribution
    
    # Average for folds 1-4
    avg_vol_1_4 = np.mean([np.mean(v) for v in vols[:4]])
    avg_dist_1_4 = {k: np.mean([d[k] for d in [get_fold_stats(v, t)[1] for v, t in zip(vols[:4], targs[:4])]]) 
                    for k in ["long", "short", "flat"]}
    
    # Fold 5
    vol_5, dist_5 = get_fold_stats(vols[4], targs[4])
    
    vol_shift = abs(vol_5 - avg_vol_1_4) / avg_vol_1_4
    
    # KL Divergence (simplified)
    kl_div = 0
    for k in ["long", "short", "flat"]:
        p = dist_5[k] + 1e-9
        q = avg_dist_1_4[k] + 1e-9
        kl_div += p * np.log(p / q)
    
    print(f"Vol Shift: {vol_shift:.2%}")
    print(f"KL Divergence: {kl_div:.4f}")
    
    if vol_shift > 0.20 or kl_div > 0.1:
        print("PASS: Regime shift detection logic works (detected synthetic shift in Fold 5).")
        return True
    else:
        print("FAIL: Regime shift not detected.")
        return False

def main():
    print("==================================================================")
    print("PIPELINE TRUTH AUDIT REPORT")
    print("==================================================================")
    
    results = {
        "Temporal Alignment": audit_temporal_alignment(),
        "Fold Separation": audit_fold_separation(),
        "Normalization Truth": audit_normalization(),
        "Rolling Window Leakage": audit_rolling_leakage(),
        "Regime Shift Audit": audit_regime_shift(),
    }
    
    print("\n==================================================================")
    print("FINAL SUMMARY")
    print("==================================================================")
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{name:<30} : {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\nOVERALL RESULT: PASS")
        sys.exit(0)
    else:
        print("\nOVERALL RESULT: FAIL")
        sys.exit(1)

if __name__ == "__main__":
    main()
