# clip_audit.py  — run ONCE on your training data to calibrate robust_clip_bound
# Usage: python clip_audit.py

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import config
from train import _make_feature_config, process_dataset, FeatureEngineer

def run_audit():
    # ── 1. Load actual training data ──────────────────────────────────────────
    print("Loading actual training data...")
    fe_config = _make_feature_config()
    fe = FeatureEngineer(config=fe_config)
    
    # process_dataset returns (asset_id, features, targets) per file
    asset_data_list, feature_cols = process_dataset(config.DATA_FILE, fe)
    
    # Identify robust columns (using data_loader logic)
    from data_loader import _col_bucket
    robust_cols_idx = [i for i, col in enumerate(feature_cols) if _col_bucket(col) == "robust"]
    robust_col_names = [feature_cols[i] for i in robust_cols_idx]
    
    if not robust_col_names:
        print("No robust-scaled columns found. Check data_loader._col_bucket() rules.")
        return

    print(f"Found {len(robust_col_names)} robust columns: {robust_col_names}")
    
    # Aggregate ALL training data across all assets
    # For each asset, we only use the training split (0 to train_end)
    train_features_list = []
    for asset_id, feat, target in asset_data_list:
        total_len = len(feat)
        train_end = int(total_len * config.TRAIN_RATIO)
        if train_end > 0:
            train_features_list.append(feat[:train_end, robust_cols_idx])
    
    if not train_features_list:
        print("No training data available for audit.")
        return
        
    all_train_robust = np.concatenate(train_features_list, axis=0) # (N, num_robust)
    
    # ── 2. Audit each robust column ───────────────────────────────────────────
    for i, col_name in enumerate(robust_col_names):
        data = all_train_robust[:, i].reshape(-1, 1)
        scaler = RobustScaler()
        transformed = scaler.fit_transform(data).flatten()
        raw_data = data.flatten()

        percentiles = [90, 95, 98, 99, 99.5, 99.9, 99.99, 100]
        
        print(f"\n{'='*65}")
        print(f"{'ROBUST-SCALED PERCENTILE AUDIT':^65}")
        print(f"{'Column: ' + col_name:^65}")
        print(f"  Scaler median={scaler.center_[0]:.2f}, IQR={scaler.scale_[0]:.2f}")
        print(f"{'='*65}")
        print(f"  {'Percentile':>12} | {'IQR-units':>12} | {'Raw value':>12}")
        print(f"  {'-'*42}")

        for p in percentiles:
            iqr_val = np.percentile(transformed, p)
            raw_val = np.percentile(raw_data, p)
            print(f"  {p:>11.2f}% | {iqr_val:>12.3f} | {raw_val:>12.1f}")

        # Symmetric (absolute value) percentiles
        print(f"\n  Symmetric |x| percentiles:")
        print(f"  {'-'*42}")
        abs_col = np.abs(transformed)
        for p in percentiles:
            print(f"  {p:>11.2f}% | {np.percentile(abs_col, p):>12.3f} IQR-units")

        # ── 4. Clip-rate table for candidate bounds ───────────────────────────
        print(f"\n{'='*65}")
        print(f"{'CLIP RATE BY CANDIDATE BOUND':^65}")
        print(f"{'='*65}")
        print(f"  {'Bound (IQR)':>12} | {'% clipped':>10} | {'# samples clipped':>18}")
        print(f"  {'-'*45}")

        for bound in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 10.0]:
            clipped = np.sum(abs_col > bound)
            rate    = clipped / len(transformed) * 100
            print(f"  {bound:>12.1f} | {rate:>9.3f}% | {clipped:>18,d}")

        print(f"\n  Total training samples: {len(transformed):,}")
        print(f"  Recommendation: choose bound where clip rate is 0.5–1.5%")
        print(f"  (clips pathological spikes, keeps economically meaningful extremes)")

if __name__ == "__main__":
    run_audit()
