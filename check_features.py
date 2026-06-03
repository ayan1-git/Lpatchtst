# check_features.py
import os
import sys
import pandas as pd
import numpy as np

# Ensure we can import from current directory
sys.path.insert(0, os.getcwd())

import config
from train import _make_feature_config, FeatureEngineer

def check_features():
    # 1. Setup Configuration
    print("\n" + "="*100)
    print(f"  FEATURE NORMALIZATION CHECK  (INPUT_MODE={config.INPUT_MODE}, USE_TALIB={config.USE_TALIB})")
    print("="*100)

    fe_config = _make_feature_config()
    fe = FeatureEngineer(config=fe_config)

    # 2. Load Data
    data_files = config.DATA_FILE
    if isinstance(data_files, list):
        file_path = data_files[0]
    else:
        file_path = data_files

    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    print(f"📂 Loading data: {file_path}")
    df_raw = pd.read_csv(file_path)
    
    # 3. Build Features
    print("🛠️  Engineering features...")
    # Standardize column names to lowercase to match FeatureEngineer expectations
    df_raw.columns = [c.lower() for c in df_raw.columns]
    
    # Note: We use fe.build with dropna=True to see final model inputs
    df = fe.build(df_raw['close'], ohlc=df_raw, include_target=False, dropna=True)
    feature_cols = df.columns.tolist()
    
    # 4. Statistical Summary
    print("\n" + "-"*100)
    print(f"  {'Feature Name':<35} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8} | {'Non-NaN':>8}")
    print("-"*100)

    stats = []
    for col in feature_cols:
        s = df[col]
        stats.append({
            "name": col,
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "max": s.max(),
            "count": s.count()
        })

    # Sort by name for easier scanning
    stats.sort(key=lambda x: x["name"])

    for item in stats:
        print(f"  {item['name']:<35} | {item['mean']:8.4f} | {item['std']:8.4f} | {item['min']:8.4f} | {item['max']:8.4f} | {item['count']:8d}")

    print("-"*100)
    print(f"  Total Features: {len(feature_cols)}")
    print(f"  Total Rows:     {len(df)}")
    
    # Check for potential issues
    print("\n🔍 Anomalies / Warnings:")
    found_issue = False
    for item in stats:
        if abs(item['mean']) > 5.0 and "vs_factor" not in item['name']:
            print(f"  ⚠️  High Mean: {item['name']} ({item['mean']:.4f})")
            found_issue = True
        if item['count'] == 0:
            print(f"  ⚠️  Empty Feature: {item['name']}")
            found_issue = True
        elif item['std'] < 1e-6:
            # We ignore volume features if volume is all zeros
            if "mfi" in item['name'] or "ad" in item['name']:
                continue
            print(f"  ⚠️  Zero Variance: {item['name']} (std={item['std']:.6f})")
            found_issue = True

    if not found_issue:
        print("  ✅ All features appear within reasonable bounds.")
    
    print("="*100 + "\n")

if __name__ == "__main__":
    check_features()
