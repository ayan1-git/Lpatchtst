# investigate_outliers.py
import os
import numpy as np
import pandas as pd
import config
from train import _make_feature_config, _build_features, FeatureEngineer

def investigate():
    # ── 1. Setup ─────────────────────────────────────────────────────────────
    print("🚀 Initializing outlier investigation...")
    fe_config = _make_feature_config()
    fe = FeatureEngineer(config=fe_config)
    
    file_path = "Data /NIFTY 50_30minute.csv"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    print(f"📂 Loading data: {file_path}")
    df_raw = pd.read_csv(file_path)
    
    # ── 2. Engineer Features ──────────────────────────────────────────────────
    print("🛠️  Engineering features...")
    df, feature_cols = _build_features(df_raw, fe)
    
    # ── 3. Find Outliers ──────────────────────────────────────────────────────
    print("\n🔍 Searching for outliers...")
    
    # Target 1: ret_norm_1d min (~ -9.49)
    col_ret = "ret_norm_1d"
    min_ret_idx = df[col_ret].idxmin()
    min_ret_val = df[col_ret].min()
    
    # Target 2: macd_8_24 min (~ -5.68)
    col_macd = "macd_8_24"
    min_macd_idx = df[col_macd].idxmin()
    min_macd_val = df[col_macd].min()
    
    print(f"\n[OUTLIER 1] {col_ret}")
    print(f"Minimum Value: {min_ret_val:.4f}")
    print(f"Timestamp:     {min_ret_idx}")
    
    # Show surrounding context
    ctx_start = df.index.get_loc(min_ret_idx) - 5
    ctx_end   = df.index.get_loc(min_ret_idx) + 5
    print("\nContext (OHLC + Feature):")
    print(df.iloc[ctx_start:ctx_end][["open", "high", "low", "close", "atr", col_ret]])
    
    print(f"\n[OUTLIER 2] {col_macd}")
    print(f"Minimum Value: {min_macd_val:.4f}")
    print(f"Timestamp:     {min_macd_idx}")
    
    ctx_start = df.index.get_loc(min_macd_idx) - 5
    ctx_end   = df.index.get_loc(min_macd_idx) + 5
    print("\nContext (OHLC + Feature):")
    print(df.iloc[ctx_start:ctx_end][["open", "high", "low", "close", "atr", col_macd]])

    # ── 4. Verify Magnitude ───────────────────────────────────────────────────
    # For ret_norm_1d, check the raw return vs the EWMA vol
    row = df.loc[min_ret_idx]
    # In features.py: ret_norm = raw_ret / (vol * sqrt(h))
    # raw_ret = close_t / close_{t-h} - 1
    # We need to find the previous price.
    
    prev_idx = df.index[df.index.get_loc(min_ret_idx) - 1]
    raw_ret = (df.loc[min_ret_idx, "close"] / df.loc[prev_idx, "close"]) - 1
    # Note: ret_norm_1d in features.py is computed on a 1-bar horizon by default if h=1.
    # Actually FE_RETURN_HORIZONS = [1, 3, 6, ...]
    
    print(f"\n[VERIFICATION] {min_ret_idx}")
    print(f"Raw Return: {raw_ret:.4%}")
    # The vol used is ewma_vol_span260
    vol = df.loc[min_ret_idx, f"ewma_vol_span{config.FE_VOL_LONG_PERIOD}"]
    print(f"EWMA Vol (σ): {vol:.6f}")
    calc_norm = raw_ret / vol
    print(f"Calculated Norm: {calc_norm:.4f} (Expected ~ {min_ret_val:.4f})")

if __name__ == "__main__":
    investigate()
