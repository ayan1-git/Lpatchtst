# check_features.py — v2 feature scale audit
#
# The v2 feature set is passed through UNTOUCHED (NO_SCALE bucket) by the
# scaler, so this audit answers: are all 10 features at a comparable level,
# or does one dominate another?
import os
import sys
import pandas as pd
import numpy as np

# Ensure we can import config/features/train from core/
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "core"))

import config
from features import FeatureEngineer
from train import _make_feature_config


def check_features():
    print("\n" + "=" * 100)
    print(f"  FEATURE SCALE CHECK (v2)  (INPUT_MODE={config.INPUT_MODE})")
    print("=" * 100)

    fe_config = _make_feature_config()
    fe = FeatureEngineer(config=fe_config)

    # 1. Load Data
    data_files = config.DATA_FILE
    file_path = data_files[0] if isinstance(data_files, list) else data_files

    if not os.path.exists(file_path):
        # DATA_FILE entries are relative to the repo root — resolve there
        alt = os.path.join(_ROOT, file_path)
        if os.path.exists(alt):
            file_path = alt
        else:
            print(f"Error: {file_path} not found.")
            return

    print(f"Loading data: {file_path}")
    df_raw = pd.read_csv(file_path)
    df_raw.columns = [c.lower() for c in df_raw.columns]

    # 2. Build Features (dropna=True → final model inputs after warm-up cut)
    print("Engineering features...")
    df = fe.build(df_raw["close"], ohlc=df_raw, include_target=False, dropna=True)
    feature_cols = df.columns.tolist()

    # 3. Statistical Summary
    print("\n" + "-" * 118)
    print(f"  {'Feature Name':<20} | {'Mean':>8} | {'Std':>8} | {'Min':>9} | "
          f"{'Max':>9} | {'p1':>8} | {'p99':>8} | {'|Mean|/Std':>10}")
    print("-" * 118)

    stats = []
    for col in feature_cols:
        s = df[col]
        stats.append({
            "name": col,
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "max": s.max(),
            "p1": s.quantile(0.01),
            "p99": s.quantile(0.99),
            "count": s.count(),
        })

    # Sort: raw return first, then normalized returns, vols, momentum
    order = {c: i for i, c in enumerate(fe_config.feature_columns)}
    stats.sort(key=lambda x: order.get(x["name"], 99))

    for item in stats:
        centering = abs(item["mean"]) / item["std"] if item["std"] > 0 else float("inf")
        print(f"  {item['name']:<20} | {item['mean']:8.4f} | {item['std']:8.4f} | "
              f"{item['min']:9.4f} | {item['max']:9.4f} | {item['p1']:8.4f} | "
              f"{item['p99']:8.4f} | {centering:10.4f}")

    print("-" * 118)
    print(f"  Total Features: {len(feature_cols)}   Rows: {len(df)}")

    # 4. Dominance analysis — do any features dwarf the others?
    stds = np.array([it["std"] for it in stats])
    means = np.array([abs(it["mean"]) for it in stats])
    valid = stds > 1e-12

    print("\nDominance Analysis (all features share the NO_SCALE bucket, so")
    print("their raw magnitudes ARE what the model sees):")

    if valid.any():
        std_ratio = stds[valid].max() / stds[valid].min()
        dom_std = max(stats, key=lambda x: x["std"] if x["std"] > 1e-12 else -1)["name"]
        weak_std = min((x for x in stats if x["std"] > 1e-12),
                       key=lambda x: x["std"])["name"]
        print(f"  Std spread (max/min): {std_ratio:.2f}x  "
              f"(largest: {dom_std}, smallest: {weak_std})")
        if std_ratio > 5:
            print("  WARNING: >5x std spread — the large-scale feature will dominate")
            print("           the loss/gradients unless the model handles it internally.")
        else:
            print("  OK: features are within a comparable magnitude band (<5x).")

    # 5. Per-feature anomalies
    print("\nAnomalies / Warnings:")
    found_issue = False
    for item in stats:
        if item["count"] == 0:
            print(f"  Empty Feature: {item['name']}")
            found_issue = True
            continue
        if item["std"] < 1e-6:
            print(f"  Zero Variance: {item['name']} (std={item['std']:.8f})")
            found_issue = True
            continue
        centering = abs(item["mean"]) / item["std"]
        # A feature whose mean dwarfs its std is effectively a constant offset
        # to the network and can dominate dot-product attention scores.
        if centering > 1.0:
            print(f"  Off-Center:    {item['name']}  |mean|/std = {centering:.3f} "
                  f"(mean={item['mean']:.4f}, std={item['std']:.4f})")
            found_issue = True
        tail = max(abs(item["min"]), abs(item["max"]))
        if tail > 50:
            print(f"  Heavy Tail:    {item['name']}  max|value| = {tail:.2f}")
            found_issue = True

    if not found_issue:
        print("  All features appear within reasonable bounds.")

    print("=" * 100 + "\n")


if __name__ == "__main__":
    check_features()
