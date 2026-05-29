import torch
import numpy as np
import pandas as pd
import config
from data_loader import _compute_sample_weights
from oracle import generate_targets
from torch.utils.data import WeightedRandomSampler
from collections import Counter

class MockConfig:
    def __init__(self, bias_correction_power=0.0):
        self.BIAS_CORRECTION_POWER = bias_correction_power

def load_actual_targets(file_path):
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df = df.iloc[1:]
    
    # Compute ATR as done in train.py
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    atr = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(config.ATR_PERIOD).mean()
    
    target_vals = generate_targets(
        df["open"].values,
        df["high"].values,
        df["low"].values,
        df["close"].values,
        atr.values,
    )
    
    # Target alignment: zero-out based on SAMPLER_THRESHOLD
    thresh = getattr(config, "SAMPLER_THRESHOLD", 0.0)
    target_vals[np.abs(target_vals) < thresh] = 0.0
    
    return target_vals, thresh

def run_audit(targets, thresh, power, num_samples=100000):
    config_mock = MockConfig(bias_correction_power=power)
    weights = _compute_sample_weights(targets, thresh, config=config_mock, use_sqrt=True)
    
    sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
    sampled_indices = list(sampler)
    sampled_targets = targets[sampled_indices]
    
    classes = np.zeros_like(sampled_targets, dtype=np.int32)
    classes[sampled_targets < -thresh] = 0
    classes[np.abs(sampled_targets) < thresh] = 1
    classes[sampled_targets > thresh] = 2
    
    counts = Counter(classes)
    proportions = {k: v / num_samples for k, v in counts.items()}
    
    # Theoretical calculation
    cls = np.zeros_like(targets, dtype=np.int32)
    cls[targets < -thresh] = 0
    cls[np.abs(targets) < thresh] = 1
    cls[targets > thresh] = 2
    actual_counts = np.bincount(cls, minlength=3)
    
    maj_idx = np.argmax(actual_counts)
    c_maj = actual_counts[maj_idx]
    
    expected_mass = []
    for i in range(3):
        c_i = actual_counts[i]
        w_base_i = 1.0 / np.sqrt(c_i) if c_i > 0 else 0
        if i == maj_idx:
            mass = c_i * w_base_i
        else:
            mass = c_i * w_base_i * ((c_i / c_maj) ** power)
        expected_mass.append(mass)
    
    total_mass = sum(expected_mass)
    expected_proportions = {i: m / total_mass for i, m in enumerate(expected_mass)}
    
    return proportions, expected_proportions

def main():
    file_path = "Data/NIFTY 100_30minute.csv"
    targets, thresh = load_actual_targets(file_path)
    
    powers = [0.0, -0.5, 1.0]
    
    print(f"\nActual Dataset Distribution:")
    cls = np.zeros_like(targets, dtype=np.int32)
    cls[targets < -thresh] = 0
    cls[np.abs(targets) < thresh] = 1
    cls[targets > thresh] = 2
    counts = np.bincount(cls, minlength=3)
    for i, name in enumerate(["Short", "Flat", "Long"]):
        print(f"  {name}: {counts[i]} ({counts[i]/len(targets)*100:.2f}%)")
    
    print(f"Threshold: {thresh}")
    print("-" * 80)
    print(f"{'Power':<10} | {'Class':<10} | {'Expected %':<15} | {'Observed %':<15} | {'Diff':<10}")
    print("-" * 80)
    
    for p in powers:
        obs, exp = run_audit(targets, thresh, p)
        for i in range(3):
            class_name = ["Short", "Flat", "Long"][i]
            o = obs.get(i, 0) * 100
            e = exp.get(i, 0) * 100
            print(f"{p:<10.2f} | {class_name:<10} | {e:<15.2f} | {o:<15.2f} | {abs(o-e):<10.2f}")
        print("-" * 80)

if __name__ == "__main__":
    main()
