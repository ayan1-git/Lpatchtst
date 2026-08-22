import torch
import numpy as np
from data_loader import _compute_sample_weights
from torch.utils.data import WeightedRandomSampler
from collections import Counter

class MockConfig:
    def __init__(self, bias_correction_power=0.0):
        self.BIAS_CORRECTION_POWER = bias_correction_power

def run_audit(targets, thresh, power, num_samples=100000):
    config = MockConfig(bias_correction_power=power)
    weights = _compute_sample_weights(targets, thresh, config=config, use_sqrt=True)
    
    # We use WeightedRandomSampler to see what actually gets picked
    sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
    
    # Sample indices
    sampled_indices = list(sampler)
    sampled_targets = targets[sampled_indices]
    
    # Assign classes based on the same logic as in _compute_sample_weights
    classes = np.zeros_like(sampled_targets, dtype=np.int32)
    classes[sampled_targets < -thresh] = 0
    classes[np.abs(sampled_targets) < thresh] = 1
    classes[sampled_targets > thresh] = 2
    
    counts = Counter(classes)
    proportions = {k: v / num_samples for k, v in counts.items()}
    
    # Theoretical calculation
    actual_counts = np.bincount(
        np.zeros_like(targets, dtype=np.int32) + 
        (targets < -thresh) * 0 + 
        (np.abs(targets) < thresh) * 1 + 
        (targets > thresh) * 2, 
        minlength=3
    )
    # Note: the above is a bit clunky, let's just use the same logic as the function
    cls = np.zeros_like(targets, dtype=np.int32)
    cls[targets < -thresh] = 0
    cls[np.abs(targets) < thresh] = 1
    cls[targets > thresh] = 2
    actual_counts = np.bincount(cls, minlength=3)
    
    # Weights calculation
    # W_base = 1/sqrt(C)
    # W_maj = W_base[maj]
    # W_i = W_base[i] * (C_i/C_maj)^power
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
    # Scenario: Highly imbalanced targets
    # Short: 100, Flat: 1000, Long: 100
    # Total: 1200
    np.random.seed(42)
    targets = np.concatenate([
        np.random.uniform(-1.0, -0.2, 100), # Short
        np.random.uniform(-0.1, 0.1, 1000), # Flat
        np.random.uniform(0.2, 1.0, 100),   # Long
    ])
    np.random.shuffle(targets)
    
    thresh = 0.15
    powers = [0.0, -0.5, 1.0]
    
    print(f"Dataset Distribution: Short=100, Flat=1000, Long=100 (Total=1200)")
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
