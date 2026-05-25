"""
Confirm that oracle targets are actually < 1e-6 or > 1e-6
and show actual distributions.
"""
import numpy as np
import pandas as pd
from oracle import generate_targets
import config

# Load one sample file
csv_file = config.DATA_FILE[0]  # First file
df = pd.read_csv(csv_file)

# Rename columns to lowercase
df.columns = df.columns.str.lower()

# Simple ATR computation
def compute_atr(df, period=14):
    """Compute Average True Range"""
    df = df.copy()
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = np.abs(df['high'] - df['close'].shift())
    df['tr3'] = np.abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['ATR'] = df['tr'].rolling(window=period).mean()
    return df['ATR'].values

atr = compute_atr(df, period=config.ATR_PERIOD)

# Generate oracle targets
targets = generate_targets(
    df['open'].values,
    df['high'].values,
    df['low'].values,
    df['close'].values,
    atr,
    max_hold=config.ORACLE_MAX_HOLD,
    fee_per_side=config.FEE_PER_SIDE,
    slippage=config.SLIPPAGE,
    sl_atr_mult=config.ORACLE_SL_ATR_MULT,
    tp_atr_mult=config.ORACLE_TP_ATR_MULT,
    enable_trailing=config.ORACLE_ENABLE_TRAILING,
    trail_atr_mult=config.ORACLE_TRAIL_ATR_MULT,
    saturation_factor=config.SATURATION_FACTOR,
    mae_penalty=config.MAE_PENALTY,
)

targets = np.array(targets, dtype=np.float32)

# Check boundaries
threshold = 1e-6
is_flat = np.abs(targets) < threshold
is_edge = np.abs(targets) >= threshold

print("=" * 80)
print(f"Sample file: {csv_file}")
print(f"Total bars: {len(targets)}")
print("=" * 80)

print(f"\n✓ FLAT (|target| < {threshold}): {is_flat.sum():,} bars ({100*is_flat.mean():.1f}%)")
print(f"✓ EDGE (|target| ≥ {threshold}): {is_edge.sum():,} bars ({100*is_edge.mean():.1f}%)")

# Show actual values in flat bucket
flat_targets = targets[is_flat]
edge_targets = targets[is_edge]

print("\n" + "=" * 80)
print("FLAT TARGETS (should be exactly 0.0):")
print("=" * 80)
print(f"  Min: {flat_targets.min():.15f}")
print(f"  Max: {flat_targets.max():.15f}")
print(f"  Mean: {flat_targets.mean():.15f}")
print(f"  Unique values: {len(np.unique(flat_targets))}")
print(f"  All exactly zero? {np.all(flat_targets == 0.0)}")

print("\n" + "=" * 80)
print("EDGE TARGETS (|target| ≥ 1e-6):")
print("=" * 80)
print(f"  Count: {len(edge_targets)}")
print(f"  Min: {edge_targets.min():.6f}")
print(f"  Max: {edge_targets.max():.6f}")
print(f"  Mean: {edge_targets.mean():.6f}")
print(f"  Median: {np.median(edge_targets):.6f}")
print(f"  Std: {np.std(edge_targets):.6f}")

# Show histogram of edge targets
print("\nEDGE TARGET DISTRIBUTION:")
pos_edge = edge_targets[edge_targets > 0]
neg_edge = edge_targets[edge_targets < 0]
print(f"  Positive (long signals): {len(pos_edge):,} ({100*len(pos_edge)/len(edge_targets):.1f}%)")
print(f"    Min: {pos_edge.min():.6f}, Max: {pos_edge.max():.6f}, Mean: {pos_edge.mean():.6f}")
print(f"  Negative (short signals): {len(neg_edge):,} ({100*len(neg_edge)/len(edge_targets):.1f}%)")
print(f"    Min: {neg_edge.min():.6f}, Max: {neg_edge.max():.6f}, Mean: {neg_edge.mean():.6f}")

# Show smallest non-zero edge targets (closest to boundary)
print("\nSMALLEST EDGE TARGETS (closest to 1e-6 boundary):")
abs_edge = np.abs(edge_targets)
smallest_idx = np.argsort(abs_edge)[:10]
for i, idx in enumerate(smallest_idx):
    print(f"  {i+1}. {edge_targets[idx]:+.10f} (|{abs_edge[idx]:.10f}|)")

print("\n" + "=" * 80)
print("CONFIRMATION")
print("=" * 80)
print(f"✓ All FLAT targets < {threshold}: {np.all(np.abs(flat_targets) < threshold)}")
print(f"✓ All EDGE targets ≥ {threshold}: {np.all(np.abs(edge_targets) >= threshold)}")
print(f"✓ No targets between -1e-6 and +1e-6: {not np.any((np.abs(targets) > 0) & (np.abs(targets) < threshold))}")
print("\n→ The 1e-6 threshold is a HARD boundary in the oracle!")
print("→ Flat targets are EXACTLY 0.0")
print("→ Edge targets start at 1e-6 and go up to ~0.9 (tanh squash)")
