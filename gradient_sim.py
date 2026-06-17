"""
Gradient simulation for Asymmetric Number-Line Loss v6.
Tests gradient magnitudes at various (pred, target) combinations to verify
the loss rebalancing (FLAT_L1_WEIGHT 0.8→0.2, EDGE_MAG_WEIGHT 0.5→3.0)
actually produces balanced gradients.
"""

import torch
import sys
sys.path.insert(0, '/home/ayanmarvin124/Lpatchtst')
import config

# Import the loss with OLD weights (v5) and NEW weights (v6)
# We'll manually set the weights to compare

def compute_gradients(pred_val, target_val, flat_l1_w, edge_mag_w, label=""):
    """Compute total gradient at a single (pred, target) point."""
    pred = torch.tensor([pred_val], requires_grad=True)
    target = torch.tensor([target_val])

    # Direction loss (asymmetric quadratic, all bars)
    alpha, beta = 2.0, 1.5
    gap = pred - target
    is_undershoot = (gap * target) < 0
    scale = alpha if is_undershoot.item() else beta
    mag_weight = target.abs().pow(0.5).clamp(min=0.7, max=2.0)
    loss_dir = (scale * gap.pow(2) * mag_weight)  # mean over 1 element = itself

    # Flat L1 penalty (flat bars only: target == 0)
    is_flat = target.abs() < 1e-8
    loss_flat_l1 = torch.tensor(0.0)
    if is_flat.item():
        loss_flat_l1 = flat_l1_w * pred.abs()

    # Flat quadratic penalty (flat bars above dead zone)
    loss_flat_q = torch.tensor(0.0)
    if is_flat.item():
        over = (pred.abs() - 0.08).clamp(min=0.0)
        loss_flat_q = 0.5 * over.pow(2)

    # Edge magnitude auxiliary (edge bars only: target != 0)
    is_edge = ~is_flat
    loss_edge = torch.tensor(0.0)
    if is_edge.item():
        diff = (pred.abs() - target.abs()).abs()
        loss_edge = edge_mag_w * (diff.pow(2) * 0.5 * (diff < 1).float()
                                   + (diff - 0.5) * (diff >= 1).float())

    total = loss_dir + loss_flat_l1 + loss_flat_q + loss_edge
    total.backward()

    grad = pred.grad.item()
    return {
        'label': label,
        'pred': pred_val,
        'target': target_val,
        'grad_dir': (2 * scale * gap * mag_weight).item(),
        'grad_flat_l1': (flat_l1_w * (1.0 if pred_val > 0 else -1.0)) if is_flat.item() else 0.0,
        'grad_flat_q': (2 * 0.5 * (abs(pred_val) - 0.08) * (1.0 if pred_val > 0 else -1.0)) if (is_flat.item() and abs(pred_val) > 0.08) else 0.0,
        'grad_edge': grad - (2 * scale * gap * mag_weight).item() - (flat_l1_w * (1.0 if pred_val > 0 else -1.0) if is_flat.item() else 0.0) - ((2 * 0.5 * (abs(pred_val) - 0.08) * (1.0 if pred_val > 0 else -1.0)) if (is_flat.item() and abs(pred_val) > 0.08) else 0.0),
        'grad_total': grad,
    }


def print_header(title):
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")


def print_results(results, flat_l1_w, edge_mag_w):
    print(f"\n  Weights: FLAT_L1_WEIGHT={flat_l1_w}, EDGE_MAG_WEIGHT={edge_mag_w}")
    print(f"  {'Case':>30s}  {'Pred':>6s}  {'Tgt':>6s}  {'Grad_dir':>10s}  {'Grad_flatL1':>12s}  {'Grad_flatQ':>11s}  {'Grad_edge':>10s}  {'Grad_TOTAL':>10s}")
    print(f"  {'-'*30}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*11}  {'-'*10}  {'-'*10}")
    for r in results:
        print(f"  {r['label']:>30s}  {r['pred']:>6.2f}  {r['target']:>+6.2f}  {r['grad_dir']:>+10.3f}  {r['grad_flat_l1']:>+12.3f}  {r['grad_flat_q']:>+11.3f}  {r['grad_edge']:>+10.3f}  {r['grad_total']:>+10.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: Flat bars (target = 0) — gradient should push pred toward 0
# ═══════════════════════════════════════════════════════════════════════════════
print_header("TEST 1: FLAT BARS (target=0) — gradient pushes pred → 0")

for flat_l1_w, edge_mag_w, version in [(0.8, 0.5, "v5 (OLD)"), (0.2, 3.0, "v6 (NEW)")]:
    results = []
    for pred_val in [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
        r = compute_gradients(pred_val, 0.0, flat_l1_w, edge_mag_w,
                              f"flat pred={pred_val:.2f}")
        results.append(r)
    print_results(results, flat_l1_w, edge_mag_w)

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: Edge bars — gradient should push pred toward target magnitude
# ═══════════════════════════════════════════════════════════════════════════════
print_header("TEST 2: EDGE BARS (|target|=0.5) — gradient pushes pred → target")

for flat_l1_w, edge_mag_w, version in [(0.8, 0.5, "v5 (OLD)"), (0.2, 3.0, "v6 (NEW)")]:
    results = []
    target = 0.5
    for pred_val in [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.53, 0.60, 0.80, 1.0, 1.5]:
        r = compute_gradients(pred_val, target, flat_l1_w, edge_mag_w,
                              f"edge pred={pred_val:.2f} tgt={target}")
        results.append(r)
    print_results(results, flat_l1_w, edge_mag_w)

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: Edge bars with negative target
# ═══════════════════════════════════════════════════════════════════════════════
print_header("TEST 3: EDGE BARS (target=-0.5) — gradient pushes pred → target")

for flat_l1_w, edge_mag_w, version in [(0.8, 0.5, "v5 (OLD)"), (0.2, 3.0, "v6 (NEW)")]:
    results = []
    target = -0.5
    for pred_val in [0.0, -0.05, -0.10, -0.20, -0.30, -0.40, -0.50, -0.53, -0.60, -0.80, -1.0]:
        r = compute_gradients(pred_val, target, flat_l1_w, edge_mag_w,
                              f"edge pred={pred_val:.2f} tgt={target}")
        results.append(r)
    print_results(results, flat_l1_w, edge_mag_w)

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: Wrong direction (undershoot) — should get alpha penalty
# ═══════════════════════════════════════════════════════════════════════════════
print_header("TEST 4: WRONG DIRECTION — target=+0.5, pred<0 (undershoot, alpha=2.0)")

for flat_l1_w, edge_mag_w, version in [(0.8, 0.5, "v5 (OLD)"), (0.2, 3.0, "v6 (NEW)")]:
    results = []
    target = 0.5
    for pred_val in [-0.50, -0.30, -0.10, -0.05, 0.0]:
        r = compute_gradients(pred_val, target, flat_l1_w, edge_mag_w,
                              f"wrong-dir pred={pred_val:.2f} tgt={target}")
        results.append(r)
    print_results(results, flat_l1_w, edge_mag_w)

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: Overshoot — target=+0.3, pred > target
# ═══════════════════════════════════════════════════════════════════════════════
print_header("TEST 5: OVERSHOOT — target=+0.3, pred>target (beta=1.5)")

for flat_l1_w, edge_mag_w, version in [(0.8, 0.5, "v5 (OLD)"), (0.2, 3.0, "v6 (NEW)")]:
    results = []
    target = 0.3
    for pred_val in [0.30, 0.40, 0.50, 0.60, 0.80, 1.0]:
        r = compute_gradients(pred_val, target, flat_l1_w, edge_mag_w,
                              f"overshoot pred={pred_val:.2f} tgt={target}")
        results.append(r)
    print_results(results, flat_l1_w, edge_mag_w)

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: Gradient ratio analysis — flat vs edge at equilibrium
# ═══════════════════════════════════════════════════════════════════════════════
print_header("TEST 6: GRADIENT RATIO ANALYSIS — flat vs edge gradient magnitudes")

print("\n  At the old equilibrium point (pred≈0.1 for everything):")
print(f"  {'Version':>10s}  {'Flat grad (|pred|=0.1)':>22s}  {'Edge grad (|pred|=0.1, |tgt|=0.5)':>34s}  {'Ratio flat/edge':>16s}")
print(f"  {'-'*10}  {'-'*22}  {'-'*34}  {'-'*16}")

for flat_l1_w, edge_mag_w, version in [(0.8, 0.5, "v5 (OLD)"), (0.2, 3.0, "v6 (NEW)")]:
    flat_r = compute_gradients(0.10, 0.0, flat_l1_w, edge_mag_w, "flat")
    edge_r = compute_gradients(0.10, 0.5, flat_l1_w, edge_mag_w, "edge")
    ratio = abs(flat_r['grad_total']) / (abs(edge_r['grad_total']) + 1e-9)
    print(f"  {version:>10s}  {flat_r['grad_total']:>+22.3f}  {edge_r['grad_total']:>+34.3f}  {ratio:>16.2f}x")

print("\n  At the new expected equilibrium (pred≈0.2 for edge, pred≈0.03 for flat):")
print(f"  {'Version':>10s}  {'Flat grad (|pred|=0.03)':>22s}  {'Edge grad (|pred|=0.2, |tgt|=0.5)':>34s}  {'Ratio flat/edge':>16s}")
print(f"  {'-'*10}  {'-'*22}  {'-'*34}  {'-'*16}")

for flat_l1_w, edge_mag_w, version in [(0.8, 0.5, "v5 (OLD)"), (0.2, 3.0, "v6 (NEW)")]:
    flat_r = compute_gradients(0.03, 0.0, flat_l1_w, edge_mag_w, "flat")
    edge_r = compute_gradients(0.20, 0.5, flat_l1_w, edge_mag_w, "edge")
    ratio = abs(flat_r['grad_total']) / (abs(edge_r['grad_total']) + 1e-9)
    print(f"  {version:>10s}  {flat_r['grad_total']:>+22.3f}  {edge_r['grad_total']:>+34.3f}  {ratio:>16.2f}x")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 7: Effective gradient per class (accounting for sampler weights)
# ═══════════════════════════════════════════════════════════════════════════════
print_header("TEST 7: EFFECTIVE GRADIENT WITH SAMPLER WEIGHTS")

# From the logs: train has ~50% flat, ~27% long, ~23% short
# Sampler uses 4th-root inverse frequency:
#   flat weight ≈ (1/0.50)^0.25 = 1.189
#   long weight ≈ (1/0.27)^0.25 = 1.384
#   short weight ≈ (1/0.23)^0.25 = 1.449
# After normalization, directional bars get ~1.2x the flat weight

sampler_flat_w = 1.0
sampler_edge_w = 1.2  # approximate boost for directional bars

print(f"\n  Sampler weights: flat={sampler_flat_w:.2f}, edge={sampler_edge_w:.2f}")
print(f"  Class distribution: flat~50%, edge~50% (long~27%, short~23%)")
print()

for flat_l1_w, edge_mag_w, version in [(0.8, 0.5, "v5 (OLD)"), (0.2, 3.0, "v6 (NEW)")]:
    # Average gradient contribution per bar, weighted by class frequency and sampler
    # Flat bars: pred~0.10 (from logs)
    flat_r = compute_gradients(0.10, 0.0, flat_l1_w, edge_mag_w, "flat")
    # Edge bars: pred~0.10, target~0.5 (strong edge from logs)
    edge_r = compute_gradients(0.10, 0.5, flat_l1_w, edge_mag_w, "edge")

    # Effective gradient = sum over classes of (freq * sampler_weight * grad)
    eff_flat = 0.50 * sampler_flat_w * flat_r['grad_total']
    eff_edge = 0.50 * sampler_edge_w * edge_r['grad_total']
    eff_total = eff_flat + eff_edge

    print(f"  {version}:")
    print(f"    Flat contribution: 0.50 × {sampler_flat_w:.2f} × {flat_r['grad_total']:+.3f} = {eff_flat:+.4f}")
    print(f"    Edge contribution: 0.50 × {sampler_edge_w:.2f} × {edge_r['grad_total']:+.3f} = {eff_edge:+.4f}")
    print(f"    Net effective gradient: {eff_total:+.4f}")
    print(f"    Interpretation: {'PRED PUSHED TOWARD ZERO (magnitude collapse)' if eff_total > 0 else 'PRED PUSHED TOWARD TARGET (magnitude recovery)'}")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 8: Edge magnitude gradient breakdown at various pred/target combos
# ═══════════════════════════════════════════════════════════════════════════════
print_header("TEST 8: EDGE MAGNITUDE GRADIENT — how strongly does edge_mag loss push?")

print(f"\n  {'Pred':>6s}  {'|Tgt':>6s}  {'|Pred|-|Tgt':>10s}  {'v5 edge_grad':>12s}  {'v6 edge_grad':>12s}  {'Improvement':>12s}")
print(f"  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}")

test_cases = [
    (0.05, 0.5), (0.10, 0.5), (0.20, 0.5), (0.30, 0.5),
    (0.40, 0.5), (0.50, 0.5), (0.60, 0.5), (0.80, 0.5),
    (0.10, 0.3), (0.20, 0.3), (0.30, 0.3), (0.50, 0.3),
    (0.05, 1.0), (0.30, 1.0), (0.50, 1.0), (1.0, 1.0),
    (1.5, 1.0), (2.0, 1.0),
]

for pred_val, tgt_val in test_cases:
    r5 = compute_gradients(pred_val, tgt_val, 0.8, 0.5, "v5")
    r6 = compute_gradients(pred_val, tgt_val, 0.2, 3.0, "v6")
    diff = abs(pred_val) - abs(tgt_val)
    # Extract just the edge component
    edge5 = r5['grad_edge']
    edge6 = r6['grad_edge']
    ratio = abs(edge6) / (abs(edge5) + 1e-9)
    print(f"  {pred_val:>6.2f}  {abs(tgt_val):>6.2f}  {diff:>+10.2f}  {edge5:>+12.3f}  {edge6:>+12.3f}  {ratio:>10.1f}x")

print(f"\n  Note: edge_grad is the gradient contribution from EDGE_MAG_WEIGHT alone.")
print(f"  v6 edge gradient is ~6x stronger (3.0/0.5 = 6x weight multiplier)")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 9: Verify direction loss gradient is unchanged between v5 and v6
# ═══════════════════════════════════════════════════════════════════════════════
print_header("TEST 9: DIRECTION LOSS GRADIENT — should be identical (weights unchanged)")

print(f"\n  {'Pred':>6s}  {'Tgt':>6s}  {'v5 dir_grad':>12s}  {'v6 dir_grad':>12s}  {'Match':>8s}")
print(f"  {'-'*6}  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*8}")

for pred_val, tgt_val in [(0.1, 0.5), (-0.1, 0.5), (0.3, 0.5), (0.6, 0.5), (0.1, -0.5), (0.0, 0.3)]:
    r5 = compute_gradients(pred_val, tgt_val, 0.8, 0.5, "v5")
    r6 = compute_gradients(pred_val, tgt_val, 0.2, 3.0, "v6")
    match = "✓" if abs(r5['grad_dir'] - r6['grad_dir']) < 1e-6 else "✗"
    print(f"  {pred_val:>6.2f}  {tgt_val:>+6.2f}  {r5['grad_dir']:>+12.3f}  {r6['grad_dir']:>+12.3f}  {match:>8s}")

print(f"\n  Direction loss is unchanged — only flat L1 and edge mag weights changed.")
