#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
GRADIENT DEEP SIMULATION — Asymmetric Number-Line Loss v6
═══════════════════════════════════════════════════════════════════════════════

For each (target, prediction) pair, we compute the analytical gradient of every
loss component with respect to the prediction:

  L_dir    = scale_asymmetry * gap^2 * mag_weight       (all bars)
  L_flat1  = lambda * |pred|                             (flat bars only)
  L_flat2  = w * max(0, |pred| - margin)^2              (flat bars only)
  L_edge   = w_edge * smooth_l1(|pred|, |target|)       (edge bars only)

Where:
  gap = pred - target
  is_undershoot = (gap * target) < 0
  scale = alpha if undershoot else beta
  mag_weight = |target|^0.5 clamped to [0.7, 2.0]

Gradients:
  dL_dir/dpred    = 2 * scale * gap * mag_weight
  dL_flat1/dpred  = lambda * sign(pred)          (constant!)
  dL_flat2/dpred  = 2 * w * max(0, |pred|-margin) * sign(pred)
  dL_edge/dpred   = w_edge * smooth_l1_grad(|pred|, |target|) * sign(pred)

The smooth_l1 gradient:
  if |diff| < 1:  grad = diff          (where diff = |pred| - |target|)
  else:           grad = sign(diff)

We simulate single-bar scenarios (mean over 1 element = the value itself).
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator

# ── Loss constants (must match loss.py) ──────────────────────────────────────
ALPHA           = 2.0
BETA            = 1.5
FLAT_L1_WEIGHT  = 0.2
FLAT_MARGIN     = 0.08
FLAT_PENALTY_W  = 0.5
EDGE_MAG_WEIGHT = 3.0


# ═══════════════════════════════════════════════════════════════════════════════
# Analytical gradient computation
# ═══════════════════════════════════════════════════════════════════════════════

def mag_weight(target):
    """|target|^0.5 clamped to [0.7, 2.0]"""
    return np.clip(np.abs(target) ** 0.5, 0.7, 2.0)


def is_undershoot(pred, target):
    """(gap * target) < 0"""
    gap = pred - target
    return (gap * target) < 0


def scale_factor(pred, target):
    """alpha for undershoot, beta otherwise"""
    return np.where(is_undershoot(pred, target), ALPHA, BETA)


def grad_direction(pred, target):
    """d/dpred [scale * (pred-target)^2 * mw] = 2 * scale * (pred-target) * mw"""
    mw = mag_weight(target)
    sc = scale_factor(pred, target)
    return 2.0 * sc * (pred - target) * mw


def grad_flat_l1(pred):
    """d/dpred [lambda * |pred|] = lambda * sign(pred)"""
    return FLAT_L1_WEIGHT * np.sign(pred)


def grad_flat_quad(pred):
    """d/dpred [w * max(0, |pred|-margin)^2] = 2*w*max(0,|pred|-margin)*sign(pred)"""
    over = np.maximum(np.abs(pred) - FLAT_MARGIN, 0.0)
    return 2.0 * FLAT_PENALTY_W * over * np.sign(pred)


def grad_edge_mag(pred, target):
    """Gradient of smooth_l1(|pred|, |target|) w.r.t. pred"""
    p_abs = np.abs(pred)
    t_abs = np.abs(target)
    diff = p_abs - t_abs
    abs_diff = np.abs(diff)

    # smooth_l1 gradient w.r.t. diff
    grad_diff = np.where(abs_diff < 1.0, diff, np.sign(diff))

    # chain rule: d|pred|/dpred = sign(pred)
    return EDGE_MAG_WEIGHT * grad_diff * np.sign(pred)


def compute_all_grads(pred, target):
    """
    Returns dict with all gradient components and total.
    For flat bars (target==0): L_dir + L_flat_l1 + L_flat_quad
    For edge bars (target!=0): L_dir + L_edge_mag
    """
    eps = 1e-8
    flat = np.abs(target) < eps

    g_dir   = grad_direction(pred, target)
    g_flat1 = grad_flat_l1(pred) if flat else 0.0
    g_flat2 = grad_flat_quad(pred) if flat else 0.0
    g_edge  = grad_edge_mag(pred, target) if not flat else 0.0

    g_total = g_dir + g_flat1 + g_flat2 + g_edge

    return {
        "g_dir":    g_dir,
        "g_flat1":  g_flat1,
        "g_flat2":  g_flat2,
        "g_edge":   g_edge,
        "g_total":  g_total,
        "flat":     flat,
    }


def compute_all_losses(pred, target):
    """Compute actual loss values for each component."""
    eps = 1e-8
    flat = np.abs(target) < eps
    gap = pred - target
    sc = scale_factor(pred, target)
    mw = mag_weight(target)

    l_dir = sc * gap**2 * mw

    l_flat1 = FLAT_L1_WEIGHT * np.abs(pred) if flat else 0.0
    over = np.maximum(np.abs(pred) - FLAT_MARGIN, 0.0)
    l_flat2 = FLAT_PENALTY_W * over**2 if flat else 0.0

    l_edge = 0.0
    if not flat:
        p_abs = np.abs(pred)
        t_abs = np.abs(target)
        diff = np.abs(p_abs - t_abs)
        l_edge = EDGE_MAG_WEIGHT * np.where(
            diff < 1.0,
            0.5 * diff**2,
            diff - 0.5,
        )

    l_total = l_dir + l_flat1 + l_flat2 + l_edge

    return {
        "l_dir":   l_dir,
        "l_flat1": l_flat1,
        "l_flat2": l_flat2,
        "l_edge":  l_edge,
        "l_total": l_total,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Numerical gradient verification
# ═══════════════════════════════════════════════════════════════════════════════

def numerical_grad(pred, target, eps=1e-5):
    """Finite-difference gradient for verification."""
    def total_loss(p, t):
        gap = p - t
        sc = ALPHA if ((gap * t) < 0) else BETA
        mw = np.clip(np.abs(t) ** 0.5, 0.7, 2.0)
        l = sc * gap**2 * mw

        if np.abs(t) < 1e-8:
            l += FLAT_L1_WEIGHT * np.abs(p)
            over = np.maximum(np.abs(p) - FLAT_MARGIN, 0.0)
            l += FLAT_PENALTY_W * over**2
        else:
            diff = np.abs(np.abs(p) - np.abs(t))
            l += EDGE_MAG_WEIGHT * np.where(diff < 1.0, 0.5 * diff**2, diff - 0.5)
        return l

    return (total_loss(pred + eps, target) - total_loss(pred - eps, target)) / (2 * eps)


# ═══════════════════════════════════════════════════════════════════════════════
# Test scenarios
# ═══════════════════════════════════════════════════════════════════════════════

# Scenario A: target = 0 (flat bars), pred from 0.0 to 0.6
preds_flat = [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60]
target_flat = 0.0

# Scenario B: target = 0.5 (positive edge), pred from 0.0 to 1.0
preds_edge_pos = [0.00, 0.03, 0.10, 0.20, 0.30, 0.40, 0.50, 0.53, 0.60, 0.70, 0.80, 0.90, 1.00]
target_edge_pos = 0.5

# Scenario C: target = -0.5 (negative edge), pred from -1.0 to 0.0
preds_edge_neg = [-1.00, -0.90, -0.80, -0.70, -0.60, -0.53, -0.50, -0.40, -0.30, -0.20, -0.10, -0.03, 0.00]
target_edge_neg = -0.5

# Scenario D: target = 0.1 (weak edge), pred from -0.2 to 0.4
preds_weak = [-0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
target_weak = 0.1

# Scenario E: target = 0.8 (strong edge), pred from 0.0 to 1.2
preds_strong = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20]
target_strong = 0.8


# ═══════════════════════════════════════════════════════════════════════════════
# Run simulations
# ═══════════════════════════════════════════════════════════════════════════════

def run_scenario(preds, target, label):
    """Compute gradients and losses for all (pred, target) pairs."""
    results = []
    for pred in preds:
        g = compute_all_grads(pred, target)
        l = compute_all_losses(pred, target)
        ng = numerical_grad(pred, target)
        results.append({
            "pred": pred,
            "target": target,
            **g,
            **l,
            "g_numerical": ng,
        })
    return results


scenarios = {
    f"FLAT (target={target_flat})": run_scenario(preds_flat, target_flat, "flat"),
    f"EDGE+ (target={target_edge_pos})": run_scenario(preds_edge_pos, target_edge_pos, "edge_pos"),
    f"EDGE- (target={target_edge_neg})": run_scenario(preds_edge_neg, target_edge_neg, "edge_neg"),
    f"WEAK EDGE (target={target_weak})": run_scenario(preds_weak, target_weak, "weak"),
    f"STRONG EDGE (target={target_strong})": run_scenario(preds_strong, target_strong, "strong"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Print detailed tables
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 120)
print("  GRADIENT DEEP SIMULATION — Loss v6")
print("=" * 120)

for scenario_name, results in scenarios.items():
    is_flat = results[0]["flat"]

    print(f"\n{'─' * 120}")
    print(f"  SCENARIO: {scenario_name}")
    print(f"{'─' * 120}")

    if is_flat:
        header = f"  {'pred':>7} │ {'L_dir':>8} {'L_fL1':>8} {'L_fQ':>8} {'L_total':>8} │ {'g_dir':>9} {'g_fL1':>9} {'g_fQ':>9} {'g_total':>9} │ {'g_num':>9} {'err':>8}"
        print(header)
        print(f"  {'─'*7}─┼─{'─'*8}─{'─'*8}─{'─'*8}─{'─'*8}─┼─{'─'*9}─{'─'*9}─{'─'*9}─{'─'*9}─┼─{'─'*9}─{'─'*8}")
    else:
        header = f"  {'pred':>7} │ {'L_dir':>8} {'L_edge':>8} {'L_total':>8} │ {'g_dir':>9} {'g_edge':>9} {'g_total':>9} │ {'g_num':>9} {'err':>8}"
        print(header)
        print(f"  {'─'*7}─┼─{'─'*8}─{'─'*8}─{'─'*8}─┼─{'─'*9}─{'─'*9}─{'─'*9}─┼─{'─'*9}─{'─'*8}")

    for r in results:
        pred = r["pred"]
        err = abs(r["g_total"] - r["g_numerical"])

        if is_flat:
            print(f"  {pred:+.2f}   │ {r['l_dir']:8.4f} {r['l_flat1']:8.4f} {r['l_flat2']:8.4f} {r['l_total']:8.4f} │ "
                  f"{r['g_dir']:+9.4f} {r['g_flat1']:+9.4f} {r['g_flat2']:+9.4f} {r['g_total']:+9.4f} │ "
                  f"{r['g_numerical']:+9.4f} {err:.2e}")
        else:
            print(f"  {pred:+.2f}   │ {r['l_dir']:8.4f} {r['l_edge']:8.4f} {r['l_total']:8.4f} │ "
                  f"{r['g_dir']:+9.4f} {r['g_edge']:+9.4f} {r['g_total']:+9.4f} │ "
                  f"{r['g_numerical']:+9.4f} {err:.2e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Gradient balance analysis
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'=' * 120}")
print("  GRADIENT BALANCE ANALYSIS")
print(f"{'=' * 120}")

print("""
  Key questions:
  1. FLAT BARS: Is the total gradient strong enough to push pred→0 at all pred values?
     - At pred=0.01, is the gradient comparable to edge-bar gradients?
     - Does the L1 term provide the constant push it's designed for?
  2. EDGE BARS: Is the gradient balanced between direction and magnitude?
     - At pred=target, is the gradient zero (perfect equilibrium)?
     - For undershoot vs overshoot, is the alpha/beta asymmetry working?
  3. CROSS-REGIME: Are flat-bar gradients and edge-bar gradients in the same
     ballpark, so the encoder doesn't over-optimize one at the expense of the other?
""")

# Compare gradient magnitudes at key points
print("  ┌─────────────────────────────────────────────────────────────────────────────────┐")
print("  │ CROSS-REGIME GRADIENT COMPARISON                                                │")
print("  ├─────────────────────────────────────────────────────────────────────────────────┤")

# Flat bar at various preds
for pred_val in [0.01, 0.03, 0.10, 0.30, 0.50]:
    r = compute_all_grads(pred_val, 0.0)
    print(f"  │ FLAT  pred={pred_val:+.2f} target=0.00  →  g_total={r['g_total']:+.4f}  "
          f"(dir={r['g_dir']:+.4f}  L1={r['g_flat1']:+.4f}  Q={r['g_flat2']:+.4f})")

print("  │")

# Edge bar at target
for tval in [0.1, 0.3, 0.5, 0.8]:
    r = compute_all_grads(tval, tval)
    print(f"  │ EDGE  pred={tval:+.2f} target={tval:+.2f}  →  g_total={r['g_total']:+.4f}  "
          f"(dir={r['g_dir']:+.4f}  edge={r['g_edge']:+.4f})  [should be ~0]")

print("  │")

# Edge bar undershoot (pred=0, target>0)
for tval in [0.1, 0.3, 0.5, 0.8]:
    r = compute_all_grads(0.0, tval)
    print(f"  │ UNDR  pred=+0.00 target={tval:+.2f}  →  g_total={r['g_total']:+.4f}  "
          f"(dir={r['g_dir']:+.4f}  edge={r['g_edge']:+.4f})")

print("  │")

# Edge bar overshoot (pred > target)
for tval in [0.3, 0.5]:
    pval = tval + 0.2
    r = compute_all_grads(pval, tval)
    print(f"  │ OVER  pred={pval:+.2f} target={tval:+.2f}  →  g_total={r['g_total']:+.4f}  "
          f"(dir={r['g_dir']:+.4f}  edge={r['g_edge']:+.4f})")

print("  └─────────────────────────────────────────────────────────────────────────────────┘")


# ═══════════════════════════════════════════════════════════════════════════════
# Equilibrium analysis: where is g_total = 0?
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'=' * 120}")
print("  EQUILIBRIUM ANALYSIS — Where does g_total = 0?")
print(f"{'=' * 120}")

# For flat bars: g_dir = 2*beta*pred*0.7 (since target=0, gap=pred, undershoot=False because gap*target=0)
# At target=0: is_undershoot = (pred*0)<0 = False → scale = beta = 1.5
# mag_weight = 0^0.5 clamped to 0.7
# g_dir = 2 * 1.5 * pred * 0.7 = 2.1 * pred
# g_flat1 = 0.2 * sign(pred)
# g_flat2 = 2 * 0.5 * max(0, |pred|-0.08) * sign(pred) = max(0, |pred|-0.08) * sign(pred)
# For pred > 0.08: g_total = 2.1*pred + 0.2 + (pred-0.08) = 3.1*pred + 0.12 — ALWAYS POSITIVE
# For 0 < pred < 0.08: g_total = 2.1*pred + 0.2 — ALWAYS POSITIVE (min 0.2 at pred→0)
# For pred < 0: g_total = 2.1*pred - 0.2 + ... — can be negative

print("\n  FLAT BAR (target=0) equilibrium:")
print("    For pred > 0: g_dir = 2*β*pred*0.7 = 2.1*pred (positive, pushes pred UP)")
print("                  g_flat1 = +0.2 (positive, pushes pred UP ← WRONG DIRECTION!)")
print("                  g_flat2 = max(0, pred-0.08) (positive above margin)")
print("    → For pred > 0: ALL components push POSITIVE. No equilibrium at pred=0!")
print("    → The L1 gradient has the WRONG SIGN for flat bars.")
print()
print("    WAIT — let's re-examine. L_flat1 = λ*|pred|.")
print("    d/dpred |pred| = sign(pred). For pred > 0: sign = +1.")
print("    So g_flat1 = +λ. This pushes pred UP, away from zero!")
print()
print("    For pred < 0: sign = -1, g_flat1 = -λ. This also pushes pred DOWN, away from zero!")
print("    → The L1 penalty on |pred| has gradient AWAY from zero on both sides!")
print("    → This is the OPPOSITE of what we want!")
print()
print("    CORRECTION NEEDED: The flat penalty should be L = λ * |pred|,")
print("    but d/dpred|pred| = sign(pred), which for pred>0 is +1.")
print("    This means the gradient pushes pred to INCREASE, not decrease!")
print()
print("    Actually wait - reconsider. The loss is λ*|pred|.")
print("    For pred > 0: d/dpred(λ*pred) = λ > 0. Gradient descent: pred -= lr * λ.")
print("    So pred DECREASES. The gradient is positive, but gradient DESCENT subtracts it.")
print("    → Gradient descent: pred_new = pred - lr * g_total")
print("    → If g_total > 0: pred decreases → pushes toward zero ✓")
print("    → If g_total < 0: pred increases → pushes toward zero ✓")
print()
print("    So: positive gradient → descent pushes pred down → toward zero ✓")
print("    The L1 term IS working correctly for gradient descent!")

print("\n  Let's verify with actual gradient descent steps:")
print("  ┌──────────────────────────────────────────────────────────────────────┐")

for start_pred in [0.30, 0.10, 0.03]:
    pred = start_pred
    lr = 0.1
    print(f"  │ Starting pred={start_pred:.2f}, target=0.0, lr={lr}:")
    for step in range(20):
        g = compute_all_grads(pred, 0.0)
        if abs(g["g_total"]) < 1e-6:
            print(f"  │   Step {step:2d}: pred={pred:.6f}  g_total={g['g_total']:+.6f}  → CONVERGED")
            break
        pred = pred - lr * g["g_total"]
        pred = max(pred, 0.0)  # clamp to non-negative for display
        if step < 5 or step % 5 == 0:
            print(f"  │   Step {step:2d}: pred={pred:.6f}  g_total={g['g_total']:+.6f}")
    print(f"  │   Final: pred={pred:.6f}")
    print(f"  │")

print("  └──────────────────────────────────────────────────────────────────────┘")


# ═══════════════════════════════════════════════════════════════════════════════
# Edge bar equilibrium
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n  EDGE BAR equilibrium (where g_total should be zero):")
print("  ┌──────────────────────────────────────────────────────────────────────┐")

for tval in [0.1, 0.3, 0.5, 0.8]:
    # At pred = target: gap = 0, so g_dir = 0
    # g_edge: diff = |pred| - |target| = tval - tval = 0, so g_edge = 0
    r = compute_all_grads(tval, tval)
    print(f"  │ pred={tval:.2f}, target={tval:.2f}: g_dir={r['g_dir']:+.6f}, g_edge={r['g_edge']:+.6f}, g_total={r['g_total']:+.6f}")

    # Also check pred slightly below and above target
    for delta in [-0.05, 0.05]:
        p = tval + delta
        r = compute_all_grads(p, tval)
        direction = "undershoot" if ((p - tval) * tval) < 0 else "overshoot"
        print(f"  │ pred={p:+.2f}, target={tval:.2f} ({direction}): g_total={r['g_total']:+.6f}")

print("  └──────────────────────────────────────────────────────────────────────┘")


# ═══════════════════════════════════════════════════════════════════════════════
# Comprehensive visualization
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'=' * 120}")
print("  GENERATING VISUALIZATION...")
print(f"{'=' * 120}")

# Create a fine grid for smooth curves
pred_fine_flat = np.linspace(0.0, 0.6, 300)
pred_fine_pos = np.linspace(-0.2, 1.2, 400)
pred_fine_neg = np.linspace(-1.2, 0.2, 400)

def compute_curve(preds, target):
    """Compute gradient curves over a fine grid."""
    g_dir = np.array([grad_direction(p, target) for p in preds])
    g_total_num = np.array([numerical_grad(p, target) for p in preds])

    eps = 1e-8
    is_flat = abs(target) < eps

    if is_flat:
        g_flat1 = np.array([grad_flat_l1(p) for p in preds])
        g_flat2 = np.array([grad_flat_quad(p) for p in preds])
        g_edge = np.zeros_like(preds)
        g_total = g_dir + g_flat1 + g_flat2
    else:
        g_flat1 = np.zeros_like(preds)
        g_flat2 = np.zeros_like(preds)
        g_edge = np.array([grad_edge_mag(p, target) for p in preds])
        g_total = g_dir + g_edge

    l_dir = np.array([compute_all_losses(p, target)["l_dir"] for p in preds])
    l_total = np.array([compute_all_losses(p, target)["l_total"] for p in preds])

    return {
        "preds": preds, "g_dir": g_dir, "g_flat1": g_flat1,
        "g_flat2": g_flat2, "g_edge": g_edge, "g_total": g_total,
        "g_total_num": g_total_num, "l_dir": l_dir, "l_total": l_total,
        "is_flat": is_flat, "target": target,
    }


# Compute curves
curve_flat = compute_curve(pred_fine_flat, 0.0)
curve_pos = compute_curve(pred_fine_pos, 0.5)
curve_neg = compute_curve(pred_fine_neg, -0.5)
curve_weak = compute_curve(np.linspace(-0.3, 0.5, 400), 0.1)
curve_strong = compute_curve(np.linspace(-0.2, 1.4, 400), 0.8)

# ── Figure 1: Flat bar gradient breakdown ───────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("Gradient Deep Simulation — Loss v6\n", fontsize=16, fontweight="bold")

# Panel 1a: Flat bar — gradient components
ax = axes[0, 0]
ax.set_title("FLAT BAR (target=0.0) — Gradient Components", fontsize=13, fontweight="bold")
ax.plot(curve_flat["preds"], curve_flat["g_dir"], "b-", linewidth=2, label="g_dir (direction)")
ax.plot(curve_flat["preds"], curve_flat["g_flat1"], "r-", linewidth=2, label="g_flat1 (L1)")
ax.plot(curve_flat["preds"], curve_flat["g_flat2"], "g-", linewidth=2, label="g_flat2 (quad)")
ax.plot(curve_flat["preds"], curve_flat["g_total"], "k-", linewidth=2.5, label="g_total (analytical)")
ax.plot(curve_flat["preds"], curve_flat["g_total_num"], "k--", linewidth=1, alpha=0.5, label="g_total (numerical)")
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0.08, color="green", linewidth=0.8, linestyle=":", alpha=0.7, label="margin=0.08")
ax.set_xlabel("Prediction")
ax.set_ylabel("Gradient (dL/dpred)")
ax.legend(fontsize=9, loc="upper left")
ax.set_xlim(0, 0.6)
ax.grid(True, alpha=0.3)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

# Panel 1b: Flat bar — loss components
ax = axes[0, 1]
ax.set_title("FLAT BAR (target=0.0) — Loss Components", fontsize=13, fontweight="bold")
l_flat1 = FLAT_L1_WEIGHT * pred_fine_flat
over = np.maximum(pred_fine_flat - FLAT_MARGIN, 0.0)
l_flat2 = FLAT_PENALTY_W * over**2
l_dir = BETA * pred_fine_flat**2 * 0.7  # mag_weight = 0.7 for target=0
ax.plot(pred_fine_flat, l_dir, "b-", linewidth=2, label="L_dir")
ax.plot(pred_fine_flat, l_flat1, "r-", linewidth=2, label="L_flat1 (L1)")
ax.plot(pred_fine_flat, l_flat2, "g-", linewidth=2, label="L_flat2 (quad)")
ax.plot(pred_fine_flat, l_dir + l_flat1 + l_flat2, "k-", linewidth=2.5, label="L_total")
ax.axvline(0.08, color="green", linewidth=0.8, linestyle=":", alpha=0.7, label="margin=0.08")
ax.set_xlabel("Prediction")
ax.set_ylabel("Loss")
ax.legend(fontsize=9, loc="upper left")
ax.set_xlim(0, 0.6)
ax.grid(True, alpha=0.3)

# Panel 1c: Positive edge — gradient components
ax = axes[1, 0]
ax.set_title("EDGE BAR (target=+0.5) — Gradient Components", fontsize=13, fontweight="bold")
ax.plot(curve_pos["preds"], curve_pos["g_dir"], "b-", linewidth=2, label="g_dir (direction)")
ax.plot(curve_pos["preds"], curve_pos["g_edge"], "r-", linewidth=2, label="g_edge (magnitude)")
ax.plot(curve_pos["preds"], curve_pos["g_total"], "k-", linewidth=2.5, label="g_total (analytical)")
ax.plot(curve_pos["preds"], curve_pos["g_total_num"], "k--", linewidth=1, alpha=0.5, label="g_total (numerical)")
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0.5, color="blue", linewidth=0.8, linestyle=":", alpha=0.7, label="target=0.5")
ax.axvline(0.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.5, label="pred=0 (undershoot)")
ax.set_xlabel("Prediction")
ax.set_ylabel("Gradient (dL/dpred)")
ax.legend(fontsize=9, loc="upper left")
ax.set_xlim(-0.2, 1.2)
ax.grid(True, alpha=0.3)

# Panel 1d: Negative edge — gradient components
ax = axes[1, 1]
ax.set_title("EDGE BAR (target=-0.5) — Gradient Components", fontsize=13, fontweight="bold")
ax.plot(curve_neg["preds"], curve_neg["g_dir"], "b-", linewidth=2, label="g_dir (direction)")
ax.plot(curve_neg["preds"], curve_neg["g_edge"], "r-", linewidth=2, label="g_edge (magnitude)")
ax.plot(curve_neg["preds"], curve_neg["g_total"], "k-", linewidth=2.5, label="g_total (analytical)")
ax.plot(curve_neg["preds"], curve_neg["g_total_num"], "k--", linewidth=1, alpha=0.5, label="g_total (numerical)")
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(-0.5, color="blue", linewidth=0.8, linestyle=":", alpha=0.7, label="target=-0.5")
ax.axvline(0.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.5, label="pred=0 (undershoot)")
ax.set_xlabel("Prediction")
ax.set_ylabel("Gradient (dL/dpred)")
ax.legend(fontsize=9, loc="upper left")
ax.set_xlim(-1.2, 0.2)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("gradient_simulation_1.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved gradient_simulation_1.png")


# ── Figure 2: Cross-regime comparison ────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("Cross-Regime Gradient Comparison — Loss v6\n", fontsize=16, fontweight="bold")

# Panel 2a: Total gradient comparison (all regimes)
ax = axes[0, 0]
ax.set_title("Total Gradient Across Regimes", fontsize=13, fontweight="bold")
ax.plot(curve_flat["preds"], curve_flat["g_total"], "k-", linewidth=2.5, label="Flat (tgt=0)")
ax.plot(curve_weak["preds"], curve_weak["g_total"], "g-", linewidth=2, label="Weak edge (tgt=0.1)")
ax.plot(curve_pos["preds"], curve_pos["g_total"], "b-", linewidth=2, label="Edge (tgt=0.5)")
ax.plot(curve_strong["preds"], curve_strong["g_total"], "r-", linewidth=2, label="Strong edge (tgt=0.8)")
ax.axhline(0, color="gray", linewidth=0.5)
ax.set_xlabel("Prediction")
ax.set_ylabel("Total Gradient (dL/dpred)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2b: Gradient at pred=0 for different targets
ax = axes[0, 1]
ax.set_title("Gradient at pred=0 vs Target Value", fontsize=13, fontweight="bold")
targets_sweep = np.linspace(0.0, 1.0, 200)
g_at_zero = [compute_all_grads(0.0, t)["g_total"] for t in targets_sweep]
g_dir_at_zero = [compute_all_grads(0.0, t)["g_dir"] for t in targets_sweep]
g_edge_at_zero = [compute_all_grads(0.0, t)["g_edge"] for t in targets_sweep]
ax.plot(targets_sweep, g_at_zero, "k-", linewidth=2.5, label="g_total")
ax.plot(targets_sweep, g_dir_at_zero, "b--", linewidth=1.5, label="g_dir")
ax.plot(targets_sweep, g_edge_at_zero, "r--", linewidth=1.5, label="g_edge")
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0.0, color="gray", linewidth=0.5)
ax.set_xlabel("Target Value")
ax.set_ylabel("Gradient at pred=0")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2c: Gradient magnitude heatmap (pred vs target)
ax = axes[1, 0]
ax.set_title("Total Gradient Magnitude |g_total| — Heatmap", fontsize=13, fontweight="bold")
pred_grid = np.linspace(-0.8, 1.0, 200)
tgt_grid = np.linspace(0.0, 0.8, 200)
P, T = np.meshgrid(pred_grid, tgt_grid)
G = np.zeros_like(P)
for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        G[i, j] = abs(compute_all_grads(P[i, j], T[i, j])["g_total"])

im = ax.pcolormesh(P, T, G, cmap="hot_r", shading="auto", vmin=0, vmax=3.0)
plt.colorbar(im, ax=ax, label="|g_total|")
ax.set_xlabel("Prediction")
ax.set_ylabel("Target")
ax.plot([0, 1], [0, 1], "c--", linewidth=1, label="pred=target")
ax.plot([0, -0.8], [0, 0.8], "c--", linewidth=1)  # pred = -target line
ax.legend(fontsize=9, loc="upper left")

# Panel 2d: Gradient descent trajectory from various starting points
ax = axes[1, 1]
ax.set_title("Gradient Descent Trajectories (target=0.5)", fontsize=13, fontweight="bold")
lr = 0.05
for start, color, ls in [(0.0, "red", "-"), (0.2, "orange", "--"), (0.8, "green", "-."), (1.0, "blue", ":")]:
    pred = start
    trajectory = [pred]
    for _ in range(50):
        g = compute_all_grads(pred, 0.5)
        pred = pred - lr * g["g_total"]
        trajectory.append(pred)
    ax.plot(trajectory, color=color, linestyle=ls, linewidth=2, label=f"start={start:.1f}")
ax.axhline(0.5, color="black", linewidth=1, linestyle=":", label="target=0.5")
ax.set_xlabel("GD Step")
ax.set_ylabel("Prediction")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("gradient_simulation_2.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved gradient_simulation_2.png")


# ── Figure 3: Detailed gradient balance at key points ────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle("Gradient Balance at Critical Points — Loss v6\n", fontsize=16, fontweight="bold")

# Panel 3a: Bar chart of gradient components at key flat-bar preds
ax = axes[0]
ax.set_title("Flat Bar: Gradient Components", fontsize=13, fontweight="bold")
key_preds = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]
x = np.arange(len(key_preds))
width = 0.25
g_dirs = [compute_all_grads(p, 0.0)["g_dir"] for p in key_preds]
g_l1s = [compute_all_grads(p, 0.0)["g_flat1"] for p in key_preds]
g_quads = [compute_all_grads(p, 0.0)["g_flat2"] for p in key_preds]
g_totals = [compute_all_grads(p, 0.0)["g_total"] for p in key_preds]

bars1 = ax.bar(x - width, g_dirs, width, label="g_dir", color="steelblue")
bars2 = ax.bar(x, g_l1s, width, label="g_flat1 (L1)", color="coral")
bars3 = ax.bar(x + width, g_quads, width, label="g_flat2 (quad)", color="mediumseagreen")
ax.plot(x, g_totals, "ko-", linewidth=2, markersize=6, label="g_total")
ax.axhline(0, color="gray", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f"{p:.2f}" for p in key_preds], rotation=45)
ax.set_xlabel("Prediction (target=0)")
ax.set_ylabel("Gradient")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# Panel 3b: Bar chart of gradient components at key edge-bar preds
ax = axes[1]
ax.set_title("Edge Bar (tgt=0.5): Gradient Components", fontsize=13, fontweight="bold")
key_preds_edge = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
x = np.arange(len(key_preds_edge))
g_dirs_e = [compute_all_grads(p, 0.5)["g_dir"] for p in key_preds_edge]
g_edges = [compute_all_grads(p, 0.5)["g_edge"] for p in key_preds_edge]
g_totals_e = [compute_all_grads(p, 0.5)["g_total"] for p in key_preds_edge]

bars1 = ax.bar(x - 0.2, g_dirs_e, 0.4, label="g_dir", color="steelblue")
bars2 = ax.bar(x + 0.2, g_edges, 0.4, label="g_edge", color="coral")
ax.plot(x, g_totals_e, "ko-", linewidth=2, markersize=6, label="g_total")
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(4.0, color="blue", linewidth=0.8, linestyle=":", alpha=0.7)  # target=0.5
ax.set_xticks(x)
ax.set_xticklabels([f"{p:.2f}" for p in key_preds_edge], rotation=45)
ax.set_xlabel("Prediction (target=0.5)")
ax.set_ylabel("Gradient")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# Panel 3c: Gradient ratio analysis — flat vs edge
ax = axes[2]
ax.set_title("Gradient Ratio: Flat vs Edge at Same |pred|", fontsize=13, fontweight="bold")
common_preds = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
g_flat_vals = [abs(compute_all_grads(p, 0.0)["g_total"]) for p in common_preds]
g_edge_vals = [abs(compute_all_grads(p, 0.5)["g_total"]) for p in common_preds]
ratios = [f / (e + 1e-9) for f, e in zip(g_flat_vals, g_edge_vals)]

ax.bar(range(len(common_preds)), ratios, color="mediumpurple", alpha=0.8)
ax.axhline(1.0, color="red", linewidth=1.5, linestyle="--", label="ratio=1 (balanced)")
ax.set_xticks(range(len(common_preds)))
ax.set_xticklabels([f"{p:.2f}" for p in common_preds], rotation=45)
ax.set_xlabel("|pred|")
ax.set_ylabel("|g_flat| / |g_edge|")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
ax.set_yscale("log")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("gradient_simulation_3.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved gradient_simulation_3.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary statistics
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n\n{'=' * 120}")
print("  SUMMARY STATISTICS")
print(f"{'=' * 120}")

print("""
  ┌─────────────────────────────────────────────────────────────────────────────────────┐
  │                        GRADIENT MAGNITUDE SUMMARY                                   │
  ├─────────────────────────────────────────────────────────────────────────────────────┤""")

for scenario_name, results in scenarios.items():
    g_totals = [abs(r["g_total"]) for r in results]
    g_dirs = [abs(r["g_dir"]) for r in results]
    print(f"  │ {scenario_name:<30s} │ g_total: [{min(g_totals):.4f} .. {max(g_totals):.4f}]  "
          f"│ g_dir: [{min(g_dirs):.4f} .. {max(g_dirs):.4f}] │")

print(f"  └─────────────────────────────────────────────────────────────────────────────────────┘")

# Verify numerical gradient matches analytical
print(f"\n  Numerical gradient verification (max error across all test points):")
max_err = 0
for scenario_name, results in scenarios.items():
    for r in results:
        err = abs(r["g_total"] - r["g_numerical"])
        max_err = max(max_err, err)
print(f"    Max analytical vs numerical gradient error: {max_err:.2e}")
print(f"    {'✓ PASS' if max_err < 1e-6 else '✗ FAIL'} (threshold: 1e-6)")

print(f"\n  Simulation complete. 3 plots saved.")
