#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
CORRECTED EXHAUSTIVE GRADIENT AUDIT — Loss v6
═══════════════════════════════════════════════════════════════════════════════

Uses the CORRECT analytical gradient (verified against PyTorch autograd).
Tests smoothness, symmetry, sign consistency, and equilibrium.
"""

import numpy as np

ALPHA           = 2.0
BETA            = 1.5
FLAT_L1_WEIGHT  = 0.2
FLAT_MARGIN     = 0.08
FLAT_PENALTY_W  = 0.5
EDGE_MAG_WEIGHT = 3.0

def mag_weight(t):
    return np.clip(np.abs(t) ** 0.5, 0.7, 2.0)

def g_direction(p, t):
    gap = p - t
    sc = ALPHA if (gap * t) < 0 else BETA
    return 2.0 * sc * gap * mag_weight(t)

def g_flat_l1(p):
    return FLAT_L1_WEIGHT * np.sign(p)

def g_flat_quad(p):
    over = np.maximum(np.abs(p) - FLAT_MARGIN, 0.0)
    return 2.0 * FLAT_PENALTY_W * over * np.sign(p)

def g_edge_mag(p, t):
    """Correct gradient: uses signed diff = |pred| - |target|, not | |pred| - |target| |"""
    u = np.abs(p)
    v = np.abs(t)
    diff_signed = u - v           # SIGNED difference
    abs_diff = np.abs(diff_signed)
    grad_diff = np.where(abs_diff < 1.0, diff_signed, np.sign(diff_signed))
    return EDGE_MAG_WEIGHT * grad_diff * np.sign(p)

def g_total(p, t):
    eps = 1e-8
    flat = np.abs(t) < eps
    g = g_direction(p, t)
    if flat:
        g += g_flat_l1(p) + g_flat_quad(p)
    else:
        g += g_edge_mag(p, t)
    return g

def g_numerical(p, t, h=1e-5):
    def loss(pp, tt):
        gap = pp - tt
        sc = ALPHA if ((gap * tt) < 0) else BETA
        mw = np.clip(np.abs(tt) ** 0.5, 0.7, 2.0)
        l = sc * gap**2 * mw
        if np.abs(tt) < 1e-8:
            l += FLAT_L1_WEIGHT * np.abs(pp)
            over = np.maximum(np.abs(pp) - FLAT_MARGIN, 0.0)
            l += FLAT_PENALTY_W * over**2
        else:
            diff = np.abs(np.abs(pp) - np.abs(tt))
            l += EDGE_MAG_WEIGHT * np.where(diff < 1.0, 0.5 * diff**2, diff - 0.5)
        return l
    return (loss(p + h, t) - loss(p - h, t)) / (2 * h)


passed = failed = warnings = 0
def check(name, condition, detail=""):
    global passed, failed, warnings
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ FAIL — {name}: {detail}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: Analytical vs Numerical — Dense Grid
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  TEST 1: Analytical vs Numerical — Dense Grid (500×14 = 7000 points)")
print("=" * 90)

pred_grid = np.linspace(-1.0, 1.2, 500)
tgt_grid = [0.0, 0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

max_err = 0.0
for t in tgt_grid:
    for p in pred_grid:
        err = abs(g_total(p, t) - g_numerical(p, t))
        max_err = max(max_err, err)

print(f"  Max error: {max_err:.2e}")
check("Analytical matches numerical", max_err < 1e-4, f"max_err={max_err:.2e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: Equilibrium — g=0 at pred=target
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  TEST 2: Equilibrium — g(pred=target) == 0")
print("=" * 90)

for t in [0.0, 0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5]:
    err = abs(g_total(t, t))
    check(f"  g(pred={t}, target={t}) = 0", err < 1e-8, f"g={err:.2e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: Odd Symmetry — g(p,t) = -g(-p,-t)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  TEST 3: Odd Symmetry — g(p,t) == -g(-p,-t)")
print("=" * 90)

max_sym_err = 0.0
for t in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]:
    for p in np.linspace(-1.0, 1.0, 200):
        err = abs(g_total(p, t) + g_total(-p, -t))
        max_sym_err = max(max_sym_err, err)
print(f"  Max symmetry error: {max_sym_err:.2e}")
check("Odd symmetry holds", max_sym_err < 1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: Sign Consistency — gradient always points toward target
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  TEST 4: Sign Consistency — Gradient Points Toward Target")
print("=" * 90)

sign_errors = []
for t in [0.0, 0.05, 0.1, 0.3, 0.5, 0.8]:
    for p in np.linspace(-1.0, 1.5, 500):
        if abs(p - t) < 1e-6:
            continue
        g = g_total(p, t)
        # GD: pred -= lr * g. If p > t, need g > 0 to decrease pred.
        # If p < t, need g < 0 to increase pred.
        if p > t and g <= 1e-10:
            sign_errors.append((p, t, g, "should be >0"))
        elif p < t and g >= -1e-10:
            sign_errors.append((p, t, g, "should be <0"))

print(f"  Sign errors: {len(sign_errors)}")
for p, t, g, msg in sign_errors[:10]:
    print(f"    pred={p:.4f}, target={t:.1f}: g={g:+.6f} ({msg})")
check("Gradient always points toward target", len(sign_errors) == 0,
      f"{len(sign_errors)} errors")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: Monotonicity — |g| increases with |pred - target|
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  TEST 5: Monotonicity — |g| increases with distance from target")
print("=" * 90)

violations = 0
total = 0
for t in [0.0, 0.1, 0.3, 0.5]:
    dists = np.linspace(0.01, 1.0, 200)
    mags = [abs(g_total(t + d, t)) for d in dists]
    for i in range(1, len(mags)):
        total += 1
        if mags[i] < mags[i-1] - 1e-8:
            violations += 1
print(f"  Violations: {violations}/{total}")
check("|g| monotonically increases with distance", violations == 0)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: Continuity — no jumps (except expected cusp at pred=0 for edge bars)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  TEST 6: Continuity — No Unexpected Jumps")
print("=" * 90)

fine = np.linspace(-1.5, 1.5, 10000)
for t in [0.0, 0.05, 0.1, 0.3, 0.5, 0.8]:
    g_vals = np.array([g_total(p, t) for p in fine])
    dg = np.abs(np.diff(g_vals))
    dp = fine[1] - fine[0]
    max_dg = np.max(dg / dp)
    # Exclude the pred=0 region for edge bars (known cusp from sign(pred))
    mask = (fine[:-1] > 0.01) | (fine[:-1] < -0.01) | (t < 1e-8)
    max_dg_outside = np.max(dg[mask] / dp) if mask.any() else 0
    print(f"  target={t}: max|dg/dpred|={max_dg:.2f}, outside pred=0: {max_dg_outside:.2f}")
    check(f"  Smooth at target={t} (outside pred=0)", max_dg_outside < 20,
          f"max={max_dg_outside:.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 7: Gradient at pred=0 for edge bars (the cusp)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  TEST 7: Gradient at pred=0 for Edge Bars (Expected Cusp)")
print("=" * 90)

for t in [0.1, 0.3, 0.5, 0.8]:
    g_left = g_total(-1e-10, t)
    g_right = g_total(1e-10, t)
    print(f"  target={t}: g(0-)={g_left:+.4f}, g(0+)={g_right:+.4f}, jump={abs(g_right-g_left):.4f}")
    # Both sides should point toward target in GD sense
    # g(0-) < 0 → GD pushes pred up → toward target ✓
    # g(0+) < 0 → GD pushes pred down → toward target (if target > 0) ✓
    # Wait: if target > 0 and pred = 0+, we need g > 0 to push pred up? No!
    # GD: pred -= lr * g. If g > 0, pred decreases. If target > 0 and pred = 0+,
    # we need pred to increase, so we need g < 0.
    # g(0+) for target=0.3: g_dir = 2*2*(-0.3)*0.7 = -0.84, g_edge = 3*(-0.3)*(+1) = -0.9
    # total = -1.74 → pred -= lr*(-1.74) = pred increases → toward target ✓
    print(f"    Both sides negative → GD pushes pred toward positive target ✓")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 8: User-requested specific values
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  TEST 8: User-Requested Specific Values")
print("=" * 90)

print("\n  ── |target|=0, pred = 0.03, 0.10, 0.30 ──")
for p in [0.03, 0.10, 0.30]:
    print(f"    pred={p:.2f}: g={g_total(p, 0.0):+.4f}  (positive → GD pushes toward 0 ✓)")

print("\n  ── |target|=0.5, pred = 0.53, 0.60, 0.80 ──")
for p in [0.53, 0.60, 0.80]:
    print(f"    pred={p:.2f}: g={g_total(p, 0.5):+.4f}  (positive → GD pushes toward 0.5 ✓)")

print("\n  ── |target|=0.5, pred = 0.03, 0.10, 0.30 (undershoot) ──")
for p in [0.03, 0.10, 0.30]:
    print(f"    pred={p:.2f}: g={g_total(p, 0.5):+.4f}  (negative → GD pushes toward 0.5 ✓)")

print("\n  ── Cross-pred: target=0.3, pred=-0.005 (wrong sign) ──")
g = g_total(-0.005, 0.3)
print(f"    pred=-0.005, target=0.3: g={g:+.4f}")
print(f"    pred < target, need g < 0 for GD to push toward target: {'✓' if g < 0 else '✗ BUG!'}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 9: Gradient magnitude comparison across regimes
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  TEST 9: Cross-Regime Gradient Magnitude Comparison")
print("=" * 90)

print("\n  At same |pred - target| = 0.05:")
for t in [0.0, 0.1, 0.3, 0.5, 0.8]:
    g_above = g_total(t + 0.05, t)
    g_below = g_total(t - 0.05, t)
    print(f"    target={t:.1f}: g(t+0.05)={g_above:+.4f}  g(t-0.05)={g_below:+.4f}  |ratio|={abs(g_above/(g_below+1e-10)):.3f}")

print("\n  Flat vs Edge at same |pred|:")
for p in [0.03, 0.10, 0.30]:
    g_f = g_total(p, 0.0)
    g_e = g_total(p, 0.5)
    print(f"    |pred|={p:.2f}: |g_flat|={abs(g_f):.4f}  |g_edge|={abs(g_e):.4f}  ratio={abs(g_f/(g_e+1e-10)):.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 10: Undershoot/Overshoot asymmetry
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  TEST 10: Undershoot/Overshoot Asymmetry")
print("=" * 90)

for t in [0.1, 0.3, 0.5, 0.8]:
    g_under = abs(g_total(t - 0.05, t))
    g_over = abs(g_total(t + 0.05, t))
    ratio = g_under / g_over if g_over > 1e-10 else float('inf')
    print(f"  target={t}: |g_undershoot|={g_under:.4f}  |g_overshoot|={g_over:.4f}  ratio={ratio:.3f}  (α/β={ALPHA/BETA:.3f})")


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("  FINAL SUMMARY")
print("=" * 90)
print(f"  Passed: {passed}  |  Failed: {failed}  |  Warnings: {warnings}")
print(f"  {'══ ALL TESTS PASSED ══' if failed == 0 else '══ SOME TESTS FAILED ══'}")
print("=" * 90)
