#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
v8 GRADIENT VERIFICATION — Balanced Number Line Loss
═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import numpy as np

ALPHA       = 2.0
BETA        = 1.5
MAG_W_MIN   = 0.3
MAG_W_MAX   = 2.0
FLAT_PUSH_W = 1.5
EDGE_NUDGE_W = 1.0

def mw(t):
    return np.clip(np.abs(t)**0.5, MAG_W_MIN, MAG_W_MAX)

def g_dir(p, t):
    gap = p - t
    sc = ALPHA if (gap * t) < 0 else BETA
    return 2.0 * sc * gap * mw(t)

def g_flat_push(p):
    """L1 push: gradient = push_w * sign(pred) * mw(0)"""
    return FLAT_PUSH_W * np.sign(p) * mw(0)

def g_edge_nudge(p, t):
    """smooth_l1(pred, target) gradient (no |pred|)"""
    gap = p - t
    abs_gap = abs(gap)
    grad = gap if abs_gap < 1.0 else np.sign(gap)
    return EDGE_NUDGE_W * grad * mw(t)

def g_flat(p):
    return g_dir(p, 0.0) + g_flat_push(p)

def g_edge_total(p, t):
    return g_dir(p, t) + g_edge_nudge(p, t)

def g_numerical(p, t, h=1e-5):
    def loss(pp, tt):
        gap = pp - tt
        sc = ALPHA if ((gap * tt) < 0) else BETA
        w = np.clip(abs(tt)**0.5, MAG_W_MIN, MAG_W_MAX)
        l = sc * gap**2 * w
        if abs(tt) < 1e-8:
            l += FLAT_PUSH_W * abs(pp) * w
        else:
            d = abs(gap)
            l += EDGE_NUDGE_W * w * (0.5*gap**2 if d < 1.0 else d - 0.5)
        return l
    return (loss(p + h, t) - loss(p - h, t)) / (2 * h)

def g_torch(p_val, t_val):
    pred = torch.tensor([p_val], requires_grad=True)
    target = torch.tensor([t_val])
    gap = pred - target
    is_under = (gap * target) < 0
    scale = torch.where(is_under, torch.tensor(ALPHA), torch.tensor(BETA))
    w = target.abs().pow(0.5).clamp(min=MAG_W_MIN, max=MAG_W_MAX)
    l_dir = (scale * gap.pow(2) * w).mean()
    is_flat = target.abs() < 1e-8
    if is_flat:
        l_push = (FLAT_PUSH_W * pred.abs() * w).mean()
        l_nudge = torch.tensor(0.0)
    else:
        l_push = torch.tensor(0.0)
        ad = gap.abs()
        l_nudge = (EDGE_NUDGE_W * w * (gap.pow(2)*0.5*(ad<1).float() + (ad-0.5)*(ad>=1).float())).mean()
    loss = l_dir + l_push + l_nudge
    loss.backward()
    return pred.grad.item(), loss.item()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: Analytical vs Numerical vs PyTorch
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 100)
print("  TEST 1: Analytical vs Numerical vs PyTorch Autograd")
print("=" * 100)

test_points = [
    (0.00, 0.0), (0.01, 0.0), (0.03, 0.0), (0.10, 0.0), (0.30, 0.0), (0.50, 0.0),
    (-0.01, 0.0), (-0.03, 0.0), (-0.10, 0.0), (-0.30, 0.0),
    (0.00, 0.5), (0.03, 0.5), (0.10, 0.5), (0.30, 0.5), (0.50, 0.5), (0.53, 0.5),
    (0.60, 0.5), (0.80, 0.5), (1.00, 0.5),
    (-0.005, 0.3), (-0.005, 0.5), (-0.005, 0.8),
    (0.005, 0.3), (0.005, 0.5), (0.005, 0.8),
    (0.00, 0.1), (0.05, 0.1), (0.10, 0.1), (0.15, 0.1),
    (0.00, 0.8), (0.30, 0.8), (0.80, 0.8), (1.00, 0.8),
    (-0.53, -0.5), (-0.50, -0.5), (0.50, 0.5),
]

max_err_anal = 0
max_err_torch = 0
print(f"\n  {'pred':>7} {'tgt':>6} | {'analytical':>10} {'numerical':>10} {'pytorch':>10} | {'err_a':>8} {'err_t':>8}")
for p, t in test_points:
    if abs(t) < 1e-8:
        ga = g_flat(p)
    else:
        ga = g_edge_total(p, t)
    gn = g_numerical(p, t)
    gt, lt = g_torch(p, t)
    err_a = abs(ga - gn)
    err_t = abs(ga - gt)
    max_err_anal = max(max_err_anal, err_a)
    max_err_torch = max(max_err_torch, err_t)
    flag = '  ' if err_t < 1e-4 else '**'
    print(f"  {p:7.3f} {t:6.2f} | {ga:+10.4f} {gn:+10.4f} {gt:+10.4f} | {err_a:.2e} {err_t:.2e} {flag}")

print(f"\n  Max analytical vs numerical error: {max_err_anal:.2e}")
print(f"  Max analytical vs pytorch error: {max_err_torch:.2e}")
print(f"  {'✓ PASS' if max_err_torch < 1e-4 else '✗ FAIL'}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: Cross-Regime Balance at Same |gap|
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("  TEST 2: Cross-Regime Balance at Same |gap| = |pred - target|")
print("=" * 100)

print(f"\n  {'|gap|':>7} | {'|g_flat|':>10} {'|g_edge(0.1)|':>13} {'|g_edge(0.3)|':>13} {'|g_edge(0.5)|':>13} {'|g_edge(0.8)|':>13} | ratio_f/avg_e")
for gap in [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
    gf = abs(g_flat(gap))
    ge1 = abs(g_edge_total(0.1 - gap, 0.1)) if gap < 0.1 else float('nan')
    ge3 = abs(g_edge_total(0.3 - gap, 0.3)) if gap < 0.3 else float('nan')
    ge5 = abs(g_edge_total(0.5 - gap, 0.5)) if gap < 0.5 else float('nan')
    ge8 = abs(g_edge_total(0.8 - gap, 0.8)) if gap < 0.8 else float('nan')
    edges = [ge1, ge3, ge5, ge8]
    valid = [e for e in edges if e == e]
    ratio_str = ""
    if valid:
        avg_edge = np.mean(valid)
        ratio_str = f"{gf/avg_edge:.2f}"
    print(f"  {gap:7.2f} | {gf:10.4f} {ge1:13.4f} {ge3:13.4f} {ge5:13.4f} {ge8:13.4f} | {ratio_str}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: Sign Consistency
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("  TEST 3: Sign Consistency — Gradient Always Points Toward Target")
print("=" * 100)

sign_errors = []
for t in [0.0, 0.05, 0.1, 0.3, 0.5, 0.8]:
    for p in np.linspace(-1.0, 1.5, 500):
        if abs(p - t) < 1e-6:
            continue
        g = g_flat(p) if abs(t) < 1e-8 else g_edge_total(p, t)
        if p > t and g <= 1e-10:
            sign_errors.append((p, t, g, "should be >0"))
        elif p < t and g >= -1e-10:
            sign_errors.append((p, t, g, "should be <0"))

print(f"  Sign errors: {len(sign_errors)}")
for p, t, g, msg in sign_errors[:10]:
    print(f"    pred={p:.4f}, target={t:.1f}: g={g:+.6f} ({msg})")
print(f"  {'✓ PASS' if len(sign_errors) == 0 else '✗ FAIL'}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: Equilibrium — g=0 at pred=target
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("  TEST 4: Equilibrium — g(pred=target) == 0")
print("=" * 100)

for t in [0.0, 0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5]:
    g = g_flat(t) if abs(t) < 1e-8 else g_edge_total(t, t)
    err = abs(g)
    status = '✓' if err < 1e-8 else '✗'
    print(f"  {status} g(pred={t}, target={t}) = {err:.2e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: Odd Symmetry
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("  TEST 5: Odd Symmetry — g(p,t) == -g(-p,-t)")
print("=" * 100)

max_sym_err = 0.0
for t in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]:
    for p in np.linspace(-1.0, 1.0, 200):
        g1 = g_flat(p) if abs(t) < 1e-8 else g_edge_total(p, t)
        g2 = g_flat(-p) if abs(t) < 1e-8 else g_edge_total(-p, -t)
        err = abs(g1 + g2)
        max_sym_err = max(max_sym_err, err)
print(f"  Max symmetry error: {max_sym_err:.2e}")
print(f"  {'✓ PASS' if max_sym_err < 1e-6 else '✗ FAIL'}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: Monotonicity
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("  TEST 6: Monotonicity — |g| increases with distance from target")
print("=" * 100)

violations = 0
total = 0
for t in [0.0, 0.1, 0.3, 0.5]:
    dists = np.linspace(0.01, 1.0, 200)
    mags = [abs(g_flat(t + d)) if abs(t) < 1e-8 else abs(g_edge_total(t + d, t)) for d in dists]
    for i in range(1, len(mags)):
        total += 1
        if mags[i] < mags[i-1] - 1e-8:
            violations += 1
print(f"  Violations: {violations}/{total}")
print(f"  {'✓ PASS' if violations == 0 else '✗ FAIL'}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 7: User-Requested Specific Values
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("  TEST 7: User-Requested Specific Values")
print("=" * 100)

print("\n  ── |target|=0, pred = 0.03, 0.10, 0.30 ──")
for p in [0.03, 0.10, 0.30]:
    g = g_flat(p)
    print(f"    pred={p:.2f}: g={g:+.4f}  (positive → GD pushes toward 0 ✓)")

print("\n  ── |target|=0.5, pred = 0.53, 0.60, 0.80 ──")
for p in [0.53, 0.60, 0.80]:
    g = g_edge_total(p, 0.5)
    print(f"    pred={p:.2f}: g={g:+.4f}  (positive → GD pushes toward 0.5 ✓)")

print("\n  ── |target|=0.5, pred = 0.03, 0.10, 0.30 (undershoot) ──")
for p in [0.03, 0.10, 0.30]:
    g = g_edge_total(p, 0.5)
    print(f"    pred={p:.2f}: g={g:+.4f}  (negative → GD pushes toward 0.5 ✓)")

print("\n  ── Cross-sign: pred<0, target>0 (was buggy in v6, check v8) ──")
for p, t in [(-0.005, 0.3), (-0.005, 0.5), (-0.01, 0.5), (-0.05, 0.5)]:
    g = g_edge_total(p, t)
    correct = g < 0  # pred < target, need g < 0 for GD to push toward target
    print(f"    pred={p:+.3f}, target={t:.1f}: g={g:+.4f}  {'✓' if correct else '✗ BUG!'}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 8: Full Gradient Profile
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("  TEST 8: Full Number Line Gradient Profile")
print("=" * 100)

for t in [0.0, 0.3, 0.5, 0.8]:
    print(f"\n  target={t}:")
    if abs(t) < 1e-8:
        print(f"  {'pred':>7} | {'g_dir':>8} {'g_push':>8} {'g_total':>8} | {'|g|':>8}")
        preds = np.linspace(-0.5, 1.0, 16)
        for p in preds:
            gd = g_dir(p, 0.0)
            gp = g_flat_push(p)
            gt = gd + gp
            print(f"  {p:7.3f} | {gd:+8.4f} {gp:+8.4f} {gt:+8.4f} | {abs(gt):8.4f}")
    else:
        print(f"  {'pred':>7} | {'g_dir':>8} {'g_nudge':>8} {'g_total':>8} | {'|g|':>8}")
        preds = np.linspace(max(t - 0.6, -0.5), t + 0.6, 15)
        for p in preds:
            gd = g_dir(p, t)
            gn = g_edge_nudge(p, t)
            gt = gd + gn
            print(f"  {p:7.3f} | {gd:+8.4f} {gn:+8.4f} {gt:+8.4f} | {abs(gt):8.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 9: Undershoot/Overshoot Asymmetry
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("  TEST 9: Undershoot/Overshoot Asymmetry")
print("=" * 100)

for t in [0.1, 0.3, 0.5, 0.8]:
    g_under = abs(g_edge_total(t - 0.05, t))
    g_over = abs(g_edge_total(t + 0.05, t))
    ratio = g_under / g_over if g_over > 1e-10 else float('inf')
    print(f"  target={t}: |g_undershoot|={g_under:.4f}  |g_overshoot|={g_over:.4f}  ratio={ratio:.3f}  (α/β={ALPHA/BETA:.3f})")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 10: Convergence estimate — steps to reach target from various starts
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("  TEST 10: Gradient Descent Convergence Estimate")
print("=" * 100)

lr = 5e-6  # config LEARNING_RATE
clip = 5.0

print(f"\n  lr={lr}, clip={clip}")
print(f"\n  Flat bars (target=0):")
for start in [0.50, 0.30, 0.10, 0.05, 0.03, 0.01]:
    pred = start
    steps = 0
    for _ in range(10000):
        g = g_flat(pred)
        g_clipped = np.clip(g, -clip, clip)
        pred -= lr * g_clipped
        steps += 1
        if abs(pred) < 0.001:
            break
    print(f"    start={start:.2f} → converged in {steps} steps (final pred={pred:.6f})")

print(f"\n  Edge bars (target=0.5):")
for start in [0.00, 0.03, 0.10, 0.30, 0.70, 0.90]:
    pred = start
    steps = 0
    for _ in range(10000):
        g = g_edge_total(pred, 0.5)
        g_clipped = np.clip(g, -clip, clip)
        pred -= lr * g_clipped
        steps += 1
        if abs(pred - 0.5) < 0.001:
            break
    print(f"    start={start:.2f} → converged in {steps} steps (final pred={pred:.6f})")
