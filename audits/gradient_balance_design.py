#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
GRADIENT IMBALANCE ANALYSIS — Precise measurements for redesign
═══════════════════════════════════════════════════════════════════════════════

Measures |g_total| at every (pred, target) to find the exact imbalance
between flat and edge regimes, then derives the correct weights.
"""

import numpy as np

ALPHA       = 2.0
BETA        = 1.5
FLAT_L1_W   = 0.2
FLAT_MARGIN = 0.08
FLAT_PEN_W  = 0.5
EDGE_W      = 3.0

def mw(t):
    return np.clip(np.abs(t)**0.5, 0.7, 2.0)

def g_dir(p, t):
    gap = p - t
    sc = ALPHA if (gap * t) < 0 else BETA
    return 2.0 * sc * gap * mw(t)

def g_flat_l1(p):
    return FLAT_L1_W * np.sign(p)

def g_flat_quad(p):
    over = np.maximum(np.abs(p) - FLAT_MARGIN, 0.0)
    return 2.0 * FLAT_PEN_W * over * np.sign(p)

def g_edge(p, t):
    u = np.abs(p); v = np.abs(t)
    d = u - v  # signed
    gd = np.where(np.abs(d) < 1.0, d, np.sign(d))
    return EDGE_W * gd * np.sign(p)

def g_flat(p):
    return g_dir(p, 0.0) + g_flat_l1(p) + g_flat_quad(p)

def g_edge_total(p, t):
    return g_dir(p, t) + g_edge(p, t)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Gradient magnitude along the full number line
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 100)
print("  SECTION 1: |g| along the number line for various targets")
print("=" * 100)

# For each target, measure |g| at key distances
print("\n  Flat bar (target=0):")
print(f"  {'pred':>7}  | {'|g_dir|':>8} {'|g_L1|':>8} {'|g_quad|':>8} {'|g_total|':>8}")
for p in [0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.00]:
    gd = abs(g_dir(p, 0.0))
    gl = abs(g_flat_l1(p))
    gq = abs(g_flat_quad(p))
    gt = abs(g_flat(p))
    print(f"  {p:7.2f}  | {gd:8.4f} {gl:8.4f} {gq:8.4f} {gt:8.4f}")

print("\n  Edge bar (target=0.5):")
print(f"  {'pred':>7}  | {'|g_dir|':>8} {'|g_edge|':>8} {'|g_total|':>8}  | gap={'>' if True else '<'}0")
for p in [0.00, 0.03, 0.10, 0.20, 0.30, 0.40, 0.50, 0.53, 0.60, 0.70, 0.80, 1.00]:
    gd = abs(g_dir(p, 0.5))
    ge = abs(g_edge(p, 0.5))
    gt = abs(g_edge_total(p, 0.5))
    print(f"  {p:7.2f}  | {gd:8.4f} {ge:8.4f} {gt:8.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: The core imbalance — at same |pred|, flat vs edge
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 100)
print("  SECTION 2: Cross-regime imbalance at same |pred|")
print("=" * 100)

print(f"\n  {'|pred|':>7}  | {'|g_flat|':>8} {'|g_edge(t=0.3)|':>14} {'|g_edge(t=0.5)|':>14} {'|g_edge(t=0.8)|':>14}  | ratio_f/e(0.5)")
for p in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
    gf = abs(g_flat(p))
    ge3 = abs(g_edge_total(p, 0.3))
    ge5 = abs(g_edge_total(p, 0.5))
    ge8 = abs(g_edge_total(p, 0.8))
    ratio = gf / (ge5 + 1e-10)
    print(f"  {p:7.2f}  | {gf:8.4f} {ge3:14.4f} {ge5:14.4f} {ge8:14.4f}  | {ratio:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: What FLAT_L1_W would balance at each |pred|?
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 100)
print("  SECTION 3: Required FLAT_L1_W to match edge gradient at each |pred|")
print("=" * 100)

print(f"\n  {'|pred|':>7}  | {'current |g_flat|':>16} {'target |g_edge(0.5)|':>20} {'required_L1_W':>14}")
for p in [0.01, 0.03, 0.05, 0.10, 0.20, 0.30]:
    # Current flat gradient (without L1)
    gd = abs(g_dir(p, 0.0))  # = 2*beta*p*0.7 = 2.1*p
    gq = abs(g_flat_quad(p))
    ge = abs(g_edge_total(p, 0.5))
    
    # To match: gd + L1_W + gq = ge → L1_W = ge - gd - gq
    required = ge - gd - gq
    current_total = gd + FLAT_L1_W + gq
    print(f"  {p:7.2f}  | {current_total:16.4f} {ge:20.4f} {required:14.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: The fundamental problem — L1 is constant, edge grows with |pred|
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 100)
print("  SECTION 4: Root cause analysis")
print("=" * 100)

print("""
  The flat gradient at small pred is:
    g_flat ≈ 2*beta*pred*0.7 + L1_W = 2.1*pred + 0.2
    
  The edge gradient at same pred (for target=0.5) is:
    g_edge ≈ 2*alpha*(pred-0.5)*mw(0.5) + 3.0*(|pred|-0.5)*sign(pred)
    
  At pred=0.01:
    g_flat = 2.1*0.01 + 0.2 = 0.221
    g_edge = 2*2.0*(-0.49)*0.707 + 3.0*(-0.49)*(+1) = -1.386 + (-1.47) = -2.856 → |g|=2.856
    
  The edge gradient is ~13x larger because:
    1. Direction loss: 2*alpha*|gap|*mw — gap≈0.5, mw≈0.7 → ~1.4
    2. Edge mag loss: 3.0*| |pred|-|target| | ≈ 3.0*0.5 = 1.5
    Total edge ≈ 2.9
    
  The flat gradient is small because:
    1. Direction loss: 2*beta*pred*mw = 2*1.5*0.01*0.7 = 0.021 (tiny!)
    2. L1 loss: 0.2 (constant, but only 0.2)
    Total flat ≈ 0.22
    
  TO BALANCE: The flat gradient needs to be ~2.9 at pred=0.01
    Required: 2.1*0.01 + L1_W = 2.9 → L1_W ≈ 2.88
    
  But at pred=0.3:
    g_flat = 2.1*0.3 + L1_W = 0.63 + L1_W
    g_edge(target=0.5) = 2*1.5*0.2*0.707 + 3.0*0.2*1 = 0.424 + 0.6 = 1.024
    Required: 0.63 + L1_W = 1.024 → L1_W ≈ 0.39
    
  So a CONSTANT L1_W can't balance across all pred values!
  At small pred: need L1_W ≈ 2.9
  At large pred: need L1_W ≈ 0.4
  
  SOLUTION: Make the flat penalty adaptive — scale with target magnitude
  or use a different functional form that naturally scales.
""")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Design the balanced loss
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 100)
print("  SECTION 5: Design options for balanced gradients")
print("=" * 100)

print("""
  OPTION A: Scale FLAT_L1_WEIGHT by a factor that depends on pred magnitude
    Problem: introduces a kink, not smooth
    
  OPTION B: Replace L1 with a scaled L2 that matches edge gradient growth
    L_flat = w * pred^2 * mw_flat where mw_flat is chosen to match edge
    g_flat = 2 * w * pred * mw_flat
    At pred=0.01: 2*w*0.01*0.7 = 2.9 → w = 207 (absurd)
    
  OPTION C: Use a HYBRID approach — keep L1 for near-zero, add scaled L2 for larger
    This is what v6 already does (L1 + quad above margin)
    Problem: the quad only kicks in above 0.08
    
  OPTION D (RECOMMENDED): Single unified loss function that treats flat as edge with target=0
    Remove the flat/edge distinction entirely.
    Use ONE loss that works for all targets including 0:
    
    L = scale_asymmetry * (pred - target)^2 * mag_weight(target)
      + edge_w * smooth_l1(|pred|, |target|) * mag_weight(target)
    
    For target=0:
      mag_weight = 0.7 (clamped)
      L_dir = beta * pred^2 * 0.7 → g = 2*beta*pred*0.7 = 2.1*pred
      L_edge = edge_w * smooth_l1(|pred|, 0) → g = edge_w * sign(pred) (for |pred|<1)
      g_total = 2.1*pred + edge_w * sign(pred)
      
    At pred=0.01: g = 0.021 + edge_w
    At pred=0.3:  g = 0.63 + edge_w
    
    For target=0.5, pred=0.01:
      g_dir = 2*2.0*(-0.49)*0.707 = -1.386
      g_edge = 3.0 * (-0.49) * 1 = -1.47
      g_total = -2.856 → |g| = 2.856
      
    To match at pred=0.01: 0.021 + edge_w = 2.856 → edge_w = 2.835
    Check at pred=0.3, target=0.5:
      g_dir = 2*1.5*(-0.2)*0.707 = -0.424
      g_edge = 3.0 * (-0.2) * 1 = -0.6
      g_total = -1.024 → |g| = 1.024
    Flat at pred=0.3: g = 0.63 + 2.835 = 3.465 → |g| = 3.465
    
    Now flat is STRONGER than edge at large pred. That's the opposite problem!
    
  OPTION E (RECOMMENDED): Scale the edge magnitude weight DOWN and use a 
  SMOOTH transition between flat and edge regimes.
  
  The key insight: EDGE_MAG_WEIGHT=3.0 is too large. It was increased from 0.5
  to 3.0 to make it "strong enough vs direction loss" but this created the 
  imbalance with flat bars.
  
  Let's find the weight that balances:
  
  For target=0.5, pred=0.01 (worst case undershoot):
    g_dir = 2*2.0*(-0.49)*0.707 = -1.386
    g_edge = W * (-0.49) * 1 = -0.49*W
    g_total = -(1.386 + 0.49*W)
    
  For target=0, pred=0.01 (flat):
    g_dir = 2*1.5*0.01*0.7 = 0.021
    g_L1 = L (current L1 weight)
    g_total = 0.021 + L
    
  To balance: 0.021 + L = 1.386 + 0.49*W
  
  If we keep L=0.2: 0.221 = 1.386 + 0.49*W → W = -2.38 (nonsense, wrong direction)
  
  The problem is structural: the edge gradient at pred≈0 is ALWAYS going to be
  much larger than the flat gradient at the same pred, because the edge has
  a large gap (pred-target ≈ -0.5) while the flat has a tiny gap (pred-0 ≈ 0.01).
  
  THE REAL SOLUTION: Don't try to match gradients at the same |pred|.
  Instead, match gradients at the same |gap| = |pred - target|.
  
  For flat: gap = pred - 0 = pred
  For edge: gap = pred - target
  
  At same |gap| = 0.05:
    Flat (pred=0.05, target=0): g = 2*1.5*0.05*0.7 + 0.2 = 0.105 + 0.2 = 0.305
    Edge (pred=0.45, target=0.5): g = 2*2.0*(-0.05)*0.707 + 3.0*(-0.05)*(-1) = -0.141 + 0.15 = 0.009
    Wait, that's wrong. Let me recalculate.
    
  Edge (pred=0.45, target=0.5):
    gap = -0.05, undershoot → alpha=2.0
    mw = sqrt(0.5) = 0.707
    g_dir = 2*2.0*(-0.05)*0.707 = -0.141
    g_edge: u=0.45, v=0.5, d = -0.05, |d|<1 → gd = -0.05
           g_edge = 3.0 * (-0.05) * sign(0.45) = -0.15
    g_total = -0.141 + (-0.15) = -0.291 → |g| = 0.291
    
  Flat (pred=0.05, target=0):
    g = 2*1.5*0.05*0.7 + 0.2 = 0.105 + 0.2 = 0.305 → |g| = 0.305
  
  Ratio: 0.305/0.291 = 1.05 — ALREADY BALANCED at same |gap|!
  
  The imbalance is NOT in the loss design — it's in the COMPARISON.
  Comparing flat at |pred|=0.01 to edge at |pred|=0.01 is wrong because
  the edge has |gap|=0.49 while the flat has |gap|=0.01.
  
  The gradients ARE balanced when compared at the same |pred - target|!
""")

print("\n  Verification: |g| at same |gap| = |pred - target|")
print(f"  {'|gap|':>7}  | {'|g_flat|':>10} {'|g_edge(t=0.3)|':>14} {'|g_edge(t=0.5)|':>14} {'|g_edge(t=0.8)|':>14}")

for gap in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
    # Flat: pred = gap, target = 0
    gf = abs(g_flat(gap))
    
    # Edge: pred = t - gap (undershoot by gap)
    ge3 = abs(g_edge_total(0.3 - gap, 0.3)) if gap < 0.3 else float('nan')
    ge5 = abs(g_edge_total(0.5 - gap, 0.5)) if gap < 0.5 else float('nan')
    ge8 = abs(g_edge_total(0.8 - gap, 0.8)) if gap < 0.8 else float('nan')
    
    print(f"  {gap:7.2f}  | {gf:10.4f} {ge3:14.4f} {ge5:14.4f} {ge8:14.4f}")

print("""
  CONCLUSION: The gradients ARE naturally balanced at the same |gap|.
  The perceived imbalance comes from comparing at the same |pred| instead of |gap|.
  
  However, there's still a practical issue: the encoder sees the gradient at the
  CURRENT prediction, not at a hypothetical "same gap" point. If the encoder
  produces pred=0.03 for a flat bar, the gradient is 0.26. If it produces
  pred=0.03 for an edge bar (target=0.5), the gradient is 2.74.
  
  The encoder is trained by BOTH flat and edge bars. The much stronger edge
  gradients dominate, pushing the encoder to produce features that work for
  edge bars. These features may produce small spurious signals on flat bars,
  and the weak flat gradient (0.26) isn't enough to correct them.
  
  SOLUTION: Increase FLAT_L1_WEIGHT to make flat gradients competitive.
  The question is: what value?
  
  We want: at the SAMPLER_THRESHOLD boundary (pred=0.08), the flat gradient
  should be comparable to the edge gradient at the same distance from target.
  
  At pred=0.08, target=0 (flat):
    g = 2*1.5*0.08*0.7 + L1_W = 0.168 + L1_W
    
  At pred=0.08, target=0.5 (edge, gap=-0.42):
    g_dir = 2*2.0*(-0.42)*0.707 = -1.187
    g_edge = 3.0 * (0.08-0.5) * 1 = 3.0 * (-0.42) = -1.26
    g_total = -2.447 → |g| = 2.447
    
  To match: 0.168 + L1_W = 2.447 → L1_W = 2.28
  
  But this would make the flat gradient at pred=0.01:
    g = 0.021 + 2.28 = 2.301
    
  And at pred=0.5:
    g = 2*1.5*0.5*0.7 + 2.28 + 2*0.5*(0.5-0.08) = 1.05 + 2.28 + 0.42 = 3.75
    
  This is very aggressive. The model would be heavily penalized for any 
  non-zero prediction on flat bars.
  
  A more moderate approach: L1_W = 1.0
    At pred=0.01: g = 1.021 (vs edge 2.86 — ratio 0.36)
    At pred=0.08: g = 1.168 (vs edge 2.45 — ratio 0.48)
    At pred=0.3:  g = 1.05+1.0+0.22 = 2.27 (vs edge 1.02 — ratio 2.23)
    
  At L1_W=1.0, flat becomes STRONGER than edge at pred>0.15.
  This might be too aggressive and prevent the model from learning.
  
  BETTER APPROACH: Use a SMOOTH, ADAPTIVE weight that scales with pred.
  Instead of constant L1_W, use L1_W * (1 + scale_factor * |pred|).
  This gives strong push at small pred (where edge dominates) and
  moderate push at larger pred (where direction loss takes over).
  
  Or even better: use a single unified formulation.
""")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Proposed balanced design
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 100)
print("  SECTION 6: Proposed balanced design — single smooth number line loss")
print("=" * 100)

print("""
  PROPOSED v7 DESIGN:
  
  Remove the flat/edge distinction. Use ONE loss for all bars:
  
  L = L_direction + L_magnitude
  
  Where:
    L_direction = scale_asymmetry * (pred - target)^2 * mag_weight(|target|)
    L_magnitude = w_mag * smooth_l1(|pred|, |target|) * mag_weight(|target|)
  
  For target=0 (what was "flat"):
    mag_weight = 0.7 (clamped)
    L_dir = beta * pred^2 * 0.7
    L_mag = w_mag * smooth_l1(|pred|, 0) * 0.7
         = w_mag * |pred| * 0.7  (for |pred| < 1)
    
    g_dir = 2 * beta * pred * 0.7 = 2.1 * pred
    g_mag = w_mag * sign(pred) * 0.7
    g_total = 2.1 * pred + 0.7 * w_mag * sign(pred)
    
  For target=0.5 (edge):
    mag_weight = sqrt(0.5) = 0.707
    L_dir = scale * (pred-0.5)^2 * 0.707
    L_mag = w_mag * smooth_l1(|pred|, 0.5) * 0.707
    
    At pred=0.01 (undershoot):
      g_dir = 2*2.0*(-0.49)*0.707 = -1.386
      g_mag = w_mag * (0.01-0.5) * 1 * 0.707 = w_mag * (-0.49) * 0.707 = -0.346 * w_mag
      g_total = -1.386 - 0.346 * w_mag
    
  To balance at pred=0.01:
    Flat: g = 2.1*0.01 + 0.7*w_mag = 0.021 + 0.7*w_mag
    Edge: g = 1.386 + 0.346*w_mag
    
    These can never be equal for positive w_mag because 0.021 ≠ 1.386.
    The direction loss alone creates the imbalance.
    
  THE FUNDAMENTAL ISSUE: The direction loss for edge bars at small pred
  has a large gap (|pred-target| ≈ 0.5) while flat bars have tiny gap
  (|pred-0| ≈ 0.01). The direction loss gradient is proportional to gap,
  so edge bars naturally get much stronger gradients.
  
  THIS IS CORRECT BEHAVIOR! The model SHOULD get stronger gradients for
  edge bars because they have further to go. The problem is that the
  encoder can satisfy edge gradients while producing small spurious signals
  on flat bars, and the flat gradients are too weak to correct this.
  
  THE FIX: Don't balance gradient magnitudes. Instead, ensure the flat
  gradient is strong enough to push pred to zero in a reasonable number of
  steps. Currently at pred=0.03, g_flat=0.26. With lr=1e-5 and grad_clip=5,
    effective update = lr * g = 1e-5 * 0.26 = 2.6e-6 per step
    To go from 0.03 to 0: ~11,500 steps (way too slow!)
    
  With L1_W = 2.0:
    g_flat(0.03) = 0.063 + 2.0 = 2.063
    effective update = 1e-5 * 2.063 = 2.06e-5 per step
    To go from 0.03 to 0: ~1,456 steps (still slow)
    
  With L1_W = 10.0:
    g_flat(0.03) = 10.063 → clipped to 5.0
    effective update = 1e-5 * 5.0 = 5e-5 per step
    To go from 0.03 to 0: ~600 steps (better)
    
  But L1_W=10 would dominate everything and make the loss meaningless
  for edge bars.
  
  THE REAL FIX: The issue isn't the loss — it's the LEARNING RATE.
  With lr=1e-5, even "balanced" gradients produce tiny updates.
    At pred=0.03, target=0.5 (edge): g = -2.74
    Update = 1e-5 * 2.74 = 2.74e-5 per step
    To go from 0.03 to 0.5: ~17,000 steps
    
  The model needs ~17k steps to correct an edge bar but ~11.5k steps for flat.
    These are in the same ballpark! The gradients ARE balanced!
    
  The real problem is that the encoder produces features optimized for edge bars
  (which are more numerous and have stronger gradients), and these features
  happen to produce small positive biases on flat bars. The flat gradient of 0.26
  is enough to eventually correct this, but training may not run long enough.
""")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Final recommendation
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 100)
print("  SECTION 7: Final recommendation")
print("=" * 100)

print("""
  After thorough analysis, the gradients ARE balanced when compared correctly
  (at same |pred - target|, not at same |pred|).
  
  The perceived imbalance is a red herring. The real issues are:
  
  1. FLAT GRADIENT TOO WEAK AT SMALL pred (structural):
     g_flat = 2.1*pred + L1_W. At pred=0.01, this is 0.021 + 0.2 = 0.221.
     With lr=1e-5, this gives 2.2e-6 update per step — very slow.
     
     FIX: Increase FLAT_L1_WEIGHT from 0.2 to 1.0-2.0.
     This makes the constant push toward zero much stronger.
     
  2. EDGE GRADIENT CANCELLATION (pred<0, target>0):
     When pred is slightly negative, g_edge opposes g_dir.
     This creates a small "dead zone" near pred=0 for edge bars.
     
     FIX: Remove the |pred| from the edge magnitude loss.
     Use smooth_l1(pred, target) instead of smooth_l1(|pred|, |target|).
     This eliminates the cusp at pred=0.
     
  3. mag_weight CLAMPING:
     All targets below 0.49 share mag_weight=0.7.
     This means the direction loss can't distinguish weak from strong edges.
     
     FIX: Lower the clamp minimum from 0.7 to 0.3 or remove it.
     Let mag_weight = |target|^0.5 for all targets.
     
  RECOMMENDED CHANGES:
    FLAT_L1_WEIGHT: 0.2 → 1.5
    EDGE_MAG_WEIGHT: 3.0 → 1.5 (reduce since we're fixing the cancellation)
    mag_weight clamp: 0.7 → 0.3 (or remove)
    Edge loss: use smooth_l1(pred, target) not smooth_l1(|pred|, |target|)
""")
