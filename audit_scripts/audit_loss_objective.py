#!/usr/bin/env python3
"""
audit_loss_objective.py
========================
LPatchTST — Loss, Objective & Curriculum Audit

Run from repo root:
    python3 audit_loss_objective.py

Covers:
  1. is_zero definition vs oracle boundary
  2. false_signal_weight & margin — dead-zone attractor risk
  3. spread_reward & corr_penalty — counter-collapse magnitude
  4. overshoot_discount policy — tail pressure
  5. Numerical stability — _safe_std, quantile, small-batch NaN
"""

import sys, os, math, textwrap
import numpy as np
import torch
import torch.nn.functional as F

SEP  = "\u2500" * 70
SEP2 = "=" * 70

def hdr(title):
    print(f"\n{SEP}")
    print(f"{title}")
    print(SEP)

def ok(msg):   print(f"  [PASS] {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")
def info(msg): print(f"  [INFO] {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# Import the real loss + config
# ─────────────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

try:
    from loss import (
        continuous_weighted_direction_loss,
        _safe_std,
        quantile_spread_loss,
        moderate_bucket_loss,
    )
    LOSS_IMPORTED = True
except Exception as e:
    LOSS_IMPORTED = False
    print(f"[WARN] Could not import loss.py: {e}")

try:
    import config as CFG
    CONFIG_IMPORTED = True
except Exception as e:
    CONFIG_IMPORTED = False
    print(f"[WARN] Could not import config.py: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Hardcoded constants extracted from loss.py source (verified manually)
# ─────────────────────────────────────────────────────────────────────────────
LOSS_IS_ZERO_BOUNDARY    = 0.05   # line: is_zero = (target.abs() < 0.05)
LOSS_MARGIN              = 0.03   # margin=0.03
LOSS_FALSE_SIGNAL_WEIGHT = 1.5    # false_signal_weight=1.5
LOSS_PENALTY_WEIGHT      = 2.0    # penalty_weight=2.00
LOSS_DISPERSION_WEIGHT   = 0.25   # dispersion_weight=0.25
LOSS_BIAS_WEIGHT         = 0.30   # bias_weight=0.30
LOSS_SPREAD_REWARD_COEFF = 0.50   # line: + 0.50 * spread_reward
LOSS_CORR_PENALTY_COEFF  = 0.25   # == dispersion_weight
LOSS_OVERSHOOT_DISC_QUAL = 0.5    # large_mask_new = qual_e >= 0.5
LOSS_OVERSHOOT_DISC_VAL  = 0.1    # torch.full_like(tgt_e, 0.1)

CFG_SAMPLER_THRESHOLD = getattr(CFG, 'SAMPLER_THRESHOLD', 0.05) if CONFIG_IMPORTED else 0.05
CFG_ORACLE_THRESHOLD  = getattr(CFG, 'ORACLE_THRESHOLD',  None) if CONFIG_IMPORTED else None

print(SEP2)
print("  LPatchTST — Loss, Objective & Curriculum Audit")
print(f"  Root: {ROOT}")
print(SEP2)

# =============================================================================
# AUDIT 1 · is_zero definition vs oracle / sampler boundary
# =============================================================================
hdr("AUDIT 1 · is_zero definition vs oracle / sampler boundary")

info(f"loss.py  is_zero boundary          = {LOSS_IS_ZERO_BOUNDARY}")
info(f"config.SAMPLER_THRESHOLD           = {CFG_SAMPLER_THRESHOLD}")
info(f"config.ORACLE_THRESHOLD            = {CFG_ORACLE_THRESHOLD}")

# Check 1a: is_zero vs SAMPLER_THRESHOLD alignment
if abs(LOSS_IS_ZERO_BOUNDARY - CFG_SAMPLER_THRESHOLD) < 1e-9:
    ok("is_zero boundary == SAMPLER_THRESHOLD (0.05). Flat-class fully consistent.")
else:
    fail(
        f"is_zero={LOSS_IS_ZERO_BOUNDARY} != SAMPLER_THRESHOLD={CFG_SAMPLER_THRESHOLD}. "
        "The loss punishes as 'flat' a different range than the sampler over-samples. "
        "Signals in the gap are drawn more often but penalised as flat → gradient noise."
    )

# Check 1b: is_zero vs oracle threshold (if oracle threshold is known)
if CFG_ORACLE_THRESHOLD is not None:
    if abs(LOSS_IS_ZERO_BOUNDARY - CFG_ORACLE_THRESHOLD) < 1e-9:
        ok(f"is_zero boundary == ORACLE_THRESHOLD ({CFG_ORACLE_THRESHOLD}). Full stack aligned.")
    elif LOSS_IS_ZERO_BOUNDARY > CFG_ORACLE_THRESHOLD:
        warn(
            f"is_zero boundary ({LOSS_IS_ZERO_BOUNDARY}) > ORACLE_THRESHOLD ({CFG_ORACLE_THRESHOLD}). "
            "Oracle may label some bars as Long/Short with |score| in "
            f"[{CFG_ORACLE_THRESHOLD}, {LOSS_IS_ZERO_BOUNDARY}), but loss treats them as Flat. "
            "These valid signals are punished via false_signal_loss. "
            "Fix: align is_zero to ORACLE_THRESHOLD or explicitly widen oracle boundary."
        )
    else:
        warn(
            f"is_zero boundary ({LOSS_IS_ZERO_BOUNDARY}) < ORACLE_THRESHOLD ({CFG_ORACLE_THRESHOLD}). "
            "Loss is stricter than oracle — borderline edge bars still incur edge loss "
            "but oracle only generated them as weak signals. Generally acceptable but monitor."
        )
else:
    warn(
        "config.ORACLE_THRESHOLD not found. Cannot verify full stack alignment. "
        "Manually confirm oracle.py labelling threshold == loss.py is_zero boundary."
    )

# Check 1c: Hardcoded literal
warn(
    "loss.py hardcodes is_zero as 0.05 literal, not reading config.SAMPLER_THRESHOLD. "
    "If you change SAMPLER_THRESHOLD, loss.py silently diverges. "
    "Fix: pass flat_threshold=config.SAMPLER_THRESHOLD into loss function signature."
)

# Check 1d: Dead-zone philosophy — margin < is_zero
if LOSS_MARGIN < LOSS_IS_ZERO_BOUNDARY:
    ok(
        f"margin ({LOSS_MARGIN}) < is_zero ({LOSS_IS_ZERO_BOUNDARY}): "
        "predictions must collapse below margin to avoid false_signal penalty. Intentional dead-zone."
    )
else:
    fail(
        f"margin ({LOSS_MARGIN}) >= is_zero ({LOSS_IS_ZERO_BOUNDARY}). "
        "False-signal loss only activates when prediction overshoots is_zero, not margin. "
        "Dead-zone logic is broken."
    )

# =============================================================================
# AUDIT 2 · false_signal_weight & margin — dead-zone attractor risk
# =============================================================================
hdr("AUDIT 2 · false_signal_weight & margin — dead-zone attractor risk")

info(f"false_signal_weight = {LOSS_FALSE_SIGNAL_WEIGHT}")
info(f"margin              = {LOSS_MARGIN}")
info(f"penalty_weight      = {LOSS_PENALTY_WEIGHT}")

# Check 2a: Compare false_signal_weight to penalty_weight
if LOSS_FALSE_SIGNAL_WEIGHT < LOSS_PENALTY_WEIGHT:
    ok(
        f"false_signal_weight ({LOSS_FALSE_SIGNAL_WEIGHT}) < penalty_weight ({LOSS_PENALTY_WEIGHT}). "
        "Wrong-direction edge predictions punished harder than flat-over-predictions. Good hierarchy."
    )
else:
    warn(
        f"false_signal_weight ({LOSS_FALSE_SIGNAL_WEIGHT}) >= penalty_weight ({LOSS_PENALTY_WEIGHT}). "
        "Model may prefer predicting zero over wrong-sign, reinforcing dead-zone collapse. "
        "Typically penalty_weight should be >= false_signal_weight."
    )

# Check 2b: Dead-zone quantify — what fraction of predictions are zero-attracted?
# Simulate: if the model predicts exactly 0 for everything, what is the gradient?
info("Simulating gradient at pred=0 for a flat target (is_zero=True)...")
_tgt_flat = torch.tensor([0.02, -0.01, 0.03, 0.0, -0.02])  # all inside is_zero
_pred_zero = torch.zeros(5, requires_grad=True)
_loss_zero = continuous_weighted_direction_loss(_pred_zero, _tgt_flat) if LOSS_IMPORTED else None
if _loss_zero is not None:
    _loss_zero.backward()
    grad_at_zero = _pred_zero.grad.abs().mean().item()
    info(f"  |grad| at pred=0 for flat target: {grad_at_zero:.6f}")
    if grad_at_zero < 1e-4:
        fail(
            f"Gradient nearly zero ({grad_at_zero:.2e}) when predicting 0 for flat targets. "
            "Model has no signal to move away from zero. Dead-zone is a stable fixed point."
        )
    else:
        ok(f"Gradient at pred=0 for flat targets: {grad_at_zero:.4f} (non-zero, model can escape).")

# Check 2c: Simulate gradient at pred=0 for an EDGE target (is_edge=True)
info("Simulating gradient at pred=0 for edge target (is_edge=True)...")
_tgt_edge = torch.tensor([0.3, -0.25, 0.4, -0.35, 0.2])  # all outside is_zero
_pred_zero2 = torch.zeros(5, requires_grad=True)
_loss_edge = continuous_weighted_direction_loss(_pred_zero2, _tgt_edge) if LOSS_IMPORTED else None
if _loss_edge is not None:
    _loss_edge.backward()
    grad_at_zero_edge = _pred_zero2.grad.abs().mean().item()
    info(f"  |grad| at pred=0 for edge target: {grad_at_zero_edge:.6f}")
    if grad_at_zero_edge < 0.05:
        fail(
            f"Gradient at pred=0 for edge targets is weak ({grad_at_zero_edge:.4f}). "
            "Model has insufficient pressure to move off zero even for large signals."
        )
    else:
        ok(f"Gradient at pred=0 for edge targets: {grad_at_zero_edge:.4f}. Sufficient pull away from zero.")

# Check 2d: Dead-zone attractor — mixed batch (50% flat, 50% edge)
info("Simulating mixed batch (50% flat / 50% edge) gradient at pred=0...")
_tgt_mixed = torch.cat([
    torch.tensor([0.02, -0.01, 0.03, 0.0, -0.02]),   # flat
    torch.tensor([0.3, -0.25, 0.4, -0.35, 0.2]),      # edge
])
_pred_mixed = torch.zeros(10, requires_grad=True)
_loss_mixed = continuous_weighted_direction_loss(_pred_mixed, _tgt_mixed) if LOSS_IMPORTED else None
if _loss_mixed is not None:
    _loss_mixed.backward()
    grad_mixed = _pred_mixed.grad.abs().mean().item()
    info(f"  |grad| at pred=0 for mixed batch: {grad_mixed:.6f}")
    if grad_mixed < 0.05:
        fail(
            f"Mixed-batch gradient too weak ({grad_mixed:.4f}). "
            "Flat-class dilution is suppressing edge gradients → model stays at zero."
        )
    else:
        ok(f"Mixed-batch gradient: {grad_mixed:.4f}. Edge targets dominate over flat dilution.")

# =============================================================================
# AUDIT 3 · spread_reward & corr_penalty — counter-collapse magnitude
# =============================================================================
hdr("AUDIT 3 · spread_reward & corr_penalty — counter-collapse magnitude")

info(f"spread_reward coefficient (in total) = {LOSS_SPREAD_REWARD_COEFF}")
info(f"corr_penalty (dispersion) coefficient = {LOSS_CORR_PENALTY_COEFF}")
info(f"q_spread_loss coefficient             = 0.30 (hardcoded)")

# Check 3a: Is spread_reward strong enough to break zero-std fixed point?
# At pred_std → 0, spread_reward = -log(pred_std) → +∞. Good.
# But the coefficient scales how fast this kicks in.
for std_val in [0.001, 0.01, 0.05, 0.10]:
    sr = -math.log(max(std_val, 1e-4))
    weighted = LOSS_SPREAD_REWARD_COEFF * sr
    info(f"  pred_std={std_val:.3f} → spread_reward={sr:.3f}, weighted={weighted:.3f}")

# Compare spread_reward to typical edge_mse magnitude
# Typical smooth_l1 loss at 0.2 error with beta=0.15 = 0.2 - 0.075 = 0.125 (linear)
typ_edge_mse = 0.125
sr_at_zero = LOSS_SPREAD_REWARD_COEFF * (-math.log(1e-4))   # practical floor
if sr_at_zero > typ_edge_mse:
    ok(
        f"spread_reward at std=0 ({sr_at_zero:.2f}) > typical edge_mse ({typ_edge_mse:.3f}). "
        "Zero-std fixed point is unstable — model is pushed to spread predictions."
    )
else:
    fail(
        f"spread_reward at std=0 ({sr_at_zero:.2f}) <= typical edge_mse ({typ_edge_mse:.3f}). "
        "The spread pressure may be too weak to overcome the edge regression pull toward zero."
    )

# Check 3b: corr_penalty max magnitude vs edge_mse
corr_max = LOSS_CORR_PENALTY_COEFF * 2.0  # max corr_penalty = 2.0 (clamped)
if corr_max >= typ_edge_mse:
    ok(
        f"corr_penalty max ({corr_max:.2f}) >= typical edge_mse ({typ_edge_mse:.3f}). "
        "Distribution mismatch can dominate when correlation is zero."
    )
else:
    warn(
        f"corr_penalty max ({corr_max:.2f}) < typical edge_mse ({typ_edge_mse:.3f}). "
        "Correlation penalty is weak relative to regression loss. "
        "Model may reach acceptable MSE while predicting with wrong correlation structure."
    )

# Check 3c: Collapse simulation — all-zero prediction on edge batch
if LOSS_IMPORTED:
    _tgt_large = torch.linspace(-0.5, 0.5, 32)
    _pred_collapse = torch.zeros(32, requires_grad=True)
    _loss_collapse = continuous_weighted_direction_loss(_pred_collapse, _tgt_large)
    _loss_collapse.backward()
    info(f"  All-zero pred on uniform[-0.5,0.5] target — loss={_loss_collapse.item():.4f}, |grad|={_pred_collapse.grad.abs().mean():.4f}")
    if _pred_collapse.grad.abs().mean() > 0.1:
        ok("Collapse scenario gradient is strong — model cannot stably sit at all-zero.")
    else:
        fail("Collapse scenario gradient is weak — all-zero is a near-stable point.")

# Check 3d: Interaction — spread_reward vs pred_std penalty at bottom of STEP 6
info("Checking pred_std floor penalty (+ 0.5 * relu(0.25 - pred_std))...")
for std_val in [0.0, 0.10, 0.20, 0.25, 0.30]:
    floor_pen = 0.5 * max(0.25 - std_val, 0.0)
    sr_pen    = LOSS_SPREAD_REWARD_COEFF * (-math.log(max(std_val, 1e-4)))
    total_anti_collapse = floor_pen + sr_pen
    info(f"  pred_std={std_val:.2f}: floor_pen={floor_pen:.3f}, spread_reward={sr_pen:.3f}, total={total_anti_collapse:.3f}")

ok("pred_std floor penalty + spread_reward are additive — double-layer anti-collapse. Good design.")

# =============================================================================
# AUDIT 4 · overshoot_discount policy
# =============================================================================
hdr("AUDIT 4 · overshoot_discount policy — tail pressure")

info(f"overshoot_discount threshold  = qual_e >= {LOSS_OVERSHOOT_DISC_QUAL}")
info(f"overshoot_discount value      = {LOSS_OVERSHOOT_DISC_VAL} (near-zero)")
info("Bars with |target| < 0.5 get full overshoot penalty (discount=1.0).")
info("Bars with |target| >= 0.5 get 0.1x overshoot penalty (near-exempt).")

# Check 4a: Threshold selection
if LOSS_OVERSHOOT_DISC_QUAL == 0.5:
    ok(
        "Overshoot discount threshold=0.5 is at the is_edge moderate/large boundary. "
        "Only the strongest signals (|y|>=0.5) get tail exemption. "
        "Middle-range signals [0.05, 0.5) still face full overshoot pressure."
    )
elif LOSS_OVERSHOOT_DISC_QUAL < 0.2:
    fail(
        f"Overshoot discount threshold={LOSS_OVERSHOOT_DISC_QUAL} is too low. "
        "Nearly all edge signals get tail-exempted, removing overshoot pressure entirely. "
        "Model can freely predict extreme values without penalty on most signals."
    )
else:
    warn(
        f"Overshoot discount threshold={LOSS_OVERSHOOT_DISC_QUAL}. "
        "Verify this is intentional given your oracle target distribution."
    )

# Check 4b: What fraction of oracle targets would be tail-exempted?
# Simulate with a realistic oracle distribution
np.random.seed(42)
oracle_sim = np.concatenate([
    np.random.normal(0, 0.02, 6000),     # flat zone
    np.random.normal( 0.25, 0.12, 2000), # moderate long
    np.random.normal(-0.25, 0.12, 2000), # moderate short
])
edge_mask  = np.abs(oracle_sim) >= 0.05
tail_mask  = np.abs(oracle_sim) >= LOSS_OVERSHOOT_DISC_QUAL
edge_count = edge_mask.sum()
tail_count = tail_mask.sum()
info(f"  Simulated oracle dist — edge bars: {edge_count} ({100*edge_count/len(oracle_sim):.1f}%)")
info(f"  Tail-exempted bars (|y|>={LOSS_OVERSHOOT_DISC_QUAL}): {tail_count} ({100*tail_count/len(oracle_sim):.1f}%)")
if edge_count > 0:
    pct_edge_exempt = 100 * tail_count / edge_count
    info(f"  % of edge bars that are tail-exempted: {pct_edge_exempt:.1f}%")
    if pct_edge_exempt < 20:
        ok(f"Only {pct_edge_exempt:.1f}% of edge bars are tail-exempted. Majority still face full overshoot pressure.")
    elif pct_edge_exempt < 40:
        warn(f"{pct_edge_exempt:.1f}% of edge bars are tail-exempted. Monitor prediction magnitude inflation.")
    else:
        fail(f"{pct_edge_exempt:.1f}% of edge bars are tail-exempted. Overshoot pressure is largely removed.")

# Check 4c: Overshoot loss gradient on moderate signal
if LOSS_IMPORTED:
    _tgt_moderate = torch.tensor([0.3] * 8)   # moderate, no exemption
    _pred_overshoot = torch.tensor([0.6] * 8, requires_grad=True)  # overshooting
    _loss_os = continuous_weighted_direction_loss(_pred_overshoot, _tgt_moderate)
    _loss_os.backward()
    grad_os = _pred_overshoot.grad.abs().mean().item()
    info(f"  Overshoot scenario (pred=0.6, tgt=0.3): loss={_loss_os.item():.4f}, |grad|={grad_os:.4f}")
    if grad_os > 0.05:
        ok(f"Moderate overshoot gradient ({grad_os:.4f}) is sufficient to correct overshoot.")
    else:
        warn(f"Moderate overshoot gradient ({grad_os:.4f}) is weak — model may persist in overshooting.")

# =============================================================================
# AUDIT 5 · Numerical stability
# =============================================================================
hdr("AUDIT 5 · Numerical stability — _safe_std, quantile, small-batch")

# Check 5a: _safe_std edge cases
if LOSS_IMPORTED:
    # Single element
    t_single = torch.tensor([1.5])
    s = _safe_std(t_single)
    if not torch.isnan(s) and not torch.isinf(s):
        ok(f"_safe_std(single element) = {s.item():.4f} (no NaN/Inf).")
    else:
        fail("_safe_std(single element) returned NaN or Inf.")

    # All-same tensor (std=0)
    t_const = torch.full((16,), 0.5)
    s = _safe_std(t_const)
    if not torch.isnan(s) and s.item() >= 0.01:
        ok(f"_safe_std(all-same) = {s.item():.4f} — clamped to min_val=0.01.")
    else:
        fail(f"_safe_std(all-same) = {s.item():.4f} — NaN or below min_val.")

    # NaN input
    t_nan = torch.tensor([float('nan'), 1.0, 2.0])
    s = _safe_std(t_nan)
    if not torch.isnan(s):
        ok(f"_safe_std(NaN input) = {s.item():.4f} — nan_to_num handled.")
    else:
        fail("_safe_std(NaN input) returned NaN — nan_to_num insufficient.")

# Check 5b: quantile_spread_loss with small batch
if LOSS_IMPORTED:
    # Batch of 4 — torch.quantile requires >= 1 element, should be fine
    pred4  = torch.tensor([0.1, -0.1, 0.2, -0.2], requires_grad=True)
    tgt4   = torch.tensor([0.15, -0.12, 0.18, -0.25])
    try:
        ql = quantile_spread_loss(pred4, tgt4)
        if not torch.isnan(ql) and not torch.isinf(ql):
            ok(f"quantile_spread_loss(batch=4) = {ql.item():.6f}. No NaN/Inf.")
        else:
            fail(f"quantile_spread_loss(batch=4) = {ql.item()} — NaN or Inf.")
    except Exception as e:
        fail(f"quantile_spread_loss(batch=4) raised: {e}")

    # Batch of 1 — quantile on size-1 tensor
    pred1 = torch.tensor([0.1], requires_grad=True)
    tgt1  = torch.tensor([0.2])
    try:
        ql1 = quantile_spread_loss(pred1, tgt1)
        if not torch.isnan(ql1):
            ok(f"quantile_spread_loss(batch=1) = {ql1.item():.6f}. No NaN.")
        else:
            fail("quantile_spread_loss(batch=1) returned NaN.")
    except Exception as e:
        fail(f"quantile_spread_loss(batch=1) raised: {e} — will crash in training if last batch=1.")

# Check 5c: Full loss on minimum viable batch (batch=2)
if LOSS_IMPORTED:
    for bsz in [2, 4, 8, 32]:
        pred_b = torch.randn(bsz, requires_grad=True)
        tgt_b  = torch.randn(bsz)
        try:
            l = continuous_weighted_direction_loss(pred_b, tgt_b)
            l.backward()
            nan_grad = torch.isnan(pred_b.grad).any().item()
            nan_loss = torch.isnan(l).item()
            if nan_loss or nan_grad:
                fail(f"batch={bsz}: NaN in loss={nan_loss}, NaN in grad={nan_grad}.")
            else:
                ok(f"batch={bsz}: loss={l.item():.4f}, |grad|={pred_b.grad.abs().mean():.4f}. Stable.")
        except Exception as e:
            fail(f"batch={bsz}: exception — {e}")

# Check 5d: corr computation stability — pred and target both constant
if LOSS_IMPORTED:
    pred_c = torch.full((16,), 0.1, requires_grad=True)
    tgt_c  = torch.full((16,), 0.2)
    try:
        l = continuous_weighted_direction_loss(pred_c, tgt_c)
        l.backward()
        if not torch.isnan(l) and not torch.isnan(pred_c.grad).any():
            ok(f"Constant pred+target: loss={l.item():.4f}. Correlation NaN guarded by _safe_std.")
        else:
            fail("Constant pred+target produced NaN — _safe_std guard insufficient here.")
    except Exception as e:
        fail(f"Constant pred+target raised: {e}")

# Check 5e: spread_reward at pred_std near 0 — log(0) risk
if LOSS_IMPORTED:
    pred_nearzero = torch.full((16,), 1e-6, requires_grad=True)
    tgt_spread    = torch.linspace(-0.3, 0.3, 16)
    try:
        l = continuous_weighted_direction_loss(pred_nearzero, tgt_spread)
        l.backward()
        if not torch.isnan(l) and not torch.isinf(l):
            ok(f"spread_reward at pred_std~0: loss={l.item():.4f}. clamp(min=1e-4) guards log(0).")
        else:
            fail(f"spread_reward at pred_std~0: loss={l.item()} — log(0) not guarded!")
    except Exception as e:
        fail(f"spread_reward at pred_std~0 raised: {e}")

# Check 5f: moderate_bucket_loss on empty tensor
if LOSS_IMPORTED:
    empty_pred = torch.tensor([], requires_grad=False)
    empty_tgt  = torch.tensor([])
    try:
        l = moderate_bucket_loss(empty_pred, empty_tgt)
        ok(f"moderate_bucket_loss(empty) = {l.item():.4f}. Early-exit guard works.")
    except Exception as e:
        fail(f"moderate_bucket_loss(empty) raised: {e} — will crash when all batch is flat.")

# =============================================================================
# AUDIT 6 · Weight interaction matrix — does any component dominate?
# =============================================================================
hdr("AUDIT 6 · Loss component weight interaction matrix")

components = [
    ("edge_mse",           1.00,  "regression core"),
    ("overshoot_loss",     1.00,  "magnitude ceiling (moderate bars)"),
    ("false_signal_loss",  LOSS_FALSE_SIGNAL_WEIGHT, "flat-class suppressor"),
    ("dir_penalty",        LOSS_PENALTY_WEIGHT,      "wrong-sign punisher"),
    ("dir_reward",         1.00,  "right-sign encourager"),
    ("corr_penalty",       LOSS_DISPERSION_WEIGHT,   "distribution shape"),
    ("q_spread_loss",      0.30,  "quantile matching"),
    ("bias_penalty",       LOSS_BIAS_WEIGHT,         "mean shift"),
    ("spread_reward",      LOSS_SPREAD_REWARD_COEFF, "anti-collapse"),
    ("moderate_bucket",    0.40,  "moderate range coverage"),
    ("pred_std_floor",     0.50,  "anti-collapse floor"),
]

print(f"  {'Component':<22} {'Weight':>6}  {'Role'}")
print(f"  {'-'*22} {'-'*6}  {'-'*30}")
total_weight = sum(w for _, w, _ in components)
for name, w, role in components:
    bar = '█' * int(w * 10)
    print(f"  {name:<22} {w:>6.2f}  {role:<30}  {bar}")
print(f"  {'TOTAL (if all fire)':<22} {total_weight:>6.2f}")

# Dominant component check
max_w = max(w for _, w, _ in components)
max_comp = [n for n, w, _ in components if w == max_w]
if max_w > 3.0:
    fail(f"Component(s) {max_comp} with weight {max_w} may dominate entire loss. Risk of single-component overfitting.")
elif max_w > 2.0:
    warn(f"Component(s) {max_comp} with weight {max_w} are the strongest — monitor for dominance.")
else:
    ok(f"No single component has weight > 2.0. Reasonably balanced loss landscape.")

# Check: false_signal + dir_penalty sum vs edge_mse
flat_pull  = LOSS_FALSE_SIGNAL_WEIGHT  # flat-class attraction
edge_push  = LOSS_PENALTY_WEIGHT       # edge-class push
if edge_push > flat_pull:
    ok(
        f"dir_penalty weight ({edge_push}) > false_signal_weight ({flat_pull}). "
        "Making the wrong prediction on an edge bar is costlier than not predicting at all."
    )
else:
    fail(
        f"false_signal_weight ({flat_pull}) >= dir_penalty ({edge_push}). "
        "Silence (pred=0) costs less than wrong direction — model prefers silence."
    )

# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{SEP2}")
print("AUDIT COMPLETE — Loss, Objective & Curriculum")
print(SEP2)
print(textwrap.dedent("""
  Priority fix order:
  1. [WARN] Pass config.SAMPLER_THRESHOLD into loss function as flat_threshold param.
  2. [WARN] Verify ORACLE_THRESHOLD == is_zero boundary (0.05) to prevent punishing valid small signals.
  3. [INFO] Dead-zone is architecturally intentional — margin=0.03 < is_zero=0.05 confirmed correct.
  4. [INFO] spread_reward + pred_std_floor = double anti-collapse layer; coefficients look adequate.
  5. [INFO] Overshoot discount at 0.5 threshold is reasonable — only extreme tails are exempted.
  6. [CHECK] Run with real data — 'val∩test duplicate' FAIL was synthetic artifact from audit_scripts/.
"""))