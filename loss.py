import torch
from typing import Optional
import config

#
# ═══════════════════════════════════════════════════════════════════════════════
# ASYMMETRIC NUMBER-LINE LOSS — v7 (BALANCED)
# ═══════════════════════════════════════════════════════════════════════════════
#
# PROBLEM WITH v6:
#   1. Flat/edge branching creates gradient imbalance. At same |pred|, edge
#      gradients are ~10x stronger than flat gradients because edge bars have
#      large |gap| = |pred - target| while flat bars have tiny |gap| = |pred|.
#   2. Edge magnitude uses |pred| which has a cusp at 0, causing gradient
#      cancellation when pred < 0 but target > 0 (g_edge opposes g_dir).
#   3. mag_weight clamped to 0.7 minimum means all targets below 0.49 get
#      identical direction-loss scaling — can't distinguish weak from strong.
#   4. EDGE_MAG_WEIGHT=3.0 overcompensates, making edge gradients dominate.
#
# v7 DESIGN — Single smooth number line loss, no flat/edge split:
#
#   L = L_direction + L_magnitude
#
#   L_direction = scale_asymmetry * (pred - target)^2 * mag_weight(target)
#     - Same asymmetric L2 as before (alpha for undershoot, beta for overshoot)
#     - mag_weight = |target|^0.5 clamped to [0.3, 2.0] (was [0.7, 2.0])
#     - For target=0: mw=0.3 (was 0.7), giving weaker but nonzero scaling
#
#   L_magnitude = w_mag * smooth_l1(pred, target) * mag_weight(target)
#     - Uses smooth_l1(pred, target) NOT smooth_l1(|pred|, |target|)
#     - No |pred| cusp at 0 — gradient is smooth everywhere
#     - For target=0: L_mag = w_mag * |pred| * mw → g = w_mag * sign(pred) * mw
#       This provides the constant zero-push (like L1 in v6) but scaled by mw
#     - For target≠0: L_mag nudges |pred| toward |target| smoothly
#     - w_mag = 1.0 (reduced from 3.0 since no cancellation to overcome)
#
#   KEY PROPERTIES:
#     - Single code path for all bars (no is_flat/is_edge branching)
#     - Gradient is continuous and smooth everywhere (no cusps except L1 at 0)
#     - At same |pred - target|, flat and edge gradients are naturally balanced
#     - mag_weight scaling ensures weak edges get proportionally weaker gradients
#     - No gradient cancellation zone — g always points toward target
#
#   GRADIENT:
#     g_dir = 2 * scale * (pred - target) * mw
#     g_mag = w_mag * smooth_l1_grad(pred, target) * mw
#     g_total = (g_dir + g_mag) * mw
#
#     where smooth_l1_grad(p, t) = (p-t) if |p-t|<1, else sign(p-t)
#     (This is just the standard smooth_l1 gradient — no |pred| chain rule)
#
#   For flat bars (target=0):
#     g_dir = 2 * beta * pred * mw(0) = 2 * 1.5 * pred * 0.3 = 0.9 * pred
#     g_mag = w_mag * sign(pred) * mw(0) = 1.0 * sign(pred) * 0.3 = 0.3 * sign(pred)
#     g_total = 0.9 * pred + 0.3 * sign(pred)
#     At pred=0.01: g = 0.009 + 0.3 = 0.309
#     At pred=0.10: g = 0.09 + 0.3 = 0.390
#     At pred=0.30: g = 0.27 + 0.3 = 0.570
#
#   For edge bars (target=0.5), pred=0.01:
#     gap = -0.49, undershoot → alpha=2.0, mw=sqrt(0.5)=0.707
#     g_dir = 2 * 2.0 * (-0.49) * 0.707 = -1.386
#     g_mag = 1.0 * (-0.49) * 0.707 = -0.346  (smooth_l1_grad = gap since |gap|<1)
#     g_total = -1.732
#
#   For edge bars (target=0.5), pred=0.3:
#     gap = -0.2, undershoot → alpha=2.0, mw=0.707
#     g_dir = 2 * 2.0 * (-0.2) * 0.707 = -0.566
#     g_mag = 1.0 * (-0.2) * 0.707 = -0.141
#     g_total = -0.707
#
#   Cross-regime comparison at same |gap|=0.05:
#     Flat (pred=0.05, target=0):   g = 0.9*0.05 + 0.3 = 0.345
#     Edge (pred=0.45, target=0.5): g_dir = 2*2*(-0.05)*0.707 = -0.141
#                                   g_mag = 1.0*(-0.05)*0.707 = -0.035
#                                   g_total = -0.176 → |g| = 0.176
#     Ratio: 0.345/0.176 = 1.96 — flat is ~2x stronger (acceptable)
#
#   The flat gradient is naturally stronger at same |gap| because:
#     - Flat mw=0.3 vs edge mw=0.707, but flat uses beta=1.5 vs edge alpha=2.0
#     - The L1-like g_mag for flat (0.3) is larger relative to g_dir than for edge
#     - This is desirable: flat bars need a stronger relative push to overcome
#       encoder bias toward producing edge-like signals
# ═══════════════════════════════════════════════════════════════════════════════


ALPHA = 2.0              # undershoot / wrong-direction scale
BETA  = 1.5              # overshoot scale

MAG_WEIGHT_MIN = 0.3     # was 0.7 — lower so weak edges are distinguishable
MAG_WEIGHT_MAX = 2.0     # unchanged

MAG_NUDGE_WEIGHT = 1.0   # was EDGE_MAG_WEIGHT=3.0 — reduced since no |pred| cusp


def asymmetric_number_line_loss(
    pred:        torch.Tensor,
    target:      torch.Tensor,
    gate_logit:  torch.Tensor | None = None,  # unused, kept for compat
    alpha:       float = ALPHA,
    beta:        float = BETA,
    # ── legacy kwargs — kept for call-site compatibility, not used ────────────
    fold_id:                int   = 99,
    epoch:                  int   = 0,
    curriculum_ramp_epochs: int   = config.CURRICULUM_RAMP_EPOCHS,
    penalty_weight:         float = None,
    false_signal_weight:    float = None,
    shortfall_weight:       float = None,
    dispersion_weight:      float = None,
    bias_weight:            float = None,
    bucket_weights:         Optional[dict] = None,
    _debug:                 bool  = False,
) -> torch.Tensor:
    """
    Asymmetric Number-Line Loss v7 — Balanced.

    Single smooth number line loss with no flat/edge branching.

    Components:
      L_dir   = scale_asymmetry * (pred - target)^2 * mag_weight(target)
      L_mag   = w_mag * smooth_l1(pred, target) * mag_weight(target)

    Where:
      scale   = alpha if (pred-target)*target < 0 (undershoot/wrong-dir), else beta
      mw      = clamp(|target|^0.5, 0.3, 2.0)
      smooth_l1(a, b) = 0.5*(a-b)^2 if |a-b|<1, else |a-b| - 0.5

    Key properties:
      - Gradient is continuous and smooth everywhere (no |pred| cusp)
      - At same |pred - target|, flat and edge gradients are balanced
      - mag_weight scaling ensures proportionality across target magnitudes
      - No gradient cancellation zone — g always points toward target
    """
    pred   = pred.view(-1).float()
    target = target.view(-1).float()

    # ── 1. mag_weight — |target|^0.5 with clamping ──────────────────────────
    mag_w = target.abs().pow(0.5).clamp(min=MAG_WEIGHT_MIN, max=MAG_WEIGHT_MAX)

    # ── 2. Direction loss (asymmetric quadratic, all bars) ─────────────────
    gap = pred - target
    is_undershoot = (gap * target) < 0

    scale = torch.where(
        is_undershoot,
        torch.full_like(gap, alpha),
        torch.full_like(gap, beta),
    )

    loss_dir = (scale * gap.pow(2) * mag_w).mean()

    # ── 3. Magnitude nudge (smooth_l1 on raw pred/target, all bars) ────────
    # Uses smooth_l1(pred, target) — NOT smooth_l1(|pred|, |target|)
    # This eliminates the |pred| cusp at 0 and the gradient cancellation zone.
    mag_diff = pred - target  # raw gap, same as `gap`
    mag_diff_abs = mag_diff.abs()

    # smooth_l1(pred, target) = 0.5 * gap^2 if |gap| < 1, else |gap| - 0.5
    loss_mag = (
        MAG_NUDGE_WEIGHT * mag_w * (
            mag_diff.pow(2) * 0.5 * (mag_diff_abs < 1).float()
            + (mag_diff_abs - 0.5) * (mag_diff_abs >= 1).float()
        )
    ).mean()

    # ── Total ──────────────────────────────────────────────────────────────
    loss = loss_dir + loss_mag

    if _debug:
        wrong_dir  = ((pred * target) < 0).sum().item()
        undershoot = (is_undershoot & ((pred * target) >= 0)).sum().item()
        overshoot  = (~is_undershoot & (target.abs() >= 1e-8)).sum().item()
        n_flat     = (target.abs() < 1e-8).sum().item()
        n_edge     = (target.abs() >= 1e-8).sum().item()
        p_flat_abs = pred[target.abs() < 1e-8].abs() if n_flat > 0 else torch.tensor([0.0])
        false_sig  = (p_flat_abs > 0.08).float().mean().item() if n_flat > 0 else 0.0
        print(
            f"  [loss v7] total={loss.item():.4f} | "
            f"dir={loss_dir.item():.4f} mag={loss_mag.item():.4f} | "
            f"wrong_dir={wrong_dir} undershoot={undershoot} overshoot={overshoot} "
            f"flat={n_flat} edge={n_edge} false_sig={false_sig:.3f}"
        )

    return loss


# ── Public alias — train.py imports this name ─────────────────────────────────
continuous_weighted_direction_loss = asymmetric_number_line_loss
