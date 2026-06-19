import torch
from typing import Optional
import config

#
# ═══════════════════════════════════════════════════════════════════════════════
# ASYMMETRIC NUMBER-LINE LOSS — v9 (BALANCED)
# ═══════════════════════════════════════════════════════════════════════════════
#
# LESSON FROM v7 FAILURE:
#   smooth_l1(pred, 0) = 0.5*pred^2 (quadratic), NOT |pred| (L1).
#   Gradient = pred (vanishes at 0), NOT sign(pred) (constant).
#   This destroyed the zero-push that made v6 work.
#   Result: false_sig_rate=98.7%, model never learns flat bars.
#
# LESSON FROM v8 TRAINING (epochs 1-41):
#   v8 achieved false_sig=28% (down from 98.7%), flat_decisions=65%, precision=65-69%.
#   But mag_calibration stuck at 0.36-0.39x (severe under-prediction).
#   Root cause: mag_weight_min=0.3 scales weak-edge gradients too aggressively.
#   Weak edges (target=0.1) get mw=0.3 vs strong edges (target=0.5) get mw=0.707.
#   The 2.36x gradient ratio causes the model to under-predict magnitudes.
#
# v9 DESIGN — Calibrated weights based on v8 training analysis:
#
#   FLAT BARS (target ≈ 0):
#     L_dir   = beta * pred^2 * mw(0)           → g_dir = 2*beta*pred*mw
#     L_push  = push_w * |pred| * mw(0)         → g_push = push_w * sign(pred) * mw  (CONSTANT!)
#     Total g = 2*beta*pred*mw + push_w*sign(pred)*mw
#
#     The L1 push term is ESSENTIAL — it provides constant gradient toward zero
#     regardless of pred magnitude. This is what v6 had and v7 lost.
#     push_w is scaled by mw so flat bars get proportionally weaker push
#     (just like edge bars get proportionally weaker gradients for small targets).
#
#   EDGE BARS (target ≠ 0):
#     L_dir   = scale * (pred-target)^2 * mw(t)  → g_dir = 2*scale*gap*mw
#     L_nudge = nudge_w * smooth_l1(pred, target) * mw(t)
#             → g_nudge = nudge_w * smooth_l1_grad(pred, target) * mw
#
#     KEY FIX vs v6: smooth_l1(pred, target) NOT smooth_l1(|pred|, |target|)
#     This eliminates the |pred| cusp at 0 and the gradient cancellation zone.
#     For edge bars, pred and target have the same sign (the model learns this),
#     so smooth_l1(pred, target) ≈ smooth_l1(|pred|, |target|) in practice,
#     but without the pathological cusp.
#
#   WEIGHTS (calibrated from v8 training analysis):
#     mw_min = 0.5 (was 0.3 in v8, was 0.7 in v6)
#       — Compromise: weak edges get more gradient than v8, less than v6
#       — target=0.1: mw=0.5 (was 0.3), target=0.5: mw=0.707 (unchanged)
#       — Ratio: 0.707/0.5 = 1.41x (was 2.36x in v8, was 1.0x in v6)
#     push_w = 2.0 (was 1.5 in v8, was 0.2 in v6)
#       — Stronger zero-push to further reduce false sig rate
#       — Scaled by mw(0)=0.5, effective push = 2.0*0.5 = 1.0
#     nudge_w = 1.5 (was 1.0 in v8, was 3.0 in v6)
#       — Slightly stronger edge magnitude nudge to improve mag calibration
#
#   GRADIENT BALANCE:
#     Flat at pred=0.03: g = 2*1.5*0.03*0.5 + 2.0*1*0.5 = 0.045 + 1.0 = 1.045
#     Edge at pred=0.03, target=0.5: g = 2*2*(-0.47)*0.707 + 1.5*(-0.47)*0.707 = -1.329 - 0.498 = -1.827
#     Ratio at same |pred|: 1.045/1.827 = 0.57 — flat is weaker but same order of magnitude
#
#     At same |gap|=0.05:
#       Flat (pred=0.05): g = 2*1.5*0.05*0.5 + 2.0*0.5 = 0.075 + 1.0 = 1.075
#       Edge (pred=0.45, target=0.5): g = 2*2*(-0.05)*0.707 + 1.5*(-0.05)*0.707 = -0.141 - 0.053 = -0.194
#       Ratio: 1.075/0.194 = 5.5 — flat is ~5.5x stronger (desirable: flat needs more push)
#
#   The flat gradient is intentionally stronger at same |gap| because:
#     1. Flat bars need a stronger relative push to overcome encoder bias
#     2. The encoder is trained primarily by edge bars and produces edge-like features
#     3. The L1 push (1.0) dominates at small pred, ensuring convergence to zero
# ═══════════════════════════════════════════════════════════════════════════════


ALPHA = 2.0              # undershoot / wrong-direction scale
BETA  = 1.5              # overshoot scale

MAG_WEIGHT_MIN = 0.5     # was 0.3 in v8, was 0.7 in v6 — calibrated from training
MAG_WEIGHT_MAX = 2.0     # unchanged

FLAT_PUSH_WEIGHT = 2.0   # was 1.5 in v8, was 0.2 in v6 — stronger zero-push
                         # Scaled by mw(0)=0.5, effective push = 2.0*0.5 = 1.0

EDGE_NUDGE_WEIGHT = 1.5  # was 1.0 in v8, was 3.0 in v6 — balanced magnitude nudge


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
    Asymmetric Number-Line Loss v9 — Balanced.

    Flat bars (target≈0):
      L_dir  = beta * pred^2 * mw(0)
      L_push = push_w * |pred| * mw(0)     ← L1, constant gradient toward zero

    Edge bars (target≠0):
      L_dir   = scale * (pred-target)^2 * mw(target)
      L_nudge = nudge_w * smooth_l1(pred, target) * mw(target)
              ← smooth_l1 on RAW pred/target (no |pred| cusp)

    Key properties:
      - Flat: constant L1 push (doesn't vanish at small pred)
      - Edge: smooth_l1(pred, target) eliminates |pred| cusp and cancellation
      - mag_weight scaling ensures proportionality across target magnitudes
      - Weights calibrated from v8 training analysis for balanced gradients
    """
    pred   = pred.view(-1).float()
    target = target.view(-1).float()

    # ── Masking ────────────────────────────────────────────────────────────
    _eps    = 1e-8
    is_flat = target.abs() < _eps
    is_edge = ~is_flat

    # ── mag_weight — |target|^0.5 with clamping ────────────────────────────
    mag_w = target.abs().pow(0.5).clamp(min=MAG_WEIGHT_MIN, max=MAG_WEIGHT_MAX)

    # ── 1. Direction loss (asymmetric quadratic, all bars) ─────────────────
    gap = pred - target
    is_undershoot = (gap * target) < 0

    scale = torch.where(
        is_undershoot,
        torch.full_like(gap, alpha),
        torch.full_like(gap, beta),
    )

    loss_dir = (scale * gap.pow(2) * mag_w).mean()

    # ── 2. Flat bars: L1 push toward zero ──────────────────────────────────
    # L_push = push_w * |pred| * mw(0)
    # g_push = push_w * sign(pred) * mw(0) — CONSTANT, doesn't vanish at 0
    loss_flat_push = torch.tensor(0.0, device=pred.device)
    n_flat = 0
    if is_flat.any():
        loss_flat_push = FLAT_PUSH_WEIGHT * pred[is_flat].abs() * mag_w[is_flat]
        loss_flat_push = loss_flat_push.mean()
        n_flat = is_flat.sum().item()

    # ── 3. Edge bars: smooth_l1 magnitude nudge (no |pred| cusp) ──────────
    # Uses smooth_l1(pred, target) — NOT smooth_l1(|pred|, |target|)
    # This eliminates the |pred| cusp at 0 and the gradient cancellation zone.
    loss_edge_nudge = torch.tensor(0.0, device=pred.device)
    n_edge = 0
    if is_edge.any():
        p_e = pred[is_edge]
        t_e = target[is_edge]
        w_e = mag_w[is_edge]

        # smooth_l1(pred, target) using raw values (no abs on pred/target)
        diff = p_e - t_e
        diff_abs = diff.abs()
        loss_edge_nudge = (
            EDGE_NUDGE_WEIGHT * w_e * (
                diff.pow(2) * 0.5 * (diff_abs < 1).float()
                + (diff_abs - 0.5) * (diff_abs >= 1).float()
            )
        ).mean()
        n_edge = is_edge.sum().item()

    # ── Total ──────────────────────────────────────────────────────────────
    loss = loss_dir + loss_flat_push + loss_edge_nudge

    if _debug:
        wrong_dir  = ((pred * target) < 0).sum().item()
        undershoot = (is_undershoot & ((pred * target) >= 0)).sum().item()
        overshoot  = (~is_undershoot & is_edge).sum().item()
        p_flat_abs = pred[is_flat].abs() if is_flat.any() else torch.tensor([0.0])
        false_sig  = (p_flat_abs > 0.08).float().mean().item() if is_flat.any() else 0.0
        print(
            f"  [loss v9] total={loss.item():.4f} | "
            f"dir={loss_dir.item():.4f} flat_push={loss_flat_push.item():.4f} "
            f"edge_nudge={loss_edge_nudge.item():.4f} | "
            f"wrong_dir={wrong_dir} undershoot={undershoot} overshoot={overshoot} "
            f"flat={n_flat} edge={n_edge} false_sig={false_sig:.3f}"
        )

    return loss


# ── Public alias — train.py imports this name ─────────────────────────────────
continuous_weighted_direction_loss = asymmetric_number_line_loss
