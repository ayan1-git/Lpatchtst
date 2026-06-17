import torch
from typing import Optional
import config

#
# ═══════════════════════════════════════════════════════════════════════════════
# ASYMMETRIC NUMBER-LINE LOSS — v6
# ═══════════════════════════════════════════════════════════════════════════════
#
# Gradient investigation revealed the fundamental problem:
#
# The L2 direction loss L = beta * pred^2 * mw has gradient dL/dpred = 2*beta*pred*mw
# At pred=0: gradient = 0. The model has NO force pushing flat bars to zero.
#
# Meanwhile, edge bars at pred=0 have gradient = -alpha * target * mw ≈ -0.84.
# The encoder learns features that push ALL bars toward edge targets.
# Flat bars get caught in this push and accumulate at pred=0.1-0.5.
#
# The flat penalty (v3/v3b) tried to compensate but:
# 1. It only activates above 0.08 (dead zone)
# 2. Its gradient is linear (2*w*(pred-margin)), too weak near the boundary
# 3. It fights against the encoder which is trained by much stronger edge gradients
#
# v5 fix: Add an L1 term for flat bars: L_flat_L1 = lambda * |pred|
# Gradient: dL/dpred = lambda * sign(pred) — CONSTANT at all pred values.
# At pred=0.01: grad = lambda (vs 0.028 for L2 alone)
# At pred=0.0: subgradient includes [-lambda, +lambda] — strong push to zero
#
# This gives the model a constant "push toward zero" on flat bars,
# regardless of the current prediction value.
# ═══════════════════════════════════════════════════════════════════════════════


ALPHA = 2.0              # undershoot / wrong-direction scale
BETA  = 1.5              # overshoot scale (restored from v1)

FLAT_L1_WEIGHT  = 0.2    # L1 penalty on flat bars: lambda * |pred|
FLAT_MARGIN     = 0.08   # dead zone = SAMPLER_THRESHOLD
FLAT_PENALTY_W  = 0.5    # reduced quadratic penalty outside dead zone (safety net)
EDGE_MAG_WEIGHT = 3.0    # magnitude nudge on edge bars (was 0.5 — too weak vs direction loss)


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
    Asymmetric Number-Line Loss v6.

    Components:
      L_dir    = scale_asymmetry * gap^2 * mag_weight       (all bars, L2)
      L_flat1  = lambda * |pred|                             (flat bars only, L1)
      L_flat2  = w * max(0, |pred| - margin)^2              (flat bars only, safety net)
      L_edge   = w_edge * smooth_l1(|pred|, |target|)       (edge bars only)

    The L1 term is the key fix: it provides a constant gradient pushing
    flat bars toward zero at ALL prediction values, including near zero
    where the L2 gradient vanishes.
    """
    pred   = pred.view(-1).float()
    target = target.view(-1).float()

    # ── Masking ────────────────────────────────────────────────────────────
    _eps    = 1e-8
    is_flat = target.abs() < _eps
    is_edge = ~is_flat

    # ── 1. Asymmetric quadratic (direction loss, all bars) ────────────────
    gap = pred - target
    is_undershoot = (gap * target) < 0

    scale = torch.where(
        is_undershoot,
        torch.full_like(gap, alpha),
        torch.full_like(gap, beta),
    )

    mag_weight = target.abs().pow(0.5).clamp(min=0.7, max=2.0)
    loss_dir = (scale * gap.pow(2) * mag_weight).mean()

    # ── 2. Flat-bar L1 penalty (PRIMARY zero-push) ───────────────────────
    # L1 = lambda * |pred| → gradient = lambda * sign(pred) (constant!)
    # This is the key fix: constant gradient at ALL pred values
    loss_flat_l1 = torch.tensor(0.0, device=pred.device)
    n_flat = 0
    if is_flat.any():
        loss_flat_l1 = FLAT_L1_WEIGHT * pred[is_flat].abs().mean()
        n_flat = is_flat.sum().item()

    # ── 3. Flat-bar quadratic penalty (safety net above dead zone) ────────
    loss_flat_q = torch.tensor(0.0, device=pred.device)
    if is_flat.any():
        p_flat  = pred[is_flat].abs()
        over    = (p_flat - FLAT_MARGIN).clamp(min=0.0)
        loss_flat_q = FLAT_PENALTY_W * over.pow(2).mean()

    # ── 4. Edge-magnitude auxiliary loss ──────────────────────────────────
    loss_edge = torch.tensor(0.0, device=pred.device)
    n_edge = 0
    if is_edge.any():
        p_e = pred[is_edge].abs()
        t_e = target[is_edge].abs()
        diff = (p_e - t_e).abs()
        loss_edge = (
            EDGE_MAG_WEIGHT * (diff.pow(2) * 0.5 * (diff < 1).float()
                               + (diff - 0.5) * (diff >= 1).float())
        ).mean()
        n_edge = is_edge.sum().item()

    # ── Total ──────────────────────────────────────────────────────────────
    loss = loss_dir + loss_flat_l1 + loss_flat_q + loss_edge

    if _debug:
        wrong_dir  = ((pred * target) < 0).sum().item()
        undershoot = (is_undershoot & ((pred * target) >= 0)).sum().item()
        overshoot  = (~is_undershoot & is_edge).sum().item()
        p_zero_abs = pred[is_flat].abs() if is_flat.any() else torch.tensor([0.0])
        false_sig  = (p_zero_abs > FLAT_MARGIN).float().mean().item() if is_flat.any() else 0.0
        print(
            f"  [loss v6] total={loss.item():.4f} | "
            f"dir={loss_dir.item():.4f} flat_L1={loss_flat_l1.item():.4f} "
            f"flat_q={loss_flat_q.item():.4f} edge_mag={loss_edge.item():.4f} | "
            f"wrong_dir={wrong_dir} undershoot={undershoot} overshoot={overshoot} "
            f"flat={n_flat} edge={n_edge} false_sig={false_sig:.3f}"
        )

    return loss


# ── Public alias — train.py imports this name ─────────────────────────────────
continuous_weighted_direction_loss = asymmetric_number_line_loss
