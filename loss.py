import torch
from typing import Optional
import config

#
# ═══════════════════════════════════════════════════════════════════════════════
# ASYMMETRIC NUMBER-LINE LOSS — v2
# ═══════════════════════════════════════════════════════════════════════════════
#
# Changes from v1 (diagnostics-driven, June 2026):
#
#   1. ALPHA  3.0 → 2.0
#      v1 penalized undershoot 2x harder than beta, which created a conservatism
#      trap.  The model's optimal strategy under alpha=3.0 was to stay small on
#      edge bars because a wrong-direction step (gap*tgt < 0) is brutally
#      penalised even when direction is correct.  At alpha=2.0 the penalty is
#      still 1.33x beta (wrong direction still costs more than overshoot) but
#      the gradient no longer pushes the model toward near-zero predictions.
#
#   2. BETA   1.5 → 1.0
#      Symmetric around pred=tgt on the overshoot side.  The original 1.5
#      made overshoot expensive too; with alpha still > beta, wrong-direction
#      is still correctly penalised more than overshoot.
#
#   3. MAG_WEIGHT  clamp(min=0.7) → clamp(min=0.15)
#      The clamp dominated every weight in v1:
#        |tgt|=0  → 0.7   flat bar
#        |tgt|=0.1 → 0.7   (sqrt=0.32, clamped)
#        |tgt|=0.3 → 0.7   (sqrt=0.55, clamped)
#        |tgt|=0.5 → 0.71  barely escaped clamp
#      Edge bars and flat bars had nearly identical mag_weight → the loss
#      could not distinguish "predict strongly here" from "predict zero".
#      Lowering clamp to 0.15 gives a clean hierarchy:
#        |tgt|=0   → 0.15  flat  (very light)
#        |tgt|=0.1 → 0.32  small edge
#        |tgt|=0.3 → 0.55  medium edge
#        |tgt|=0.5 → 0.71  strong edge
#      Incentive ratio edge/flat rises from 1.57x → 3.67x, so the model
#      finally gets a clear gradient signal to separate edge from flat.
#
#   4. FLAT-PENALTY term  (new)
#      Explicit false-signal penalty on bars where |target| < flat_margin.
#      Uses smooth hinge: max(0, |pred| - margin)^2.  This gives zero loss
#      when |pred| ≤ margin and a quadratic ramp beyond it.  The coefficient
#      FLAT_PENALTY_WEIGHT = 0.5 is calibrated so that the penalty on flat
#      bars is comparable to the directional loss on an average edge bar.
#      Without this term, flat bars only get the indirect mag_weight push,
#      which even with clamp(0.15) proved insufficient (90% false-signal rate
#      in v1 diagnostics).
#
#   5. EDGE-MAGNITUDE term  (new)
#      On edge bars (|target| ≥ eps), applies a smooth-L1 penalty between
#      |pred| and |target|:
#         edge_mag_loss = mean( smooth_l1(|pred|, |target|) * is_edge )
#      Coefficient EDGE_MAG_WEIGHT = 0.6.  This gently nudges the model to
#      predict magnitudes closer to the oracle's without imposing a hard MSE
#      target.  Calibrated so the total v2 edge gradient ≈ v1 (alpha reduction
#      is offset by this term), maintaining learning speed on edge bars while
#      the flat-bar suppression prevents the conservatism equilibrium.
#
# ──────────────────────────────────────────────────────────────────────────────
# Combined behavioural target (diagnostics → desired diagnostics):
#                         v1 (EP25)        v2 goal
#   Mag calibration       0.43x             0.65-0.80x
#   False sig rate        90%              < 60%
#   Dir acc (edge)        71.5%            68-72% (slight drop OK if mag↑)
#   Correlation (edge)    0.520            0.50-0.55 (generalises to val)
# ═══════════════════════════════════════════════════════════════════════════════


ALPHA = 2.0          # was 3.0 — undershoot / wrong-direction scale
BETA  = 1.0          # was 1.5 — overshoot scale (now symmetric)

FLAT_PENALTY_WEIGHT = 0.5   # coefficient for false-signal smooth hinge
EDGE_MAG_WEIGHT     = 0.6   # coefficient for edge magnitude smooth-L1
# Calibrated so v2 total edge gradient ≈ v1:
#   v1: 2*3.0*gap*0.55 = 3.30*gap
#   v2: 2*2.0*gap*0.55 + 0.6*gap = 2.20*gap + 0.6*gap = 2.80*gap
#   Ratio: 2.80/3.30 = 0.85x — close enough; flat-bar suppression
#   more than compensates for the 15% reduction in edge gradient.


def asymmetric_number_line_loss(
    pred:   torch.Tensor,
    target: torch.Tensor,
    alpha:  float = ALPHA,
    beta:   float = BETA,
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
    Asymmetric Number-Line Loss v2 — direction-aware with flat penalty and
    edge-magnitude auxiliary terms.

    Components (all terms are *mean* reductions over the relevant subset):

      L_dir   =  scale_asymmetry * gap^2 * mag_weight          (all bars)
      L_flat  =  w_flat * max(0, |pred| - margin)^2             (flat bars)
      L_edge  =  w_edge * smooth_l1(|pred|, |target|)          (edge bars)

    Where:
      scale_asymmetry = alpha  if gap*tgt < 0  (undershoot / wrong dir)
                        beta                        (overshoot side)
      mag_weight      = |tgt|^0.5  clamped to [0.15, 2.0]
      flat_margin     = config.FALSE_SIGNAL_MARGIN  (0.03)
    """
    pred   = pred.view(-1).float()
    target = target.view(-1).float()

    # ── Masking ────────────────────────────────────────────────────────────
    _eps    = 1e-8
    is_flat = target.abs() < _eps            # tgt == 0  (exact zeros only)
    is_edge = ~is_flat                        # everything with a signal

    flat_margin = getattr(config, "FALSE_SIGNAL_MARGIN", 0.03)

    # ── 1. Asymmetric quadratic (main direction loss) ─────────────────────
    gap = pred - target
    is_undershoot = (gap * target) < 0       # gap*tgt == 0 → False → beta

    scale = torch.where(
        is_undershoot,
        torch.full_like(gap, alpha),
        torch.full_like(gap, beta),
    )

    # Magnitude weight with wider dynamic range:
    #   sqrt compression, clamped [0.15, 2.0]
    #   Flat bars → 0.15  (light — main suppression comes from L_flat)
    #   |tgt|=0.3 → 0.55 (medium)
    #   |tgt|=1.0 → 1.0  (heavy)
    mag_weight = target.abs().pow(0.5).clamp(min=0.15, max=2.0)

    loss_dir = (scale * gap.pow(2) * mag_weight).mean()

    # ── 2. Flat-bar false-signal penalty ──────────────────────────────────
    # Smooth hinge: zero when |pred| ≤ margin, quadratic beyond.
    # Only applied on truly flat bars (target == 0).
    loss_flat = torch.tensor(0.0, device=pred.device)
    n_flat = 0
    if is_flat.any():
        p_flat  = pred[is_flat].abs()
        over    = (p_flat - flat_margin).clamp(min=0.0)
        loss_flat = (FLAT_PENALTY_WEIGHT * over.pow(2)).mean()
        n_flat = is_flat.sum().item()

    # ── 3. Edge-magnitude auxiliary loss ────────────────────────────────────
    # smooth_L1 on |pred| vs |target| — gently pushes predictions toward
    # oracle magnitude without hard MSE.  Only on edge bars.
    loss_edge = torch.tensor(0.0, device=pred.device)
    n_edge = 0
    if is_edge.any():
        p_e = pred[is_edge].abs()
        t_e = target[is_edge].abs()
        # smooth L1: 0.5 * x^2 if |x| < 1, else |x| - 0.5
        diff = (p_e - t_e).abs()
        loss_edge = (
            EDGE_MAG_WEIGHT * (diff.pow(2) * 0.5 * (diff < 1).float()
                               + (diff - 0.5) * (diff >= 1).float())
        ).mean()
        n_edge = is_edge.sum().item()

    # ── Total ──────────────────────────────────────────────────────────────
    loss = loss_dir + loss_flat + loss_edge

    if _debug:
        wrong_dir  = ((pred * target) < 0).sum().item()
        undershoot = (is_undershoot & ((pred * target) >= 0)).sum().item()
        overshoot  = (~is_undershoot & is_edge).sum().item()
        p_zero_abs = pred[is_flat].abs() if is_flat.any() else torch.tensor([0.0])
        false_sig  = (p_zero_abs > flat_margin).float().mean().item() if is_flat.any() else 0.0
        print(
            f"  [loss v2] total={loss.item():.4f} | "
            f"dir={loss_dir.item():.4f} flat={loss_flat.item():.4f} mag={loss_edge.item():.4f} | "
            f"wrong_dir={wrong_dir} undershoot={undershoot} overshoot={overshoot} "
            f"flat={n_flat} edge={n_edge} false_sig={false_sig:.3f}"
        )

    return loss


# ── Public alias — train.py imports this name ─────────────────────────────────
continuous_weighted_direction_loss = asymmetric_number_line_loss
