import torch
from typing import Optional
import config


# ═══════════════════════════════════════════════════════════════════════════════
# ASYMMETRIC NUMBER-LINE LOSS
# ═══════════════════════════════════════════════════════════════════════════════
#
# For any target (including zero), loss is a function of one quantity:
#       gap = pred - tgt
#
# Loss is zero when pred == tgt (perfect prediction).
# On either side of tgt the penalty grows quadratically at different rates:
#
#   Undershoot side  (gap * tgt < 0):  scale = ALPHA  (steeper)
#   Overshoot side   (gap * tgt >= 0): scale = BETA   (gentler)
#
# The condition (gap * tgt < 0) works for all targets:
#
#   tgt = +0.5, pred = +0.1  → gap=-0.4, gap*tgt=-0.2 < 0 → alpha  (undershoot)
#   tgt = +0.5, pred = +0.9  → gap=+0.4, gap*tgt=+0.2 > 0 → beta   (overshoot)
#   tgt = +0.5, pred = -0.3  → gap=-0.8, gap*tgt=-0.4 < 0 → alpha  (wrong dir, large gap)
#   tgt = -0.5, pred = -0.1  → gap=+0.4, gap*tgt=-0.2 < 0 → alpha  (undershoot)
#   tgt = -0.5, pred = -0.9  → gap=-0.4, gap*tgt=+0.2 > 0 → beta   (overshoot)
#   tgt =  0.0, pred = +0.3  → gap=+0.3, gap*tgt= 0.0 → 0 → beta   (symmetric V at zero)
#   tgt =  0.0, pred = -0.3  → gap=-0.3, gap*tgt= 0.0 → 0 → beta   (symmetric V at zero)
#
# Wrong-direction is NOT a separate case — it simply lands on the undershoot
# side with a very large gap², earning a proportionally large penalty naturally.
#
# tgt == 0 bars get a symmetric beta-scaled V pinned at pred=0. No separate
# false-signal term is needed.
#
# ─────────────────────────────────────────────────────────────────────────────
# ALPHA / BETA — tuning guide
# ─────────────────────────────────────────────────────────────────────────────
# ALPHA (default 2.0)
#   Steepness on the undershoot + wrong-direction side.
#   Raise if: model chronically undershoots, or wrong-direction errors persist.
#   Lower if: model is too aggressive and overshoots to compensate.
#
# BETA (default 0.8)
#   Steepness on the overshoot side.
#   Raise if: Long-bias explodes across folds (model overshoots freely).
#   Lower if: model is too conservative and undershoots on strong signals.
#
#   ALPHA/BETA ratio = 2.5x → undershoot penalised 2.5x harder than overshoot.
#   Set ALPHA == BETA to recover standard MSE.
#
# mag_weight = |tgt|^0.5 clamped to min 0.3
#   Larger targets pull harder on the optimizer (sqrt compression).
#   clamp(0.3) gives flat/small-target bars a minimum gradient voice
#   so the model is still pushed toward zero on no-trade bars.
# ═══════════════════════════════════════════════════════════════════════════════


ALPHA = 2.0   # undershoot + wrong-direction scale
BETA  = 1.5   # overshoot scale


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
    Unified asymmetric number-line loss.

    Parameters
    ----------
    pred   : raw model output, shape (N,) or (N, 1)
    target : oracle labels,    shape (N,) or (N, 1)
    alpha  : undershoot / wrong-direction scale  (default 2.0)
    beta   : overshoot scale                     (default 0.8)

    All other kwargs are legacy call-site arguments and are ignored.
    """
    pred   = pred.view(-1).float()
    target = target.view(-1).float()

    gap = pred - target

    # Undershoot side: pred has not yet reached tgt on the number line.
    # gap * tgt < 0 captures this for both positive and negative targets.
    # gap * tgt == 0 when tgt == 0 → always beta (symmetric V at zero). 
    is_undershoot_side = (gap * target) < 0

    scale = torch.where(
        is_undershoot_side,
        torch.full_like(gap, alpha),
        torch.full_like(gap, beta),
    )

    # Magnitude weight: continuous importance scaling by target size.
    # sqrt compression: |tgt|=1.0 → weight=1.0, |tgt|=0.1 → weight=0.32.
    # clamp(min=0.3): flat bars (tgt=0) still receive a gradient push toward zero.
    mag_weight = target.abs().pow(0.5).clamp(min=0.5)

    loss = (scale * gap.pow(2) * mag_weight).mean()

    if _debug:
        wrong_dir  = ((pred * target) < 0).sum().item()
        undershoot = (is_undershoot_side & ((pred * target) >= 0)).sum().item()
        overshoot  = (~is_undershoot_side & (target != 0)).sum().item()
        flat       = (target == 0.0).sum().item()
        print(
            f"  [loss] total={loss.item():.4f} | "
            f"wrong_dir={wrong_dir} undershoot={undershoot} "
            f"overshoot={overshoot} flat={flat}"
        )

    return loss


# ── Public alias — train.py imports this name ─────────────────────────────────
continuous_weighted_direction_loss = asymmetric_number_line_loss