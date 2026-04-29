# loss.py
import torch


def continuous_weighted_direction_loss(pred, target, penalty_weight: float = 2.0):
    """
    Combined focal-MSE + soft direction penalty loss.

    Changes vs previous version:
    - REMOVED: dead `mse` variable (was computed but never returned).
    - FIXED: direction penalty now uses a margin-based soft formulation
      so it always contributes non-zero gradient even when pred ≈ 0.
      The old relu(-pred*target) equals 0 when pred=0 (sits on the relu kink),
      meaning the penalty term contributed ZERO gradient at initialization
      and the model stalled in a flat-loss region.
    - focal_weight floor kept at 0.05 to preserve gradient on near-zero targets.
    """
    pred   = pred.view(-1)
    target = target.view(-1)

    # --- Focal MSE ---
    # Down-weights near-zero (neutral) targets so the majority class
    # does not dominate the gradient signal.
    focal_weight = torch.abs(target).clamp(min=0.05)
    focal_mse    = torch.mean(focal_weight * (pred - target) ** 2)

    # --- Soft Direction Penalty ---
    # margin=0.1 means: we stop penalising only when pred and target agree
    # AND pred is at least 10% of target's magnitude in the right direction.
    # Guarantees non-zero gradient even when pred=0:
    #   relu(0.1 - 0 * target) = relu(0.1) = 0.1  ← always positive at init
    # so the direction term actively pushes pred away from 0 from epoch 1.
    margin           = 0.1
    direction        = torch.relu(margin - pred * target)
    weighted_penalty = torch.mean(direction * torch.abs(target))

    return focal_mse + penalty_weight * weighted_penalty