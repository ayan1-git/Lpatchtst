# loss.py
import torch
import torch.nn.functional as F


def continuous_weighted_direction_loss(
    pred, target,
    penalty_weight: float = 0.5,
    false_signal_weight: float = 0.3,
    margin: float = 0.1,
    dispersion_weight: float = 0.05,
):
    pred   = pred.view(-1)
    target = target.view(-1)

    is_zero = (target.abs() < 1e-6)
    is_edge = ~is_zero
    quality = target.abs()

    # ── 1. FOCAL MSE — weighted by oracle quality ────────────────────────────
    if is_edge.any():
        focal_mse = torch.mean(
            quality[is_edge] * (pred[is_edge] - target[is_edge]) ** 2
        )
    else:
        focal_mse = torch.tensor(0.0, device=pred.device)

    # ── 2. FALSE SIGNAL PENALTY — scaled by model conviction ────────────────
    if is_zero.any():
        false_signal = torch.relu(pred[is_zero].abs() - margin)
        edge_frac    = is_edge.float().mean().clamp(min=0.01)
        imbalance    = edge_frac / (1.0 - edge_frac + 1e-8)
        false_signal_loss = torch.mean(false_signal) * imbalance
    else:
        false_signal_loss = torch.tensor(0.0, device=pred.device)

    # ── 3. DIRECTION PENALTY — scaled by oracle quality ─────────────────────
    if is_edge.any():
        pred_e = pred[is_edge]
        tgt_e  = target[is_edge]
        direction   = torch.relu(margin - pred_e * tgt_e)
        move_weight = torch.log1p(quality[is_edge] / margin)
        dir_penalty = torch.mean(direction * move_weight)
    else:
        dir_penalty = torch.tensor(0.0, device=pred.device)

    # ── 4. DISPERSION — edge bars only ──────────────────────────────────────
    if is_edge.sum() > 1:
        var_ratio          = pred[is_edge].var() / (target[is_edge].var() + 1e-8)
        dispersion_penalty = torch.relu(1.0 - var_ratio)
    else:
        dispersion_penalty = torch.tensor(0.0, device=pred.device)

    return (
        focal_mse
        + false_signal_weight * false_signal_loss
        + penalty_weight      * dir_penalty
        + dispersion_weight   * dispersion_penalty
    )