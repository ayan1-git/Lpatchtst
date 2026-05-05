%%writefile loss.py
# loss.py
import torch
import torch.nn.functional as F
# loss.py
import torch


def continuous_weighted_direction_loss(
    pred, target,
    penalty_weight: float = 0.8,
    false_signal_weight: float = 0.1,
    margin: float = 0.05,
    dispersion_weight: float = 0.05,   # ← halved from 0.1
    bias_weight: float = 0.05,
    _debug: bool = False,
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

    # ── 4. CORRELATION PENALTY ───────────────────────────────────────────────────
    if is_edge.sum() > 1:
        pred_e = pred[is_edge].float()          # ← cast to float32 before stats
        tgt_e  = target[is_edge].float()

        pred_centered = pred_e - pred_e.mean()
        tgt_centered  = (tgt_e - tgt_e.mean()).detach()

        cov      = (pred_centered * tgt_centered).mean()
        pred_std = pred_e.std().clamp(min=0.01)          # ← floor at 0.01, not 1e-8
        tgt_std  = tgt_e.std().clamp(min=0.01).detach()

        corr         = cov / (pred_std * tgt_std)
        corr_penalty = (1.0 - corr).clamp(min=0.0, max=2.0)   # ← explicit bounds

        bias_penalty = (pred_e.mean() - tgt_e.mean().detach()).abs()
    else:
        corr_penalty = torch.tensor(0.0, device=pred.device)
        bias_penalty = torch.tensor(0.0, device=pred.device)

    total = (
        focal_mse
        + false_signal_weight * false_signal_loss
        + penalty_weight      * dir_penalty
        + dispersion_weight   * corr_penalty
        + bias_weight         * bias_penalty
    )

    if _debug:
        print(
            f"  Loss breakdown → "
            f"focal={focal_mse:.4f} | "
            f"false_sig={false_signal_weight * false_signal_loss:.4f} | "
            f"dir={penalty_weight * dir_penalty:.4f} | "
            f"corr={dispersion_weight * corr_penalty:.4f} | "
            f"bias={bias_weight * bias_penalty:.4f}"
        )

    return total