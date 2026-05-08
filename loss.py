%%writefile loss.py
# loss.py — v4 (pred-normalized direction/focal, pure corr dispersion)
import torch


def continuous_weighted_direction_loss(
    pred, target,
    penalty_weight: float = 1.0,
    false_signal_weight: float = 0.1,
    margin: float = 0.1,
    dispersion_weight: float = 0.15,
    bias_weight: float = 0.03,
    _debug: bool = False,
):
    pred   = pred.view(-1)
    target = target.view(-1)

    is_zero = (target.abs() < 1e-6)
    is_edge = ~is_zero
    quality = target.abs()

    # ── 1. FALSE SIGNAL PENALTY (unchanged) ─────────────────────────────────
    if is_zero.any():
        false_signal = torch.relu(pred[is_zero].abs() - margin)
        edge_frac    = is_edge.float().mean().clamp(min=0.01)
        imbalance    = edge_frac / (1.0 - edge_frac + 1e-8)
        false_signal_loss = torch.mean(false_signal) * imbalance
    else:
        false_signal_loss = torch.tensor(0.0, device=pred.device)

    # ── 2. SCALE-NORMALIZED FOCAL + DIRECTION ────────────────────────────────
    # Root cause of v1-v3 instability: pred_std and dir_penalty fight each
    # other. Fix: rescale pred to match tgt_std before computing focal/dir.
    # This decouples "spread" from "direction" completely.
    if is_edge.any():
        pred_e = pred[is_edge]
        tgt_e  = target[is_edge]

        tgt_std_val  = tgt_e.float().std().clamp(min=0.01).detach()
        pred_std_val = pred_e.float().std().clamp(min=0.01)
        scale        = (tgt_std_val / pred_std_val).detach()  # stop gradient on scale

        pred_scaled = pred_e * scale   # rescale predictions to target spread

        # Focal MSE on rescaled predictions
        focal_mse = torch.mean(
            quality[is_edge] * (pred_scaled - tgt_e) ** 2
        )

        # Direction penalty on rescaled predictions
        direction   = torch.relu(margin - pred_scaled * tgt_e)
        move_weight = torch.log1p(quality[is_edge] / margin)
        dir_penalty = torch.mean(direction * move_weight)
    else:
        focal_mse   = torch.tensor(0.0, device=pred.device)
        dir_penalty = torch.tensor(0.0, device=pred.device)

    # ── 3. DISPERSION — pure correlation penalty ─────────────────────────────
    # No std wrestling. Just penalize low correlation.
    # std_penalty caused 3 rounds of instability; removing it entirely.
    if is_edge.sum() > 1:
        pred_e_f = pred[is_edge].float()
        tgt_e_f  = target[is_edge].float()

        pred_c = pred_e_f - pred_e_f.mean()
        tgt_c  = (tgt_e_f - tgt_e_f.mean()).detach()

        cov      = (pred_c * tgt_c).mean()
        pred_std = pred_e_f.std().clamp(min=0.01)
        tgt_std  = tgt_e_f.std().clamp(min=0.01).detach()

        corr         = cov / (pred_std * tgt_std)
        corr_penalty = (1.0 - corr).clamp(min=0.0, max=2.0)

        bias_penalty = (pred_e_f.mean() - tgt_e_f.mean().detach()).abs()
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
        ps = pred[is_edge].float().std() if is_edge.any() else torch.tensor(0.0)
        ts = target[is_edge].float().std() if is_edge.any() else torch.tensor(0.0)
        print(
            f"  Loss breakdown → "
            f"focal={focal_mse:.4f} | "
            f"false_sig={false_signal_weight * false_signal_loss:.4f} | "
            f"dir={penalty_weight * dir_penalty:.4f} | "
            f"corr={dispersion_weight * corr_penalty:.4f} | "
            f"bias={bias_weight * bias_penalty:.4f} | "
            f"pred_std={ps:.4f} | "
            f"tgt_std={ts:.4f} | "
            f"scale={scale.item():.4f}"
        )

    return total