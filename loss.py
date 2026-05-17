# loss.py — v7 (tanh-aware, balanced magnitudes, bug fixes)
import torch


def _safe_std(t: torch.Tensor, min_val: float = 0.01) -> torch.Tensor:
    if t.numel() <= 1:
        return torch.tensor(min_val, device=t.device, dtype=torch.float32)
    s = t.float().std()
    return s.nan_to_num(nan=min_val).clamp(min=min_val)


def continuous_weighted_direction_loss(
    pred, target,
    penalty_weight: float = 0.5,
    false_signal_weight: float = 0.8,
    margin: float = 0.10,
    dispersion_weight: float = 0.1,
    bias_weight: float = 0.1,
    fold_id: int = 99,  # Default to full loss (e.g. for pre-training)
    _debug: bool = False,
):
    """
    Combined loss for bounded predictions (tanh output) and sparse targets.

    Targets come from oracle.py: most are 0.0 (no trade), non-zero values
    are tanh-saturated R-multiples in [-1, 1].  Predictions are also in
    [-1, 1] due to the tanh output activation.

    FOLD LOGIC:
        Regime A (Fold 0, 1): Positive reward only for all three: long, short, and flat.
                  - False signal penalty = 0.
                  - Direction penalty = 0.
                  - Focal MSE replaced by -relu(pred * target) * quality + Gaussian flat reward.
        Regime B (Fold 2, 3, 4 & Pre-train): Full loss function (MSE + False Signal + Direction)
                  + Gaussian flat reward.
    """
    pred   = pred.view(-1)
    target = target.view(-1)

    is_zero = (target.abs() < 1e-6)
    is_edge = ~is_zero
    quality = target.abs()

    # ── 1. COMPUTE THE SELF-BALANCING GAUSSIAN FLAT REWARD (ALL FOLDS) ───────
    flat_focal_loss = torch.tensor(0.0, device=pred.device)
    if is_zero.any() and is_edge.any():
        pred_z = pred[is_zero]
        # Gaussian reward centered at 0 with standard deviation equal to margin
        flat_reward = torch.exp(-(pred_z ** 2) / (2 * (margin ** 2)))
        
        # Dynamic Class-Balancing Weight
        n_edge = is_edge.sum().float()
        n_zero = is_zero.sum().float()
        alpha  = 0.10  # 10% weight relative to edge trades
        w_flat = alpha * (n_edge / (n_zero + 1e-8))
        
        flat_focal_loss = -w_flat * torch.mean(flat_reward)

    # ── 2. COMPUTE MAIN FOLD LOSS REGIMES ────────────────────────────────────
    if fold_id in [0, 1]:
        # ── REGIME A (FOLDS 0 & 1): POSITIVE REWARD ONLY (LONG, SHORT, FLAT) ──
        false_signal_loss = torch.tensor(0.0, device=pred.device)
        dir_penalty       = torch.tensor(0.0, device=pred.device)

        if is_edge.any():
            pred_e = pred[is_edge]
            tgt_e  = target[is_edge]
            # Reward is pred*target if sign matches, else 0
            reward = pred_e * tgt_e
            pos_reward = torch.relu(reward)
            # Scaling by quality to reward strong directional calls
            focal_mse = -torch.mean(quality[is_edge] * pos_reward)
        else:
            focal_mse = torch.tensor(0.0, device=pred.device)
            
        # Add the flat reward to focal_mse
        focal_mse = focal_mse + flat_focal_loss

    else:
        # ── REGIME B (FOLDS 2, 3, 4 & PRE-TRAIN): FULL LOSS + FLAT REWARD ───
        # 1. FALSE SIGNAL PENALTY
        if is_zero.any():
            false_signal = torch.relu(pred[is_zero].abs() - margin)
            edge_frac    = is_edge.float().mean().clamp(min=0.01)
            imbalance    = (1.0 - edge_frac) / (edge_frac + 1e-8)
            false_signal_loss = torch.mean(false_signal) * imbalance.clamp(max=10.0)
        else:
            false_signal_loss = torch.tensor(0.0, device=pred.device)

        # 2. FOCAL MSE + DIRECTION
        if is_edge.any():
            pred_e = pred[is_edge]
            tgt_e  = target[is_edge]
            focal_mse = torch.mean(quality[is_edge] * (pred_e - tgt_e) ** 2)
            direction   = torch.relu(margin - pred_e * tgt_e)
            move_weight = torch.log1p(quality[is_edge] / margin)
            dir_penalty = torch.mean(direction * move_weight)
        else:
            focal_mse   = torch.tensor(0.0, device=pred.device)
            dir_penalty = torch.tensor(0.0, device=pred.device)

        # Add the flat reward alongside full loss terms
        focal_mse = focal_mse + flat_focal_loss

    # ── 3. DISPERSION (Correlation + Bias) ───────────────────────────────────
    # We keep dispersion and bias for ALL folds to prevent prediction collapse.
    if is_edge.sum() > 1:
        pred_e_f = pred[is_edge].float()
        tgt_e_f  = target[is_edge].float()

        pred_c = pred_e_f - pred_e_f.mean()
        tgt_c  = tgt_e_f  - tgt_e_f.mean().detach()

        cov      = (pred_c * tgt_c.detach()).mean()
        pred_std = _safe_std(pred_e_f)
        tgt_std  = _safe_std(tgt_e_f).detach()

        corr         = cov / (pred_std * tgt_std)
        corr_penalty = (1.0 - corr).clamp(min=0.0, max=2.0)

        global_bias = (pred.mean() - target.mean().detach()).abs()
        edge_bias   = (pred_e_f.mean() - tgt_e_f.mean().detach()).abs()
        bias_penalty = 0.5 * global_bias + 0.5 * edge_bias
    else:
        corr_penalty = torch.tensor(0.0, device=pred.device)
        bias_penalty = (pred.mean() - target.mean().detach()).abs()

    total = (
        focal_mse
        + false_signal_weight * false_signal_loss
        + penalty_weight      * dir_penalty
        + dispersion_weight   * corr_penalty
        + bias_weight         * bias_penalty
    )

    if _debug:
        ps = _safe_std(pred[is_edge].float()) if is_edge.any() else torch.tensor(0.0)
        ts = _safe_std(target[is_edge].float()) if is_edge.any() else torch.tensor(0.0)
        print(
            f"  fold={fold_id} | focal={focal_mse:.4f} | "
            f"false_sig={false_signal_weight * false_signal_loss:.4f} | "
            f"dir={penalty_weight * dir_penalty:.4f} | "
            f"corr={dispersion_weight * corr_penalty:.4f} | "
            f"bias={bias_weight * bias_penalty:.4f} | "
            f"pred_std={ps:.4f} | tgt_std={ts:.4f}"
        )

    return total


def bit_balance_loss(z_continuous):
    """
    Penalizes any bit whose mean activation deviates from 0.
    Target: E[sign(z_i)] = 0  ↔  E[z_i] = 0
    """
    return z_continuous.mean(dim=[0, 1]).pow(2).mean()