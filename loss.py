# loss.py — v6  (short-bias corrected, bias penalty rescaled, dispersion strengthened)
import torch


def _safe_std(t: torch.Tensor, min_val: float = 0.01) -> torch.Tensor:
    if t.numel() <= 1:
        return torch.tensor(min_val, device=t.device, dtype=torch.float32)
    s = t.float().std()
    return s.nan_to_num(nan=min_val).clamp(min=min_val)


def continuous_weighted_direction_loss(
    pred, target,
    penalty_weight: float = 1.0,
    false_signal_weight: float = 0.6,     # ← was 0.3; doubled — false signal rate was 89% on val
    margin: float = 0.05,                  # ← was 0.1; tightened — pred must commit sooner
    dispersion_weight: float = 0.5,        # unchanged — fine
    bias_weight: float = 0.5,             # ← was 0.08; raised 6× — bias_penalty must actually sting
    short_penalty_scale: float = 1.4,     # ← NEW: multiplier on direction penalty when pred<0 & target>0
    _debug: bool = False,
):
    pred   = pred.view(-1)
    target = target.view(-1)

    is_zero = (target.abs() < 1e-6)
    is_edge = ~is_zero
    quality = target.abs()

    # ── 1. FALSE SIGNAL PENALTY ──────────────────────────────────────────────────
    # Root cause: false_signal_weight=0.3 was too low given 89% false signal on val.
    # Imbalance multiplier already in place, so just raising the weight directly scales it.
    if is_zero.any():
        false_signal = torch.relu(pred[is_zero].abs() - margin)
        edge_frac    = is_edge.float().mean().clamp(min=0.01)
        imbalance    = edge_frac / (1.0 - edge_frac + 1e-8)
        false_signal_loss = torch.mean(false_signal) * imbalance
    else:
        false_signal_loss = torch.tensor(0.0, device=pred.device)

    # ── 2. FOCAL + DIRECTION (asymmetric short penalty) ──────────────────────────
    # Root cause: the direction penalty treated Long and Short errors symmetrically.
    # Val data is bullish (target_mean ≈ +0.04 to +0.07). Penalizing wrong-Short
    # harder directly counteracts the sampler's Short-class overrepresentation.
    if is_edge.any():
        pred_e = pred[is_edge]
        tgt_e  = target[is_edge]

        focal_mse = torch.mean(quality[is_edge] * (pred_e - tgt_e) ** 2)

        direction   = torch.relu(margin - pred_e * tgt_e)
        move_weight = torch.log1p(quality[is_edge] / margin)

        # Asymmetric scale: extra penalty when we predict Short (pred<0) but target is Long (tgt>0)
        # short_penalty_scale=1.4 adds 40% penalty on the worst failure mode (missed longs)
        wrong_short_mask = ((pred_e < 0) & (tgt_e > 0)).float()
        asymmetry        = 1.0 + (short_penalty_scale - 1.0) * wrong_short_mask

        dir_penalty = torch.mean(direction * move_weight * asymmetry)
    else:
        focal_mse   = torch.tensor(0.0, device=pred.device)
        dir_penalty = torch.tensor(0.0, device=pred.device)

    # ── 3. DISPERSION (Correlation + Bias) ───────────────────────────────────────
    # Root cause: bias_weight=0.08 contributed ~0.003 to total loss — effectively zero.
    # With pred_mean ≈ −0.4 and target_mean ≈ +0.05, bias_penalty ≈ 0.45.
    # At weight=0.08 → 0.036 contribution. At weight=0.5 → 0.225 contribution.
    # Now it actually competes with the other terms (~0.3–0.6 range).
    if is_edge.sum() > 1:
        pred_e_f = pred[is_edge].float()
        tgt_e_f  = target[is_edge].float()

        pred_c = pred_e_f - pred_e_f.mean()
        tgt_c  = (tgt_e_f - tgt_e_f.mean()).detach()

        cov      = (pred_c * tgt_c).mean()
        pred_std = _safe_std(pred_e_f)
        tgt_std  = _safe_std(tgt_e_f).detach()

        corr         = cov / (pred_std * tgt_std)
        corr_penalty = (1.0 - corr).clamp(min=0.0, max=2.0)

        # Signed bias: penalize negative bias harder than positive (market is bullish regime)
        raw_bias     = pred_e_f.mean() - tgt_e_f.mean().detach()
        # Extra 50% penalty when pred_mean < target_mean (under-predicting / shorting)
        bias_sign_scale = torch.where(raw_bias < 0,
                                      torch.tensor(1.5, device=pred.device),
                                      torch.tensor(1.0, device=pred.device))
        bias_penalty = raw_bias.abs() * bias_sign_scale
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
        ps = _safe_std(pred[is_edge].float()) if is_edge.any() else torch.tensor(0.0)
        ts = _safe_std(target[is_edge].float()) if is_edge.any() else torch.tensor(0.0)
        print(
            f"  focal={focal_mse:.4f} | "
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
    bit_means    = z_continuous.mean(dim=[0, 1])
    balance_loss = (bit_means ** 2).mean()
    return balance_loss