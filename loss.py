import torch


def _safe_std(t: torch.Tensor, min_val: float = 0.01) -> torch.Tensor:
    if t.numel() <= 1:
        return torch.tensor(min_val, device=t.device, dtype=torch.float32)
    s = t.float().std()
    return s.nan_to_num(nan=min_val).clamp(min=min_val)


def continuous_weighted_direction_loss(
    pred, target,
    penalty_weight: float = 0.125,
    false_signal_weight: float = 0.3,
    margin: float = 0.10,
    dispersion_weight: float = 0.1,
    bias_weight: float = 0.1,
    fold_id: int = 99,
    bucket_weights: dict = None,   # {"flat": w1, "moderate": w2, "large": w3}
    epoch: int = 0,                
    curriculum_ramp_epochs: int = 20,  
    moderate_reward_scale: float = 1.0, # 👈 Added parameter: Increase this to scale up the booster (old default was 0.5)
    _debug: bool = False,
):
    """
    Combined loss for bounded predictions (tanh output) and sparse targets.
    """
    pred   = pred.view(-1)
    target = target.view(-1)

    is_zero = (target.abs() < 1e-6)
    is_edge = ~is_zero
    quality = target.abs()

    # —— 0. CURRICULUM RAMP (Regime B only) —————————————————————————————————
    if fold_id not in [0, 1] and fold_id != 99:
        curriculum_ramp = min(1.0, (epoch + 1) / curriculum_ramp_epochs)
    else:
        curriculum_ramp = 1.0
        
    _false_signal_weight = false_signal_weight * curriculum_ramp
    _penalty_weight      = penalty_weight      * curriculum_ramp

    # —— 1. COMPUTE THE SELF-BALANCING GAUSSIAN FLAT REWARD (ALL FOLDS) ———————
    flat_focal_loss = torch.tensor(0.0, device=pred.device)
    if is_zero.any() and is_edge.any():
        pred_z = pred[is_zero]
        flat_reward = torch.exp(-(pred_z ** 2) / (2 * (margin ** 2)))
        
        n_edge = is_edge.sum().float()
        n_zero = is_zero.sum().float()
        alpha  = 0.10  
        w_flat = alpha * (n_edge / (n_zero + 1e-8))
        
        flat_focal_loss = -w_flat * torch.mean(flat_reward)

    # —— 2. COMPUTE MAIN FOLD LOSS REGIMES ——————————————————————————————————
    if fold_id in [0, 1]:
        # —— REGIME A (FOLDS 0 & 1): POSITIVE REWARD ONLY (LONG, SHORT, FLAT) ——
        false_signal_loss = torch.tensor(0.0, device=pred.device)
        dir_penalty       = torch.tensor(0.0, device=pred.device)

        if is_edge.any():
            pred_e    = pred[is_edge]
            tgt_e     = target[is_edge]
            quality_e = quality[is_edge]

            bw_mod   = bucket_weights.get("moderate", 1.0) if bucket_weights else 1.0
            bw_large = bucket_weights.get("large",    1.0) if bucket_weights else 1.0

            large_mask    = quality_e >= 0.5
            moderate_mask = (quality_e > 0.1) & (quality_e < 0.5)
            small_mask    = quality_e <= 0.1

            reward     = pred_e * tgt_e
            pos_reward = torch.relu(reward)

            weighted_reward = quality_e * pos_reward
            if moderate_mask.any():
                weighted_reward = weighted_reward.clone()
                weighted_reward[moderate_mask] = weighted_reward[moderate_mask] * bw_mod
            if large_mask.any():
                weighted_reward = weighted_reward.clone()
                weighted_reward[large_mask] = weighted_reward[large_mask] * bw_large

            focal_mse = -torch.mean(weighted_reward)

            # —— Moderate-trade booster ————————————————————————————————————————
            if moderate_mask.any():
                pred_m = pred_e[moderate_mask]
                tgt_m  = tgt_e[moderate_mask]
                qual_m = quality_e[moderate_mask]
                soft_direction  = torch.tanh(pred_m * tgt_m / margin)
                moderate_reward = torch.relu(soft_direction) * qual_m
                
                # 👈 Swapped hardcoded 0.5 for moderate_reward_scale
                focal_mse = focal_mse - moderate_reward_scale * bw_mod * torch.mean(moderate_reward)
        else:
            focal_mse = torch.tensor(0.0, device=pred.device)
            
        focal_mse = focal_mse + flat_focal_loss

    else:
        # —— REGIME B (FOLDS 2, 3, 4 & PRE-TRAIN): FULL LOSS + FLAT REWARD ———
        bw_mod   = bucket_weights.get("moderate", 1.0) if bucket_weights else 1.0
        bw_large = bucket_weights.get("large",    1.0) if bucket_weights else 1.0
        bw_flat  = bucket_weights.get("flat",     1.0) if bucket_weights else 1.0

        if is_zero.any():
            false_signal = torch.relu(pred[is_zero].abs() - margin)
            edge_frac    = is_edge.float().mean().clamp(min=0.01)
            imbalance    = (1.0 - edge_frac) / (edge_frac + 1e-8)
            false_signal_loss = torch.mean(false_signal) * imbalance.clamp(max=10.0) * bw_flat
        else:
            false_signal_loss = torch.tensor(0.0, device=pred.device)

        if is_edge.any():
            pred_e = pred[is_edge]
            tgt_e  = target[is_edge]

            large_mask_b    = quality[is_edge] >= 0.5
            moderate_mask_b = (quality[is_edge] > 0.1) & (quality[is_edge] < 0.5)

            per_sample_loss = quality[is_edge] * (pred_e - tgt_e) ** 2
            if moderate_mask_b.any():
                per_sample_loss = per_sample_loss.clone()
                per_sample_loss[moderate_mask_b] = per_sample_loss[moderate_mask_b] * bw_mod
            if large_mask_b.any():
                per_sample_loss = per_sample_loss.clone()
                per_sample_loss[large_mask_b] = per_sample_loss[large_mask_b] * bw_large
            focal_mse = torch.mean(per_sample_loss)

            direction   = torch.relu(margin - pred_e * tgt_e)
            move_weight = torch.log1p(quality[is_edge] / margin)
            dir_penalty = torch.mean(direction * move_weight)
        else:
            focal_mse   = torch.tensor(0.0, device=pred.device)
            dir_penalty = torch.tensor(0.0, device=pred.device)

        focal_mse = focal_mse + flat_focal_loss

    # —— 3. DISPERSION (Correlation + Bias) —————————————————————————————————
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
        + _false_signal_weight * false_signal_loss
        + _penalty_weight      * dir_penalty
        + dispersion_weight    * corr_penalty
        + bias_weight          * bias_penalty
    )

    if _debug:
        ps = _safe_std(pred[is_edge].float()) if is_edge.any() else torch.tensor(0.0)
        ts = _safe_std(target[is_edge].float()) if is_edge.any() else torch.tensor(0.0)
        print(
            f"  fold={fold_id} | ep={epoch} | ramp={curriculum_ramp:.2f} | "
            f"focal={focal_mse:.4f} | "
            f"false_sig={_false_signal_weight * false_signal_loss:.4f} | "
            f"dir={_penalty_weight * dir_penalty:.4f} | "
            f"corr={dispersion_weight * corr_penalty:.4f} | "
            f"bias={bias_weight * bias_penalty:.4f} | "
            f"pred_std={ps:.4f} | tgt_std={ts:.4f} | "
            f"bw={bucket_weights}"
        )

    return total


def bit_balance_loss(z_continuous):
    return z_continuous.mean(dim=[0, 1]).pow(2).mean()