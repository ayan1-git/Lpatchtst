import torch
import torch.nn.functional as F

def _safe_std(t: torch.Tensor, min_val: float = 0.01) -> torch.Tensor:
    if t.numel() <= 1:
        return torch.tensor(min_val, device=t.device, dtype=torch.float32)
    s = t.float().std()
    return s.nan_to_num(nan=min_val).clamp(min=min_val)

def continuous_weighted_direction_loss(
    pred, target,
    penalty_weight: float = 1.50,       # INCREASED from 0.25: Direction is paramount
    false_signal_weight: float = 0.85,  # DECREASED from 1.0: Soften the fear of flat markets
    margin: float = 0.10,
    dispersion_weight: float = 1.0,     # INCREASED from 0.3: Force distribution matching
    bias_weight: float = 0.75,          # INCREASED from 0.1: Force prediction mean to match target mean
    fold_id: int = 99,
    bucket_weights: dict = None,   
    epoch: int = 0,                
    curriculum_ramp_epochs: int = 20,  
    _debug: bool = False,
):
    pred   = pred.view(-1)
    target = target.view(-1)

    is_zero = (target.abs() < 1e-6)
    is_edge = ~is_zero
    quality = target.abs()

    # 1. CURRICULUM RAMP
    ramp_factor = min(1.0, (epoch + 1) / curriculum_ramp_epochs) if fold_id != 99 else 1.0
    _fs_weight = false_signal_weight * ramp_factor
    _pen_weight = penalty_weight * ramp_factor

    # 2. FALSE SIGNAL LOSS (Flat Targets)
    false_signal_loss = torch.tensor(0.0, device=pred.device)
    if is_zero.any():
        bw_flat = bucket_weights.get("flat", 1.0) if bucket_weights else 1.0
        raw_abs_error = pred[is_zero].abs()
        
        # ALIGN WITH CONFIG: Set dead-zone exactly to SAMPLER_THRESHOLD (0.08)
        false_signal_loss = torch.mean(torch.relu(raw_abs_error - 0.08)) * bw_flat

    # 3. EDGE TARGET LOSS
    edge_mse          = torch.tensor(0.0, device=pred.device)
    overshoot_loss    = torch.tensor(0.0, device=pred.device)
    dir_penalty       = torch.tensor(0.0, device=pred.device)

    if is_edge.any():
        pred_e = pred[is_edge]
        tgt_e  = target[is_edge]
        qual_e = quality[is_edge]

        bw_mod   = bucket_weights.get("moderate", 1.0) if bucket_weights else 1.0
        bw_large = bucket_weights.get("large",    1.0) if bucket_weights else 1.0

        moderate_mask = (qual_e > 0.1) & (qual_e < 0.5)
        large_mask    = qual_e >= 0.5

        # A. Pure Huber Distance (Removed qual_e multiplier to save intermediate gradients)
        base_error = F.smooth_l1_loss(pred_e, tgt_e, beta=0.1, reduction='none')
        
        weighted_error = base_error.clone()
        if moderate_mask.any():
            weighted_error = torch.where(moderate_mask, weighted_error * bw_mod, weighted_error)
        if large_mask.any():
            weighted_error = torch.where(large_mask, weighted_error * bw_large, weighted_error)
        
        edge_mse = torch.mean(weighted_error)

        # B. Linear Overshoot Penalty (CHANGED from squared to smooth_l1)
        # Prevents the model from being terrified to predict > 0.5
        overshoot = torch.relu(pred_e.abs() - tgt_e.abs())
        overshoot_loss = F.smooth_l1_loss(overshoot, torch.zeros_like(overshoot), beta=0.1)

        # C. Static Directional Penalty (Removed dynamic inverse weighting)
        # Simply penalize mismatch, scaled naturally by target size
        sign_mismatch = torch.relu(-pred_e * tgt_e)
        dir_penalty = torch.mean(sign_mismatch * qual_e)
    # 4. DISPERSION & BIAS
    # activation_gravity removed — was a zero-attractor that double-penalised magnitude
    # alongside WEIGHT_DECAY = 0.05. OOS stability relies on dropout + weight decay alone.

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
        var_penalty = (pred_std - tgt_std).pow(2)

        global_bias = (pred.mean() - target.mean().detach()).abs()
        edge_bias   = (pred_e_f.mean() - tgt_e_f.mean().detach()).abs()
        bias_penalty = 0.5 * global_bias + 0.5 * edge_bias
    else:
        corr_penalty = torch.tensor(0.0, device=pred.device)
        var_penalty  = torch.tensor(0.0, device=pred.device)
        bias_penalty = (pred.mean() - target.mean().detach()).abs()

    total = (
        edge_mse
        + overshoot_loss
        + _fs_weight * false_signal_loss
        + _pen_weight * dir_penalty
        + dispersion_weight * corr_penalty
        + dispersion_weight * var_penalty
        + bias_weight * bias_penalty
    )

    return total