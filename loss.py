import torch
import torch.nn.functional as F

def _safe_std(t: torch.Tensor, min_val: float = 0.01) -> torch.Tensor:
    if t.numel() <= 1:
        return torch.tensor(min_val, device=t.device, dtype=torch.float32)
    s = t.float().std()
    return s.nan_to_num(nan=min_val).clamp(min=min_val)

def continuous_weighted_direction_loss(
    pred, target,
    penalty_weight: float = 0.25,
    false_signal_weight: float = 0.5,
    margin: float = 0.10,
    dispersion_weight: float = 0.1,
    bias_weight: float = 0.1,
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

    # 1. CURRICULUM RAMP (Smoothed and Capped to prevent shock)
    ramp_factor = min(1.0, (epoch + 1) / curriculum_ramp_epochs) if fold_id != 99 else 1.0
    _fs_weight = false_signal_weight * ramp_factor
    _pen_weight = penalty_weight * ramp_factor

    # 2. FALSE SIGNAL LOSS (Flat Targets)
    false_signal_loss = torch.tensor(0.0, device=pred.device)
    if is_zero.any():
        bw_flat = bucket_weights.get("flat", 1.0) if bucket_weights else 1.0
        
        # Use L1 Loss with a margin. If it predicts exactly 0, loss is 0. 
        # Constant gradient pushes it inward aggressively.
        raw_abs_error = pred[is_zero].abs()
        false_signal_loss = torch.mean(torch.relu(raw_abs_error - (margin * 0.5))) * bw_flat

    # 3. EDGE TARGET LOSS (Long / Short Targets)
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

        # --- A. Adaptive Huber Error (Protects Moderate Trades) ---
        # Smooth L1 prevents large targets from producing massive squared gradients
        # that drown out moderate target errors.
        base_error = F.smooth_l1_loss(pred_e, tgt_e, beta=0.2, reduction='none')
        
        weighted_error = base_error * qual_e
        if moderate_mask.any():
            weighted_error = torch.where(moderate_mask, weighted_error * bw_mod, weighted_error)
        if large_mask.any():
            weighted_error = torch.where(large_mask, weighted_error * bw_large, weighted_error)
        
        edge_mse = torch.mean(weighted_error)

        # --- B. Overshoot Penalty (Stops Boundary Exploitation) ---
        # Only penalize if the model predicts a magnitude larger than the true target.
        overshoot = torch.relu(pred_e.abs() - tgt_e.abs())
        overshoot_loss = torch.mean((overshoot ** 2) * qual_e)

        # --- C. Balanced Directional Penalty (Stops Short Abandonment) ---
        # Penalizes strict sign mismatch, bypassing the moderate margin trap.
        sign_mismatch = torch.relu(-pred_e * tgt_e)
        
        # Calculate batch class frequency to protect the minority class (usually shorts)
        num_pos = (tgt_e > 0).sum().float().clamp(min=1.0)
        num_neg = (tgt_e < 0).sum().float().clamp(min=1.0)
        total_edge = num_pos + num_neg
        
        # Invert weights: if shorts are rare, penalty for missing them increases,
        # but penalty for taking a false short decreases.
        weight_pos = total_edge / (2.0 * num_pos)
        weight_neg = total_edge / (2.0 * num_neg)
        
        dir_weights = torch.where(tgt_e > 0, weight_pos, weight_neg)
        dir_penalty = torch.mean(sign_mismatch * dir_weights * qual_e)

    # 4. DISPERSION & BIAS
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

        # Variance matching prevents the std blowouts seen in Folds 0 & 1
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