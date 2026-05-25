import torch
import torch.nn.functional as F
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════════
# SAFETY UTILITY
# ═══════════════════════════════════════════════════════════════════════════════
def _safe_std(t: torch.Tensor, min_val: float = 0.01) -> torch.Tensor:
    """
    Compute standard deviation with safety bounds.
    
    Purpose: Prevent division by zero and NaN propagation when computing 
    correlation. If tensor has ≤1 element or std() is NaN, return min_val.
    
    Args:
        t: Input tensor (typically centered predictions or targets)
        min_val: Floor for std (default 0.01) to prevent 0 division in correlation
    
    Returns:
        Clamped standard deviation
    """
    if t.numel() <= 1:
        return torch.tensor(min_val, device=t.device, dtype=torch.float32)
    s = t.float().std()
    return s.nan_to_num(nan=min_val).clamp(min=min_val)

def continuous_weighted_direction_loss(
    pred, target,
    penalty_weight: float = 2.00,       
    false_signal_weight: float = 2.00,  
    margin: float = 0.05,               # DECREASED from 0.10: Close the 0.11 loophole
    dispersion_weight: float = 0.25,    # ← FIX 1: was 0.40. Reduce to stop corr_penalty dominating
    bias_weight: float = 0.30,          # ← FIX 1: was 0.50. Correlated with var_penalty, reduce
    overshoot_discount_long: float = 0.40,   # Tail exemption for long (|y| >= 0.5)
    overshoot_discount_short: float = 0.30,  # Tail exemption for short (|y| >= 0.5)
    fold_id: int = 99,                  
    bucket_weights: Optional[dict] = None,        
    epoch: int = 0,                     
    curriculum_ramp_epochs: int = 10,   
    _debug: bool = True,               
):
    pred   = pred.view(-1)
    target = target.view(-1)

    # STEP 1: CATEGORIZE
    is_zero = (target.abs() < 1e-1)     
    is_edge = ~is_zero                  
    quality = target.abs()              

    # STEP 2: PENALTIES (CURRICULUM RAMP ERADICATED)
    # The model is pre-trained. We demand 100% strictness from Epoch 0 
    # to prevent "spray and pray" memorization in early folds.
    _fs_weight = false_signal_weight    
    _pen_weight = penalty_weight        

    # STEP 3: FALSE SIGNAL LOSS 
    false_signal_loss = torch.tensor(0.0, device=pred.device)
    if is_zero.any():
        bw_flat = bucket_weights.get("flat", 1.0) if bucket_weights else 1.0
        raw_abs_error = pred[is_zero].abs()
        
        # STRICT DEAD-ZONE: Force predictions to actually drop below margin
        false_signal_loss = torch.mean(torch.relu(raw_abs_error - margin)) * bw_flat
        
    # STEP 4: EDGE TARGET LOSS 
    edge_mse          = torch.tensor(0.0, device=pred.device)
    overshoot_loss    = torch.tensor(0.0, device=pred.device)
    dir_penalty       = torch.tensor(0.0, device=pred.device)
    dir_reward        = torch.tensor(0.0, device=pred.device)

    if is_edge.any():
        pred_e = pred[is_edge]          
        tgt_e  = target[is_edge]        
        qual_e = quality[is_edge]       
        
        bw_mod   = bucket_weights.get("moderate", 1.0) if bucket_weights else 1.0
        bw_large = bucket_weights.get("large",    1.0) if bucket_weights else 1.0
        
        moderate_mask = (qual_e > 0.1) & (qual_e < 0.5)
        large_mask    = qual_e >= 0.5
 
        # COMPONENT A: MAGNITUDE ERROR 
        base_error = F.smooth_l1_loss(pred_e, tgt_e, beta=0.15, reduction='none')
        
        weighted_error = base_error.clone()
        if moderate_mask.any():
            weighted_error = torch.where(moderate_mask, weighted_error * bw_mod, weighted_error)
        if large_mask.any():
            weighted_error = torch.where(large_mask, weighted_error * bw_large, weighted_error)
        
        edge_mse = torch.mean(weighted_error)

        # COMPONENT B: OVERSHOOT PENALTY WITH TAIL EXEMPTION
        overshoot = torch.relu(pred_e.abs() - tgt_e.abs())
        base_overshoot = F.smooth_l1_loss(overshoot, torch.zeros_like(overshoot), beta=1.0, reduction='none')
        
        # TAIL EXEMPTION: If the target is massive (|y| >= 0.5), discount overshoot penalty.
        # This cures the "shoulder bulge" by making it mathematically safe to predict extremes.
        # Separate discounts for long and short positions allow independent tuning.
        is_long = tgt_e > 0
        is_short = tgt_e < 0
        
        overshoot_discount = torch.ones_like(tgt_e)
        if (large_mask & is_long).any():
            overshoot_discount = torch.where(large_mask & is_long, overshoot_discount_long, overshoot_discount)
        if (large_mask & is_short).any():
            overshoot_discount = torch.where(large_mask & is_short, overshoot_discount_short, overshoot_discount)
        
        overshoot_loss = torch.mean(base_overshoot * overshoot_discount)

        # COMPONENT C: DIRECTIONAL PENALTY 
        sign_mismatch = torch.relu(-pred_e * tgt_e)
        dir_penalty = torch.mean(sign_mismatch * qual_e)

        # COMPONENT D: DIRECTIONAL REWARD
        sign_correct = torch.sign(pred_e) * torch.sign(tgt_e)
        dir_reward = -torch.mean(sign_correct * qual_e)

    # STEP 5: DISTRIBUTION MATCHING 
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

        # Spread encouragement: actively reward predictions that have width.
        # This is the positive counterpart of var_penalty — instead of only
        # penalizing too-wide predictions, also reward any spread at all.
        # This breaks the zero-variance fixed point that var_penalty alone cannot escape.
        spread_reward = -torch.log(pred_std.clamp(min=1e-4))   # → 0 when pred_std=1.0, large when std→0

        global_bias = (pred.mean() - target.mean().detach()).abs()
        edge_bias   = (pred_e_f.mean() - tgt_e_f.mean().detach()).abs()
        bias_penalty = 0.5 * global_bias + 0.5 * edge_bias
    else:
        corr_penalty = torch.tensor(0.0, device=pred.device)
        var_penalty  = torch.tensor(0.0, device=pred.device)
        spread_reward = torch.tensor(0.0, device=pred.device)
        bias_penalty = (pred.mean() - target.mean().detach()).abs()

    # STEP 6: COMBINE ALL LOSS COMPONENTS
    total = (
        edge_mse                                
        + overshoot_loss                        
        + _fs_weight * false_signal_loss        
        + _pen_weight * dir_penalty             
        + 1.0 * dir_reward                      # ← ADD: dir_reward term
        + dispersion_weight * corr_penalty      
        + dispersion_weight * var_penalty       
        + bias_weight * bias_penalty            
        + 0.10 * spread_reward              # ← ADD: breaks zero-std fixed point
    )
    
    # Prediction std penalty
    pred_std = pred.std() if pred.numel() > 1 else torch.tensor(0.0, device=pred.device)
    total = total + 0.5 * F.relu(0.25 - pred_std)
    
    return total