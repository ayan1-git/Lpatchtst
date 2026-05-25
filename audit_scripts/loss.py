import torch
import torch.nn.functional as F
from typing import Optional

import config

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

def quantile_spread_loss(pred, target, quantiles=[0.1, 0.25, 0.75, 0.9]):
    loss = 0.0
    for q in quantiles:
        pred_q = torch.quantile(pred, q)
        tgt_q  = torch.quantile(target.detach(), q)
        loss += (pred_q - tgt_q).pow(2)
    return loss / len(quantiles)

def moderate_bucket_loss(pred, target):
    """Force predictions to populate the [0.1, 0.5] and [-0.5, -0.1] ranges."""
    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    # Pinball loss at q=0.15 and q=0.85 quantiles
    q_low, q_high = 0.15, 0.85
    tgt_low  = torch.quantile(target.detach(), q_low)   # ≈ -0.35 for your dist
    tgt_high = torch.quantile(target.detach(), q_high)  # ≈ +0.35 for your dist
    
    loss_low  = torch.mean(torch.max(q_low  * (tgt_low  - pred), 
                                      (1-q_low)  * (pred - tgt_low)))
    loss_high = torch.mean(torch.max(q_high * (tgt_high - pred), 
                                      (1-q_high) * (pred - tgt_high)))
    return (loss_low + loss_high) * 0.5

def continuous_weighted_direction_loss(
    pred, target,
    flat_threshold: float = 0.05,
    penalty_weight: float = 2.00,       
    false_signal_weight: float = 1.5,  
    margin: float = 0.03,               # DECREASED from 0.10: Close the 0.11 loophole
    dispersion_weight: float = 0.25,    # ← FIX 1: was 0.40. Reduce to stop corr_penalty dominating
    bias_weight: float = 0.30,          # ← FIX 1: was 0.50. Correlated with var_penalty, reduce
    overshoot_discount_long: float = 0.30,   # Tail exemption for long (|y| >= 0.5)
    overshoot_discount_short: float = 0.35,  # Tail exemption for short (|y| >= 0.5)
    fold_id: int = 99,                  
    bucket_weights: Optional[dict] = None,        
    epoch: int = 0,                     
    curriculum_ramp_epochs: int = config.CURRICULUM_RAMP_EPOCHS,   
    _debug: bool = True,               
):
    pred   = pred.view(-1).float()
    target = target.view(-1).float()

    # STEP 1: CATEGORIZE
    is_zero = (target.abs() < flat_threshold)
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
        
        moderate_mask = qual_e < 0.5
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
        
        # TAIL EXEMPTION: exempt everything above 0.1
        # This cures the "shoulder bulge" by making it mathematically safe to predict extremes.
        overshoot_discount = torch.ones_like(tgt_e)
        large_mask_new = qual_e >= 0.5   # was: qual_e >= 0.5
        overshoot_discount = torch.where(large_mask_new, 
                                          torch.full_like(tgt_e, 0.1),  # near-zero penalty
                                          overshoot_discount)
        
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
        q_spread_loss = quantile_spread_loss(pred_e_f, tgt_e_f)

        # Spread encouragement: actively reward predictions that have width.
        # This is the positive counterpart of quantile_spread_loss — instead of only
        # penalizing incorrect shape, also reward any spread at all.
        # This breaks the zero-variance fixed point that spread constraints alone cannot escape.
        spread_reward = -torch.log(pred_std.clamp(min=1e-7))   # → 0 when pred_std=1.0, large when std→0

        global_bias = (pred.mean() - target.mean().detach()).abs()
        edge_bias   = (pred_e_f.mean() - tgt_e_f.mean().detach()).abs()
        bias_penalty = 0.5 * global_bias + 0.5 * edge_bias
    else:
        corr_penalty = torch.tensor(0.0, device=pred.device)
        q_spread_loss = torch.tensor(0.0, device=pred.device)
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
        + 0.30 * q_spread_loss                  # Replaced var_penalty with Quantile Spread Loss
        + bias_weight * bias_penalty            
        + 0.50 * spread_reward              # was 0.10, breaks zero-std fixed point
        + 0.40 * moderate_bucket_loss(pred[is_edge], target[is_edge])
    )
    
    # Prediction std penalty
    pred_std = pred.std() if pred.numel() > 1 else torch.tensor(0.0, device=pred.device)
    total = total + 0.5 * F.relu(0.25 - pred_std)
    
    return total