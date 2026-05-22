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
    # ─── LOSS COMPONENT WEIGHTS (multipliers for each term) ───────────────────
    # These control HOW MUCH each penalty contributes to the total loss.
    # Increase weight → model pays more attention to that term → changes behavior.
    
    penalty_weight: float = 1.50,       # DIRECTION PENALTY WEIGHT
                                        # Current: 1.50 (very high)
                                        # ↑ Increase: Strongly penalize wrong sign (e.g., pred < 0 but target > 0)
                                        # ↓ Decrease: Allow more "confused" predictions if magnitude is right
    
    false_signal_weight: float = 0.75,  # FALSE SIGNAL PENALTY WEIGHT (flat markets)
                                        # Current: 0.75 (medium)
                                        # ↑ Increase: Terrify model into NOT predicting on flat targets
                                        # ↓ Decrease: Soften penalty, allow small predictions on flat targets
    
    margin: float = 0.10,               # NOT DIRECTLY USED in current implementation
                                        # (legacy parameter, kept for compatibility)
    
    dispersion_weight: float = 0.8,     # DISTRIBUTION MATCHING WEIGHT
                                        # Current: 0.8 (medium)
                                        # Controls: correlation_penalty + variance_penalty
                                        # ↑ Increase: Force predictions to have same variance and correlation as targets
                                        # ↓ Decrease: Allow prediction distribution to deviate from target distribution
    
    bias_weight: float = 0.75,          # MEAN MATCHING WEIGHT (fights artificial drift)
                                        # Current: 0.75 (very high - major change!)
                                        # ↑ Increase: Magnetic pull on prediction mean → target mean
                                        #            (eliminates short/long bias)
                                        # ↓ Decrease: Allow predictions to drift away from target mean
    
    # ─── CURRICULUM & FOLD INFO ──────────────────────────────────────────────
    fold_id: int = 99,                  # Fold identifier (99 = pretrain/no curriculum)
    bucket_weights: Optional[dict] = None,        # Per-bucket inverse-frequency weights
    epoch: int = 0,                     # Current epoch (used for curriculum ramping)
    curriculum_ramp_epochs: int = 10,   # Number of epochs to ramp up from 0 → 1
    _debug: bool = False,               # Debug flag (reserved)
):
    pred   = pred.view(-1)
    target = target.view(-1)

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1: CATEGORIZE SAMPLES INTO BUCKETS
    # ═══════════════════════════════════════════════════════════════════════════
    is_zero = (target.abs() < 1e-6)     # "FLAT" targets: oracle said do NOT trade (|y| ≈ 0)
    is_edge = ~is_zero                  # "EDGE" targets: oracle said DO trade (|y| > 1e-6)
    quality = target.abs()              # Magnitude of signal (used to scale penalties)
    
    # These masks separate loss computation into two paths:
    # - Flat targets: penalize false signals (predictions on no-trade zones)
    # - Edge targets: penalize magnitude error AND directional error

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2: CURRICULUM LEARNING RAMP
    # ═══════════════════════════════════════════════════════════════════════════
    # Early epochs: Reduce penalty_weight and false_signal_weight to allow
    # model to explore without being over-constrained.
    # 
    # ramp_factor: 0.0 at epoch 0 → 1.0 at epoch curriculum_ramp_epochs
    # Effect: penalty_weight * ramp_factor → gradually increases constraint
    
    ramp_factor = min(1.0, (epoch + 1) / curriculum_ramp_epochs) if fold_id != 99 else 1.0
    _fs_weight = false_signal_weight * ramp_factor    # Ramped false signal penalty
    _pen_weight = penalty_weight * ramp_factor        # Ramped directional penalty
    
    # Example: if penalty_weight=1.5, curriculum_ramp_epochs=20
    #   Epoch 1:  _pen_weight = 1.5 * (1/20)  = 0.075
    #   Epoch 10: _pen_weight = 1.5 * (10/20) = 0.75
    #   Epoch 20+: _pen_weight = 1.5 * (20/20) = 1.50 (full strength)

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3: FALSE SIGNAL LOSS (Penalize predictions on flat targets)
    # ═══════════════════════════════════════════════════════════════════════════
    # Purpose: Prevent model from HALLUCINATING trades on flat market days.
    # 
    # When target.abs() < 1e-6 (oracle said: do not trade):
    #   - Ideal prediction: |pred| ≈ 0 (quiet agreement)
    #   - Punished prediction: |pred| >> 0.08 (noisy disagreement)
    # 
    # Dead-zone = 0.08: Predictions with |pred| < 0.08 are considered "close enough"
    # (aligned with SAMPLER_THRESHOLD from config)
    
    false_signal_loss = torch.tensor(0.0, device=pred.device)
    if is_zero.any():
        bw_flat = bucket_weights.get("flat", 1.0) if bucket_weights else 1.0
        raw_abs_error = pred[is_zero].abs()
        
        # FORMULA: max(|pred| - 0.08, 0)
        # Interpretation:
        #   - If |pred| < 0.08:  penalty = 0 (tolerated)
        #   - If |pred| = 0.15:  penalty = 0.07 (slightly penalized)
        #   - If |pred| = 0.50:  penalty = 0.42 (heavily penalized)
        
        false_signal_loss = torch.mean(torch.relu(raw_abs_error - 0.08)) * bw_flat
        
        # bw_flat: inverse-frequency weight from compute_bucket_weights()
        # - If flat targets are RARE (15% of data): bw_flat ≈ 2.0 (2x penalty)
        # - If flat targets are COMMON (50% of data): bw_flat ≈ 0.67 (0.67x penalty)

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 4: EDGE TARGET LOSS (Penalize errors on tradeable targets)
    # ═══════════════════════════════════════════════════════════════════════════
    # Purpose: When oracle says TRADE (|target| > 1e-6), penalize:
    #   A. Magnitude error (how far off the prediction magnitude is)
    #   B. Overshoot penalty (prevent extreme predictions)
    #   C. Directional error (wrong sign)
    
    edge_mse          = torch.tensor(0.0, device=pred.device)
    overshoot_loss    = torch.tensor(0.0, device=pred.device)
    dir_penalty       = torch.tensor(0.0, device=pred.device)

    if is_edge.any():
        pred_e = pred[is_edge]          # Predictions on tradeable targets
        tgt_e  = target[is_edge]        # Actual tradeable targets
        qual_e = quality[is_edge]       # Magnitudes (|target|) for this subset
        
        # Bucket weights: applied differently based on target magnitude
        bw_mod   = bucket_weights.get("moderate", 1.0) if bucket_weights else 1.0
        bw_large = bucket_weights.get("large",    1.0) if bucket_weights else 1.0
        
        # Split into two magnitude tiers:
        # - MODERATE: 0.1 < |target| < 0.5 (small moves)
        # - LARGE:    |target| >= 0.5 (big moves)
        moderate_mask = (qual_e > 0.1) & (qual_e < 0.5)
        large_mask    = qual_e >= 0.5

        # ─────────────────────────────────────────────────────────────────────
        # COMPONENT A: MAGNITUDE ERROR (Smooth L1 / Huber loss)
        # ─────────────────────────────────────────────────────────────────────
        # Smooth L1: transitions from L1 (linear) → L2 (quadratic) at beta=0.1
        # 
        # FORMULA: smooth_l1_loss(pred, target, beta=0.1)
        #   |error| ≤ 0.1:  loss = 0.5 * error^2 (smooth)
        #   |error| > 0.1:  loss = error - 0.005 (linear tail)
        #
        # Effect: Penalizes small errors smoothly, large errors linearly
        #   - Prevents gradients from exploding on large misses
        #   - Doesn't minimize large errors to death (allows some freedom)
        
        base_error = F.smooth_l1_loss(pred_e, tgt_e, beta=0.1, reduction='none')
        
        # Apply bucket weights: rare move types get higher weight
        weighted_error = base_error.clone()
        if moderate_mask.any():
            weighted_error = torch.where(moderate_mask, weighted_error * bw_mod, weighted_error)
        if large_mask.any():
            weighted_error = torch.where(large_mask, weighted_error * bw_large, weighted_error)
        
        edge_mse = torch.mean(weighted_error)

        # ─────────────────────────────────────────────────────────────────────
        # COMPONENT B: OVERSHOOT PENALTY (prevent magnitude explosion)
        # ─────────────────────────────────────────────────────────────────────
        # CHANGED: from squared (L2) to smooth_l1 (linear) to prevent fear
        # of large predictions.
        #
        # FORMULA: overshoot = max(|pred| - |target|, 0)
        #   - If |pred| ≤ |target|: penalty = 0 (good: under-predicting)
        #   - If |pred| > |target|: penalty > 0 (bad: over-predicting)
        #
        # smooth_l1_loss(overshoot, 0, beta=0.1) applies linear penalty
        # to overshoots, making it much less scary than squared (x^2)
        #
        # Example with target=0.3:
        #   pred=0.2: overshoot=0,   loss ≈ 0      (safe)
        #   pred=0.5: overshoot=0.2, loss ≈ 0.15   (slight penalty)
        #   pred=1.0: overshoot=0.7, loss ≈ 0.65   (moderate penalty, not catastrophic)
        #
        # Compare to squared: loss ≈ 0.49 (0.7^2) — much scarier!
        # Squared would terrify model into never predicting > 0.5
        
        overshoot = torch.relu(pred_e.abs() - tgt_e.abs())
        overshoot_loss = F.smooth_l1_loss(overshoot, torch.zeros_like(overshoot), beta=0.1)

        # ─────────────────────────────────────────────────────────────────────
        # COMPONENT C: DIRECTIONAL PENALTY (sign mismatch)
        # ─────────────────────────────────────────────────────────────────────
        # Purpose: If target is positive but prediction is negative (or vice versa),
        # apply strong penalty scaled by target magnitude.
        #
        # FORMULA: sign_mismatch = max(-pred * target, 0)
        #   - If pred and target have SAME sign: pred * target > 0 → penalty = 0 ✓
        #   - If pred and target have DIFFERENT sign: pred * target < 0 → penalty > 0 ✗
        #
        # Example:
        #   target=+0.3, pred=+0.1: -pred*target = -(-0.03) = 0 (same sign ✓)
        #   target=+0.3, pred=-0.1: -pred*target = -(+0.03) = 0.03 (wrong sign ✗)
        #
        # Scaled by qual_e (|target|):
        #   - Small target (0.1): penalty reduced 10x
        #   - Large target (1.0): penalty full strength
        # This makes directional error less critical for small moves
        
        sign_mismatch = torch.relu(-pred_e * tgt_e)
        dir_penalty = torch.mean(sign_mismatch * qual_e)
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 5: DISTRIBUTION MATCHING (Dispersion & Bias penalties)
    # ═══════════════════════════════════════════════════════════════════════════
    # Purpose: Force the prediction distribution to match the target distribution.
    # This prevents systematic biases (e.g., always predicting too short).
    
    # Note: activation_gravity removed — was a zero-attractor that double-penalised 
    # magnitude alongside WEIGHT_DECAY = 0.05. OOS stability relies on 
    # dropout + weight decay alone, not gravity.

    if is_edge.sum() > 1:
        # Extract edge targets/predictions as floats for stats computation
        pred_e_f = pred[is_edge].float()
        tgt_e_f  = target[is_edge].float()
        
        # Center both distributions (subtract mean)
        pred_c = pred_e_f - pred_e_f.mean()      # Centered predictions
        tgt_c  = tgt_e_f  - tgt_e_f.mean().detach()  # Centered targets (detached to not change target stats)

        # ─────────────────────────────────────────────────────────────────────
        # CORRELATION PENALTY: Force prediction variance pattern to match target
        # ─────────────────────────────────────────────────────────────────────
        cov      = (pred_c * tgt_c.detach()).mean()  # Covariance
        pred_std = _safe_std(pred_e_f)                # Prediction std (safe)
        tgt_std  = _safe_std(tgt_e_f).detach()        # Target std (safe, detached)
        
        # Pearson correlation: cov / (std_pred * std_target)
        corr         = cov / (pred_std * tgt_std)
        
        # FORMULA: corr_penalty = max(1 - correlation, 0)
        # - If corr = 1.0:  penalty = 0 (perfect match ✓)
        # - If corr = 0.5:  penalty = 0.5 (half-match, some penalty)
        # - If corr = -0.5: penalty = 1.5 (but clamped to [0, 2])
        #
        # Effect: If prediction & target don't co-vary, this penalty kicks in.
        # Example: predictions are high when targets are low → penalty → corrects it
        
        corr_penalty = (1.0 - corr).clamp(min=0.0, max=2.0)
        
        # ─────────────────────────────────────────────────────────────────────
        # VARIANCE PENALTY: Force prediction spread to match target spread
        # ─────────────────────────────────────────────────────────────────────
        # FORMULA: var_penalty = (pred_std - target_std)^2
        #
        # Example:
        #   pred_std=0.1, tgt_std=0.3: penalty = (0.1 - 0.3)^2 = 0.04
        #   pred_std=0.3, tgt_std=0.3: penalty = 0 (match ✓)
        #   pred_std=0.5, tgt_std=0.3: penalty = (0.5 - 0.3)^2 = 0.04
        #
        # Effect: If predictions cluster too tightly (low variance) or spread too
        # widely (high variance), this penalty corrects it.
        
        var_penalty = (pred_std - tgt_std).pow(2)

        # ─────────────────────────────────────────────────────────────────────
        # BIAS PENALTY: Force prediction mean to match target mean
        # ─────────────────────────────────────────────────────────────────────
        # FORMULA: bias_penalty = 0.5 * |global_mean_diff| + 0.5 * |edge_mean_diff|
        #
        # global_bias: difference in means across ALL predictions vs targets
        # edge_bias: difference in means on EDGE targets only
        #
        # Why 50/50 split?
        # - Global: catches systematic overall shift (e.g., short bias)
        # - Edge: catches systematic shift in tradeable moves
        #
        # Example: pred.mean()=0.05, target.mean()=-0.02
        #   penalty = 0.5 * |0.05 - (-0.02)| + 0.5 * |edge_diff|
        #           = 0.5 * 0.07 + 0.5 * edge_diff
        #           ≈ 0.035 + edge_contribution
        #
        # THIS IS HUGE WITH bias_weight=0.75 (was 0.1 before):
        # With weight=0.75, even 0.07 mean diff → 0.75 * 0.035 = 0.026 loss units
        # This MAGNETIC pull fights short/long drift effectively.
        
        global_bias = (pred.mean() - target.mean().detach()).abs()
        edge_bias   = (pred_e_f.mean() - tgt_e_f.mean().detach()).abs()
        bias_penalty = 0.5 * global_bias + 0.5 * edge_bias
    else:
        # Edge cases: insufficient edge targets
        corr_penalty = torch.tensor(0.0, device=pred.device)
        var_penalty  = torch.tensor(0.0, device=pred.device)
        bias_penalty = (pred.mean() - target.mean().detach()).abs()

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 6: COMBINE ALL LOSS COMPONENTS
    # ═══════════════════════════════════════════════════════════════════════════
    # Final loss is a WEIGHTED SUM of all penalties.
    # Each weight controls relative importance.
    #
    # Current weights (v4):
    #   edge_mse:             1.0 (unweighted)
    #   overshoot_loss:       1.0 (unweighted)
    #   false_signal_loss:    0.85 (ramped: 0 → 0.85)
    #   dir_penalty:          1.50 (ramped: 0 → 1.50) ← DIRECTION IS KEY
    #   corr_penalty:         1.0  (via dispersion_weight)
    #   var_penalty:          1.0  (via dispersion_weight)
    #   bias_penalty:         0.75 (HIGH: fights drift) ← FIGHTS SHORT BIAS
    #
    # ─────────────────────────────────────────────────────────────────────────
    # INTERPRETATION OF WEIGHT CHANGES:
    # ─────────────────────────────────────────────────────────────────────────
    # 
    # INCREASE penalty_weight (was 0.25, now 1.50):
    #   → Model will HATE wrong-sign predictions
    #   → More conservative, follows oracle direction strictly
    #   → May miss magnitude nuance for perfect direction
    #
    # INCREASE dispersion_weight (was 0.3, now 1.0):
    #   → Model will match target distribution more closely
    #   → Fixes miscalibration (predictions too bunched or too spread)
    #   → Prevents overfitting to specific scale
    #
    # INCREASE bias_weight (was 0.1, now 0.75):
    #   → Model's average prediction MAGNETICALLY tracks target average
    #   → Eliminates systematic short/long drift
    #   → This was the KEY missing piece causing short bias!
    #
    # DECREASE false_signal_weight (was 1.0, now 0.85):
    #   → Soften penalty on flat-market predictions
    #   → Allow model to be slightly uncertain on quiet days
    #   → Prevent over-cautiousness that hurts signal on edge cases
    
    total = (
        edge_mse                                # Raw magnitude error (always active on edges)
        + overshoot_loss                        # Penalty for over-predicting magnitude
        + _fs_weight * false_signal_loss        # Penalty for predicting on flat targets (ramped)
        + _pen_weight * dir_penalty             # Penalty for wrong sign (ramped)
        + dispersion_weight * corr_penalty      # Force prediction ↔ target correlation
        + dispersion_weight * var_penalty       # Force prediction ↔ target variance
        + bias_weight * bias_penalty            # MAGNETIC: mean(pred) → mean(target)
    )
    
    # ─────────────────────────────────────────────────────────────────────────
    # DEBUGGING: To understand loss contribution breakdown, inspect individually:
    # ─────────────────────────────────────────────────────────────────────────
    #   print(f"edge_mse={edge_mse:.4f}")
    #   print(f"overshoot={overshoot_loss:.4f}")
    #   print(f"false_signal={_fs_weight * false_signal_loss:.4f}")
    #   print(f"dir_penalty={_pen_weight * dir_penalty:.4f}")
    #   print(f"corr={dispersion_weight * corr_penalty:.4f}")
    #   print(f"var={dispersion_weight * var_penalty:.4f}")
    #   print(f"bias={bias_weight * bias_penalty:.4f}")
    #   print(f"total={total:.4f}")

    return total