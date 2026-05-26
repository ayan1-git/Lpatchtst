import torch
import torch.nn.functional as F
from typing import Optional

import config

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_std(t: torch.Tensor, min_val: float = 0.01) -> torch.Tensor:
    """
    Standard deviation with a hard floor to prevent division-by-zero in
    correlation computation.  Returns min_val when the tensor has ≤1 element
    or when std() produces NaN.
    """
    if t.numel() <= 1:
        return torch.tensor(min_val, device=t.device, dtype=torch.float32)
    return t.float().std().nan_to_num(nan=min_val).clamp(min=min_val)


def _quantile_spread_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: list = [0.10, 0.25, 0.75, 0.90],
) -> torch.Tensor:
    """
    Penalise mismatch between the predicted and target *shape* at four
    quantile points.  Forces the prediction distribution to span the same
    range as the target distribution, not just match the mean.
    """
    loss = torch.tensor(0.0, device=pred.device)
    for q in quantiles:
        pred_q = torch.quantile(pred, q)
        tgt_q  = torch.quantile(target.detach(), q)
        loss   = loss + (pred_q - tgt_q).pow(2)
    return loss / len(quantiles)


def _moderate_bucket_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Pinball loss at q=0.15 and q=0.85 applied to the MODERATE edge population
    only (|tgt| in [0, 0.5)).

    Caller is responsible for passing the already-filtered moderate-only
    pred and target tensors.  Returns 0 on an empty input.
    """
    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device)

    q_low, q_high   = 0.15, 0.85
    tgt_low  = torch.quantile(target.detach(), q_low)
    tgt_high = torch.quantile(target.detach(), q_high)

    loss_low  = torch.mean(torch.max(
        q_low       * (tgt_low  - pred),
        (1 - q_low) * (pred     - tgt_low),
    ))
    loss_high = torch.mean(torch.max(
        q_high       * (tgt_high - pred),
        (1 - q_high) * (pred     - tgt_high),
    ))
    return (loss_low + loss_high) * 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOSS
# ═══════════════════════════════════════════════════════════════════════════════

def continuous_weighted_direction_loss(
    pred:   torch.Tensor,
    target: torch.Tensor,
    # ── directional weights ───────────────────────────────────────────────────
    penalty_weight:       float = 2.00,   # dir_penalty coefficient
    false_signal_weight:  float = 1.50,   # false-signal (exact-zero) coefficient
    # ── distribution-matching weights ────────────────────────────────────────
    dispersion_weight:    float = 0.25,   # correlation penalty coefficient
    bias_weight:          float = 0.30,   # global + edge bias penalty coefficient
    # ── magnitude weights ────────────────────────────────────────────────────
    shortfall_weight:     float = 1.50,   # underprediction-shortfall coefficient
    # ── bucket weights (passed from train; fall back to 1.0) ─────────────────
    bucket_weights: Optional[dict] = None,
    # ── unused legacy args kept for call-site compatibility ──────────────────
    fold_id:              int   = 99,
    epoch:                int   = 0,
    curriculum_ramp_epochs: int = config.CURRICULUM_RAMP_EPOCHS,
    _debug:               bool  = True,
) -> torch.Tensor:
    """
    Continuous regression loss for Oracle-labelled trade targets in [-1, 1].

    Design contract
    ───────────────
    • target == 0.0 exactly  →  Oracle no-trade bar.  Only false_signal_loss
      fires here; the bar is excluded from the edge pool entirely so it cannot
      corrupt MSE / direction gradients.

    • target != 0.0          →  Real directional signal.  All edge terms fire:
      MSE (bucket-weighted), sign-aware overshoot, direction penalty/reward,
      magnitude shortfall, distribution matching.

    Loss ordering guarantee (for target = +0.5):
      C4 wrong-sign over (−0.7) > C3 wrong-sign under (−0.2)
        > C5 collapse (0.0) > C1 correct under (+0.2) > C2 correct over (+0.7)
    The shortfall term closes the C1 > C2 gap; sign-aware overshoot prevents
    wrong-sign predictions from getting a free ride via the large-target
    tail exemption.
    """
    pred   = pred.view(-1).float()
    target = target.view(-1).float()

    # ── bucket multipliers ────────────────────────────────────────────────────
    bw_flat     = bucket_weights.get("flat",     1.0) if bucket_weights else 1.0
    bw_mod      = bucket_weights.get("moderate", 1.0) if bucket_weights else 1.0
    bw_large    = bucket_weights.get("large",    1.0) if bucket_weights else 1.0

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — CATEGORISE
    # exact_zero : Oracle no-trade bars  (target == 0.0 exactly)
    # is_edge    : everything else  (real directional signal, nonzero target)
    # ─────────────────────────────────────────────────────────────────────────
    exact_zero = (target == 0.0)
    is_edge    = ~exact_zero
    qual       = target.abs()          # quality weight  (= |target|)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — FALSE-SIGNAL LOSS  (exact-zero Oracle bars only)
    #
    # Semantic: "you predicted a trade where the Oracle found none."
    # smooth_l1 with beta=0.1 gives quadratic gradient for small predictions
    # and linear for large ones — stable even for pred=±2 or ±3.
    # ─────────────────────────────────────────────────────────────────────────
    false_signal_loss = torch.tensor(0.0, device=pred.device)
    if exact_zero.any():
        false_signal_loss = F.smooth_l1_loss(
            pred[exact_zero].abs(),
            torch.zeros(exact_zero.sum(), device=pred.device),
            beta=0.1,
            reduction="mean",
        ) * bw_flat

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — EDGE-TARGET LOSSES  (nonzero Oracle targets only)
    # ─────────────────────────────────────────────────────────────────────────
    edge_mse       = torch.tensor(0.0, device=pred.device)
    overshoot_loss = torch.tensor(0.0, device=pred.device)
    dir_penalty    = torch.tensor(0.0, device=pred.device)
    dir_reward     = torch.tensor(0.0, device=pred.device)
    shortfall_loss = torch.tensor(0.0, device=pred.device)

    if is_edge.any():
        pred_e = pred[is_edge]
        tgt_e  = target[is_edge]
        qual_e = qual[is_edge]

        moderate_mask = qual_e < 0.5
        large_mask    = qual_e >= 0.5

        # ── A: WEIGHTED MSE ──────────────────────────────────────────────────
        # Bucket weights amplify moderate and large targets relative to
        # micro-signal bars near-but-not-exactly zero.
        base_error     = (pred_e - tgt_e).pow(2)
        weighted_error = torch.where(large_mask, base_error * bw_large,
                         torch.where(moderate_mask, base_error * bw_mod,
                                     base_error))
        edge_mse = weighted_error.mean()

        # ── B: SIGN-AWARE OVERSHOOT ───────────────────────────────────────────
        # Only fires when the prediction is in the CORRECT direction but
        # exceeds the target magnitude.  Wrong-sign overpredictions are
        # fully handled by dir_penalty (C) — applying overshoot on top with
        # a 0.1 discount produced near-zero gradient and masked the problem.
        same_sign  = (pred_e * tgt_e) > 0
        os_raw     = torch.relu(pred_e.abs() - tgt_e.abs()) * same_sign.float()
        os_huber   = F.smooth_l1_loss(os_raw, torch.zeros_like(os_raw),
                                       beta=1.0, reduction="none")
        # Large-target tail exemption: large correct-direction overshoots are
        # tolerated (the model is encouraged to commit to strong signals).
        os_discount = torch.where(large_mask,
                                   torch.full_like(tgt_e, 0.1),
                                   torch.ones_like(tgt_e))
        overshoot_loss = (os_huber * os_discount).mean()

        # ── C: DIRECTIONAL PENALTY ────────────────────────────────────────────
        # relu(-pred*tgt) > 0 only when signs differ.
        # Scaled by qual_e so large-magnitude wrong-sign predictions
        # receive proportionally stronger correction.
        dir_penalty = torch.mean(torch.relu(-pred_e * tgt_e) * qual_e)

        # ── D: DIRECTIONAL REWARD ─────────────────────────────────────────────
        # Negative term: reduces total loss when prediction is in the right
        # direction, rewarding the model for correct sign.
        # sign(0) = 0 for any exact-zero target — but exact_zero bars are
        # excluded from is_edge, so this is safe.
        dir_reward = -torch.mean(torch.sign(pred_e) * torch.sign(tgt_e) * qual_e)

        # ── E: MAGNITUDE SHORTFALL ────────────────────────────────────────────
        # Fires when |pred| < |tgt| — the prediction is "underpowered."
        # Scales with qual_e: a large-target bar demands a proportionally
        # large prediction.  Silent when |pred| >= |tgt| (no over-penalty
        # here; overshoot (B) handles that case).
        # smooth_l1 beta=0.1: quadratic for small shortfalls, linear for large.
        sf_raw = torch.relu(tgt_e.abs() - pred_e.abs())
        sf_smooth = F.smooth_l1_loss(sf_raw, torch.zeros_like(sf_raw),
                                      beta=0.1, reduction="none")
        shortfall_loss = (sf_smooth * qual_e).mean()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4 — DISTRIBUTION MATCHING  (edge pool only)
    #
    # corr_penalty  : penalise low Pearson correlation between pred and target
    # q_spread_loss : penalise quantile-shape mismatch
    # spread_reward : reward prediction spread (prevents zero-variance collapse)
    #                 — gated to bars with non-trivially-zero targets so the
    #                   MSE-vs-spread conflict on flat Oracle bars is avoided
    # bias_penalty  : penalise global and edge-level mean shift
    # ─────────────────────────────────────────────────────────────────────────
    corr_penalty  = torch.tensor(0.0, device=pred.device)
    q_spread_loss = torch.tensor(0.0, device=pred.device)
    spread_reward = torch.tensor(0.0, device=pred.device)
    bias_penalty  = (pred.mean() - target.mean().detach()).abs()

    if is_edge.sum() > 1:
        pred_e_f = pred[is_edge].float()
        tgt_e_f  = target[is_edge].float()

        # Correlation
        pred_c   = pred_e_f - pred_e_f.mean()
        tgt_c    = tgt_e_f  - tgt_e_f.mean().detach()
        pred_std = _safe_std(pred_e_f)
        tgt_std  = _safe_std(tgt_e_f).detach()
        corr         = (pred_c * tgt_c.detach()).mean() / (pred_std * tgt_std)
        corr_penalty = (1.0 - corr).clamp(min=0.0, max=2.0)

        # Quantile shape
        q_spread_loss = _quantile_spread_loss(pred_e_f, tgt_e_f)

        # Spread reward — gated to bars with real (nonzero) targets only,
        # preventing conflict with MSE on near-zero edge targets.
        real_signal_mask = tgt_e_f.abs() > 0.01
        if real_signal_mask.any():
            spread_reward = -torch.log(
                _safe_std(pred_e_f[real_signal_mask]).clamp(min=1e-7)
            )

        # Bias
        global_bias  = (pred.mean()    - target.mean().detach()).abs()
        edge_bias    = (pred_e_f.mean() - tgt_e_f.mean().detach()).abs()
        bias_penalty = 0.5 * global_bias + 0.5 * edge_bias

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5 — MODERATE BUCKET LOSS  (moderate edge pool only: |tgt| < 0.5)
    #
    # Pinball loss at q=0.15/0.85 to push predictions into the ±[0.1, 0.5]
    # range.  Filtered to moderate-only BEFORE passing to the function so
    # the quantile anchors are not shifted by large-target bars.
    # ─────────────────────────────────────────────────────────────────────────
    mod_bucket_loss = torch.tensor(0.0, device=pred.device)
    if is_edge.any():
        mod_mask = (target[is_edge].abs() < 0.5)
        if mod_mask.any():
            mod_bucket_loss = _moderate_bucket_loss(
                pred[is_edge][mod_mask],
                target[is_edge][mod_mask],
            )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6 — GLOBAL STD FLOOR
    #
    # Penalise degenerate collapse of the full prediction distribution.
    # Fires when pred.std() falls below 0.35.  Acts as a backstop against
    # the optimizer finding a low-variance fixed point.
    # ─────────────────────────────────────────────────────────────────────────
    global_std_floor = torch.tensor(0.0, device=pred.device)
    if pred.numel() > 1:
        global_std_floor = F.relu(0.35 - pred.std())

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7 — COMBINE
    # ─────────────────────────────────────────────────────────────────────────
    total = (
        # ── magnitude ────────────────────────────────────────────────────────
        edge_mse                                   # A: weighted MSE on edge targets
        + overshoot_loss                           # B: correct-sign overshoot only
        + shortfall_weight  * shortfall_loss       # E: underprediction shortfall
        # ── direction ────────────────────────────────────────────────────────
        + penalty_weight    * dir_penalty          # C: wrong-sign penalty
        + 1.0               * dir_reward           # D: correct-sign reward (reduces loss)
        # ── false-signal ─────────────────────────────────────────────────────
        + false_signal_weight * false_signal_loss  # Step 2: exact-zero Oracle bars
        # ── distribution ─────────────────────────────────────────────────────
        + dispersion_weight * corr_penalty         # correlation penalty
        + 0.30              * q_spread_loss        # quantile shape penalty
        + bias_weight       * bias_penalty         # global + edge bias penalty
        + 0.50              * spread_reward        # std reward (anti-collapse)
        # ── moderate bucket ───────────────────────────────────────────────────
        + 0.40              * mod_bucket_loss      # pinball on moderate-only edge
        # ── global floor ──────────────────────────────────────────────────────
        + 1.0               * global_std_floor     # std floor backstop
    )

    return total