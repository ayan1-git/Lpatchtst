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
    pred:      torch.Tensor,
    target:    torch.Tensor,
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
    Pinball loss at q=0.40 and q=0.60 applied exclusively to the moderate
    edge population (|tgt| < 0.5).  Caller must pre-filter to moderate-only
    tensors before calling — this function does not filter internally.
    Returns 0.0 on an empty input.
    """
    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device)

    q_low, q_high = 0.40, 0.60
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
    penalty_weight:      float = 2.00,  # wrong-sign penalty coefficient
    false_signal_weight: float = 2.00,  # false-signal (exact-zero) coefficient
    # ── magnitude weights ─────────────────────────────────────────────────────
    shortfall_weight:    float = 1.50,  # underprediction shortfall coefficient
    # ── distribution-matching weights ────────────────────────────────────────
    dispersion_weight:   float = 0.25,  # correlation penalty coefficient
    bias_weight:         Optional[float] = None,  # global + edge bias penalty coefficient
    # ── legacy args kept for call-site compatibility (unused internally) ──────
    bucket_weights:      Optional[dict] = None,   # retired — see importance weight below
    fold_id:             int   = 99,
    epoch:               int   = 0,
    curriculum_ramp_epochs: int = config.CURRICULUM_RAMP_EPOCHS,
    _debug:              bool  = True,
) -> torch.Tensor:
    """
    Continuous regression loss for Oracle-labelled trade targets in [-1, 1].

    Design contract
    ═══════════════
    target == 0.0 exactly  →  Oracle no-trade bar.
        Only false_signal_loss fires.  The bar is excluded from the edge pool
        so it cannot corrupt MSE / direction / shortfall gradients.

    target != 0.0  →  Real directional signal.
        All edge terms fire: weighted MSE, sign-aware overshoot, direction
        penalty/reward, magnitude shortfall, distribution matching,
        moderate pinball, global std floor.

    Importance weighting (replaces discrete buckets)
    ─────────────────────────────────────────────────
    importance = sqrt(|target|).clamp(min=0.3)

    • Continuous, no cliff at |t|=0.5.
    • sqrt compression: a 0.8-target gets 0.89×, a 0.1-target gets 0.32×.
    • clamp(0.3) gives micro-signal bars a minimum gradient voice.
    • Applied to both edge_mse and shortfall_loss for consistent scaling.
    • Replaces bw_mod, bw_large, moderate_mask, large_mask, and the entire
      compute_bucket_weights() call in train_pretrain_fine.py.

    Loss ordering guarantee (for target = +0.5, pred collapse → 0.0):
        wrong-sign over > wrong-sign under > collapse > correct under > correct over
    The shortfall term closes the under > over gap; sign-aware overshoot prevents
    wrong-sign predictions from getting a free ride via the tail exemption.
    """
    pred   = pred.view(-1).float()
    target = target.view(-1).float()

    if bias_weight is None:
        bias_weight = 0.30 if fold_id == 99 else (0.30 + 0.10 * fold_id)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — CATEGORISE
    #
    # exact_zero : Oracle no-trade bars  (target == 0.0 exactly, as set by
    #              the Oracle when both long_r_net ≤ 0 and short_r_net ≤ 0)
    # is_edge    : every other bar  (nonzero target, real directional signal)
    # ─────────────────────────────────────────────────────────────────────────
    exact_zero = (target == 0.0)
    is_edge    = ~exact_zero
    qual       = target.abs()           # scalar quality weight per sample

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — FALSE-SIGNAL LOSS  (exact-zero Oracle bars only)
    #
    # Semantic: "you predicted a trade where the Oracle found none."
    # smooth_l1 beta=0.1 → quadratic for small errors, linear for large —
    # stable gradient even when pred = ±2 or ±3.
    # ─────────────────────────────────────────────────────────────────────────
    false_signal_loss = torch.tensor(0.0, device=pred.device)
    if exact_zero.any():
        false_signal_loss = F.smooth_l1_loss(
            pred[exact_zero].abs(),
            torch.zeros(exact_zero.sum(), device=pred.device),
            beta=0.1,
            reduction="mean",
        )

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

        # ── CONTINUOUS IMPORTANCE WEIGHT (replaces discrete bucket weights) ──
        # sqrt(|tgt|) scales smoothly with target magnitude.
        # clamp(0.3) prevents micro-signal bars from having zero weight.
        # No cliff at 0.5 — no moderate_mask / large_mask needed.
        importance = qual_e.pow(0.5).clamp(min=0.3)

        # ── A: IMPORTANCE-WEIGHTED MSE ───────────────────────────────────────
        # Quadratic error scaled by sqrt(|tgt|): large targets pull harder
        # on the optimizer proportionally and continuously.
        base_error = (pred_e - tgt_e).pow(2)
        edge_mse   = (base_error * importance).mean()

        # ── B: SIGN-AWARE OVERSHOOT ──────────────────────────────────────────
        # Fires ONLY when prediction is in the correct direction but exceeds
        # the target magnitude.  Wrong-sign overpredictions are fully handled
        # by dir_penalty (C) — applying overshoot on top with a discount
        # produced near-zero gradient on wrong-sign large errors.
        #
        # Large-target tail exemption (discount=0.1): the model is encouraged
        # to commit to strong signals.  Only applies same-sign overshoots
        # above importance threshold (qual_e >= 0.5 proxy retained as a
        # semantic boundary for the exemption — not a gradient cliff).
        same_sign   = (pred_e * tgt_e) > 0
        os_raw      = torch.relu(pred_e.abs() - tgt_e.abs()) * same_sign.float()
        os_huber    = F.smooth_l1_loss(
            os_raw, torch.zeros_like(os_raw), beta=0.1, reduction="none"
        )
        large_signal = qual_e >= 0.5
        os_discount  = torch.where(
            large_signal,
            torch.full_like(tgt_e, 0.1),
            torch.ones_like(tgt_e),
        )
        overshoot_loss = (os_huber * os_discount).mean()

        # ── C: DIRECTIONAL PENALTY ───────────────────────────────────────────
        # relu(-pred * tgt) > 0 only when signs differ.
        # Scaled by qual_e: large wrong-sign predictions receive
        # proportionally stronger correction.
        dir_penalty = torch.mean(torch.relu(-pred_e * tgt_e) * qual_e)

        # ── D: DIRECTIONAL REWARD ────────────────────────────────────────────
        # Negative term: correct-sign predictions reduce total loss.
        # sign(tgt_e) is always ±1 here (exact_zero excluded from is_edge).
        dir_reward = -torch.mean(
            torch.sign(pred_e) * torch.sign(tgt_e) * qual_e * (pred_e.abs() / (tgt_e.abs() + 1e-6)).clamp(max=1.0)
        )

        # ── E: MAGNITUDE SHORTFALL ───────────────────────────────────────────
        # Fires when |pred| < |tgt| — prediction is underpowered.
        # Symmetric with overshoot (B): together they enforce a two-sided
        # soft envelope around the target magnitude.
        # Weighted by importance (not qual_e alone) for consistency with A.
        sf_raw = torch.relu(tgt_e.abs() - pred_e.abs())
        sf_smooth = F.smooth_l1_loss(
            sf_raw, torch.zeros_like(sf_raw), beta=0.1, reduction="none"
        )
        shortfall_loss = (sf_smooth * importance).mean()

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4 — DISTRIBUTION MATCHING  (edge pool only)
    #
    # corr_penalty  : penalise low Pearson correlation
    # q_spread_loss : penalise quantile-shape mismatch
    # spread_reward : reward prediction spread (anti-collapse), gated to
    #                 bars with non-trivially-zero targets to avoid conflict
    #                 with MSE on near-zero edge targets
    # bias_penalty  : penalise global and edge-level mean shift
    # ─────────────────────────────────────────────────────────────────────────
    corr_penalty  = torch.tensor(0.0, device=pred.device)
    q_spread_loss = torch.tensor(0.0, device=pred.device)
    spread_reward = torch.tensor(0.0, device=pred.device)
    bias_penalty  = (pred.mean() - target.mean().detach()).abs()

    if is_edge.sum() > 1:
        pred_e_f = pred[is_edge].float()
        tgt_e_f  = target[is_edge].float()

        # Pearson correlation
        pred_c   = pred_e_f - pred_e_f.mean()
        tgt_c    = tgt_e_f  - tgt_e_f.mean().detach()
        pred_std = _safe_std(pred_e_f)
        tgt_std  = _safe_std(tgt_e_f).detach()
        corr         = (pred_c * tgt_c.detach()).mean() / (pred_std * tgt_std)
        corr_penalty = (1.0 - corr).clamp(min=0.0, max=2.0)

        # Quantile shape
        q_spread_loss = _quantile_spread_loss(pred_e_f, tgt_e_f)

        # Spread reward — gated to bars where the target is genuinely nonzero,
        # preventing the optimizer from increasing pred std on flat-adjacent bars
        real_signal = tgt_e_f.abs() > 0.10
        if real_signal.any():
            spread_reward = -torch.log(
                _safe_std(pred_e_f[real_signal]).clamp(min=1e-7)
            )

        # Bias — combined global and edge-level
        global_bias  = (pred.mean()     - target.mean().detach()).abs()
        edge_bias    = (pred_e_f.mean() - tgt_e_f.mean().detach()).abs()
        bias_penalty = 0.5 * global_bias + 0.5 * edge_bias

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5 — MODERATE PINBALL LOSS  (|tgt| < 0.5 edge bars only)
    #
    # Pinball at q=0.15/0.85 pushes predictions into the ±[0.1, 0.5] range.
    # Filtered to moderate-only BEFORE calling _moderate_bucket_loss so that
    # large-target bars cannot shift the quantile anchors.
    # The large_signal boundary (0.5) is retained here as a semantic filter
    # for the pinball distribution target — not as a gradient cliff.
    # ─────────────────────────────────────────────────────────────────────────
    mod_bucket_loss = torch.tensor(0.0, device=pred.device)
    if is_edge.any():
        mod_only = target[is_edge].abs() < 0.5
        if mod_only.any():
            mod_bucket_loss = _moderate_bucket_loss(
                pred[is_edge][mod_only],
                target[is_edge][mod_only],
            )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6 — GLOBAL STD FLOOR
    #
    # Backstop against degenerate zero-variance collapse of the full
     # prediction distribution.  Fires only when pred.std() < 0.20.
    # ─────────────────────────────────────────────────────────────────────────
    global_std_floor = torch.tensor(0.0, device=pred.device)
    if pred.numel() > 1:
        global_std_floor = F.relu(0.20 - pred.std())

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7 — COMBINE
    # ─────────────────────────────────────────────────────────────────────────
    total = (
        # ── magnitude ────────────────────────────────────────────────────────
        edge_mse                                     # A: importance-weighted MSE
        + overshoot_loss                             # B: sign-aware overshoot
        + shortfall_weight   * shortfall_loss        # E: magnitude shortfall
        # ── direction ────────────────────────────────────────────────────────
        + penalty_weight     * dir_penalty           # C: wrong-sign penalty
        + 1.0                * dir_reward            # D: correct-sign reward
        # ── false-signal ─────────────────────────────────────────────────────
        + false_signal_weight * false_signal_loss    # Step 2: Oracle no-trade bars
        # ── distribution ─────────────────────────────────────────────────────
        + dispersion_weight  * corr_penalty          # Pearson correlation penalty
        + 0.30               * q_spread_loss         # quantile shape penalty
        + bias_weight        * bias_penalty          # global + edge bias
        + 0.50               * spread_reward         # std reward (anti-collapse)
        # ── moderate pinball ─────────────────────────────────────────────────
        + 0.40               * mod_bucket_loss       # pinball on moderate edge only
        # ── global floor ─────────────────────────────────────────────────────
        + 1.0                * global_std_floor      # std floor backstop
    )

    return total
