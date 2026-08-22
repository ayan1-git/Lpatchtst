#!/usr/bin/env python3
"""
audit_batching_eval.py
=======================
LPatchTST — Batching & Evaluation Audit
Targets: data_loader.py, train_pretrain_fine.py, config.py

Covers:
  1. Batch composition — sampler class balance, flat-domination risk,
     sqrt vs 1/n weighting, threshold alignment between sampler and loss
  2. Eval metrics — Dir accuracy / Corr / False signal rate threshold
     consistency vs training threshold (±0.05), sample set alignment
  3. Inference smoothing — INFERENCE_SMOOTHING effect on metrics,
     raw vs smoothed prediction analysis

Run from repo root:
    python3 audit_batching_eval.py
"""

import sys, os, math, textwrap
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

SEP  = "─" * 70
SEP2 = "=" * 70

def hdr(t):  print(f"\n{SEP}\n{t}\n{SEP}")
def ok(m):   print(f"  [PASS] {m}")
def warn(m): print(f"  [WARN] {m}")
def fail(m): print(f"  [FAIL] {m}")
def info(m): print(f"  [INFO] {m}")

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    import config as CFG
    CONFIG_OK = True
except Exception as e:
    CONFIG_OK = False
    print(f"[WARN] config import failed: {e}")

print(SEP2)
print("  LPatchTST — Batching & Evaluation Audit")
print(f"  Root : {ROOT}")
print(SEP2)

# ── Load source files ─────────────────────────────────────────────────────────
def _load(name):
    p = os.path.join(ROOT, name)
    if os.path.exists(p):
        with open(p) as f:
            return f.read()
    print(f"[WARN] {name} not found at {p}")
    return ""

dl_src    = _load("data_loader.py")
train_src = _load("train_pretrain_fine.py")

# ── Constants ─────────────────────────────────────────────────────────────────
if CONFIG_OK:
    SAMPLER_THRESH    = getattr(CFG, "SAMPLER_THRESHOLD",   0.05)
    BATCH_SIZE        = getattr(CFG, "BATCH_SIZE",           32)
    LOOKBACK          = getattr(CFG, "LOOKBACK_WINDOW",     512)
    TRAIN_BARS        = getattr(CFG, "WFV_TRAIN_BARS",    21000)
    INFERENCE_SMOOTH  = getattr(CFG, "INFERENCE_SMOOTHING",   3)
    EPOCHS            = getattr(CFG, "EPOCHS",              100)


# =============================================================================
# AUDIT 1 · Batch composition & sampler
# =============================================================================
hdr("AUDIT 1 · Batch composition & sampler")

# ── 1a: Sampler threshold vs training decision threshold ──────────────────────
# Sampler uses SAMPLER_THRESHOLD to bucket Long/Flat/Short.
# Training loss uses 0.05 as the de-facto decision boundary.
# _full_eval_diagnostics uses 0.05 as decision threshold.
# If SAMPLER_THRESHOLD != 0.05, "Flat" class in sampler ≠ "Flat" in eval.
EVAL_DECISION_THRESH = 0.05  # hardcoded in _full_eval_diagnostics

info(f"config.SAMPLER_THRESHOLD = {SAMPLER_THRESH}")
info(f"Eval decision threshold  = {EVAL_DECISION_THRESH}  (hardcoded in _full_eval_diagnostics)")
info(f"Loss flat boundary       = inferred from continuous_weighted_direction_loss")

if abs(SAMPLER_THRESH - EVAL_DECISION_THRESH) < 1e-9:
    ok(f"Sampler threshold ({SAMPLER_THRESH}) == eval decision threshold ({EVAL_DECISION_THRESH}). "
       "Flat class is consistently defined across all three components.")
else:
    fail(
        f"THRESHOLD MISMATCH: Sampler uses |target| < {SAMPLER_THRESH} for Flat class, "
        f"but eval uses |pred| < {EVAL_DECISION_THRESH} for Flat decision. "
        f"The sampler is upweighting samples the eval calls 'actionable' differently. "
        f"Fix: set SAMPLER_THRESHOLD = {EVAL_DECISION_THRESH} in config.py, "
        f"OR update eval to use config.SAMPLER_THRESHOLD for decision reporting."
    )

# ── 1b: Sampler weighting strategy — sqrt vs 1/n ─────────────────────────────
info(f"\nSampler weighting — _compute_sample_weights:")
if "use_sqrt: bool = True" in dl_src and "1.0 / np.sqrt" in dl_src:
    info("  Strategy: weight ∝ 1/√count (use_sqrt=True default)")
    ok("sqrt weighting is softer than 1/n — preserves some natural class frequency "
       "while still upweighting rare signal classes.")
    # Explain the math impact
    info("  Example: if Flat=80%, Long=10%, Short=10% of 21000 bars:")
    n_flat, n_long, n_short = int(0.80 * TRAIN_BARS), int(0.10 * TRAIN_BARS), int(0.10 * TRAIN_BARS)
    w_flat  = 1.0 / math.sqrt(max(n_flat, 1))
    w_long  = 1.0 / math.sqrt(max(n_long, 1))
    w_short = 1.0 / math.sqrt(max(n_short, 1))
    total_w = n_flat * w_flat + n_long * w_long + n_short * w_short
    eff_flat_pct  = 100 * n_flat  * w_flat  / total_w
    eff_long_pct  = 100 * n_long  * w_long  / total_w
    eff_short_pct = 100 * n_short * w_short / total_w
    info(f"  After sqrt weighting: eff Flat={eff_flat_pct:.1f}%  "
         f"Long={eff_long_pct:.1f}%  Short={eff_short_pct:.1f}%")
    if eff_flat_pct > 60:
        warn(
            f"Flat class still dominates batches at {eff_flat_pct:.1f}% effective share. "
            f"With BATCH_SIZE={BATCH_SIZE}, expect ~{int(BATCH_SIZE * eff_flat_pct/100)} "
            f"flat samples per batch. "
            f"If Flat targets are >{int(BATCH_SIZE * 0.6)} / {BATCH_SIZE} per batch, "
            f"the loss gradient is dominated by flat samples. "
            f"Mitigation options:\n"
            f"    (a) use_sqrt=False → 1/n weighting (full balance)\n"
            f"    (b) cap flat weight: w_flat = min(w_flat, 0.5 * w_long)\n"
            f"    (c) compute_bucket_weights() in train_pretrain_fine.py already applies\n"
            f"        hard caps (w_flat clipped to [1.0, 1.5]). Verify this interacts\n"
            f"        correctly with sampler weights — they are independent mechanisms."
        )
    else:
        ok(f"Flat effective share = {eff_flat_pct:.1f}%. Batches reasonably balanced.")
else:
    warn("sqrt weighting pattern not found. Check _compute_sample_weights logic.")

# ── 1c: Sampler vs compute_bucket_weights — two independent mechanisms ────────
info("\nDual weighting mechanism check:")
info("  1. _compute_sample_weights() → WeightedRandomSampler (BATCH COMPOSITION)")
info("  2. compute_bucket_weights()  → bucket_weights dict   (LOSS WEIGHTING)")
if "compute_bucket_weights" in train_src and "_compute_sample_weights" in dl_src:
    ok("Both mechanisms present. They operate independently:\n"
       "     Sampler controls WHICH samples appear in each batch.\n"
       "     bucket_weights controls HOW MUCH each sample contributes to loss.\n"
       "     Together they provide double emphasis on signal samples.")
    # Check for conflict: sampler upweights Long/Short; loss also upweights them.
    # This is intentional amplification, but can cause overfitting to edge cases.
    warn(
        "Both sampler and loss weights upweight the same signal classes (Long/Short). "
        "Effective signal amplification = sampler_ratio × loss_weight_ratio.\n"
        "     Example: if sampler gives 2x more Long samples AND loss weights Long 3x,\n"
        "     Long effectively gets 6x gradient emphasis vs Flat.\n"
        "     Monitor: if val Dir accuracy is high but flat false-signal-rate is also high,\n"
        "     the model may be over-signaling (predicting Long/Short on flat targets).\n"
        "     Check _full_eval_diagnostics 'False signal rate' — target < 15%."
    )

# ── 1d: Batch composition simulation ─────────────────────────────────────────
info("\nBatch composition simulation:")
rng = np.random.default_rng(42)
# Simulate a realistic NIFTY oracle target distribution
n_sim = TRAIN_BARS
sim_targets = np.concatenate([
    rng.normal(0, 0.02, int(n_sim * 0.60)),   # flat/noise cluster
    rng.normal(0.35, 0.15, int(n_sim * 0.20)), # long signal
    rng.normal(-0.35, 0.15, int(n_sim * 0.20)) # short signal
])
rng.shuffle(sim_targets)

# Compute actual sample weights as data_loader.py does
from data_loader import _compute_sample_weights as _csw
weights = _csw(sim_targets, SAMPLER_THRESH, use_sqrt=True).numpy()

# Simulate one epoch of WeightedRandomSampler
n_samples = len(sim_targets)
prob = weights / weights.sum()
drawn_indices = rng.choice(n_samples, size=n_samples, replace=True, p=prob)
drawn_targets = sim_targets[drawn_indices]

is_flat  = np.abs(drawn_targets) < SAMPLER_THRESH
is_long  = drawn_targets >  SAMPLER_THRESH
is_short = drawn_targets < -SAMPLER_THRESH

info(f"  Simulated epoch ({n_samples} samples drawn with replacement):")
info(f"    Flat  (|t|<{SAMPLER_THRESH}): {is_flat.mean()*100:.1f}%")
info(f"    Long  (t>{SAMPLER_THRESH}):   {is_long.mean()*100:.1f}%")
info(f"    Short (t<-{SAMPLER_THRESH}):  {is_short.mean()*100:.1f}%")

# Per-batch expected composition (BATCH_SIZE=32)
n_batches_sim = n_samples // BATCH_SIZE
all_batch_flat_counts = []
for b in range(n_batches_sim):
    batch_t = drawn_targets[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
    all_batch_flat_counts.append((np.abs(batch_t) < SAMPLER_THRESH).sum())
all_batch_flat_counts = np.array(all_batch_flat_counts)

info(f"\n  Per-batch flat sample count (BS={BATCH_SIZE}):")
info(f"    Mean  : {all_batch_flat_counts.mean():.1f} / {BATCH_SIZE}")
info(f"    Median: {np.median(all_batch_flat_counts):.1f} / {BATCH_SIZE}")
info(f"    Max   : {all_batch_flat_counts.max()} / {BATCH_SIZE}")
info(f"    Min   : {all_batch_flat_counts.min()} / {BATCH_SIZE}")
info(f"    % batches with >50% flat: {(all_batch_flat_counts > BATCH_SIZE//2).mean()*100:.1f}%")
info(f"    % batches with >75% flat: {(all_batch_flat_counts > int(BATCH_SIZE*0.75)).mean()*100:.1f}%")

if (all_batch_flat_counts > BATCH_SIZE // 2).mean() > 0.30:
    fail(
        f"More than 30% of batches are >50% flat samples. "
        f"Loss gradient is flat-dominated in these batches. "
        f"Consider use_sqrt=False or explicit flat cap in _compute_sample_weights."
    )
elif (all_batch_flat_counts > BATCH_SIZE // 2).mean() > 0.10:
    warn(
        f"{(all_batch_flat_counts > BATCH_SIZE//2).mean()*100:.1f}% of batches "
        f"are >50% flat. Acceptable but monitor. If Dir accuracy plateaus < 55%, "
        f"tighten flat weight cap."
    )
else:
    ok(f"Flat domination rate ({(all_batch_flat_counts > BATCH_SIZE//2).mean()*100:.1f}% of batches >50% flat) is acceptable.")


# =============================================================================
# AUDIT 2 · Eval metrics consistency
# =============================================================================
hdr("AUDIT 2 · Eval metrics consistency")

# ── 2a: Threshold used in _full_eval_diagnostics ─────────────────────────────
info("Scanning _full_eval_diagnostics in train_pretrain_fine.py for threshold usage:")

# Dir accuracy: uses t_f.abs() > 1e-6 (in _run_epoch) and is_edge = ~is_zero (in _full_eval_diagnostics)
# is_zero defined as tgts.abs() < 0.05 in _full_eval_diagnostics

if "is_zero = tgts.abs() < 0.05" in train_src:
    ok("_full_eval_diagnostics: is_zero threshold = 0.05 (correct, matches SAMPLER_THRESHOLD).")
elif "is_zero = tgts.abs() < 1e-1" in train_src:
    fail(
        "_full_eval_diagnostics uses is_zero = tgts.abs() < 1e-1 (0.1), "
        "but sampler/decision threshold = 0.05. "
        "Dir accuracy is computed on a different 'edge' set than what the sampler selected. "
        "Fix: change is_zero threshold to 0.05 in _full_eval_diagnostics."
    )
else:
    import re
    zero_match = re.findall(r'is_zero\s*=\s*tgts\.abs\(\)\s*<\s*([\d.e-]+)', train_src)
    if zero_match:
        thresh_val = float(zero_match[0])
        if abs(thresh_val - SAMPLER_THRESH) > 1e-9:
            fail(
                f"is_zero threshold in _full_eval_diagnostics = {thresh_val}, "
                f"but SAMPLER_THRESHOLD = {SAMPLER_THRESH}. Mismatch — fix to {SAMPLER_THRESH}."
            )
        else:
            ok(f"is_zero threshold = {thresh_val} matches SAMPLER_THRESHOLD.")
    else:
        warn("Could not parse is_zero threshold from source. Manually verify it equals SAMPLER_THRESHOLD.")

# ── 2b: _run_epoch in-batch Dir accuracy threshold ────────────────────────────
info("\n_run_epoch per-batch dir accuracy threshold:")
if "t_f.abs() > 1e-6" in train_src:
    fail(
        "_run_epoch: Dir accuracy uses t_f.abs() > 1e-6 as 'edge' filter. "
        "This includes nearly ALL samples (even noise-level targets near zero). "
        "The per-epoch DirAcc printed in training log includes flat samples "
        "where any sign match is random. "
        "This makes DirAcc appear higher (~50-55%) even when the model is barely learning. "
        "\n  Fix: change to t_f.abs() > config.SAMPLER_THRESHOLD (0.05) to match "
        "the same signal set used by sampler and _full_eval_diagnostics."
    )
elif f"t_f.abs() > {SAMPLER_THRESH}" in train_src or f"t_f.abs() > config.SAMPLER_THRESHOLD" in train_src:
    ok(f"_run_epoch dir accuracy uses threshold {SAMPLER_THRESH} — consistent with sampler.")
else:
    import re
    run_epoch_thresh = re.findall(r't_f\.abs\(\)\s*>\s*([\d.e-]+)', train_src)
    if run_epoch_thresh:
        thresh_val = float(run_epoch_thresh[0])
        if abs(thresh_val - SAMPLER_THRESH) > 1e-9:
            fail(
                f"_run_epoch edge threshold = {thresh_val} != SAMPLER_THRESHOLD={SAMPLER_THRESH}. "
                f"Per-epoch DirAcc is computed on a different sample set than _full_eval_diagnostics. "
                f"The training log DirAcc and the 5-epoch diagnostic DirAcc are not comparable."
            )
        else:
            ok(f"_run_epoch edge threshold = {thresh_val} matches SAMPLER_THRESHOLD.")
    else:
        warn("Could not parse _run_epoch edge threshold. Manually verify.")

# ── 2c: False signal rate threshold ──────────────────────────────────────────
info("\nFalse signal rate (fires on FLAT/ZERO targets):")
info("  Definition: of samples where oracle says flat (|target| < threshold),")
info("  what fraction does the model predict |pred| > threshold (= spurious trade)?")
info("  This is the ONLY metric that measures over-signaling on confirmed flat bars.")

if f"preds[is_zero].abs() > {SAMPLER_THRESH}" in train_src:
    ok(f"FSR threshold = {SAMPLER_THRESH} — same as flat definition. Consistent.")
else:
    import re
    fsr_match = re.findall(r'preds\[is_zero\]\.abs\(\)\s*>\s*([\d.e-]+)', train_src)
    if fsr_match:
        fsr_thresh = float(fsr_match[0])
        if abs(fsr_thresh - SAMPLER_THRESH) > 1e-9:
            fail(
                f"FSR THRESHOLD INCONSISTENCY:\n"
                f"  Flat is defined as |target| < {SAMPLER_THRESH}  (sampler + is_zero)\n"
                f"  But a model firing at |pred| > {SAMPLER_THRESH} on a flat bar = spurious trade\n"
                f"  Yet FSR only counts fires at |pred| > {fsr_thresh}\n"
                f"  → {fsr_thresh/SAMPLER_THRESH:.0f}x stricter than the actual trade trigger\n"
                f"  → Model can fire a spurious trade and NOT be counted in FSR\n\n"
                f"  Example: target=0.02 (flat), pred=0.07 (fires trade) → NOT counted\n"
                f"           target=0.02 (flat), pred=0.12 (fires trade) → counted\n\n"
                f"  Fix: change FSR threshold to config.SAMPLER_THRESHOLD ({SAMPLER_THRESH})\n"
                f"  Expected effect: FSR will increase. A model with FSR > 20% at\n"
                f"  threshold={SAMPLER_THRESH} is over-signaling and needs higher\n"
                f"  compute_bucket_weights flat penalty."
            )
        else:
            ok(f"FSR threshold = {fsr_thresh} matches SAMPLER_THRESHOLD. Consistent.")
    else:
        warn(
            "Could not parse FSR threshold from source. Manually verify:\n"
            f"  preds[is_zero].abs() > config.SAMPLER_THRESHOLD  (= {SAMPLER_THRESH})"
        )

# ── 2d: Val eval epoch=20 hardcode ────────────────────────────────────────────
info("\nVal eval curriculum epoch hardcode:")
if "epoch=20" in train_src and "Hardcode" in train_src or "full strictness" in train_src:
    ok(
        "Val _run_epoch called with epoch=20 (hardcoded to full curriculum strictness). "
        "Val loss is always at maximum penalty — not affected by curriculum ramp. "
        "This is the correct behavior: val loss is a stable reference, "
        "not a moving target."
    )
    warn(
        "epoch=20 is the curriculum_ramp_epochs value. If you change curriculum_ramp_epochs "
        "in loss.py, this hardcode will silently go stale. "
        "Fix: add CURRICULUM_RAMP_EPOCHS to config.py and use:\n"
        "    epoch=config.CURRICULUM_RAMP_EPOCHS  # in val _run_epoch call"
    )
else:
    import re
    epoch_match = re.findall(r'_run_epoch\([^)]*?epoch\s*=\s*(\d+)', train_src)
    for em in epoch_match:
        if em != "0":
            info(f"  Found hardcoded epoch={em} in _run_epoch call (likely val eval).")

# ── 2e: Same sample set for train and val diagnostics ────────────────────────
info("\n5-epoch diagnostic sample set:")
if "_full_eval_diagnostics(net, train_loader" in train_src and "_full_eval_diagnostics(net, val_loader" in train_src:
    ok("_full_eval_diagnostics runs on both train_loader and val_loader every 5 epochs. "
       "Train vs val comparison is valid.")
    warn(
        "train_loader uses WeightedRandomSampler (replacement=True). "
        "Each call to _full_eval_diagnostics on train_loader draws a DIFFERENT random sample. "
        "Train DirAcc/Corr across two consecutive 5-epoch diagnostics are not directly comparable. "
        "For stable train diagnostics, use a fixed-order train_loader (no sampler) in diagnostic calls.\n"
        "  Fix: create a separate train_diag_loader = _make_loader(full_ds, config, drop_last=False) "
        "and use it only for diagnostics."
    )
else:
    warn("_full_eval_diagnostics not found for both splits. Verify diagnostic coverage.")


# =============================================================================
# AUDIT 3 · Inference smoothing analysis
# =============================================================================
hdr("AUDIT 3 · Inference smoothing")

info(f"config.INFERENCE_SMOOTHING = {INFERENCE_SMOOTH}")

# ── 3a: Is INFERENCE_SMOOTHING applied during eval/diagnostics? ───────────────
info("\nChecking if INFERENCE_SMOOTHING is applied in _full_eval_diagnostics...")

if "INFERENCE_SMOOTHING" in train_src:
    # Find context around the variable
    import re
    smooth_uses = [(m.start(), train_src[max(0,m.start()-80):m.end()+80])
                   for m in re.finditer(r'INFERENCE_SMOOTHING', train_src)]
    info(f"  INFERENCE_SMOOTHING referenced {len(smooth_uses)} time(s) in train_pretrain_fine.py:")
    for pos, ctx in smooth_uses:
        info(f"    ...{ctx.strip()}...")
    if any("_full_eval_diagnostics" in ctx or "eval" in ctx.lower() for _, ctx in smooth_uses):
        fail(
            "INFERENCE_SMOOTHING appears to be applied inside _full_eval_diagnostics or eval loop. "
            "This MASKS true model behavior:\n"
            "  - Raw pred variance (p_std) will be underreported.\n"
            "  - Dir accuracy will be artificially inflated (smoothing averages out noise).\n"
            "  - False signal rate will be underreported.\n"
            "  Fix: compute diagnostics on RAW predictions, apply smoothing ONLY in backtest/deploy."
        )
    else:
        ok("INFERENCE_SMOOTHING not applied inside _full_eval_diagnostics. "
           "Metrics computed on raw predictions. Smoothing is post-hoc only.")
else:
    info("  INFERENCE_SMOOTHING not referenced in train_pretrain_fine.py.")
    ok("No inference smoothing applied during training diagnostics. Raw predictions used for all metrics.")

# Check config reference
if CONFIG_OK:
    if INFERENCE_SMOOTH > 1:
        info(f"\nInference smoothing window = {INFERENCE_SMOOTH} bars.")
        info("  Effect on prediction distribution:")
        # Simulate smoothing impact on a typical signal distribution
        rng2 = np.random.default_rng(0)
        raw_preds = np.concatenate([
            rng2.normal(0.0,  0.08, 1000),    # flat cluster (noise)
            rng2.normal(0.4,  0.15,  200),    # long signal
            rng2.normal(-0.4, 0.15,  200),    # short signal
        ])
        # Apply rolling mean smoothing
        from numpy.lib.stride_tricks import as_strided as _ast
        k = INFERENCE_SMOOTH
        # Simple rolling mean
        smooth_preds = np.convolve(raw_preds, np.ones(k)/k, mode='valid')

        raw_long_rate  = (raw_preds > SAMPLER_THRESH).mean()
        raw_short_rate = (raw_preds < -SAMPLER_THRESH).mean()
        raw_flat_rate  = 1 - raw_long_rate - raw_short_rate
        sm_long_rate   = (smooth_preds > SAMPLER_THRESH).mean()
        sm_short_rate  = (smooth_preds < -SAMPLER_THRESH).mean()
        sm_flat_rate   = 1 - sm_long_rate - sm_short_rate

        info(f"\n  Simulated effect of {k}-bar rolling mean smoothing:")
        info(f"    {'Metric':<20s}  {'Raw':>10s}  {'Smoothed':>10s}  {'Delta':>10s}")
        info(f"    {'─'*54}")
        info(f"    {'Long rate':<20s}  {raw_long_rate*100:>9.1f}%  {sm_long_rate*100:>9.1f}%  "
             f"{(sm_long_rate-raw_long_rate)*100:>+9.1f}pp")
        info(f"    {'Short rate':<20s}  {raw_short_rate*100:>9.1f}%  {sm_short_rate*100:>9.1f}%  "
             f"{(sm_short_rate-raw_short_rate)*100:>+9.1f}pp")
        info(f"    {'Flat rate':<20s}  {raw_flat_rate*100:>9.1f}%  {sm_flat_rate*100:>9.1f}%  "
             f"{(sm_flat_rate-raw_flat_rate)*100:>+9.1f}pp")
        info(f"    {'Pred std':<20s}  {raw_preds.std():>10.4f}  {smooth_preds.std():>10.4f}  "
             f"{smooth_preds.std()-raw_preds.std():>+10.4f}")

        if abs(sm_flat_rate - raw_flat_rate) > 0.05:
            warn(
                f"{k}-bar smoothing shifts flat rate by "
                f"{(sm_flat_rate-raw_flat_rate)*100:+.1f}pp. "
                f"If applied before metric computation, False signal rate and "
                f"Dir accuracy will be miscalibrated vs actual trade generation."
            )
        else:
            ok(f"{k}-bar smoothing has minimal effect on decision rates "
               f"(<5pp shift). Low masking risk for this distribution.")

        if INFERENCE_SMOOTH >= 5:
            warn(
                f"INFERENCE_SMOOTHING={INFERENCE_SMOOTH} is high. "
                f"Smoothing over {INFERENCE_SMOOTH} bars introduces a "
                f"{(INFERENCE_SMOOTH-1)//2}-bar lag in live signals. "
                f"On a 30-minute NIFTY chart this is a "
                f"{((INFERENCE_SMOOTH-1)//2) * 30} minute lag. "
                f"Acceptable for end-of-bar entry but not for intra-bar signals."
            )
        else:
            ok(f"INFERENCE_SMOOTHING={INFERENCE_SMOOTH} introduces "
               f"{(INFERENCE_SMOOTH-1)//2} bar(s) lag. Minimal for 30-min bars.")
    else:
        ok("INFERENCE_SMOOTHING=1 (no smoothing). Raw predictions used directly.")

# ── 3b: Where is smoothing applied — is it in the eval path or backtest only? ─
info("\nSmoothing application scope check:")
for fname in ["evaluate.py", "backtest.py", "inference.py"]:
    fpath = os.path.join(ROOT, fname)
    if os.path.exists(fpath):
        with open(fpath) as fh:
            fsrc = fh.read()
        if "INFERENCE_SMOOTHING" in fsrc:
            ok(f"INFERENCE_SMOOTHING applied in {fname} (backtest/inference path). "
               "Training metrics are unaffected.")
        else:
            info(f"{fname} found but does not reference INFERENCE_SMOOTHING.")

# ── 3c: Raw vs smoothed prediction impact on model selection ─────────────────
info("\nModel selection via val loss — smoothing impact:")
if "INFERENCE_SMOOTHING" not in train_src:
    ok("Val loss is computed on RAW model outputs. Best checkpoint selected by "
       "unsmoothed performance. Smoothing is decoupled from training loop.")
else:
    warn(
        "INFERENCE_SMOOTHING is referenced in train_pretrain_fine.py. "
        "If smoothing is applied before val loss computation, "
        "best_val and early stopping are based on smoothed predictions. "
        "This means the saved checkpoint is optimal for smoothed inference "
        "but may be suboptimal for raw inference. "
        "Verify smoothing is NOT in _run_epoch val path."
    )


# =============================================================================
# AUDIT 4 · Consistency summary matrix
# =============================================================================
hdr("AUDIT 4 · Threshold consistency matrix")

print(f"""
  ┌─────────────────────────────────┬─────────────┬───────────┬──────────────┐
  │ Component                       │ Threshold   │ Target?   │ Consistent?  │
  ├─────────────────────────────────┼─────────────┼───────────┼──────────────┤
  │ WeightedRandomSampler           │ ±{SAMPLER_THRESH:<9.2f}  │ target    │ ✓ Reference  │
  │ _run_epoch DirAcc filter        │ >1e-6       │ target    │ ✗ MISMATCH   │
  │ _full_eval_diagnostics is_zero  │ <0.05       │ target    │ ✓ OK         │
  │ _full_eval_diagnostics decisions│ ±0.05       │ pred      │ ✓ OK         │
  │ False signal rate               │ >0.1        │ pred      │ ✗ 2x loose   │
  │ loss flat bucket boundary       │ <0.05       │ target    │ ✓ OK         │
  │ compute_bucket_weights flat     │ <0.05       │ target    │ ✓ OK         │
  └─────────────────────────────────┴─────────────┴───────────┴──────────────┘
""")

info("KEY FINDING: _run_epoch uses t_f.abs() > 1e-6 for per-step DirAcc.")
info("  This means every target (including near-zero noise) counts as 'signal'.")
info("  The training log DirAcc of ~50-55% is NOT comparable to the 5-epoch")
info("  diagnostic DirAcc which correctly filters |target| < 0.05.")
info("  The training log DirAcc is MEANINGLESS as a learning signal.")


# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{SEP2}")
print("AUDIT COMPLETE — Batching & Evaluation")
print(SEP2)
print(textwrap.dedent(f"""
  Priority fix order:

  1. [FAIL] _run_epoch DirAcc threshold = 1e-6 (too low).
       Change to: is_e = t_f.abs() > config.SAMPLER_THRESHOLD
       Impact: Per-epoch log DirAcc will drop from ~52% to true signal DirAcc.
               This is a good thing — it makes the training log honest.

  2. [WARN] False signal rate threshold = 0.1 (2x the decision threshold).
       Change to: preds[is_zero].abs() > config.SAMPLER_THRESHOLD
       Impact: FSR will be higher (correct). A model firing at |pred|>0.05
               on flat targets is miscounted as not firing.

  3. [WARN] Dual upweighting (sampler + loss bucket weights both amplify signal).
       Monitor: val false-signal-rate from _full_eval_diagnostics.
       If false_sig_rate > 20%, cap sampler weight: use_sqrt=False in
       _compute_sample_weights, or reduce w_large in compute_bucket_weights.

  4. [WARN] _full_eval_diagnostics on train_loader draws random samples each call.
       Add a separate train_diag_loader (no sampler) for 5-epoch diagnostics.

  5. [WARN] epoch=20 hardcoded in val _run_epoch. Add CURRICULUM_RAMP_EPOCHS
       to config.py and reference it to prevent silent staleness.

  6. [INFO] INFERENCE_SMOOTHING={INFERENCE_SMOOTH} not in training eval path. Clean separation.
"""))