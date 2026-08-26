#!/usr/bin/env python3
"""
audit_model_optimization.py
===========================
LPatchTST — Model & Optimization Audit  (v2 feature pipeline, features_only)

Run from repo root:
    python3 audits/scripts/audit_model_optimization.py

Adapts the legacy audit to the current pipeline:
  • Single source of truth: core/train.py (no separate pretrain/fine file)
  • INPUT_MODE = "features_only"  → tokenizer paths skipped
  • v2 features: 9 close-only model inputs (4 OHLC raw + 9 engineered = 13 if OHLC appended)
  • QuantileHead (Falcon-2.0) instead of scalar prediction head
  • 4 param groups: head_decay / head_no_decay / enc_decay / enc_no_decay
  • AMP + GradScaler with scheduler.step() guarded on scale recovery

Covers:
  1.  LR schedule & unfreeze plan
  2.  Gradient clipping, AMP skip-step, grad norm logging
  3.  Weight decay / dropout combo, no-decay param group correctness
  4.  Parameter initialization & input dim match (v2 9-feature column count)
  5.  OneCycleLR shape simulation
  6.  WalkForward + WFV constants sanity
"""

import os
import re
import sys
import math
import textwrap
import inspect as _inspect
import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "core"))
sys.path.insert(0, _REPO)

SEP  = "─" * 70
SEP2 = "=" * 70

def hdr(t: str) -> None:
    print(f"\n{SEP}\n{t}\n{SEP}")
def ok(m:   str) -> None: print(f"  [PASS] {m}")
def warn(m: str) -> None: print(f"  [WARN] {m}")
def fail(m: str) -> None: print(f"  [FAIL] {m}")
def info(m: str) -> None: print(f"  [INFO] {m}")

# ── Imports ──────────────────────────────────────────────────────────────────
try:
    import config as CFG
    CONFIG_OK = True
except Exception as e:
    CONFIG_OK = False; print(f"[WARN] config import failed: {e}")

try:
    from model import LPatchTST, PatchTST, InputMode
    MODEL_OK = True
except Exception as e:
    MODEL_OK = False; print(f"[WARN] model import failed: {e}")

try:
    from features import FeatureConfig
    FEAT_OK = True
except Exception as e:
    FEAT_OK = False; print(f"[WARN] features import failed: {e}")


print(SEP2)
print("  LPatchTST — Model & Optimization Audit  (v2 features_only)")
print(f"  Repo : {_REPO}")
print(SEP2)


# ── Config snapshot ───────────────────────────────────────────────────────────
def _g(name, default):
    return getattr(CFG, name, default) if CONFIG_OK else default

LR_BASE          = _g("LEARNING_RATE",      1e-5)
PRETRAIN_LR      = _g("PRETRAIN_LR",         3e-5)
PRETRAIN_EPOCHS  = _g("PRETRAIN_EPOCHS",     50)
PRETRAIN_WD      = _g("PRETRAIN_WEIGHT_DECAY", 1e-2)
FINETUNE_HEAD_LR = _g("FINETUNE_HEAD_LR",    1e-5)
FINETUNE_FULL_LR = _g("FINETUNE_FULL_LR",    5e-6)
FINETUNE_WD      = _g("FINETUNE_WEIGHT_DECAY", 5e-3)
DROPOUT          = _g("DROPOUT",             0.2)
FINETUNE_DROPOUT = _g("FINETUNE_DROPOUT",    0.2)
GRAD_CLIP        = _g("GRAD_CLIP",           5.0)
PRETRAIN_CLIP    = _g("PRETRAIN_GRAD_CLIP",  5.0)
FTUNE_CLIP_A     = _g("FINETUNE_GRAD_CLIP_STAGE_A", 5.0)
FTUNE_CLIP_B     = _g("FINETUNE_GRAD_CLIP_STAGE_B", 5.0)
USE_AMP          = _g("USE_AMP",             True)
BATCH_SIZE       = _g("BATCH_SIZE",          128)
D_MODEL          = _g("D_MODEL",             64)
N_HEADS          = _g("N_HEADS",             8)
N_LAYERS         = _g("N_LAYERS",            4)
PATCH_LEN        = _g("PATCH_LEN",           16)
STRIDE           = _g("STRIDE",              12)
LSTM_LAYERS      = _g("LSTM_LAYERS",         1)
SEQ_LEN          = _g("LOOKBACK_WINDOW",     512)
INPUT_MODE       = _g("INPUT_MODE",          "features_only")
USE_LPATCHTST    = _g("USE_LPATCHTST",       True)
AGGREGATION      = _g("AGGREGATION_MODE",    "mixing")
QUANTILE_LEVELS  = _g("QUANTILE_LEVELS",     [0.02, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.98])
QUANTILE_HEAD    = _g("QUANTILE_HEAD",       True)
QUANTILE_HIDDEN  = _g("QUANTILE_HEAD_HIDDEN_MULT", 2)
WFV_ENABLED      = _g("WFV_ENABLED",         True)
WFV_TRAIN_BARS   = _g("WFV_TRAIN_BARS",      21000)
NUM_FEATURES_CFG = _g("NUM_FEATURES",        None)
SAMPLER_THR      = _g("SAMPLER_THRESHOLD",   0.08)
ORACLE_THR       = _g("ORACLE_THRESHOLD",    0.08)


# ── Source for code-pattern scans ────────────────────────────────────────────
train_py = os.path.join(_REPO, "core", "train.py")
src = ""
if os.path.exists(train_py):
    with open(train_py) as f:
        src = f.read()
    info(f"Source scan: {os.path.basename(train_py)}")
else:
    warn(f"core/train.py not found at {train_py}")


# ═════════════════════════════════════════════════════════════════════════════
# AUDIT 1 · LR schedule & unfreeze plan
# ═════════════════════════════════════════════════════════════════════════════
hdr("AUDIT 1 · LR schedule & unfreeze plan")

# Stage B enc LR is one-tenth of finetune_full (per train.py:1630 param group)
FT_STAGE_B_HEAD_MAX = FINETUNE_HEAD_LR / 10   # 1e-6
FT_STAGE_B_ENC_MAX  = FINETUNE_FULL_LR / 10   # 5e-7

# Find pretrain epochs in the callsite
pretrain_call_re = re.search(r"pretrain\s*\([^)]*epochs\s*=\s*(\d+)", src)
if pretrain_call_re:
    pretrain_epochs_actual = int(pretrain_call_re.group(1))
else:
    pretrain_epochs_actual = PRETRAIN_EPOCHS

info(f"config.LEARNING_RATE (legacy base)        = {LR_BASE:.2e}")
info(f"PRETRAIN_LR                              = {PRETRAIN_LR:.2e}")
info(f"PRETRAIN_EPOCHS (config)                 = {PRETRAIN_EPOCHS}")
info(f"PRETRAIN_EPOCHS (train.py call)          = {pretrain_epochs_actual}")
info(f"FINETUNE_HEAD_LR                         = {FINETUNE_HEAD_LR:.2e}")
info(f"FINETUNE_FULL_LR                         = {FINETUNE_FULL_LR:.2e}")
info(f"FT Stage B head max_lr (= head_lr/10)    = {FT_STAGE_B_HEAD_MAX:.2e}")
info(f"FT Stage B enc  max_lr (= full_lr/10)    = {FT_STAGE_B_ENC_MAX:.2e}")

# Check 1a: Pretrain peak LR
if PRETRAIN_LR < 1e-6:
    warn(f"PRETRAIN_LR={PRETRAIN_LR:.2e} is very conservative.")
elif PRETRAIN_LR > 1e-3:
    fail(f"PRETRAIN_LR={PRETRAIN_LR:.2e} is too high. Expect instability.")
else:
    ok(f"PRETRAIN_LR={PRETRAIN_LR:.2e} in acceptable range [1e-6, 1e-3].")

# Check 1b: Pretrain comment sanity (the config itself documents the prior failure mode)
if "grad-norm avg climbed" in src.lower() or "constant spike-clipping" in src.lower():
    info("Config comment notes prior over-LR failure at 5e-5 → reduced to current value.")

# Check 1c: Head LR (pretrain) > enc LR (Stage B) — strong head-first discipline
if FINETUNE_HEAD_LR > FT_STAGE_B_ENC_MAX * 5:
    ok(
        f"FINETUNE_HEAD_LR ({FINETUNE_HEAD_LR:.2e}) >> FT Stage B enc LR ({FT_STAGE_B_ENC_MAX:.2e}). "
        "Head-first tuning with conservative encoder unfreeze."
    )
else:
    warn(
        f"FINETUNE_HEAD_LR ({FINETUNE_HEAD_LR:.2e}) not much larger than enc LR "
        f"({FT_STAGE_B_ENC_MAX:.2e}). Encoder may be over-tuned before the head is well-calibrated."
    )

# Check 1d: Param groups detected
if 'head_decay' in src and 'head_no_decay' in src and 'enc_decay' in src and 'enc_no_decay' in src:
    ok("4 param groups detected: head_decay / head_no_decay / enc_decay / enc_no_decay.")
else:
    fail("Expected 4 param groups (head_decay/no_decay + enc_decay/no_decay) not found.")

# Check 1e: Stage A double-freeze — encoder lr=0.0 in head groups (lines ~1628)
if '"lr": 0.0' in src and 'requires_grad = False' in src:
    ok("Stage A double-freeze detected: encoder lr=0.0 in head param groups + requires_grad=False.")
elif 'optimizer.param_groups[0]["lr"] = 0.0' in src and 'requires_grad = False' in src:
    ok("Stage A double-freeze detected: dynamic lr=0.0 + requires_grad=False on encoder.")
else:
    fail("Stage A double-freeze not detected (lr=0.0 + requires_grad=False on encoder missing).")


# ═════════════════════════════════════════════════════════════════════════════
# AUDIT 2 · Gradient clipping, AMP, grad norm logging
# ═════════════════════════════════════════════════════════════════════════════
hdr("AUDIT 2 · Gradient clipping, AMP skip-step, grad norm logging")

info(f"config.GRAD_CLIP              = {GRAD_CLIP}")
info(f"config.PRETRAIN_GRAD_CLIP     = {PRETRAIN_CLIP}")
info(f"config.FINETUNE_GRAD_CLIP_STAGE_A = {FTUNE_CLIP_A}")
info(f"config.FINETUNE_GRAD_CLIP_STAGE_B = {FTUNE_CLIP_B}")
info(f"USE_AMP                       = {USE_AMP}")

# 2a: Stages exist
for label, val in [("PRETRAIN", PRETRAIN_CLIP), ("Stage A", FTUNE_CLIP_A), ("Stage B", FTUNE_CLIP_B)]:
    if val is None:
        warn(f"{label} grad clip not configured.")
    else:
        info(f"  {label} clip = {val}")

# 2b: Stages relationship (progressive tightening expected for safety)
if FTUNE_CLIP_A <= PRETRAIN_CLIP and FTUNE_CLIP_B <= FTUNE_CLIP_A:
    ok("Clip values non-increasing pretrain → A → B. Conservative unfreeze progression.")
elif FTUNE_CLIP_B > FTUNE_CLIP_A:
    warn(f"Stage B clip ({FTUNE_CLIP_B}) > Stage A clip ({FTUNE_CLIP_A}). Risky if encoder is unfrozen.")
else:
    warn(f"Stage A clip ({FTUNE_CLIP_A}) > pretrain clip ({PRETRAIN_CLIP}). Verify intent.")

# 2c: getattr fallback risk
if src:
    for label, key, fb in [
        ("PRETRAIN_GRAD_CLIP",            "PRETRAIN_GRAD_CLIP",            5.0),
        ("FINETUNE_GRAD_CLIP_STAGE_A",    "FINETUNE_GRAD_CLIP_STAGE_A",    5.0),
        ("FINETUNE_GRAD_CLIP_STAGE_B",    "FINETUNE_GRAD_CLIP_STAGE_B",    5.0),
    ]:
        pattern = f'getattr(config, "{key}", {fb})'
        alt     = pattern.replace('"', "'")
        if pattern in src or alt in src:
            info(f"  {label} read via getattr fallback {fb}.")

# 2d: AMP skip-step guard
if 'scale_before' in src and 'scaler_amp.get_scale()' in src and 'scale_before' in src and 'scheduler.step' in src:
    ok("AMP skip-step guard present: scheduler.step() gated on grad_scaler.get_scale() >= scale_before.")
else:
    fail(
        "AMP skip-step guard NOT detected. NaN batches may still advance the scheduler. "
        "Fix: wrap scheduler.step() inside `if scaler_amp.get_scale() >= scale_before:`."
    )

# 2e: clip_grad_norm_ coverage
if re.search(r'clip_grad_norm_\s*\(\s*net\.parameters\(\)', src):
    ok("clip_grad_norm_ called on net.parameters(). All groups covered.")
elif 'clip_grad_norm_' in src:
    warn("clip_grad_norm_ found but not directly on net.parameters(). Verify all param groups are covered.")
else:
    fail("clip_grad_norm_ not found. Gradients are not clipped — training unstable risk.")

# 2f: Grad accumulation
accum = any(p in src for p in ['GRAD_ACCUM_STEPS', 'accumulation_steps', 'accum_steps'])
if accum:
    warn("Gradient accumulation pattern detected. Verify clip_grad_norm_ fires after the full accumulation cycle.")
else:
    ok("No gradient accumulation. clip_grad_norm_ fires every optimizer step.")

# 2g: grad norm logging
if '_grad_norm' in src and 'grad_norms.append' in src:
    ok("Per-step grad norm logged and aggregated (avg + max per epoch). Exploding GN observable.")
else:
    warn("Grad norm logging not detected — exploding/vanishing GN cannot be diagnosed from logs.")

# 2h: finiteness guard
if 'torch.isfinite' in src:
    ok("torch.isfinite guard present (likely on total_norm). Inf safe-handled.")
else:
    warn("No torch.isfinite guard on total_norm. Inf may print as 'inf' and confuse diagnostics.")


# ═════════════════════════════════════════════════════════════════════════════
# AUDIT 3 · Weight decay / dropout combo
# ═════════════════════════════════════════════════════════════════════════════
hdr("AUDIT 3 · Weight decay / dropout / no-decay param group")

info(f"PRETRAIN_WEIGHT_DECAY    = {PRETRAIN_WD}")
info(f"FINETUNE_WEIGHT_DECAY    = {FINETUNE_WD}")
info(f"DROPOUT                  = {DROPOUT}")
info(f"FINETUNE_DROPOUT         = {FINETUNE_DROPOUT}")

# 3a: WD/LR per phase
for name, lr, wd in [
    ("Pretrain",      PRETRAIN_LR,      PRETRAIN_WD),
    ("FT Stage A",    FINETUNE_HEAD_LR, FINETUNE_WD),
    ("FT Stage B",    FT_STAGE_B_ENC_MAX, FINETUNE_WD),
]:
    ratio = wd / (lr + 1e-12)
    info(f"  {name:14s}  LR={lr:.2e}  WD={wd}  WD/LR={ratio:.0f}")
    if ratio > 10000:
        fail(f"{name}: WD/LR={ratio:.0f} — WD dominates. Reduce WD to 1e-3 or LR must rise.")
    elif ratio > 1000:
        warn(f"{name}: WD/LR={ratio:.0f} — aggressive decay.")
    else:
        ok(f"{name}: WD/LR={ratio:.0f} — balanced.")

# 3b: Cumulative WD shrinkage over Stage B
steps_per_epoch = max(WFV_TRAIN_BARS // BATCH_SIZE, 1)
# Pretrain total
pretrain_steps = pretrain_epochs_actual * steps_per_epoch
# Stage A and B together ~ (FINETUNE_EPOCHS - FREEZE_EPOCHS); use PRETRAIN_EPOCHS as proxy
stage_b_steps = max(pretrain_epochs_actual * steps_per_epoch, 1)
cum_wd = FT_STAGE_B_ENC_MAX * FINETUNE_WD * stage_b_steps
info(f"Stage B cumulative WD shrinkage ≈ {FT_STAGE_B_ENC_MAX:.2e} * {FINETUNE_WD} * {stage_b_steps} = {cum_wd:.4f}")
if cum_wd > 0.5:
    warn(f"Cumulative WD shrinkage = {cum_wd:.2f} — encoder may shrink meaningfully over Stage B.")
else:
    ok(f"Stage B cumulative WD shrinkage = {cum_wd:.4f} — bounded.")

# 3c: No-decay param group exists
if "weight_decay': 0.0" in src or 'weight_decay": 0.0' in src:
    ok("No-decay param group (weight_decay=0.0) found — bias / LayerNorm / embeddings excluded from WD.")
else:
    fail("No WD=0.0 param group detected — WD applied uniformly to bias/LayerNorm/embedding, harming training.")

# 3d: net.eval() at val
if 'net.eval()' in src:
    ok("net.eval() found in train.py — dropout disabled during validation.")
else:
    fail("net.eval() NOT found. Dropout is active during val — val loss becomes stochastic.")

# 3e: Triple regularization score
reg_score = 0
if DROPOUT       >= 0.3: reg_score += 1
if FINETUNE_WD   >= 0.05: reg_score += 1
if FT_STAGE_B_ENC_MAX <= 1e-6: reg_score += 1
info(f"Regularization intensity score: {reg_score}/3 (dropout={DROPOUT}, finetune_WD={FINETUNE_WD}, Stage_B_enc_lr={FT_STAGE_B_ENC_MAX:.0e})")
if reg_score >= 3:
    fail("Triple regularization: high dropout + high WD + tiny enc LR. Underfit risk.")
elif reg_score == 2:
    warn("Double regularization active. Watch val loss plateau.")
else:
    ok(f"Regularization intensity is moderate (score={reg_score}/3).")


# ═════════════════════════════════════════════════════════════════════════════
# AUDIT 4 · Parameter init & input dim match
# ═════════════════════════════════════════════════════════════════════════════
hdr("AUDIT 4 · Parameter init & input dim match")

# 4a: Compute expected n_features for v2 pipeline (close-only path)
n_feat_v2_close = 0
if FEAT_OK:
    fe_cfg = FeatureConfig(
        vol_ratio_spans=getattr(CFG, "FE_VOL_RATIO_SPANS", [10, 20]),
        vol_norm_span=getattr(CFG, "FE_VOL_NORM_SPAN", 60),
        ret_norm_horizons=getattr(CFG, "FE_RET_NORM_HORIZONS", [1, 5, 20]),
        mom_horizons=getattr(CFG, "FE_MOM_HORIZONS", [3, 10, 40]),
        target_clip=getattr(CFG, "FE_TARGET_CLIP", 20.0),
    )
    n_feat_v2_close = len(fe_cfg.feature_columns)
    info(f"v2 close-only features (from FeatureConfig.feature_columns): {n_feat_v2_close}")
    info(f"  → {fe_cfg.feature_columns}")
else:
    fail("Could not import features.py — cannot verify n_features.")

# 4b: Check train.py DEFAULT_FEATURE_LIST = 4 OHLC + 9 features
n_feat_with_ohlc = 4 + n_feat_v2_close
info(f"v2 + raw OHLC = {n_feat_with_ohlc} columns (default in train.py DEFAULT_FEATURE_LIST).")
info(f"config.NUM_FEATURES = {NUM_FEATURES_CFG}")
if NUM_FEATURES_CFG is not None and NUM_FEATURES_CFG not in (n_feat_v2_close, n_feat_with_ohlc):
    fail(
        f"config.NUM_FEATURES={NUM_FEATURES_CFG} does not match expected "
        f"{n_feat_v2_close} (close-only) or {n_feat_with_ohlc} (with OHLC)."
    )
elif NUM_FEATURES_CFG is None:
    info("config.NUM_FEATURES=None (set dynamically from feature_cols at build time).")

# 4c: Build the model
if MODEL_OK and CONFIG_OK:
    # We use the close-only count for the forward pass (cleanest case)
    n_feat = max(n_feat_v2_close, 1)
    cls = LPatchTST if USE_LPATCHTST else PatchTST
    try:
        net = cls(
            input_mode=INPUT_MODE,
            seq_len=SEQ_LEN,
            n_features=n_feat,
            s1_bits=_g("TOKENIZER_S1_BITS", 10),
            s2_bits=_g("TOKENIZER_S2_BITS", 10),
            d_model=D_MODEL,
            patch_len=PATCH_LEN,
            stride=STRIDE,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            lstm_layers=LSTM_LAYERS,
            dropout=DROPOUT,
            aggregation=AGGREGATION,
            vocab_size=2 ** _g("TOKENIZER_S1_BITS", 10),
        )
        n_params     = sum(p.numel() for p in net.parameters())
        n_trainable  = sum(p.numel() for p in net.parameters() if p.requires_grad)
        ok(f"Model instantiated ({'LPatchTST' if USE_LPATCHTST else 'PatchTST'}): {n_params:,} total, {n_trainable:,} trainable.")

        # 4d: num_patches
        num_patches = (SEQ_LEN - PATCH_LEN) // STRIDE + 1
        if num_patches < 4:
            warn(f"Only {num_patches} patches per sequence. Consider smaller PATCH_LEN or STRIDE.")
        else:
            ok(f"num_patches = (SEQ_LEN={SEQ_LEN} - PATCH_LEN={PATCH_LEN}) // STRIDE={STRIDE} + 1 = {num_patches}")

        # 4e: D_MODEL / N_HEADS divisibility
        if D_MODEL % N_HEADS == 0:
            ok(f"D_MODEL={D_MODEL} divisible by N_HEADS={N_HEADS}. (head_dim={D_MODEL // N_HEADS})")
        else:
            fail(f"D_MODEL={D_MODEL} NOT divisible by N_HEADS={N_HEADS}.")

        # 4f: Forward pass — features_only
        if INPUT_MODE == "features_only":
            B = 4
            dummy = torch.randn(B, SEQ_LEN, n_feat)
            try:
                net.eval()
                with torch.no_grad():
                    out = net(tokens=None, features=dummy)
                # Expect quantile head output: (B, Q)
                Q = len(QUANTILE_LEVELS)
                if out.shape == (B, Q):
                    ok(f"Forward pass OK: features → {tuple(out.shape)} (quantile head, Q={Q}).")
                    # Verify monotonicity (a key invariant of the QuantileHead)
                    diffs = (out[:, 1:] - out[:, :-1])
                    if (diffs >= 0).all():
                        ok("QuantileHead output is monotonically non-decreasing across levels.")
                    else:
                        fail("QuantileHead output is NOT monotonic — parametrization bug.")
                else:
                    fail(f"Forward pass output shape {tuple(out.shape)} != expected (B={B}, Q={Q}).")
            except Exception as e:
                fail(f"Forward pass FAILED: {e}")
        else:
            warn(f"INPUT_MODE='{INPUT_MODE}' — forward-pass test skipped (not features_only).")

        # 4g: Weight init audit
        init_issues = []
        for name, p in net.named_parameters():
            if p.numel() < 4:
                continue
            p_std = p.data.std().item()
            # LayerNorm/RMSNorm .weight tensors are *meant* to start as
            # constants (1.0 and possibly 0.0 for RMSNorm) — std=0 is normal.
            is_norm_weight = any(
                t in name
                for t in (
                    'norm1.weight', 'norm2.weight', 'norm3.weight',
                    'norm.weight', 'encoder_norm.weight', 'decoder_norm.weight',
                    'layer_norm.weight', 'LayerNorm.weight',
                )
            )
            if p_std < 1e-5 and 'bias' not in name and not is_norm_weight:
                init_issues.append((name, 'near-zero std', p_std))
            elif p_std > 5.0:
                init_issues.append((name, 'exploding init', p_std))
        if init_issues:
            for n, issue, v in init_issues:
                fail(f"Init [{issue}] in '{n}': std={v:.4f}")
        else:
            ok("All weight tensors have reasonable init std (1e-5 to 5.0).")

        # 4h: Top-5 largest param tensors
        sizes = sorted(((n, p.numel()) for n, p in net.named_parameters()), key=lambda x: -x[1])
        info("Top-5 largest parameter tensors:")
        for n, sz in sizes[:5]:
            info(f"  {n:<55s} {sz:>10,} params")

        # 4i: QuantileHead sanity
        if QUANTILE_HEAD and hasattr(net, "head"):
            head_dim = net.head.output_dim if hasattr(net.head, "output_dim") else None
            info(f"QuantileHead output_dim = {head_dim}, config.QUANTILE_LEVELS = {len(QUANTILE_LEVELS)}")
            if head_dim is not None and head_dim != len(QUANTILE_LEVELS):
                fail(f"QuantileHead output_dim={head_dim} != len(QUANTILE_LEVELS)={len(QUANTILE_LEVELS)}.")
            else:
                ok("QuantileHead output matches QUANTILE_LEVELS count.")

    except Exception as e:
        fail(f"Model instantiation failed: {e}")
        import traceback; traceback.print_exc()


# ═════════════════════════════════════════════════════════════════════════════
# AUDIT 5 · OneCycleLR shape simulation
# ═════════════════════════════════════════════════════════════════════════════
hdr("AUDIT 5 · OneCycleLR LR curve sanity check")

def _simulate_onecycle(label, max_lr, total_steps, pct_start, div_factor, final_div_factor):
    try:
        dummy = nn.Linear(4, 1)
        opt = torch.optim.AdamW(dummy.parameters(), lr=max_lr / div_factor)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=max_lr, total_steps=max(total_steps, 1),
            pct_start=pct_start, div_factor=div_factor, final_div_factor=final_div_factor,
        )
        lrs = []
        for _ in range(max(total_steps, 1)):
            lrs.append(sched.get_last_lr()[0])
            sched.step()
        arr = np.array(lrs)
        warm = int(total_steps * pct_start)
        peak = arr[min(warm, len(arr) - 1)]
        info(f"  [{label}] steps={total_steps} warmup={warm} init={arr[0]:.3e} peak={peak:.3e} final={arr[-1]:.3e}")
        if peak >= max_lr * 0.99:
            ok(f"[{label}] LR peaks at {peak:.3e} ≈ max_lr {max_lr:.3e}.")
        else:
            warn(f"[{label}] LR peak {peak:.3e} < max_lr {max_lr:.3e}.")
        if arr[-1] < arr[0] * 0.5:
            ok(f"[{label}] LR decays to {arr[-1]:.3e} (< initial {arr[0]:.3e}).")
        else:
            warn(f"[{label}] LR does not fully decay: final={arr[-1]:.3e} ≥ initial/2.")
    except Exception as e:
        warn(f"[{label}] simulation failed: {e}")

steps_per_epoch = max(WFV_TRAIN_BARS // BATCH_SIZE, 1)
info(f"steps_per_epoch ≈ {steps_per_epoch} (WFV_TRAIN_BARS={WFV_TRAIN_BARS} / BATCH_SIZE={BATCH_SIZE})")

info("Simulation 1: pretrain() — OneCycleLR peak=PRETRAIN_LR")
_simulate_onecycle(
    "Pretrain",
    max_lr=PRETRAIN_LR,
    total_steps=pretrain_epochs_actual * steps_per_epoch,
    pct_start=0.3,
    div_factor=10,
    final_div_factor=50,
)

info("Simulation 2: finetune Stage B encoder — OneCycleLR peak=full_lr/10")
_simulate_onecycle(
    "Stage B encoder",
    max_lr=FT_STAGE_B_ENC_MAX,
    total_steps=pretrain_epochs_actual * steps_per_epoch,
    pct_start=0.1,
    div_factor=5,
    final_div_factor=100,
)


# ═════════════════════════════════════════════════════════════════════════════
# AUDIT 6 · WalkForward constants sanity
# ═════════════════════════════════════════════════════════════════════════════
hdr("AUDIT 6 · WalkForward / WFV constants")

if CONFIG_OK:
    wfv_val   = _g("WFV_VAL_BARS",  2500)
    wfv_step  = _g("WFV_STEP_BARS", 2500)
    wfv_folds = _g("WFV_MIN_FOLDS", 3)
    wfv_pat   = _g("WFV_PATIENCE",  20)
    info(f"WFV_TRAIN_BARS = {WFV_TRAIN_BARS}")
    info(f"WFV_VAL_BARS   = {wfv_val}")
    info(f"WFV_STEP_BARS  = {wfv_step}")
    info(f"WFV_MIN_FOLDS  = {wfv_folds}")
    info(f"WFV_PATIENCE   = {wfv_pat}")

    if WFV_ENABLED:
        if wfv_val > WFV_TRAIN_BARS:
            fail(f"WFV_VAL_BARS ({wfv_val}) > WFV_TRAIN_BARS ({WFV_TRAIN_BARS}) — val set is bigger than train.")
        elif wfv_val < WFV_TRAIN_BARS // 10:
            warn(f"WFV_VAL_BARS ({wfv_val}) is < 10% of train — val signal may be noisy.")
        else:
            ok(f"WFV_VAL_BARS / WFV_TRAIN_BARS = {wfv_val/WFV_TRAIN_BARS:.2f} — reasonable split.")

        if wfv_step > WFV_TRAIN_BARS + wfv_val:
            warn(f"WFV_STEP_BARS ({wfv_step}) > train+val — folds barely overlap.")
        else:
            ok(f"WFV_STEP_BARS = {wfv_step} (≥ train+val ensures progressing folds).")

        if wfv_folds < 3:
            warn(f"WFV_MIN_FOLDS = {wfv_folds} — fewer than 3 folds makes trend estimation noisy.")
        else:
            ok(f"WFV_MIN_FOLDS = {wfv_folds}.")
    else:
        info("WFV disabled — single split only. Don't trust val as out-of-sample.")


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP2}")
print("AUDIT COMPLETE — Model & Optimization (v2 features_only)")
print(SEP2)
print(textwrap.dedent(f"""
  Configuration snapshot
    PRETRAIN_LR={PRETRAIN_LR:.0e}  FINETUNE_HEAD_LR={FINETUNE_HEAD_LR:.0e}  FINETUNE_FULL_LR={FINETUNE_FULL_LR:.0e}
    PRETRAIN_WD={PRETRAIN_WD}  FINETUNE_WD={FINETUNE_WD}
    D_MODEL={D_MODEL}  N_HEADS={N_HEADS}  N_LAYERS={N_LAYERS}  PATCH_LEN={PATCH_LEN}  STRIDE={STRIDE}
    DROPOUT={DROPOUT}  AMP={USE_AMP}  GRAD_CLIP={GRAD_CLIP}
    INPUT_MODE='{INPUT_MODE}'  USE_LPATCHTST={USE_LPATCHTST}  AGGREGATION='{AGGREGATION}'
    QUANTILE_HEAD={QUANTILE_HEAD}  Q={len(QUANTILE_LEVELS)}  ({QUANTILE_LEVELS})
    v2 close-only features = {n_feat_v2_close}  (+ 4 OHLC raw = {n_feat_with_ohlc} in DEFAULT_FEATURE_LIST)
    WFV={WFV_ENABLED}  TRAIN_BARS={WFV_TRAIN_BARS}  VAL={_g('WFV_VAL_BARS',2500)}  STEP={_g('WFV_STEP_BARS',2500)}

  Action items (only if any [FAIL] / [WARN] above)
    1. Address any [FAIL] before next training run.
    2. Re-run after any config change.
    3. Cross-check train.py WARN lines with current code (path constants).
"""))
