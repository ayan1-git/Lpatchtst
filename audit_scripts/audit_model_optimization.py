#!/usr/bin/env python3
"""
audit_model_optimization.py
============================
LPatchTST — Model & Optimization Audit

Run from repo root:
    python3 audit_model_optimization.py

Covers:
  1. LR schedule & unfreeze plan — OneCycleLR params, max_lr vs base_lr ratio,
     frozen tokenizer handling, param-group LR multipliers
  2. Gradient clipping & norm logging — clip value, AMP interaction,
     grad accumulation detection, skip-step on NaN
  3. Weight decay / dropout combo — WD vs LR interaction, embedding WD=0,
     dropout at inference, underfitting risk
  4. Parameter initialization & input dim mismatch — model expected vs actual
     n_features, tokenizer d_in vs ohlc_returns shape
"""

import sys, os, math, re, textwrap
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

SEP  = "\u2500" * 70
SEP2 = "=" * 70

def hdr(t): print(f"\n{SEP}\n{t}\n{SEP}")
def ok(m):   print(f"  [PASS] {m}")
def warn(m): print(f"  [WARN] {m}")
def fail(m): print(f"  [FAIL] {m}")
def info(m): print(f"  [INFO] {m}")

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    import config as CFG
    CONFIG_OK = True
except Exception as e:
    CONFIG_OK = False; print(f"[WARN] config import failed: {e}")

try:
    from model import LPatchTST, InputMode
    MODEL_OK = True
except Exception as e:
    MODEL_OK = False; print(f"[WARN] model import failed: {e}")

print(SEP2)
print("  LPatchTST — Model & Optimization Audit")
print(f"  Root : {ROOT}")
print(SEP2)

# ─────────────────────────────────────────────────────────────────────────────
# Extract constants from config (with safe defaults)
# ─────────────────────────────────────────────────────────────────────────────
if CONFIG_OK:
    LR            = getattr(CFG, 'LEARNING_RATE',   1e-5)
    GRAD_CLIP     = getattr(CFG, 'GRAD_CLIP',        2.0)
    WEIGHT_DECAY  = getattr(CFG, 'WEIGHT_DECAY',     0.05)
    DROPOUT       = getattr(CFG, 'DROPOUT',          0.2)
    BATCH_SIZE    = getattr(CFG, 'BATCH_SIZE',        32)
    EPOCHS        = getattr(CFG, 'EPOCHS',           100)
    USE_AMP       = getattr(CFG, 'USE_AMP',           True)
    D_MODEL       = getattr(CFG, 'D_MODEL',           96)
    N_HEADS       = getattr(CFG, 'N_HEADS',            4)
    N_LAYERS      = getattr(CFG, 'N_LAYERS',           5)
    PATCH_LEN     = getattr(CFG, 'PATCH_LEN',         16)
    STRIDE        = getattr(CFG, 'STRIDE',            12)
    LSTM_LAYERS   = getattr(CFG, 'LSTM_LAYERS',        1)
    SEQ_LEN       = getattr(CFG, 'LOOKBACK_WINDOW',  512)
    INPUT_MODE    = getattr(CFG, 'INPUT_MODE', 'tokens_only')
    S1_BITS       = getattr(CFG, 'TOKENIZER_S1_BITS', 10)
    S2_BITS       = getattr(CFG, 'TOKENIZER_S2_BITS', 10)
    TOK_D_IN      = getattr(CFG, 'TOKENIZER_D_IN',    6)
    PRETRAIN_CLIP = getattr(CFG, 'PRETRAIN_GRAD_CLIP',        2.0)
    FTUNE_CLIP_A  = getattr(CFG, 'FINETUNE_GRAD_CLIP_STAGE_A', 1.0)
    FTUNE_CLIP_B  = getattr(CFG, 'FINETUNE_GRAD_CLIP_STAGE_B', 1.0)
    NUM_FEATURES  = getattr(CFG, 'NUM_FEATURES',      None)
    WFV           = getattr(CFG, 'WFV_ENABLED',       False)
    TRAIN_BARS    = getattr(CFG, 'WFV_TRAIN_BARS',    21000)
else:
    LR = 1e-5; GRAD_CLIP = 2.0; WEIGHT_DECAY = 0.05; DROPOUT = 0.2
    BATCH_SIZE = 32; EPOCHS = 100; USE_AMP = True; D_MODEL = 96
    N_HEADS = 4; N_LAYERS = 5; PATCH_LEN = 16; STRIDE = 12
    LSTM_LAYERS = 1; SEQ_LEN = 512; INPUT_MODE = 'tokens_only'
    S1_BITS = 10; S2_BITS = 10; TOK_D_IN = 6
    PRETRAIN_CLIP = 2.0; FTUNE_CLIP_A = 1.0; FTUNE_CLIP_B = 1.0
    NUM_FEATURES = None; WFV = False; TRAIN_BARS = 21000

# ── Source file for all code-level scans ─────────────────────────────────────
train_py = os.path.join(ROOT, 'train_pretrain_fine.py')
if not os.path.exists(train_py):
    train_py = os.path.join(ROOT, 'train.py')  # fallback
src = ""
if os.path.exists(train_py):
    with open(train_py) as f:
        src = f.read()
    info(f"Source scan: {os.path.basename(train_py)}")
else:
    warn(f"Neither train_pretrain_fine.py nor train.py found in {ROOT}")

# LR constants matching train_pretrain_fine.py callsites
PRETRAIN_MAX_LR   = 5e-5     # pretrain(), hardcoded
FINETUNE_HEAD_LR  = 3e-5     # head_lr arg in train() -> finetune_fold()
FINETUNE_FULL_LR  = 5e-6     # full_lr arg in train()
STAGE_B_HEAD_MAX  = FINETUNE_HEAD_LR / 10   # = 3e-6
STAGE_B_ENC_MAX   = FINETUNE_FULL_LR / 5    # = 1e-6  (encoder after unfreeze)
FREEZE_EPOCHS     = 5
PRETRAIN_EPOCHS   = 30       # train() calls pretrain(epochs=30)

# =============================================================================
# AUDIT 1 · LR schedule & unfreeze plan
# =============================================================================
hdr("AUDIT 1 · LR schedule & unfreeze plan")

info(f"config.LEARNING_RATE (legacy base) = {LR:.2e}")
info(f"Pretrain OneCycleLR max_lr         = {PRETRAIN_MAX_LR:.2e}  (pretrain())")
info(f"FT Stage A head max_lr             = {FINETUNE_HEAD_LR:.2e}  (finetune_fold, encoder=0.0)")
info(f"FT Stage B head max_lr             = {STAGE_B_HEAD_MAX:.2e}  (= head_lr/10)")
info(f"FT Stage B encoder max_lr          = {STAGE_B_ENC_MAX:.2e}  (= full_lr/5)")
info(f"Freeze epochs before Stage B        = {FREEZE_EPOCHS}")

# Check 1a: Pretrain peak LR adequacy
if PRETRAIN_MAX_LR < 1e-6:
    warn(f"Pretrain max_lr={PRETRAIN_MAX_LR:.2e} is very conservative. Weights may barely move.")
elif PRETRAIN_MAX_LR > 1e-4:
    fail(f"Pretrain max_lr={PRETRAIN_MAX_LR:.2e} is too high. Expect instability.")
else:
    ok(f"Pretrain max_lr={PRETRAIN_MAX_LR:.2e} in acceptable range [1e-6, 1e-4].")

# Check 1b: Stage B encoder LR vs full_lr base
if STAGE_B_ENC_MAX < FINETUNE_FULL_LR:
    ok(
        f"Stage B encoder scheduler peak ({STAGE_B_ENC_MAX:.2e}) < full_lr ({FINETUNE_FULL_LR:.2e}). "
        "Gentle encoder warmup after unfreeze."
    )
else:
    warn(f"Stage B encoder max_lr ({STAGE_B_ENC_MAX:.2e}) not reduced below full_lr ({FINETUNE_FULL_LR:.2e}).")

# Check 1c: Stage A double-freeze (requires_grad=False AND lr=0.0 in param group)
if src and 'requires_grad = False' in src and ('"lr": 0.0' in src or "'lr': 0.0" in src):
    ok("Stage A: encoder frozen by BOTH requires_grad=False AND optimizer LR=0.0.")
elif src:
    fail(
        "Stage A double-freeze NOT detected. Encoder must be frozen via "
        "requires_grad=False AND param_group lr=0.0 simultaneously."
    )

# Check 1d: Stage B optimizer state pre-fill before unfreeze
if src and 'exp_avg_sq' in src and '1e-4' in src:
    ok("Stage B: exp_avg_sq pre-filled. No Adam denominator spike on first unfrozen step.")
elif src:
    fail(
        "Stage B unfreeze: optimizer state NOT pre-filled. "
        "First unfrozen step has infinite effective LR spike."
    )

# Check 1e: No WD=0 param group in train_pretrain_fine.py (bias/LayerNorm/embed decay)
info("Checking no-decay param groups in training source...")
if src and ('weight_decay": 0.0' in src or "weight_decay': 0.0" in src):
    ok("No-decay param group (weight_decay=0.0) found for bias/LayerNorm/embed.")
else:
    warn(
        "No weight_decay=0.0 param group detected. "
        "train_pretrain_fine.py applies config.WEIGHT_DECAY to both head and encoder groups. "
        "Recommendation: add a no-decay group for bias, LayerNorm.weight, and embedding vectors."
    )

# Check 1e: train_pretrain_fine.py clip stages
info(f"Grad clip stages (from config): pretrain={PRETRAIN_CLIP}, finetune_A={FTUNE_CLIP_A}, finetune_B={FTUNE_CLIP_B}")
if FTUNE_CLIP_A <= PRETRAIN_CLIP and FTUNE_CLIP_B <= FTUNE_CLIP_A:
    ok("Clip values decrease across stages (pretrain → fine_A → fine_B). Conservative unfreeze progression.")
elif FTUNE_CLIP_A > PRETRAIN_CLIP:
    warn(
        f"FINETUNE_GRAD_CLIP_STAGE_A ({FTUNE_CLIP_A}) > PRETRAIN_GRAD_CLIP ({PRETRAIN_CLIP}). "
        "Clip relaxes during fine-tune, which is risky if encoder is unfrozen. Monitor grad norms."
    )
else:
    ok("Clip stages exist and are defined.")

# =============================================================================
# AUDIT 2 · Gradient clipping & norm logging
# =============================================================================
hdr("AUDIT 2 · Gradient clipping & norm logging")

info(f"config.GRAD_CLIP = {GRAD_CLIP}")
info(f"USE_AMP          = {USE_AMP}")

# Check 2a: Clip value for an unfrozen transformer at this batch size
steps_per_epoch_est = max(TRAIN_BARS // BATCH_SIZE, 1)
info(f"Estimated steps/epoch = {steps_per_epoch_est} (WFV_TRAIN_BARS={TRAIN_BARS} / BS={BATCH_SIZE})")

if GRAD_CLIP <= 1.0:
    ok(f"GRAD_CLIP={GRAD_CLIP} is tight — good for stability when tokenizer is frozen.")
elif GRAD_CLIP <= 2.0:
    ok(f"GRAD_CLIP={GRAD_CLIP} is standard. Watch for max_gn > 5.0 in epoch logs — indicates LR too high.")
elif GRAD_CLIP <= 5.0:
    warn(
        f"GRAD_CLIP={GRAD_CLIP} is permissive for a frozen-encoder setup. "
        "If GradN max regularly exceeds 3.0 in logs, tighten to 1.0."
    )
else:
    fail(f"GRAD_CLIP={GRAD_CLIP} is too high. Gradient clipping provides no protection above ~5.0.")

# Check 2b: getattr clip fallbacks silently relax clipping
if src:
    for label, pattern in [
        ("PRETRAIN_GRAD_CLIP", 'getattr(config, "PRETRAIN_GRAD_CLIP", 5.0)'),
        ("FINETUNE_GRAD_CLIP_STAGE_A", 'getattr(config, "FINETUNE_GRAD_CLIP_STAGE_A", 5.0)'),
        ("FINETUNE_GRAD_CLIP_STAGE_B", 'getattr(config, "FINETUNE_GRAD_CLIP_STAGE_B", 5.0)'),
    ]:
        alt = pattern.replace('"', "'")
        if pattern in src or alt in src:
            warn(
                f"{label} uses getattr fallback 5.0 (not config.GRAD_CLIP={GRAD_CLIP}). "
                "If the config key is removed, clip silently relaxes 2.0 -> 5.0. "
                "Fix: change fallback to config.GRAD_CLIP."
            )

# Check 2c: AMP grad scaler — skip-step on NaN
info(f"Checking AMP skip-step logic in {os.path.basename(train_py) if train_py else 'train script'}...")
if os.path.exists(train_py):
    with open(train_py) as f:
        src = f.read()
    amp_guard = (
        'scale_before' in src and 'scheduler.step()' in src and
        ('scale_after >= scale_before' in src or 'get_scale() >= scale_before' in src)
    )
    if amp_guard:
        ok(
            "AMP skip-step guard found: scheduler.step() only called when "
            "grad_scaler scale did not drop. NaN batches will not advance LR."
        )
    else:
        fail(
            "AMP skip-step guard NOT detected. If a NaN batch triggers grad_scaler to skip "
            "the optimizer step, the scheduler still advances, causing LR/step desync. "
            "Fix: wrap scheduler.step() inside `if scale_after >= scale_before:`."
        )

# Check 2d: clip_grad_norm_ covers ALL parameters
if re.search(r'clip_grad_norm_\s*\(\s*net\.parameters\(\)', src):
    ok("clip_grad_norm_ called on net.parameters(). All groups covered.")
elif 'clip_grad_norm_' in src:
    warn(
        "clip_grad_norm_ found but not with net.parameters(). "
        "Verify it covers all param groups including embed_params."
    )
else:
    fail(f"clip_grad_norm_ not found in {os.path.basename(train_py)}. Gradients are not clipped — training unstable risk.")

# Check 2e: Gradient accumulation detection
accum_patterns = ['GRAD_ACCUM_STEPS', 'accumulation_steps', 'accum_steps',
                  '% ACCUM', 'if step % accum']
has_accum = any(p in src for p in accum_patterns)
if has_accum:
    warn(
        "Gradient accumulation pattern detected. Verify clip_grad_norm_ fires AFTER "
        "the full accumulation cycle, not each micro-step. Clipping per micro-step "
        "under-clips the effective gradient norm by sqrt(accum_steps)."
    )
else:
    ok("No gradient accumulation detected. clip_grad_norm_ fires every optimizer step. Correct.")

# Check 2f: grad norm logging coverage
if '_grad_norm' in src and 'grad_norms.append' in src:
    ok("Per-step grad norm logged and aggregated (avg + max per epoch). Exploding GN is observable.")
else:
    warn("Grad norm logging not detected. Add grad norm tracking to diagnose LR / unfreeze issues.")

# Check 2g: total_norm.isfinite() guard
if 'torch.isfinite(total_norm)' in src:
    ok("total_norm finiteness guard present — Inf grad norm replaced with 0.0 for safe logging.")
else:
    warn("No torch.isfinite(total_norm) guard. An Inf norm may print as 'inf' and confuse diagnostics.")

# =============================================================================
# AUDIT 3 · Weight decay / dropout combo
# =============================================================================
hdr("AUDIT 3 · Weight decay / dropout combo")

info(f"config.WEIGHT_DECAY = {WEIGHT_DECAY}")
info(f"config.DROPOUT      = {DROPOUT}")
info(f"config.LEARNING_RATE = {LR:.2e}")
info(f"config.EPOCHS        = {EPOCHS}")

# Check 3a: WD vs LR ratio per training phase (not one global config.LEARNING_RATE)
info(f"WEIGHT_DECAY = {WEIGHT_DECAY} — checking WD/LR per phase:")
for phase_name, phase_lr in [
    ("Pretrain",         PRETRAIN_MAX_LR),
    ("FT Stage A head",  FINETUNE_HEAD_LR),
    ("FT Stage B head",  STAGE_B_HEAD_MAX),
    ("FT Stage B enc",   STAGE_B_ENC_MAX),
]:
    wd_lr_ratio = WEIGHT_DECAY / phase_lr
    info(f"  {phase_name:18s} LR={phase_lr:.2e}  WD/LR={wd_lr_ratio:.0f}")
    if wd_lr_ratio > 10000:
        fail(
            f"{phase_name}: WD/LR={wd_lr_ratio:.0f} at LR={phase_lr:.2e}. "
            f"Weight decay dominates updates. "
            f"Fix: reduce WEIGHT_DECAY to 1e-3 (recommended before next training run)."
        )
    elif wd_lr_ratio > 1000:
        warn(
            f"{phase_name}: WD/LR={wd_lr_ratio:.0f}. Aggressive decay relative to LR. "
            "Monitor train/val loss plateau."
        )
    else:
        ok(f"{phase_name}: WD/LR={wd_lr_ratio:.0f}. Balanced.")

# Check 3b: AdamW decoupled WD — worst-case Stage B encoder cumulative shrinkage
effective_wd_per_step = STAGE_B_ENC_MAX * WEIGHT_DECAY
stage_b_steps = max((EPOCHS - FREEZE_EPOCHS) * steps_per_epoch_est, 1)
eff_wd_total = effective_wd_per_step * stage_b_steps
info(f"Stage B encoder effective WD/step = {STAGE_B_ENC_MAX:.2e} * {WEIGHT_DECAY} = {effective_wd_per_step:.2e}")
info(f"Cumulative WD shrinkage over Stage B ({stage_b_steps} steps est.) = {eff_wd_total:.4f}")
if eff_wd_total > 0.5:
    warn(
        f"Cumulative WD shrinkage = {eff_wd_total:.2f} over Stage B. "
        "Encoder weights may shrink significantly after unfreeze."
    )
else:
    ok(f"Stage B cumulative WD shrinkage = {eff_wd_total:.4f}. Encoder weights unlikely to collapse from decay alone.")

# Check 3c: Dropout during inference guard
info(f"Checking net.eval() is called before val loop in {os.path.basename(train_py)}...")
if 'net.eval()' in src:
    ok(f"net.eval() found in {os.path.basename(train_py)}. Dropout disabled during validation. Correct.")
else:
    fail(
        "net.eval() NOT found. Dropout is ACTIVE during validation passes. "
        "Val loss will be stochastic and higher than true val loss — model selection is broken."
    )

# Check 3d: Dropout rate for this architecture
# At D_MODEL=96, N_LAYERS=5, LSTM_LAYERS=1, DROPOUT=0.2 is standard.
# Risk: combined with WD=0.05 and Stage B enc LR=1e-6 this is a triple-regularization setup.
reg_score = 0
if DROPOUT >= 0.3:  reg_score += 1
if WEIGHT_DECAY >= 0.05: reg_score += 1
if STAGE_B_ENC_MAX <= 1e-6: reg_score += 1

info(f"Regularization intensity score: {reg_score}/3 (dropout={DROPOUT}, WD={WEIGHT_DECAY}, Stage_B_enc_lr={STAGE_B_ENC_MAX:.0e})")
if reg_score >= 3:
    fail(
        f"Triple regularization: high dropout + high WD + Stage B encoder LR={STAGE_B_ENC_MAX:.0e}. "
        "Model may underfit severely. Priority fix: reduce WEIGHT_DECAY to 1e-3 before next run."
    )
elif reg_score == 2:
    warn(
        f"Double regularization active. Watch val loss plateau in early epochs — "
        "if val loss does not fall below 1.0 by epoch 10, reduce either WD or dropout."
    )
else:
    ok(f"Regularization intensity is moderate (score={reg_score}/3).")

# =============================================================================
# AUDIT 4 · Parameter initialization & input dim mismatch
# =============================================================================
hdr("AUDIT 4 · Parameter initialization & input dim mismatch")

# Check 4a: Build the actual model and verify it accepts the right input dims
if MODEL_OK and CONFIG_OK:
    # Determine n_features for the current INPUT_MODE
    if INPUT_MODE == 'tokens_only':
        n_feat_expected = 1   # model uses embedding lookup, not raw features
        info(f"INPUT_MODE='{INPUT_MODE}' → n_features passed to model = {n_feat_expected} (token embedding path)")
    else:
        # Count feature cols same way train.py._build_feature_cols does
        from features import FeatureConfig
        fe_cfg = FeatureConfig(
            ewma_span=CFG.FE_VOL_LONG_PERIOD,
            return_horizons=CFG.FE_RETURN_HORIZONS,
            macd_pairs=CFG.FE_MACD_PAIRS,
            macd_price_std_window=CFG.FE_MACD_PRICE_STD_WIN,
            macd_signal_std_window=CFG.FE_MACD_SIGNAL_STD_WIN,
            target_clip=CFG.FE_TARGET_CLIP,
            add_session_features=CFG.FE_ADD_SESSION,
            use_talib=getattr(CFG, 'USE_TALIB', False),
        )
        # Reconstruct column count
        n_feat_expected = (
            1 +                          # ewma_vol_span
            len(CFG.FE_RETURN_HORIZONS) + # ret_norm_*
            len(CFG.FE_MACD_PAIRS) +      # macd_*
            5 +                           # feat_efficiency, icp, momentum_rsi, vol_asymmetry, local_structure
            1 +                           # feat_vol_squeeze
            (2 if CFG.FE_ADD_SESSION else 0)  # session sin/cos
        )
        info(f"INPUT_MODE='{INPUT_MODE}' → expected n_features = {n_feat_expected}")
    info(f"config.NUM_FEATURES = {NUM_FEATURES}")
    if NUM_FEATURES is not None and NUM_FEATURES != n_feat_expected:
        fail(
            f"config.NUM_FEATURES={NUM_FEATURES} != expected n_features={n_feat_expected}. "
            "Model will be built with wrong input projection dim. "
            "Silent failure: weights initialized but first layer produces garbage projections."
        )
    elif NUM_FEATURES is None:
        warn(
            "config.NUM_FEATURES=None (populated dynamically at runtime). "
            "Cannot pre-validate here. If you see 'size mismatch' at first batch, "
            "check that train.py passes len(feature_cols) to LPatchTST(n_features=...)."
        )

    # Instantiate model with known dims and do a forward pass
    try:
        n_feat_build = max(n_feat_expected, 1)
        net = LPatchTST(
            input_mode=INPUT_MODE,
            seq_len=SEQ_LEN,
            n_features=n_feat_build,
            s1_bits=S1_BITS,
            s2_bits=S2_BITS,
            d_model=D_MODEL,
            patch_len=PATCH_LEN,
            stride=STRIDE,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            lstm_layers=LSTM_LAYERS,
            dropout=DROPOUT,
            aggregation=getattr(CFG, 'AGGREGATION_MODE', 'mixing'),
        )
        n_params = sum(p.numel() for p in net.parameters())
        n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        ok(f"Model instantiated: {n_params:,} total params, {n_trainable:,} trainable.")

        # Check 4b: num_patches calculation
        num_patches = (SEQ_LEN - PATCH_LEN) // STRIDE + 1
        info(f"num_patches = (SEQ_LEN={SEQ_LEN} - PATCH_LEN={PATCH_LEN}) // STRIDE={STRIDE} + 1 = {num_patches}")
        if num_patches < 4:
            warn(f"Only {num_patches} patches per sequence. Consider reducing PATCH_LEN or STRIDE.")
        else:
            ok(f"{num_patches} patches per sequence. Sufficient temporal resolution.")

        # Check 4c: D_MODEL divisible by N_HEADS
        if D_MODEL % N_HEADS == 0:
            ok(f"D_MODEL={D_MODEL} divisible by N_HEADS={N_HEADS}. No attention head dim error.")
        else:
            fail(f"D_MODEL={D_MODEL} NOT divisible by N_HEADS={N_HEADS}. "
                 "Multi-head attention will raise at runtime.")

        # Check 4d: TOKENIZER_D_IN vs prepare_ohlc_features output shape
        # prepare_ohlc_features returns OHLC normalized returns — 5 columns (OHLCV) or 6?
        info(f"config.TOKENIZER_D_IN = {TOK_D_IN} (expected tokenizer input features)")
        # Inspect tokenizer.py for actual d_in expectation
        tok_py = os.path.join(ROOT, 'tokenizer.py')
        if os.path.exists(tok_py):
            with open(tok_py) as f:
                tok_src = f.read()
            # Check prepare_ohlc_features return shape
            import re
            # Look for column selections
            ohlc_cols_match = re.findall(r'\[[\'"](\w+)[\'"].*?\]', tok_src)
            if 'volume' in tok_src and 'log_ret' in tok_src:
                info("tokenizer.py uses log returns from OHLCV — likely 5 or 6 columns.")
            if TOK_D_IN == 5:
                ok(f"TOKENIZER_D_IN=5 (OHLCV returns). Matches standard 5-column OHLCV input.")
            elif TOK_D_IN == 6:
                ok(f"TOKENIZER_D_IN=6. Verify tokenizer.py prepare_ohlc_features returns exactly 6 columns.")
            else:
                warn(f"TOKENIZER_D_IN={TOK_D_IN}. Manually confirm prepare_ohlc_features returns {TOK_D_IN} columns.")

        # Check 4e: Forward pass with correct token dims
        if INPUT_MODE in ('tokens_only', 'combined'):
            vocab_size = 2 ** S1_BITS
            batch_sz = 4
            dummy_coarse = torch.randint(0, vocab_size, (batch_sz, SEQ_LEN))
            dummy_fine   = torch.randint(0, vocab_size, (batch_sz, SEQ_LEN))
            dummy_feat   = torch.randn(batch_sz, SEQ_LEN, n_feat_build) if INPUT_MODE == 'combined' else None
            try:
                net.eval()
                with torch.no_grad():
                    out = net(tokens=(dummy_coarse, dummy_fine), features=dummy_feat)
                ok(f"Forward pass OK: tokens input → output shape {out.shape}. No dim mismatch.")
                if out.shape == (batch_sz, 1) or out.shape == (batch_sz,):
                    ok(f"Output shape {tuple(out.shape)} correct for scalar regression.")
                else:
                    warn(f"Output shape {tuple(out.shape)} unexpected. Model may need .view(-1).")
            except Exception as e:
                fail(f"Forward pass FAILED: {e}")
        elif INPUT_MODE == 'features_only':
            batch_sz = 4
            dummy_feat = torch.randn(batch_sz, SEQ_LEN, n_feat_build)
            try:
                net.eval()
                with torch.no_grad():
                    out = net(tokens=None, features=dummy_feat)
                ok(f"Forward pass OK: features input → output shape {out.shape}.")
            except Exception as e:
                fail(f"Forward pass FAILED: {e}")

        # Check 4f: Weight initialization audit — check for zero/exploding init
        print(f"\n  ── Initial weight statistics ──")
        init_issues = []
        for name, param in net.named_parameters():
            if param.numel() < 4:
                continue
            p_std  = param.data.std().item()
            p_mean = param.data.mean().item()
            p_max  = param.data.abs().max().item()
            is_norm_weight = any(tag in name for tag in ('norm1.weight', 'norm2.weight',
                                                          'norm.weight', 'layer_norm.weight',
                                                          'LayerNorm.weight'))
            if p_std < 1e-5 and 'bias' not in name and not is_norm_weight:
                init_issues.append((name, 'near-zero std', p_std))
            elif p_std > 5.0:
                init_issues.append((name, 'exploding init', p_std))
        if init_issues:
            for name, issue, val in init_issues:
                fail(f"Init issue [{issue}] in '{name}': std={val:.4f}")
        else:
            ok("All weight tensors have reasonable init std (1e-5 to 5.0). No zero or exploding inits.")

        # Print top-5 largest param tensors
        param_sizes = [(n, p.numel()) for n, p in net.named_parameters()]
        param_sizes.sort(key=lambda x: -x[1])
        info("Top-5 largest parameter tensors:")
        for n, sz in param_sizes[:5]:
            info(f"  {n:<55s} {sz:>10,} params")

    except Exception as e:
        fail(f"Model instantiation failed: {e}")
        import traceback; traceback.print_exc()

# =============================================================================
# AUDIT 5 · OneCycleLR shape simulation (pretrain + Stage B encoder)
# =============================================================================
hdr("AUDIT 5 · OneCycleLR LR curve sanity check")

def _simulate_onecycle(label, max_lr, total_steps, pct_start, div_factor, final_div_factor):
    """Simulate a single OneCycleLR schedule and report peak/decay."""
    try:
        dummy_model = nn.Linear(4, 1)
        opt = torch.optim.AdamW(dummy_model.parameters(), lr=max_lr / div_factor)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=max_lr, total_steps=max(total_steps, 1),
            pct_start=pct_start, div_factor=div_factor,
            final_div_factor=final_div_factor,
        )
        lrs = []
        for _ in range(max(total_steps, 1)):
            lrs.append(sched.get_last_lr()[0])
            sched.step()
        warmup_steps = int(total_steps * pct_start)
        lr_arr = np.array(lrs)
        info(f"  [{label}] steps={total_steps} warmup={warmup_steps}")
        info(f"  [{label}] LR@0={lr_arr[0]:.3e} peak@{warmup_steps}={lr_arr[min(warmup_steps, len(lr_arr)-1)]:.3e} final={lr_arr[-1]:.3e}")
        peak = lr_arr[min(warmup_steps, len(lr_arr) - 1)]
        if peak >= max_lr * 0.99:
            ok(f"[{label}] LR peaks at {peak:.3e} ≈ max_lr {max_lr:.3e}.")
        else:
            warn(f"[{label}] LR peak {peak:.3e} < max_lr {max_lr:.3e}.")
        if lr_arr[-1] < lr_arr[0] * 0.5:
            ok(f"[{label}] LR decays to {lr_arr[-1]:.3e} (< initial {lr_arr[0]:.3e}).")
        else:
            warn(f"[{label}] LR does not fully decay: final={lr_arr[-1]:.3e}, initial={lr_arr[0]:.3e}.")
    except Exception as e:
        warn(f"[{label}] LR curve simulation failed: {e}")

info("Simulation 1: pretrain() — 30 epochs, max_lr=5e-5, pct_start=0.15, final_div=50")
_simulate_onecycle(
    "Pretrain",
    max_lr=PRETRAIN_MAX_LR,
    total_steps=PRETRAIN_EPOCHS * steps_per_epoch_est,
    pct_start=0.15,
    div_factor=10,
    final_div_factor=50,
)

info("Simulation 2: finetune Stage B encoder — max_lr=1e-6, pct_start=0.10, final_div=100")
_simulate_onecycle(
    "Stage B encoder",
    max_lr=STAGE_B_ENC_MAX,
    total_steps=max((EPOCHS - FREEZE_EPOCHS) * steps_per_epoch_est, 1),
    pct_start=0.10,
    div_factor=5,
    final_div_factor=100,
)

# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{SEP2}")
print("AUDIT COMPLETE — Model & Optimization")
print(SEP2)
stage_b_wd_lr = WEIGHT_DECAY / STAGE_B_ENC_MAX
print(textwrap.dedent(f"""
  Priority fix order:
  1. [CRITICAL] Stage B encoder WD/LR = {stage_b_wd_lr:.0f} — reduce WEIGHT_DECAY from {WEIGHT_DECAY} to 1e-3 before next run.
  2. [WARN if fired] getattr clip fallbacks use 5.0 not config.GRAD_CLIP ({GRAD_CLIP}) — silent clip relaxation risk.
  3. [WARN if fired] No WD=0 param group — add no-decay group for bias, LayerNorm, embeddings.
  4. [INFO] Pretrain max_lr={PRETRAIN_MAX_LR:.0e}; FT Stage A head={FINETUNE_HEAD_LR:.0e}; Stage B enc={STAGE_B_ENC_MAX:.0e}.
  5. [INFO] Stage A double-freeze + Stage B exp_avg_sq pre-fill — verify PASS above.
  6. [CHECK] Run on real data to confirm n_features matches model's n_features at construction.
"""))