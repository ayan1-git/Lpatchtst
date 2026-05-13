# train.py  (Pre-train → Fine-Tune Edition)
#
# Architecture of this file:
#   1. All imports + helpers are identical to the Diagnostic Edition.
#   2. train_fold() is replaced by:
#        - pretrain()      → single pass over all historical data, fresh model
#        - finetune_fold() → warms-start from pretrained checkpoint, per fold
#        - train()         → orchestrator: pretrain once, finetune each fold
#   3. make_rolling_folds() is upgraded from sliding → expanding window.
#   4. Every scaler is fitted on its OWN training slice — no leakage.
#   5. KronosTokenizer is FROZEN throughout (never in any optimizer).
#
# KEY INVARIANT (data leakage guard)
#   Scaler is fitted on train slice ONLY (create_fold_dataloaders already does this).
#   Pretrain uses rows [0 : pretrain_end].
#   Fold k trains on   rows [0 : fold_train_end]          (expanding).
#   Fold k validates on rows [fold_val_start : fold_val_end].
#   There is always a GAP = LOOKBACK_WINDOW bars between train end and val start.
#   The tokenizer produces per-bar tokens using only PAST bars (causal window).

import sys, os, random, math
from typing import Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import importlib

sys.path.insert(0, os.getcwd())

import config, model, features, loss, oracle, data_loader, tokenizer
for m in [config, model, tokenizer, features, data_loader, loss, oracle]:
    importlib.reload(m)

from oracle import generate_targets
from data_loader import (
    create_multi_index_dataloaders,
    create_fold_dataloaders,
    fit_scaler,
    FinancialDataset,
    _make_loader,
    _compute_sample_weights,
    ColumnSelectiveScaler,
)
from model import LPatchTST, InputMode
from loss import continuous_weighted_direction_loss
from features import FeatureConfig, FeatureEngineer
from tokenizer import prepare_ohlc_features, KronosTokenizer
from torch.utils.data import WeightedRandomSampler

PRETRAIN_CKPT = "pretrained_lpatchtst.pth"
MODEL_PATH    = "best_model_lpatchtst.pth"
OHLC_COLS     = ["open", "high", "low", "close", "volume"]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers (unchanged from Diagnostic Edition)
# ─────────────────────────────────────────────────────────────────────────────

def _set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _make_feature_config():
    return FeatureConfig(
        ewma_span=config.FE_VOL_LONG_PERIOD,
        return_horizons=config.FE_RETURN_HORIZONS,
        macd_pairs=config.FE_MACD_PAIRS,
        macd_price_std_window=config.FE_MACD_PRICE_STD_WIN,
        macd_signal_std_window=config.FE_MACD_SIGNAL_STD_WIN,
        target_clip=config.FE_TARGET_CLIP,
        momentum_period=config.FE_MOMENTUM_PERIOD,
        rsi_period=config.FE_RSI_PERIOD,
        vol_asym_window=config.FE_VOL_ASYM_WINDOW,
        icp_period=config.FE_ICP_PERIOD,
        local_structure_bars=config.FE_LOCAL_STRUCTURE_BARS,
        vol_squeeze_fast=config.FE_VOL_SQUEEZE_FAST,
        vol_squeeze_slow=config.FE_VOL_SQUEEZE_SLOW,
        atr_period=config.ATR_PERIOD,
        session_open=config.FE_SESSION_OPEN,
        session_close=config.FE_SESSION_CLOSE,
        session_tz=config.FE_SESSION_TZ,
        add_session_features=config.FE_ADD_SESSION,
        use_talib=getattr(config, "USE_TALIB", False),
    )

def _build_feature_cols(fe_config):
    no_scale_cols, robust_cols = [], []
    no_scale_cols.append(f"ewma_vol_span{fe_config.ewma_span}")
    for h in fe_config.return_horizons:
        no_scale_cols.append(f"ret_norm_{h}d")
    for s, l in fe_config.macd_pairs:
        no_scale_cols.append(f"macd_{s}_{l}")
    # vs_factor removed
    no_scale_cols += ["feat_efficiency","feat_icp","feat_momentum_rsi","feat_vol_asymmetry","feat_local_structure"]
    robust_cols.append("feat_vol_squeeze")
    if fe_config.add_session_features:
        no_scale_cols += ["feat_session_sin","feat_session_cos"]

    if fe_config.use_talib:
        try:
            from talib_features import TALIB_PASSTHROUGH, TALIB_SCALE
            # Both passthrough and scale are tanh-normalized to [-1, 1] 
            # in talib_features.py, so they go to no_scale.
            no_scale_cols += TALIB_PASSTHROUGH
            no_scale_cols += TALIB_SCALE
        except ImportError:
            pass

    return no_scale_cols, robust_cols, robust_cols + no_scale_cols

def _build_features(df, fe):
    time_col = next((c for c in df.columns if c.lower() in ("date","datetime")), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col]); df = df.set_index(time_col)
    else:
        try: df.index = pd.to_datetime(df.index)
        except: pass
    df = df.sort_index()
    feat_df = fe.build(df["close"], ohlc=df[OHLC_COLS], include_target=False, dropna=False)
    combined_df = df.join(feat_df, how="inner")
    hl = combined_df["high"] - combined_df["low"]
    hc = (combined_df["high"] - combined_df["close"].shift()).abs()
    lc = (combined_df["low"]  - combined_df["close"].shift()).abs()
    combined_df["atr"] = pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(config.ATR_PERIOD).mean()
    combined_df.dropna(inplace=True)
    _, _, all_feat_cols = _build_feature_cols(fe.config)
    return combined_df, all_feat_cols

def process_dataset(file_paths, fe):
    asset_data_list, final_feature_cols = [], []
    for f in file_paths:
        if not os.path.exists(f): continue
        print(f"Processing {f}…")
        df_raw = pd.read_csv(f)
        df, feature_cols = _build_features(df_raw, fe)
        ohlc_returns = prepare_ohlc_features(df)
        df = df.iloc[1:]
        targets = generate_targets(
            df["open"].values, df["high"].values, df["low"].values,
            df["close"].values, df["atr"].values,
            max_hold=config.ORACLE_MAX_HOLD, fee_per_side=config.FEE_PER_SIDE,
            slippage=config.SLIPPAGE, atr_mult=config.ATR_MULT,
            saturation_factor=config.SATURATION_FACTOR, mae_penalty=config.MAE_PENALTY)
        valid_len = len(targets) - config.ORACLE_MAX_HOLD
        if valid_len <= 0: continue
        feat_vals   = np.asarray(df[feature_cols].values, dtype=np.float32)[:valid_len]
        target_vals = np.asarray(targets[:valid_len],      dtype=np.float32)
        ohlc_vals   = ohlc_returns[:valid_len]
        asset_data_list.append((f, feat_vals, target_vals, ohlc_vals))
        final_feature_cols = feature_cols
        long = (target_vals > 0).mean(); short = (target_vals < 0).mean()
        print(f"  Target Distribution — Long: {long:.3f} | Short: {short:.3f} | Zero: {1-long-short:.3f}")
    return asset_data_list, final_feature_cols

def _grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return math.sqrt(total)

def _weight_norms(model):
    norms = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            norms[name] = p.detach().norm(2).item()
    return norms

@torch.no_grad()
def _full_eval_diagnostics(net, loader, device, tag="VAL"):
    net.eval()
    all_preds, all_tgts = [], []
    for tokens, feats, y in loader:
        if tokens is not None:
            tokens = (tokens[0].to(device), tokens[1].to(device))
        if feats is not None:
            feats = feats.to(device)
        y = y.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=config.USE_AMP):
            pred = net(tokens=tokens, features=feats)
        all_preds.append(pred.view(-1).float().cpu())
        all_tgts.append(y.view(-1).float().cpu())
    preds = torch.cat(all_preds)
    tgts  = torch.cat(all_tgts)
    is_zero = tgts.abs() < 1e-6
    is_edge = ~is_zero
    p_mean, p_std = preds.mean().item(), preds.std().item()
    t_mean, t_std = tgts.mean().item(),  tgts.std().item()
    if is_edge.any():
        p_e = preds[is_edge]; t_e = tgts[is_edge]
        dir_acc = ((p_e * t_e) > 0).float().mean().item()
        pc = p_e - p_e.mean(); tc = (t_e - t_e.mean())
        corr = (pc * tc).mean() / (p_e.std().clamp(min=1e-6) * t_e.std().clamp(min=1e-6))
        corr = corr.item()
        mag_over = (p_e.abs() > t_e.abs()).float().mean().item()
    else:
        dir_acc = corr = mag_over = float("nan")
    if is_zero.any():
        false_sig_rate = (preds[is_zero].abs() > 0.1).float().mean().item()
    else:
        false_sig_rate = float("nan")
    buckets = {"<-0.5":0, "-0.5:-0.1":0, "-0.1:0":0, "0:0.1":0, "0.1:0.5":0, ">0.5":0}
    for v in preds.tolist():
        if   v < -0.5:  buckets["<-0.5"]    += 1
        elif v < -0.1:  buckets["-0.5:-0.1"] += 1
        elif v < 0.0:   buckets["-0.1:0"]    += 1
        elif v < 0.1:   buckets["0:0.1"]     += 1
        elif v < 0.5:   buckets["0.1:0.5"]   += 1
        else:           buckets[">0.5"]      += 1
    total_preds = len(preds)
    bucket_pct  = {k: 100*v/total_preds for k,v in buckets.items()}
    thresh = 0.05
    n_long  = (preds >  thresh).sum().item()
    n_short = (preds < -thresh).sum().item()
    n_flat  = total_preds - n_long - n_short
    return {
        "tag": tag, "n": total_preds,
        "p_mean": p_mean, "p_std": p_std,
        "t_mean": t_mean, "t_std": t_std,
        "dir_acc": dir_acc, "corr": corr,
        "false_sig_rate": false_sig_rate, "mag_over": mag_over,
        "bucket_pct": bucket_pct,
        "n_long": n_long, "n_short": n_short, "n_flat": n_flat,
        "long_pct": 100*n_long/total_preds,
        "short_pct": 100*n_short/total_preds,
        "flat_pct": 100*n_flat/total_preds,
    }

def _print_diagnostics(d):
    print(f"\n  ┌── {d['tag']} DIAGNOSTICS (n={d['n']}) ─────────────────────────────")
    print(f"  │ Pred  : mean={d['p_mean']:+.4f}  std={d['p_std']:.4f}")
    print(f"  │ Target: mean={d['t_mean']:+.4f}  std={d['t_std']:.4f}  ratio={d['p_std']/(d['t_std']+1e-9):.3f}x")
    print(f"  │ Dir accuracy (non-zero targets): {d['dir_acc']*100:.1f}%")
    print(f"  │ Correlation (non-zero):           {d['corr']:.4f}")
    print(f"  │ False signal rate (zero targets): {d['false_sig_rate']*100:.1f}%")
    print(f"  │ Pred magnitude > tgt magnitude:   {d['mag_over']*100:.1f}%")
    bp = d["bucket_pct"]
    print(f"  │ Pred distribution:")
    print(f"  │   <-0.5  : {bp['<-0.5']:5.1f}%   -0.5:-0.1: {bp['-0.5:-0.1']:5.1f}%   -0.1:0: {bp['-0.1:0']:5.1f}%")
    print(f"  │    0:0.1 : {bp['0:0.1']:5.1f}%    0.1:0.5 : {bp['0.1:0.5']:5.1f}%    >0.5: {bp['>0.5']:5.1f}%")
    print(f"  │ Decisions (±0.05): Long={d['long_pct']:.1f}%  Short={d['short_pct']:.1f}%  Flat={d['flat_pct']:.1f}%")
    print(f"  └─────────────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# Model factory (shared so pretrain and finetune get identical architectures)
# ─────────────────────────────────────────────────────────────────────────────

def _build_model(feature_cols, device):
    net = LPatchTST(
        input_mode=config.INPUT_MODE,
        seq_len=config.LOOKBACK_WINDOW,
        n_features=len(feature_cols),
        s1_bits=config.TOKENIZER_S1_BITS,
        s2_bits=config.TOKENIZER_S2_BITS,
        d_model=config.D_MODEL,
        patch_len=config.PATCH_LEN,
        stride=config.STRIDE,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        lstm_layers=config.LSTM_LAYERS,
        dropout=config.DROPOUT,
        aggregation=config.AGGREGATION_MODE,
    ).to(device)
    if device.type == "cuda":
        try: net = torch.compile(net); print("  torch.compile: OK")
        except: pass
    return net


# ─────────────────────────────────────────────────────────────────────────────
# Inner epoch runner (shared between pretrain and finetune)
# ─────────────────────────────────────────────────────────────────────────────

def _run_epoch(net, loader, device, optimizer=None, grad_scaler=None,
               scheduler=None, is_train=True, use_amp=True, grad_clip=None):
    """Run one epoch. Returns avg_loss and per-batch stats dict."""
    if is_train:
        net.train()
    else:
        net.eval()

    total_loss  = 0.0
    batch_count = 0
    grad_norms  = []
    pred_stds   = []
    dir_accs    = []
    corrs       = []

    ctx = torch.no_grad() if not is_train else torch.enable_grad()
    with ctx:
        for step, (tokens, feats, y) in enumerate(loader):
            if tokens is not None:
                tokens = (tokens[0].to(device), tokens[1].to(device))
            if feats is not None:
                feats = feats.to(device)
            y = y.to(device)

            if is_train:
                optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pred = net(tokens=tokens, features=feats)
                batch_loss = continuous_weighted_direction_loss(pred, y)

            if is_train:
                grad_scaler.scale(batch_loss).backward()
                grad_scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(
                    net.parameters(), grad_clip if grad_clip is not None else config.GRAD_CLIP)
                if not torch.isfinite(total_norm):
                    total_norm = torch.tensor(0.0)
                grad_norms.append(total_norm.item())
                scale_before = grad_scaler.get_scale()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                if grad_scaler.get_scale() >= scale_before and scheduler is not None:
                    scheduler.step()

            total_loss  += batch_loss.item()
            batch_count += 1

            with torch.no_grad():
                p_f = pred.view(-1).float()
                t_f = y.view(-1).float()
                is_e = t_f.abs() > 1e-6
                pred_stds.append(p_f[is_e].std().item() if is_e.any() else 0.0)
                if is_e.any():
                    da = ((p_f[is_e] * t_f[is_e]) > 0).float().mean().item()
                    dir_accs.append(da)
                    pc = p_f[is_e] - p_f[is_e].mean()
                    tc = t_f[is_e] - t_f[is_e].mean()
                    c  = (pc*tc).mean() / (p_f[is_e].std().clamp(1e-6) * t_f[is_e].std().clamp(1e-6))
                    corrs.append(c.item())

    return {
        "avg_loss":  total_loss / max(batch_count, 1),
        "avg_gn":    float(np.mean(grad_norms))  if grad_norms  else 0.0,
        "max_gn":    float(np.max(grad_norms))   if grad_norms  else 0.0,
        "avg_ps":    float(np.mean(pred_stds))   if pred_stds   else 0.0,
        "avg_da":    float(np.mean(dir_accs))    if dir_accs    else 0.0,
        "avg_corr":  float(np.mean(corrs))       if corrs       else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 ── Pre-train on ALL historical data
# ─────────────────────────────────────────────────────────────────────────────
#
# WHY: Teaches the model general NIFTY patch structure (embedding table +
#      transformer encoder + LSTM) from the full history, so every fold
#      starts from an informed init rather than random noise.
#
# LEAKAGE GUARD:
#   pretrain_end = first fold's val_start - GAP
#   The scaler is fitted on features[:pretrain_end] only.
#   No future bar is ever seen during pre-training.
#
# TOKENIZER: stays eval() and is never added to any optimizer.
#            It is a frozen vocabulary, not a learnable component.

def pretrain(
    all_feat:     np.ndarray,
    all_targ:     np.ndarray,
    all_ohlc:     np.ndarray,
    feature_cols: list[str],
    tok,
    pretrain_end: int,
    device:       torch.device,
    epochs:       int = 50,
    max_lr:       float = 5e-5,
):
    """
    Train the model on rows [0 : pretrain_end].

    Parameters
    ----------
    pretrain_end : exclusive upper bound — the first bar that must NOT be seen.
                   Set by make_rolling_folds: fold_0.val_start - LOOKBACK_WINDOW.
    """
    print(f"\n{'='*65}")
    print(f"  PRE-TRAIN  |  bars [0 : {pretrain_end}]  |  epochs={epochs}")
    print(f"  Device: {device}  |  LR_max={max_lr:.1e}  |  D_MODEL={config.D_MODEL}")
    print(f"{'='*65}\n")

    # ── Build dataset ────────────────────────────────────────────────────────
    # Scaler fitted on pretrain slice only — no leakage.
    scaler = fit_scaler(all_feat[:pretrain_end], feature_cols, config=config)

    train_ds = FinancialDataset(
        all_feat[:pretrain_end],
        all_targ[:pretrain_end],
        config.LOOKBACK_WINDOW,
        ohlc_returns=all_ohlc[:pretrain_end] if all_ohlc is not None else None,
        scaler=scaler,
        tokenizer=tok,
        config=config,
    )
    # Sample weights for class balance
    start      = config.LOOKBACK_WINDOW - 1
    y_aligned  = all_targ[start : start + len(train_ds)]
    sw         = _compute_sample_weights(y_aligned, config.SAMPLER_THRESHOLD)
    sampler    = WeightedRandomSampler(sw, num_samples=len(sw)//2, replacement=True)
    loader     = _make_loader(train_ds, config, sampler=sampler, drop_last=True)

    print(f"  Pre-train windows: {len(train_ds):,}  |  Batches/epoch: {len(loader):,}")

    # ── Model, optimizer, scheduler ─────────────────────────────────────────
    net = _build_model(feature_cols, device)
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"  Model params: {total_params:,}\n")

    optimizer  = torch.optim.AdamW(
        net.parameters(), lr=max_lr / 10, weight_decay=config.WEIGHT_DECAY)
    total_steps = epochs * len(loader)
    scheduler  = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=total_steps,
        pct_start=0.15, div_factor=10, final_div_factor=50)
    scaler_amp = torch.amp.GradScaler(
        enabled=config.USE_AMP and device.type == "cuda", growth_interval=200)

    best_loss    = float("inf")
    patience     = 10
    pat_counter  = 0

    for epoch in range(epochs):
        stats = _run_epoch(
            net, loader, device,
            optimizer=optimizer, grad_scaler=scaler_amp,
            scheduler=scheduler, is_train=True, use_amp=config.USE_AMP,
            grad_clip=1.0,
        )
        lr_now = scheduler.get_last_lr()[0]

        saved = ""
        if stats["avg_loss"] < best_loss:
            best_loss   = stats["avg_loss"]
            pat_counter = 0
            torch.save(net.state_dict(), PRETRAIN_CKPT)
            saved = "  ✓ SAVED"
        else:
            pat_counter += 1

        print(
            f"  [Pre-train] Ep{epoch+1:3d} | "
            f"Loss={stats['avg_loss']:.4f} | "
            f"LR={lr_now:.2e} | "
            f"GN avg={stats['avg_gn']:.3f} max={stats['max_gn']:.3f} | "
            f"DirAcc={stats['avg_da']*100:.1f}% | "
            f"Corr={stats['avg_corr']:.3f} | "
            f"Pat={pat_counter}/{patience}"
            f"{saved}"
        )
        if pat_counter >= patience:
            print(f"  ⛔ Pre-train early stop at epoch {epoch+1}.")
            break

    print(f"\n  ✅ Pre-train done. Best loss={best_loss:.4f}  → {PRETRAIN_CKPT}\n")
    del net  # free VRAM before fine-tuning loop starts
    torch.cuda.empty_cache() if device.type == "cuda" else None


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 ── Fine-tune a single fold
# ─────────────────────────────────────────────────────────────────────────────
#
# STRATEGY:
#   Stage A (epochs 1–freeze_epochs): freeze encoder, train head only.
#              This lets the head recalibrate to the new fold's distribution
#              before the encoder weights start drifting.
#   Stage B (remaining epochs):       unfreeze all, train at very low LR.
#              Prevents catastrophic forgetting of pre-trained structure.
#
# LEAKAGE GUARD:
#   Scaler is fitted on train slice [fold_train_start : fold_train_end] only.
#   create_fold_dataloaders() already enforces this — we pass global arrays.
#   val slice starts at fold_train_end + LOOKBACK_WINDOW (gap is a full lookback).

def finetune_fold(
    fold_id:       int,
    all_feat:      np.ndarray,
    all_targ:      np.ndarray,
    all_ohlc:      np.ndarray,
    feature_cols:  list[str],
    tok,
    train_start:   int,
    train_end:     int,
    val_start:     int,
    val_end:       int,
    device:        torch.device,
    epochs:        int      = None,
    freeze_epochs: int      = 5,
    head_lr:       float    = 3e-5,
    full_lr:       float    = 5e-6,
    patience:      int      = None,
    load_path:     str      = None,
):
    """
    Fine-tune one walk-forward fold, warm-starting from PRETRAIN_CKPT.

    Parameters
    ----------
    train_start, train_end : absolute row indices for the training slice.
    val_start,   val_end   : absolute row indices for the val slice.
                             MUST satisfy val_start >= train_end + LOOKBACK_WINDOW.
    freeze_epochs          : how many epochs to keep the encoder frozen.
                             During these epochs, only the head trains (high LR).
    head_lr                : LR while encoder is frozen (head only).
    full_lr                : LR once encoder is unfrozen (very small to avoid
                             catastrophic forgetting).
    """
    epochs  = epochs  if epochs  is not None else config.EPOCHS
    patience = patience if patience is not None else config.WFV_PATIENCE

    # ── Leakage assertion ────────────────────────────────────────────────────
    gap = val_start - train_end
    assert gap >= config.LOOKBACK_WINDOW, (
        f"Fold {fold_id}: val_start - train_end = {gap} < LOOKBACK_WINDOW={config.LOOKBACK_WINDOW}. "
        "Val window overlaps with train — data leakage! "
        "Ensure make_rolling_folds inserts a full lookback as gap."
    )

    print(f"\n{'='*65}")
    print(f"  FINE-TUNE  Fold {fold_id}  |  Device: {device}")
    print(f"  Train rows: [{train_start} : {train_end}]  ({train_end-train_start:,} bars)")
    print(f"  Val   rows: [{val_start} : {val_end}]  ({val_end-val_start:,} bars)")
    print(f"  Gap (leakage buffer): {gap} bars  (LOOKBACK_WINDOW={config.LOOKBACK_WINDOW})")
    print(f"  Freeze epochs: {freeze_epochs}  |  Head LR: {head_lr:.1e}  |  Full LR: {full_lr:.1e}")
    print(f"{'='*65}\n")

    # ── Dataloaders (scaler fitted on train slice only via create_fold_dataloaders) ──
    # NOTE: We pass GLOBAL arrays (not pre-sliced) — create_fold_dataloaders
    #       slices internally and fits the scaler on the train slice only.
    train_loader, val_loader, _ = create_fold_dataloaders(
        features=all_feat,
        targets=all_targ,
        train_indices=(train_start, train_end),
        val_indices=(val_start, val_end),
        test_indices=(val_end, min(val_end + config.WFV_VAL_BARS, len(all_feat))),
        config=config,
        feature_cols=feature_cols,
        tokenizer=tok,
        ohlc_returns=all_ohlc,
    )

    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # ── Load weights (Pre-trained or Previous Fold) ─────────────────────────
    net = _build_model(feature_cols, device)
    path_to_load = load_path if load_path else PRETRAIN_CKPT

    if os.path.exists(path_to_load):
        net.load_state_dict(torch.load(path_to_load, map_location=device))
        print(f"  ✓ Loaded weights from {path_to_load}")
    else:
        print(f"  ⚠️  Checkpoint not found at {path_to_load}. "
              f"Training from scratch (ensure pretrain() or previous fold ran).")

    # ── Helper: identify head vs. encoder parameters ─────────────────────────
    # "head" = any param whose name starts with 'head' or 'fc' or 'proj'
    # Everything else = encoder (transformer, LSTM, stem/embedding).
    # Adjust the prefix list if your LPatchTST uses different naming.
    def _is_head(name: str) -> bool:
        name = name.replace("_orig_mod.", "")
        return any(name.startswith(p) for p in ("head", "fc", "proj", "output", "feature_head"))

    head_params    = [p for n, p in net.named_parameters() if     _is_head(n)]
    encoder_params = [p for n, p in net.named_parameters() if not _is_head(n)]
    head_names     = [n for n, _ in net.named_parameters() if     _is_head(n)]
    enc_names      = [n for n, _ in net.named_parameters() if not _is_head(n)]

    print(f"  Head params    ({len(head_names)}): {head_names[:3]}{'...' if len(head_names)>3 else ''}")
    print(f"  Encoder params ({len(enc_names)}): {enc_names[:3]}{'...' if len(enc_names)>3 else ''}\n")

    # ── Stage A: Frozen encoder, head-only ──────────────────────────────────
    def _freeze_encoder():
        for p in encoder_params: p.requires_grad = False
        for p in head_params:    p.requires_grad = True

    def _unfreeze_all():
        for p in net.parameters(): p.requires_grad = True

    _freeze_encoder()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=head_lr, weight_decay=config.WEIGHT_DECAY,
    )
    # Short OneCycleLR for the frozen stage
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=head_lr,
        total_steps=freeze_epochs * len(train_loader),
        pct_start=0.2, div_factor=5, final_div_factor=10,
    )
    scaler_amp = torch.amp.GradScaler(
        enabled=config.USE_AMP and device.type == "cuda", growth_interval=200)

    best_val   = float("inf")
    best_epoch = -1
    pat_counter = 0

    for epoch in range(epochs):
        # ── Switch from Stage A → Stage B ────────────────────────────────────
        if epoch == freeze_epochs:
            print(f"\n  → Unfreezing encoder at epoch {epoch+1}. LR → {full_lr:.1e}")
            _unfreeze_all()
            remaining = (epochs - freeze_epochs) * len(train_loader)
            # SPLIT param groups: encoder gets weight_decay to prevent memorisation,
            # head gets weight_decay=0 (it was already trained in Stage A).
            optimizer = torch.optim.AdamW([
                {"params": head_params,    "lr": full_lr / 10, "weight_decay": 0.0},
                {"params": encoder_params, "lr": full_lr / 10, "weight_decay": 1e-4},
            ])
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=[full_lr, full_lr],   # one max_lr per param group
                total_steps=max(remaining, 1),
                pct_start=0.05, div_factor=5, final_div_factor=100,
            )
            scaler_amp = torch.amp.GradScaler(
                enabled=config.USE_AMP and device.type == "cuda", growth_interval=200)

        stage = "A-frozen" if epoch < freeze_epochs else "B-full"

        # ── Train one epoch ───────────────────────────────────────────────────
        clip_val = 0.5 if epoch >= freeze_epochs else 1.0
        tr = _run_epoch(
            net, train_loader, device,
            optimizer=optimizer, grad_scaler=scaler_amp,
            scheduler=scheduler, is_train=True, use_amp=config.USE_AMP,
            grad_clip=clip_val,
        )
        va = _run_epoch(
            net, val_loader, device,
            is_train=False, use_amp=config.USE_AMP,
        )
        lr_now = scheduler.get_last_lr()[0]

        saved = ""
        if va["avg_loss"] < best_val:
            best_val   = va["avg_loss"]
            best_epoch = epoch + 1
            pat_counter = 0
            torch.save(net.state_dict(), MODEL_PATH)
            saved = "  ✓ SAVED"
        else:
            pat_counter += 1

        print(
            f"  [Fold {fold_id} {stage}] Ep{epoch+1:3d} | "
            f"Tr={tr['avg_loss']:.4f}  Va={va['avg_loss']:.4f} | "
            f"LR={lr_now:.2e} | "
            f"GN avg={tr['avg_gn']:.3f} max={tr['avg_gn']:.3f} | "
            f"DirAcc={tr['avg_da']*100:.1f}% | "
            f"Corr={tr['avg_corr']:.3f} | "
            f"Pat={pat_counter}/{patience}"
            f"{saved}"
        )

        if (epoch + 1) % 5 == 0:
            d_train = _full_eval_diagnostics(net, train_loader, device, tag="TRAIN")
            d_val   = _full_eval_diagnostics(net, val_loader,   device, tag="VAL")
            _print_diagnostics(d_train)
            _print_diagnostics(d_val)

        if pat_counter >= patience:
            print(f"\n  ⛔ Fold {fold_id} early stop — epoch {epoch+1}.")
            break

    print(f"\n  ✅ Fold {fold_id} done. Best epoch={best_epoch}  Best val={best_val:.4f}\n")
    return best_val


# ─────────────────────────────────────────────────────────────────────────────
# Expanding walk-forward fold builder
# ─────────────────────────────────────────────────────────────────────────────
#
# EXPANDING (not sliding):
#   fold 0: train [0 : TRAIN_BARS],          val [TRAIN_BARS+GAP : TRAIN_BARS+GAP+VAL_BARS]
#   fold 1: train [0 : TRAIN_BARS+STEP],     val [...+STEP...]
#   fold 2: train [0 : TRAIN_BARS+2*STEP],   val [...+2*STEP...]
#
# WHY EXPANDING?
#   You have ~30k bars total. Sliding throws away early market data every fold.
#   Expanding retains all history and lets the model accumulate knowledge.
#   The pre-train phase already covers all of [0 : pretrain_end], so
#   fold fine-tuning just adds the newest regime on top.
#
# GAP:
#   val_start = train_end + LOOKBACK_WINDOW
#   This ensures no OHLC bar in the val window appears in any train window.

def make_rolling_folds(total_bars: int, config) -> list[dict]:
    """
    Build expanding-window walk-forward folds.

    Returns
    -------
    list of dicts, each with keys:
        fold_id, train_start, train_end, val_start, val_end
    """
    GAP        = config.LOOKBACK_WINDOW  # causal gap = full lookback
    TRAIN_BARS = config.WFV_TRAIN_BARS
    VAL_BARS   = config.WFV_VAL_BARS
    STEP_BARS  = config.WFV_STEP_BARS

    folds = []
    fold_id = 0
    train_end = TRAIN_BARS  # expands each fold; train_start is always 0

    while True:
        val_start = train_end + GAP
        val_end   = val_start + VAL_BARS

        if val_end > total_bars:
            print(f"  [make_rolling_folds] fold {fold_id}: val_end={val_end} > "
                  f"total_bars={total_bars}. Stopping.")
            break

        folds.append({
            "fold_id":     fold_id,
            "train_start": 0,           # EXPANDING: always start from 0
            "train_end":   train_end,
            "val_start":   val_start,
            "val_end":     val_end,
        })

        fold_id  += 1
        train_end += STEP_BARS          # expand train window by one step

    if len(folds) < config.WFV_MIN_FOLDS:
        raise RuntimeError(
            f"Only {len(folds)} folds available, need at least {config.WFV_MIN_FOLDS}. "
            f"Total bars={total_bars}, TRAIN_BARS={TRAIN_BARS}, "
            f"VAL_BARS={VAL_BARS}, STEP_BARS={STEP_BARS}."
        )

    print(f"\n  Walk-forward folds ({len(folds)} total, EXPANDING window):")
    for f in folds:
        print(f"    Fold {f['fold_id']}: "
              f"train [0:{f['train_end']}] ({f['train_end']:,} bars)  "
              f"val [{f['val_start']}:{f['val_end']}] ({f['val_end']-f['val_start']:,} bars)  "
              f"gap={f['val_start']-f['train_end']} bars")
    return folds


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def train(asset_data_list, feature_cols):
    """
    Full pre-train → fine-tune pipeline.

    Steps
    -----
    1. Concatenate all assets into a single timeline (single-asset for NIFTY).
    2. Build expanding walk-forward folds.
    3. Pre-train once on rows [0 : fold_0.val_start - GAP].
    4. Fine-tune each fold from the pretrained checkpoint.
    5. Report best val per fold.
    """
    _set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # For single-asset (NIFTY 50), asset_data_list has one entry.
    # For multi-asset, extend process_dataset and this loop.
    assert len(asset_data_list) == 1, (
        "Multi-asset pretrain not yet implemented. "
        "Pass a single-asset list or extend _concat_assets()."
    )
    _, all_feat, all_targ, all_ohlc = asset_data_list[0]
    total_bars = len(all_feat)
    print(f"\n  Total bars available: {total_bars:,}")

    # ── Load frozen tokenizer (Optional if INPUT_MODE is features_only) ─────
    tok = None
    if config.INPUT_MODE != "features_only":
        tok = KronosTokenizer(
            d_in=config.TOKENIZER_D_IN,
            d_model=config.TOKENIZER_D_MODEL,
            n_heads=config.TOKENIZER_N_HEADS,
            ff_dim=config.TOKENIZER_FF_DIM,
            n_enc_layers=config.TOKENIZER_N_ENC,
            n_dec_layers=config.TOKENIZER_N_DEC,
            s1_bits=config.TOKENIZER_S1_BITS,
            s2_bits=config.TOKENIZER_S2_BITS,
            group_size=config.TOKENIZER_GROUP_SIZE,
        )
        tok_path = "tokenizer.pt"
        if os.path.exists(tok_path):
            tok.load_state_dict(torch.load(tok_path, map_location="cpu"), strict=False)
            print(f"  ✓ Tokenizer loaded from {tok_path}")
        else:
            raise FileNotFoundError(
                f"Tokenizer checkpoint not found at {tok_path}. "
                "Run train_tokenizer.py first."
            )
        tok.eval()
        for p in tok.parameters():
            p.requires_grad = False
    else:
        print("\n  ✓ INPUT_MODE is features_only. Skipping tokenizer.")

    # ── Build folds ──────────────────────────────────────────────────────────
    folds = make_rolling_folds(total_bars, config)

    # ── Pre-train boundary ───────────────────────────────────────────────────
    # Safe upper bound: first fold's val_start minus one full lookback.
    # This guarantees no bar that appears in any fold's val window is seen
    # during pre-training (even indirectly via sliding token windows).
    pretrain_end = folds[0]["val_start"] - config.LOOKBACK_WINDOW
    assert pretrain_end >= config.LOOKBACK_WINDOW, (
        f"pretrain_end={pretrain_end} is too small. "
        "Reduce WFV_TRAIN_BARS or increase total data."
    )
    print(f"\n  Pre-train boundary: rows [0 : {pretrain_end}]  "
          f"({pretrain_end:,} bars, "
          f"{pretrain_end/total_bars*100:.1f}% of total data)")
    print(f"  Val leak gap: {folds[0]['val_start'] - pretrain_end} bars "
          f"(LOOKBACK_WINDOW={config.LOOKBACK_WINDOW})\n")

    # ── Phase 1: Pre-train ───────────────────────────────────────────────────
    if not os.path.exists(PRETRAIN_CKPT):
        pretrain(
            all_feat=all_feat,
            all_targ=all_targ,
            all_ohlc=all_ohlc,
            feature_cols=feature_cols,
            tok=tok,
            pretrain_end=pretrain_end,
            device=device,
            epochs=30,          # lighter pass — fine-tuning does the heavy work
            max_lr=5e-5,
        )
    else:
        print(f"  ✓ Pre-trained checkpoint already exists at {PRETRAIN_CKPT}. Skipping pretrain.")

    # ── Phase 2: Fine-tune each fold ─────────────────────────────────────────
    fold_results = []
    current_load_path = PRETRAIN_CKPT

    for fold in folds:
        best_val = finetune_fold(
            fold_id=fold["fold_id"],
            all_feat=all_feat,
            all_targ=all_targ,
            all_ohlc=all_ohlc,
            feature_cols=feature_cols,
            tok=tok,
            train_start=fold["train_start"],
            train_end=fold["train_end"],
            val_start=fold["val_start"],
            val_end=fold["val_end"],
            device=device,
            epochs=config.EPOCHS,
            freeze_epochs=10,
            head_lr=3e-5,
            full_lr=5e-6,
            patience=config.WFV_PATIENCE,
            load_path=current_load_path,
        )
        fold_results.append((fold["fold_id"], best_val))

        # IMPORTANT: Next fold will load the BEST model from the fold just finished.
        # This implements recursive fine-tuning.
        current_load_path = MODEL_PATH
        print(f"  → Fold {fold['fold_id']} complete. Next fold will load from {current_load_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  WALK-FORWARD SUMMARY  ({len(fold_results)} folds)")
    print(f"{'='*65}")
    for fid, bv in fold_results:
        print(f"    Fold {fid}: best val = {bv:.4f}")
    avg_val = np.mean([bv for _, bv in fold_results])
    print(f"  Average val loss: {avg_val:.4f}")
    print(f"  Final model saved to: {MODEL_PATH}")
    print(f"{'='*65}\n")

    return fold_results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fe          = FeatureEngineer(_make_feature_config())
    asset_data, feat_cols = process_dataset(config.DATA_FILE, fe)
    config.NUM_FEATURES   = len(feat_cols)
    train(asset_data, feat_cols)