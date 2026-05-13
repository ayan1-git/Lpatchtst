# train.py  (Diagnostic Edition — full instrumentation)
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
from data_loader import create_multi_index_dataloaders
from model import LPatchTST, InputMode
from loss import continuous_weighted_direction_loss
from features import FeatureConfig, FeatureEngineer
from tokenizer import prepare_ohlc_features, KronosTokenizer

MODEL_PATH   = "best_model_lpatchtst.pth"
OHLC_COLS    = ["open", "high", "low", "close", "volume"]

# ── Helpers ──────────────────────────────────────────────────────────────────

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

# ── Gradient norm helper ──────────────────────────────────────────────────────

def _grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return math.sqrt(total)

# ── Per-layer weight norm helper ──────────────────────────────────────────────

def _weight_norms(model):
    norms = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            norms[name] = p.detach().norm(2).item()
    return norms

# ── Prediction calibration diagnostics ───────────────────────────────────────

@torch.no_grad()
def _full_eval_diagnostics(net, loader, device, tag="VAL"):
    """
    Run full pass over a loader and collect detailed prediction stats.
    Returns dict of metrics.
    """
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

    # Global stats
    p_mean, p_std   = preds.mean().item(), preds.std().item()
    t_mean, t_std   = tgts.mean().item(),  tgts.std().item()

    # Directional accuracy on non-zero targets
    if is_edge.any():
        p_e = preds[is_edge]; t_e = tgts[is_edge]
        dir_acc = ((p_e * t_e) > 0).float().mean().item()
        # Pearson correlation
        pc = p_e - p_e.mean(); tc = (t_e - t_e.mean())
        corr = (pc * tc).mean() / (p_e.std().clamp(min=1e-6) * t_e.std().clamp(min=1e-6))
        corr = corr.item()
        # Magnitude coverage: what fraction of targets are |tgt| < |pred|?
        mag_over = (p_e.abs() > t_e.abs()).float().mean().item()
    else:
        dir_acc = corr = mag_over = float('nan')

    # False signal rate on zero targets
    if is_zero.any():
        false_sig_rate = (preds[is_zero].abs() > 0.1).float().mean().item()
    else:
        false_sig_rate = float('nan')

    # Histogram buckets for pred distribution
    buckets = {"<-0.5":0, "-0.5:-0.1":0, "-0.1:0":0,
               "0:0.1":0, "0.1:0.5":0, ">0.5":0}
    for v in preds.tolist():
        if   v < -0.5:  buckets["<-0.5"]    += 1
        elif v < -0.1:  buckets["-0.5:-0.1"] += 1
        elif v < 0.0:   buckets["-0.1:0"]    += 1
        elif v < 0.1:   buckets["0:0.1"]     += 1
        elif v < 0.5:   buckets["0.1:0.5"]   += 1
        else:           buckets[">0.5"]      += 1
    total_preds = len(preds)
    bucket_pct = {k: 100*v/total_preds for k,v in buckets.items()}

    # Long/Short/Flat decision breakdown (threshold 0.05)
    thresh = 0.05
    n_long  = (preds >  thresh).sum().item()
    n_short = (preds < -thresh).sum().item()
    n_flat  = total_preds - n_long - n_short

    return {
        "tag": tag,
        "n": total_preds,
        "p_mean": p_mean, "p_std": p_std,
        "t_mean": t_mean, "t_std": t_std,
        "dir_acc": dir_acc, "corr": corr,
        "false_sig_rate": false_sig_rate,
        "mag_over": mag_over,
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
    bp = d['bucket_pct']
    print(f"  │ Pred distribution:")
    print(f"  │   <-0.5  : {bp['<-0.5']:5.1f}%   -0.5:-0.1: {bp['-0.5:-0.1']:5.1f}%   -0.1:0: {bp['-0.1:0']:5.1f}%")
    print(f"  │    0:0.1 : {bp['0:0.1']:5.1f}%    0.1:0.5 : {bp['0.1:0.5']:5.1f}%    >0.5: {bp['>0.5']:5.1f}%")
    print(f"  │ Decisions (±0.05): Long={d['long_pct']:.1f}%  Short={d['short_pct']:.1f}%  Flat={d['flat_pct']:.1f}%")
    print(f"  └─────────────────────────────────────────────────────────────────\n")

# ── Training ──────────────────────────────────────────────────────────────────

def train_fold(fold_id, train_loader, val_loader, feature_cols):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print(f"  Training Fold: {fold_id}  |  Mode: {config.INPUT_MODE}  |  Device: {device}")
    print(f"  Architecture : D_MODEL={config.D_MODEL}  N_LAYERS={config.N_LAYERS}  "
          f"N_HEADS={config.N_HEADS}  PATCH={config.PATCH_LEN}  STRIDE={config.STRIDE}  "
          f"LSTM={config.LSTM_LAYERS}")
    print(f"  Training     : LR={config.LEARNING_RATE}  BS={config.BATCH_SIZE}  "
          f"EPOCHS={config.EPOCHS}  WD={config.WEIGHT_DECAY}  CLIP={config.GRAD_CLIP}")
    print(f"{'='*65}\n")

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

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"  Model params : {total_params:,}")
    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # Print initial weight norms per named layer group
    wn = _weight_norms(net)
    print(f"\n  ── Initial Weight Norms (top-level) ──")
    for k, v in list(wn.items())[:8]:
        print(f"     {k:<45s} {v:.4f}")
    print()

    if device.type == "cuda":
        try: net = torch.compile(net); print("  torch.compile: OK\n")
        except: pass

    # Set embedding weight_decay=0.0 (stops the optimizer from suppressing the only feature extractor)
    embed_params = [p for n, p in net.named_parameters() if ("embed_coarse" in n or "embed_fine" in n) and p.requires_grad]
    other_params = [p for n, p in net.named_parameters() if not ("embed_coarse" in n or "embed_fine" in n) and p.requires_grad]
    
    optimizer = torch.optim.AdamW([
        {"params": embed_params, "weight_decay": 0.0},
        {"params": other_params, "weight_decay": config.WEIGHT_DECAY}
    ], lr=config.LEARNING_RATE)

    steps_per_epoch = len(train_loader)
    total_steps     = config.EPOCHS * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-6, total_steps=total_steps,
        pct_start=0.10, div_factor=10, final_div_factor=10)
    grad_scaler = torch.amp.GradScaler(enabled=config.USE_AMP and device.type == "cuda", growth_interval=200)

    best_val  = float("inf")
    best_epoch = -1
    patience_counter = 0
    PATIENCE = 15

    # ── EPOCH LOOP ────────────────────────────────────────────────────────────
    for epoch in range(config.EPOCHS):
        net.train()

        # Per-epoch accumulators
        train_loss   = 0.0
        batch_count  = 0
        grad_norms   = []
        pred_stds    = []
        focal_acc    = 0.0
        dir_acc_val  = 0.0
        corr_acc     = 0.0
        false_acc    = 0.0

        for step, (tokens, feats, y) in enumerate(train_loader):
            if tokens is not None:
                tokens = (tokens[0].to(device), tokens[1].to(device))
            if feats is not None:
                feats = feats.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=config.USE_AMP):
                pred = net(tokens=tokens, features=feats)
                batch_loss = continuous_weighted_direction_loss(pred, y)

            grad_scaler.scale(batch_loss).backward()
            grad_scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), config.GRAD_CLIP)
            if not torch.isfinite(total_norm):
                total_norm = torch.tensor(0.0)
            grad_norms.append(total_norm.item())
            scale_before = grad_scaler.get_scale()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scale_after = grad_scaler.get_scale()

            # Only step the scheduler if the optimizer actually updated (didn't skip due to NaN)
            if scale_after >= scale_before:
                scheduler.step()

            train_loss  += batch_loss.item()
            batch_count += 1

            # Collect per-batch pred diagnostics
            with torch.no_grad():
                p_f = pred.view(-1).float()
                t_f = y.view(-1).float()
                is_e = t_f.abs() > 1e-6
                pred_stds.append(p_f[is_e].std().item() if is_e.any() else 0.0)

                # direction accuracy this batch
                if is_e.any():
                    da = ((p_f[is_e] * t_f[is_e]) > 0).float().mean().item()
                    dir_acc_val += da
                    # batch corr
                    pc = p_f[is_e] - p_f[is_e].mean()
                    tc = t_f[is_e] - t_f[is_e].mean()
                    c  = (pc*tc).mean() / (p_f[is_e].std().clamp(1e-6) * t_f[is_e].std().clamp(1e-6))
                    corr_acc += c.item()

        # ── Validation ───────────────────────────────────────────────────────
        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for tokens, feats, y in val_loader:
                if tokens is not None:
                    tokens = (tokens[0].to(device), tokens[1].to(device))
                if feats is not None:
                    feats = feats.to(device)
                y = y.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=config.USE_AMP):
                    pred = net(tokens=tokens, features=feats)
                    batch_loss = continuous_weighted_direction_loss(pred, y)
                val_loss += batch_loss.item()

        avg_train = train_loss  / steps_per_epoch
        avg_val   = val_loss    / len(val_loader)
        avg_gn    = np.mean(grad_norms)
        max_gn    = np.max(grad_norms)
        avg_ps    = np.mean(pred_stds)
        avg_da    = dir_acc_val / batch_count
        avg_corr  = corr_acc    / batch_count
        current_lr = scheduler.get_last_lr()[0]

        # ── Compact per-epoch log ─────────────────────────────────────────────
        saved_marker = ""
        if avg_val < best_val:
            best_val   = avg_val
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(net.state_dict(), MODEL_PATH)
            saved_marker = "  ✓ SAVED"
        else:
            patience_counter += 1

        print(
            f"Ep{epoch+1:3d} | "
            f"Tr={avg_train:.4f}  Va={avg_val:.4f} | "
            f"LR={current_lr:.2e} | "
            f"GradN avg={avg_gn:.3f} max={max_gn:.3f} | "
            f"PredStd={avg_ps:.4f} | "
            f"DirAcc={avg_da*100:.1f}% | "
            f"Corr={avg_corr:.3f} | "
            f"Pat={patience_counter}/{PATIENCE}"
            f"{saved_marker}"
        )

        # ── Full diagnostics every 5 epochs ──────────────────────────────────
        if (epoch + 1) % 5 == 0:
            d_train = _full_eval_diagnostics(net, train_loader, device, tag="TRAIN")
            d_val   = _full_eval_diagnostics(net, val_loader,   device, tag="VAL")
            _print_diagnostics(d_train)
            _print_diagnostics(d_val)

            # Weight norm snapshot every 5 epochs
            wn = _weight_norms(net)
            print(f"  ── Weight Norms @ Ep{epoch+1} ──")
            for k, v in list(wn.items())[:8]:
                print(f"     {k:<45s} {v:.4f}")
            print()

        # ── Early stopping ────────────────────────────────────────────────────
        if patience_counter >= PATIENCE:
            print(f"\n  ⛔ Early stop — no val improvement for {PATIENCE} epochs.")
            print(f"  Best epoch: {best_epoch}  Best val: {best_val:.4f}")
            break

    print(f"\n  ✅ Training complete. Best epoch={best_epoch}  Best val={best_val:.4f}")
    print(f"  Saved to: {MODEL_PATH}\n")

    return best_val


# ── Main ──────────────────────────────────────────────────────────────────────

def make_rolling_folds(asset_data_list, config):
    """
    Build walk-forward fold specs from asset data.
    
    Each fold:
      train: [fold_train_start : fold_train_end]        absolute row indices
      val  : [fold_train_end + gap : val_end]           immediately after train
    
    No overlap. No leakage. Gap = FORECAST_HORIZON + 50 between train and val.
    
    Returns list of fold dicts:
      {
        'label'     : str,
        'train_list': [(asset_id, feat, targ, ohlc, train_end_abs), ...],
        'val_list'  : [(asset_id, feat_slice, targ_slice, ohlc_slice, None), ...]
      }
    """
    gap           = config.FORECAST_HORIZON + 50
    train_bars    = config.WFV_TRAIN_BARS
    val_bars      = config.WFV_VAL_BARS
    step_bars     = config.WFV_STEP_BARS
    min_seq       = config.LOOKBACK_WINDOW

    # Determine total length from first asset (single-asset setup)
    total_len = len(asset_data_list[0][1])   # feat array length

    folds = []
    fold_idx = 0
    train_start = 0

    while True:
        train_end = train_start + train_bars
        val_start = train_end + gap
        val_end   = val_start + val_bars

        # Stop if we'd exceed the data
        if val_end > total_len:
            break

        # Sanity: both slices must be large enough for at least one window
        if (train_end - train_start) < min_seq:
            break
        if (val_end - val_start) < min_seq:
            break

        fold_train_list = []
        fold_val_list   = []

        for asset_id, feat, targ, ohlc in asset_data_list:
            # ── TRAIN slice ──────────────────────────────────────────────────
            # Pass absolute train_end so create_multi_index_dataloaders
            # fits the scaler on feat[:train_end] — matching the slice we use.
            # We pass the full feat/targ/ohlc arrays from train_start to train_end.
            f_tr   = feat[train_start:train_end]
            t_tr   = targ[train_start:train_end]
            o_tr   = ohlc[train_start:train_end] if ohlc is not None else None
            # train_end for scaler fitting = len(f_tr) since we sliced
            fold_train_list.append((asset_id, f_tr, t_tr, o_tr, len(f_tr)))

            # ── VAL slice ────────────────────────────────────────────────────
            # Sliced absolutely — val is always the next window after train+gap.
            # No train data in this slice. Scaler fitted on train is passed in.
            f_va   = feat[val_start:val_end]
            t_va   = targ[val_start:val_end]
            o_va   = ohlc[val_start:val_end] if ohlc is not None else None
            fold_val_list.append((asset_id, f_va, t_va, o_va, None))

        folds.append({
            'label'      : f'fold_{fold_idx + 1}',
            'train_range': (train_start, train_end),
            'val_range'  : (val_start, val_end),
            'train_list' : fold_train_list,
            'val_list'   : fold_val_list,
        })

        fold_idx   += 1
        train_start += step_bars   # slide forward

    if len(folds) < config.WFV_MIN_FOLDS:
        raise ValueError(
            f"Only {len(folds)} folds generated but WFV_MIN_FOLDS={config.WFV_MIN_FOLDS}. "
            f"total_len={total_len}, train_bars={train_bars}, val_bars={val_bars}, "
            f"step_bars={step_bars}, gap={gap}. "
            f"Reduce WFV_TRAIN_BARS or WFV_STEP_BARS, or increase data."
        )

    return folds


def train():
    _set_seed(42)
    fe = FeatureEngineer(config=_make_feature_config())
    asset_data_list, feature_cols = process_dataset(config.DATA_FILE, fe)

    tok = None
    if config.INPUT_MODE in (InputMode.TOKENS_ONLY, InputMode.COMBINED):
        tok = KronosTokenizer(
            d_in=config.TOKENIZER_D_IN, d_model=config.TOKENIZER_D_MODEL,
            n_heads=config.TOKENIZER_N_HEADS, ff_dim=config.TOKENIZER_FF_DIM,
            n_enc_layers=config.TOKENIZER_N_ENC, n_dec_layers=config.TOKENIZER_N_DEC,
            s1_bits=config.TOKENIZER_S1_BITS, s2_bits=config.TOKENIZER_S2_BITS,
            group_size=config.TOKENIZER_GROUP_SIZE)
        if os.path.exists("tokenizer.pt"):
            state = torch.load("tokenizer.pt", map_location="cpu")
            missing, unexpected = tok.load_state_dict(state, strict=False)
            if unexpected: print(f"  ⚠ Ignored keys: {unexpected}")
            print("  Loaded tokenizer.pt")
        tok.eval()

    # ── Walk-Forward Folds ────────────────────────────────────────────────────
    if getattr(config, 'WFV_ENABLED', False):
        folds = make_rolling_folds(asset_data_list, config)
        print(f"\n  Walk-Forward: {len(folds)} folds  "
              f"(train={config.WFV_TRAIN_BARS} bars, "
              f"val={config.WFV_VAL_BARS} bars, "
              f"step={config.WFV_STEP_BARS} bars)\n")

        fold_results = []
        for fold in folds:
            print(f"\n  ── Fold {fold['label']} | "
                  f"train rows {fold['train_range']} | "
                  f"val rows {fold['val_range']} ──")

            # Scaler fitted on train slice only — enforced inside
            # create_multi_index_dataloaders with is_train=True
            train_loader, fitted_scalers = create_multi_index_dataloaders(
                fold['train_list'], config, feature_cols, tok, is_train=True)
            val_loader, _ = create_multi_index_dataloaders(
                fold['val_list'], config, feature_cols, tok,
                is_train=False, scalers=fitted_scalers)

            best_val = train_fold(fold['label'], train_loader, val_loader, feature_cols)
            fold_results.append({'fold': fold['label'], 'best_val': best_val,
                                  'train_range': fold['train_range'],
                                  'val_range': fold['val_range']})

        # Summary across folds
        print("\n" + "="*65)
        print("  Walk-Forward Summary")
        print("="*65)
        for r in fold_results:
            print(f"  {r['fold']} | train {r['train_range']} | "
                  f"val {r['val_range']} | best_val={r['best_val']:.4f}")
        best_fold = min(fold_results, key=lambda x: x['best_val'])
        print(f"\n  Best fold: {best_fold['fold']}  val={best_fold['best_val']:.4f}")

    else:
        # ── Original single-split path (kept for quick runs) ─────────────────
        gap = config.FORECAST_HORIZON + 50
        train_list, val_list = [], []
        for asset_id, feat, target, ohlc in asset_data_list:
            train_end = int(len(feat) * config.TRAIN_RATIO)
            val_start = train_end + gap
            val_end   = min(val_start + int(len(feat) * config.VAL_RATIO),
                            len(feat) - gap - config.LOOKBACK_WINDOW)
            if train_end > config.LOOKBACK_WINDOW:
                train_list.append((asset_id, feat, target, ohlc, train_end))
            if val_end > val_start + config.LOOKBACK_WINDOW:
                val_list.append((asset_id, feat[val_start:val_end],
                                 target[val_start:val_end],
                                 ohlc[val_start:val_end], None))

        train_loader, fitted_scalers = create_multi_index_dataloaders(
            train_list, config, feature_cols, tok, is_train=True)
        val_loader, _ = create_multi_index_dataloaders(
            val_list, config, feature_cols, tok, is_train=False, scalers=fitted_scalers)

        train_fold("baseline", train_loader, val_loader, feature_cols)


if __name__ == "__main__":
    train()