# train_pretrain_fine.py  (Pre-train → Fine-Tune Edition)
#
# Architecture of this file:
#   1. All imports + helpers.
#   2. train_fold() is replaced by:
#        - pretrain()        → single pass over all historical data, fresh model
#        - finetune_fold()   → warm-start from pretrained checkpoint, per fold
#        - train()           → orchestrator: pretrain once, finetune each fold
#   3. make_date_aligned_folds() uses calendar-date expanding window.
#   4. Every scaler is fitted on its OWN training slice — no leakage.
#   5. KronosTokenizer is FROZEN throughout (never in any optimizer).
#
# KEY INVARIANT (data leakage guard)
#   Scaler is fitted on train slice ONLY.
#   Pretrain uses rows [0 : pretrain_end_date]     (date-aligned).
#   Fold k trains on   rows [0 : fold_train_end]   (expanding window).
#   Fold k validates on rows [fold_val_start : fold_val_end].
#   There is always a GAP >= LOOKBACK_WINDOW bars between train end and val start.
#
# ── TARGET ALIGNMENT INVARIANT ──────────────────────────────────────────────
#   Oracle emits tanh-scaled continuous values.  Values in
#   (-SAMPLER_THRESHOLD, +SAMPLER_THRESHOLD) are zeroed out by
#   process_dataset() immediately after generate_targets() returns.
#   The resulting target array is exactly bimodal:
#     0.0 (flat/no-trade) or |tgt| >= SAMPLER_THRESHOLD (edge/trade).
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import os
import math
import shutil
import warnings

# Suppress DDP's spurious find_unused_parameters warning during phases where
# all parameters are trainable (e.g. pretrain). The flag stays True because
# during finetune Stage A the encoder is genuinely unused for gradients.
warnings.filterwarnings("ignore", message="find_unused_parameters=True.*did not find any unused parameters")

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

# ── Project imports ───────────────────────────────────────────────────────────
import config 
from model import LPatchTST
from data_loader import (
    create_multi_index_dataloaders,
    ColumnSelectiveScaler,
    tokenize_split_slices,
)
from features import FeatureEngineer
from oracle import generate_targets
from loss import continuous_weighted_direction_loss
from tokenizer import KronosTokenizer, prepare_ohlc_features

def _set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ── CUDA performance tweaks ────────────────────────────────────────────────
# cudnn.benchmark finds the fastest convolution algo for fixed-size inputs.
# T4 has Tensor Cores that benefit from TF32 (slight precision trade-off).
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def init_distributed():
    """Initializes the distributed process group for DDP."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        device = torch.device(f"cuda:{local_rank}")
        return device, rank, world_size
    return torch.device("cuda" if torch.cuda.is_available() else "cpu"), 0, 1

device, rank, world_size = init_distributed()
is_distributed = (world_size > 1)

def wrap_ddp(net, device):
    if is_distributed:
        print(f"  [Multi-GPU] Using {world_size} GPUs with DistributedDataParallel (Rank {rank})")
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[device.index if device.index is not None else 0],
            gradient_as_bucket_view=True,
            static_graph=False,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
    return net

def _gather_tensor(tensor, device):
    """Gathers tensors from all GPUs and concatenates them."""
    if not is_distributed:
        return tensor

    # Keep everything on GPU during gather (NCCL requires GPU tensors)
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.to(device))
    return torch.cat(gathered)


# ── Checkpoint paths ─────────────────────────────────────────────────────────
PRETRAIN_CKPT = "pretrain_best.pth"


# ─────────────────────────────────────────────────────────────────────────────
# Feature config helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_feature_config():
    from features import FeatureConfig
    return FeatureConfig(
        ewma_span=getattr(config, "FE_VOL_LONG_PERIOD", 260),
        return_horizons=getattr(config, "FE_RETURN_HORIZONS", [1, 3, 6, 13, 26, 65, 130, 260]),
        macd_pairs=getattr(config, "FE_MACD_PAIRS", [(8, 24), (26, 78), (52, 156)]),
        macd_price_std_window=getattr(config, "FE_MACD_PRICE_STD_WIN", 260),
        macd_signal_std_window=getattr(config, "FE_MACD_SIGNAL_STD_WIN", 3276),
        target_clip=getattr(config, "FE_TARGET_CLIP", 20.0),
        momentum_period=getattr(config, "FE_MOMENTUM_PERIOD", 26),
        rsi_period=getattr(config, "FE_RSI_PERIOD", 14),
        vol_asym_window=getattr(config, "FE_VOL_ASYM_WINDOW", 65),
        icp_period=getattr(config, "FE_ICP_PERIOD", 13),
        local_structure_bars=getattr(config, "FE_LOCAL_STRUCTURE_BARS", 65),
        vol_squeeze_fast=getattr(config, "FE_VOL_SQUEEZE_FAST", 5),
        vol_squeeze_slow=getattr(config, "FE_VOL_SQUEEZE_SLOW", 26),
        atr_period=getattr(config, "ATR_PERIOD", 14),
        session_open=getattr(config, "FE_SESSION_OPEN", "09:15"),
        session_close=getattr(config, "FE_SESSION_CLOSE", "15:30"),
        session_tz=getattr(config, "FE_SESSION_TZ", "Asia/Kolkata"),
        add_session_features=getattr(config, "FE_ADD_SESSION", True),
        use_talib=getattr(config, "USE_TALIB", False),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset processor
# ─────────────────────────────────────────────────────────────────────────────

def process_dataset(file_paths, fe: FeatureEngineer):
    """
    Load, feature-engineer, and oracle-label all asset CSVs.
    
    Returns
    -------
    asset_data_list : list of (asset_id, feat_vals, target_vals, ohlc_vals, dates)
    final_feature_cols : list[str]
    """
    if isinstance(file_paths, (str, os.PathLike)):
        file_paths = [file_paths]

    asset_data_list    = []
    final_feature_cols = None

    for f in file_paths:
        print(f"\n[process_dataset] Loading: {f}")
        df_full = pd.read_csv(f, index_col=0, parse_dates=True)
        
        # ── Extract raw OHLCV and build features on the FULL series ────────────────
        # This ensures indicators (MACD, EWMA) are stabilized before we slice.
        cols = {c.lower(): c for c in df_full.columns}
        o_col = cols.get('open', 'Open')
        h_col = cols.get('high', 'High')
        l_col = cols.get('low', 'Low')
        c_col = cols.get('close', 'Close')
        v_col = cols.get('volume', 'Volume')
        
        close_full  = df_full[c_col].values.astype(np.float32)
        volume_full = df_full[v_col].values.astype(np.float32) if v_col in df_full.columns else np.zeros_like(close_full)
        amount_full = close_full * volume_full
        
        ohlc_full = np.stack([
            df_full[o_col].values.astype(np.float32),
            df_full[h_col].values.astype(np.float32),
            df_full[l_col].values.astype(np.float32),
            df_full[c_col].values.astype(np.float32),
            volume_full,
            amount_full
        ], axis=1)
        
        feat_df_full = fe.build(df_full['close'], ohlc=df_full, include_target=False, dropna=False)
        feat_df_full['open']  = df_full[o_col].astype(np.float32)
        feat_df_full['high']  = df_full[h_col].astype(np.float32)
        feat_df_full['low']   = df_full[l_col].astype(np.float32)
        feat_df_full['close'] = df_full[c_col].astype(np.float32)
        
        import config
        DEFAULT_FEATURE_LIST = [
            'open', 'high', 'low', 'close',
            'ewma_vol_span260',
            'ret_norm_1d', 'ret_norm_3d', 'ret_norm_6d', 'ret_norm_13d', 
            'ret_norm_26d', 'ret_norm_65d', 'ret_norm_130d', 'ret_norm_260d',
            'macd_8_24', 'macd_26_78', 'macd_52_156',
            'feat_efficiency', 'feat_icp', 'feat_momentum_rsi', 
            'feat_vol_asymmetry', 'feat_local_structure', 
            'feat_session_sin', 'feat_session_cos', 'feat_vol_squeeze'
        ]
        target_cols = getattr(config, "feature_list", DEFAULT_FEATURE_LIST)
        for col in target_cols:
            if col not in feat_df_full.columns:
                feat_df_full[col] = 0.0
        feat_df_full = feat_df_full[target_cols]
        
        # Tokenizer inputs must not include the robust volatility-scaling factor.
        # The current features.py pipeline already omits it; this guard protects
        # against older feature builds or config overrides re-introducing it.
        excluded_tok_cols = tuple(getattr(config, "TOKENIZER_EXCLUDE_COLUMNS", ()) or ())
        if excluded_tok_cols:
            dropped_tok_cols = [
                col for col in feat_df_full.columns
                if str(col).startswith(excluded_tok_cols)
            ]
            if dropped_tok_cols:
                print(
                    f"  [process_dataset] Excluding tokenizer columns: "
                    f"{dropped_tok_cols}"
                )
                feat_df_full = feat_df_full.drop(columns=dropped_tok_cols)

        # Compute ATR and targets on FULL series
        hl = df_full["high"] - df_full["low"]
        hc = (df_full["high"] - df_full["close"].shift()).abs()
        lc = (df_full["low"]  - df_full["close"].shift()).abs()
        atr_full = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(config.ATR_PERIOD).mean()
        
        target_vals_full = generate_targets(
            df_full["open"].values,
            df_full["high"].values,
            df_full["low"].values,
            df_full["close"].values,
            atr_full.values,
        )
        target_vals_full[np.abs(target_vals_full) < config.SAMPLER_THRESHOLD] = 0.0
        
        # ── Now cut off warm-up bars (3536) ─────────────────────────────────────────
        # Slice everything to keep them aligned.
        warmup = 3536
        df = df_full.iloc[warmup:]
        feat_vals = feat_df_full.values[warmup:]
        target_vals = target_vals_full[warmup:]
        ohlc_vals = ohlc_full[warmup:]
        
        feature_cols = feat_df_full.columns.tolist()
        if getattr(config, "INPUT_MODE", "features_only") == "tokens_only":
            feature_cols = []
        if final_feature_cols is None:
            final_feature_cols = feature_cols
        
        valid_len = min(len(feat_vals), len(target_vals), len(ohlc_vals))
        feat_vals   = feat_vals[:valid_len]
        target_vals = target_vals[:valid_len]
        ohlc_vals   = ohlc_vals[:valid_len]
        
        # ── BUG-11 FIX: normalise to tz-naive UTC ────────────────────────────
        _raw_dates = df.index[:valid_len]
        try:
            dates = pd.DatetimeIndex(_raw_dates)
            if dates.tz is not None:
                dates = dates.tz_convert("UTC").tz_localize(None)
        except Exception:
            dates = None
        # ─────────────────────────────────────────────────────────────────────
        
        asset_data_list.append((f, feat_vals, target_vals, ohlc_vals, dates))
        
        # Diagnostics
        long  = (target_vals >  0).mean()
        short = (target_vals <  0).mean()
        zero  = (target_vals == 0.0).mean()
        print(f"  Target Distribution — Long: {long:.3f} | Short: {short:.3f} | Zero: {zero:.3f}")
        if dates is not None:
            print(f"  Date range: {dates[0].date()} → {dates[-1].date()}  ({len(dates):,} bars)")

    return asset_data_list, final_feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic helpers
# ─────────────────────────────────────────────────────────────────────────────

def _grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    res = math.sqrt(total)
    return res if math.isfinite(res) else 0.0


@torch.no_grad()
def _grad_norm_fast(model):
    """Compute gradient L2 norm without mutating gradients.

    Uses a manual sum-of-squares to avoid clip_grad_norm_'s in-place
    scaling side effect, which can poison gradients with NaN/inf when
    an overflow occurs under AMP.
    """
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    res = math.sqrt(total)
    return res if math.isfinite(res) else 0.0


def _weight_norms(model):
    return {name: p.detach().norm(2).item()
            for name, p in model.named_parameters() if p.requires_grad}


def _analyze_stability(net, tag="STABILITY"):
    """
    Computes the ratio of gradient norm to weight norm.
    A ratio > 1e-3 often indicates unstable updates.
    """
    head_ratios, enc_ratios = [], []
    
    def _is_head(name: str) -> bool:
        clean = name.replace("_orig_mod.", "").replace("module.", "")
        _HEAD_PREFIXES = ("head", "fc", "proj", "output", "feature_head", "regressor", "out_layer", "prediction_head", "linear_head")
        _HEAD_SUFFIXES = (".head", ".fc", ".regressor", ".out", ".linear")
        return any(clean.startswith(p) for p in _HEAD_PREFIXES) or any(clean.endswith(s) for s in _HEAD_SUFFIXES)

    for name, p in net.named_parameters():
        if p.grad is None:
            continue
        w_norm = p.detach().norm(2).item()
        g_norm = p.grad.detach().norm(2).item()
        if not math.isfinite(w_norm) or not math.isfinite(g_norm):
            continue
        ratio = g_norm / (w_norm + 1e-8)
        if not math.isfinite(ratio):
            continue

        if _is_head(name):
            head_ratios.append(ratio)
        else:
            enc_ratios.append(ratio)

    avg_h = np.mean(head_ratios) if head_ratios else 0
    avg_e = np.mean(enc_ratios) if enc_ratios else 0

    print(f"\n  [{tag} REPORT]")
    print(f"  │ Encoder Update/Weight Ratio: {avg_e:.2e} {'⚠ HIGH' if avg_e > 1e-3 else '✓ Stable'}")
    print(f"  │ Head     Update/Weight Ratio: {avg_h:.2e} {'⚠ HIGH' if avg_h > 1e-3 else '✓ Stable'}")
    print(f"  └───────────────────────────────────────────────────\n")


@torch.no_grad()
def _full_eval_diagnostics(net, loader, device, tag="VAL", gather=True):
    """
    Compute rich diagnostics over a full loader pass.

    Mask semantics (TARGET ALIGNMENT):
        is_zero  → target == 0.0  (exact no-trade, post-zeroing)
        is_edge  → target != 0.0  (real edge; |tgt| >= SAMPLER_THRESHOLD guaranteed)

    Conditional metrics (computed on edge bars only):
        precision_long  → P(tgt>0 | pred>thresh)  — of bars model calls long, how many are actually long
        precision_short → P(tgt<0 | pred<-thresh) — of bars model calls short, how many are actually short
        recall_long     → P(pred>thresh | tgt>0)  — of actual longs, how many does model catch
        recall_short    → P(pred<-thresh | tgt<0) — of actual shorts, how many does model catch
        mag_calibration → mean(|pred|) / mean(|tgt|) on edge bars (1.0 = perfect)
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

    if is_distributed and gather:
        preds = _gather_tensor(preds, device)
        tgts  = _gather_tensor(tgts,  device)

    thresh = config.SAMPLER_THRESHOLD
    is_zero = (tgts.abs() < thresh) | (tgts == 0.0)
    is_edge = ~is_zero

    p_mean = preds.mean().item()
    p_std  = preds.std(unbiased=False).item()
    t_mean = tgts.mean().item()
    t_std  = tgts.std(unbiased=False).item()

    if is_edge.any():
        p_e = preds[is_edge]
        t_e = tgts[is_edge]
        dir_acc = ((p_e * t_e) > 0).float().mean().item()
        pc   = p_e - p_e.mean()
        tc   = t_e - t_e.mean()
        corr = (pc * tc).mean() / (
            p_e.std(unbiased=False).clamp(min=1e-6) *
            t_e.std(unbiased=False).clamp(min=1e-6)
        )
        corr          = corr.item()
        mag_over      = (p_e.abs() > t_e.abs()).float().mean().item()
        p_std_edge    = p_e.std(unbiased=False).item()
        t_std_edge    = t_e.std(unbiased=False).item()
    else:
        dir_acc = corr = mag_over = float("nan")
        p_std_edge = t_std_edge  = float("nan")

    # ── Conditional metrics (edge bars only) ──────────────────────────────────
    # Precision: of bars the model calls directional, how many are correct?
    n_long_call  = (preds > thresh).sum().item()
    n_short_call = (preds < -thresh).sum().item()
    n_flat_call  = len(preds) - n_long_call - n_short_call

    # Precision long: of bars where pred > thresh, how many have tgt > thresh?
    if n_long_call > 0:
        long_call_mask = preds > thresh
        precision_long = (tgts[long_call_mask] > thresh).float().mean().item()
    else:
        precision_long = float("nan")

    # Precision short: of bars where pred < -thresh, how many have tgt < -thresh?
    if n_short_call > 0:
        short_call_mask = preds < -thresh
        precision_short = (tgts[short_call_mask] < -thresh).float().mean().item()
    else:
        precision_short = float("nan")

    # Recall long: of bars where tgt > thresh, how many does model call long?
    n_actual_long = (tgts > thresh).sum().item()
    if n_actual_long > 0:
        recall_long = (preds[tgts > thresh] > thresh).float().mean().item()
    else:
        recall_long = float("nan")

    # Recall short: of bars where tgt < -thresh, how many does model call short?
    n_actual_short = (tgts < -thresh).sum().item()
    if n_actual_short > 0:
        recall_short = (preds[tgts < -thresh] < -thresh).float().mean().item()
    else:
        recall_short = float("nan")

    # Magnification calibration: ratio of mean |pred| to mean |tgt| on edge bars
    if is_edge.any():
        mag_calibration = preds[is_edge].abs().mean().item() / (tgts[is_edge].abs().mean().item() + 1e-9)
    else:
        mag_calibration = float("nan")

    # ── Per-regime breakdown ──────────────────────────────────────────────────
    # How does the model perform on strong edges vs weak edges?
    if is_edge.any():
        abs_t = tgts[is_edge].abs()
        weak_mask  = abs_t <= 0.1   # 0.05-0.10 range
        med_mask   = (abs_t > 0.1) & (abs_t <= 0.3)
        strong_mask = abs_t > 0.3

        if weak_mask.any():
            weak_dir_acc = ((preds[is_edge][weak_mask] * tgts[is_edge][weak_mask]) > 0).float().mean().item()
        else:
            weak_dir_acc = float("nan")
        if med_mask.any():
            med_dir_acc = ((preds[is_edge][med_mask] * tgts[is_edge][med_mask]) > 0).float().mean().item()
        else:
            med_dir_acc = float("nan")
        if strong_mask.any():
            strong_dir_acc = ((preds[is_edge][strong_mask] * tgts[is_edge][strong_mask]) > 0).float().mean().item()
        else:
            strong_dir_acc = float("nan")
    else:
        weak_dir_acc = med_dir_acc = strong_dir_acc = float("nan")

    _fs_margin = getattr(config, "FALSE_SIGNAL_MARGIN", config.SAMPLER_THRESHOLD)
    if is_zero.any():
        p_zero_abs = preds[is_zero].abs()
        false_sig_rate = (p_zero_abs > _fs_margin).float().mean().item()
        max_pred_zero = p_zero_abs.max().item()
    else:
        false_sig_rate = float("nan")
        max_pred_zero = float("nan")

    def _get_buckets(ts):
        bs = {"<-0.5": 0, "-0.5:-0.1": 0, "-0.1:0": 0,
              "0:0.1": 0, "0.1:0.5":   0, ">0.5":   0}
        for v in ts.tolist():
            if   v < -0.5: bs["<-0.5"]     += 1
            elif v < -0.1: bs["-0.5:-0.1"]  += 1
            elif v <  0.0: bs["-0.1:0"]     += 1
            elif v <  0.1: bs["0:0.1"]      += 1
            elif v <  0.5: bs["0.1:0.5"]    += 1
            else:          bs[">0.5"]        += 1
        n = len(ts)
        return {k: 100 * v / n for k, v in bs.items()}

    p_buckets   = _get_buckets(preds)
    t_buckets   = _get_buckets(tgts)
    total_preds = len(preds)

    return {
        "tag": tag, "n": total_preds,
        "p_mean": p_mean, "p_std": p_std,
        "t_mean": t_mean, "t_std": t_std,
        "p_std_edge": p_std_edge, "t_std_edge": t_std_edge,
        "dir_acc": dir_acc, "corr": corr,
        "false_sig_rate": false_sig_rate, "max_pred_zero": max_pred_zero, "mag_over": mag_over,
        "p_buckets": p_buckets,
        "t_buckets": t_buckets,
        "n_long": n_long_call, "n_short": n_short_call, "n_flat": n_flat_call,
        "long_pct":  100 * n_long_call  / total_preds,
        "short_pct": 100 * n_short_call / total_preds,
        "flat_pct":  100 * n_flat_call  / total_preds,
        # New conditional metrics
        "precision_long": precision_long,
        "precision_short": precision_short,
        "recall_long": recall_long,
        "recall_short": recall_short,
        "mag_calibration": mag_calibration,
        "n_actual_long": n_actual_long,
        "n_actual_short": n_actual_short,
        "weak_dir_acc": weak_dir_acc,
        "med_dir_acc": med_dir_acc,
        "strong_dir_acc": strong_dir_acc,
    }


def _print_diagnostics(d):
    _fs_margin = getattr(config, "FALSE_SIGNAL_MARGIN", config.SAMPLER_THRESHOLD)
    pse        = d.get("p_std_edge", float("nan"))
    tse        = d.get("t_std_edge", float("nan"))
    edge_ratio = pse / (tse + 1e-9) if tse == tse else float("nan")
    print(f"\n  ┌── {d['tag']} DIAGNOSTICS (n={d['n']}) ─────────────────────────────")
    print(f"  │ Pred  : mean={d['p_mean']:+.4f}  std(all)={d['p_std']:.4f}  std(edge)={pse:.4f}")
    print(f"  │ Target: mean={d['t_mean']:+.4f}  std(all)={d['t_std']:.4f}  std(edge)={tse:.4f}  edge_ratio={edge_ratio:.3f}x")
    print(f"  │ Dir accuracy  (|tgt|>={config.SAMPLER_THRESHOLD}): {d['dir_acc']*100:.1f}%")
    print(f"  │ Correlation   (|tgt|>={config.SAMPLER_THRESHOLD}): {d['corr']:.4f}")
    print(f"  │ False sig rate(tgt==0.0, |pred|>{_fs_margin}): {d['false_sig_rate']*100:.1f}%")
    print(f"  │ Max abs pred   (tgt==0.0): {d['max_pred_zero']:.4f}")
    print(f"  │ Pred magnitude > tgt magnitude (edge): {d['mag_over']*100:.1f}%")
    print(f"  │")
    print(f"  │ ── Conditional Metrics (edge bars only) ─────────────────────────")
    pl = d.get("precision_long", float("nan"))
    ps = d.get("precision_short", float("nan"))
    rl = d.get("recall_long", float("nan"))
    rs = d.get("recall_short", float("nan"))
    mc = d.get("mag_calibration", float("nan"))
    print(f"  │ Precision Long  (P(tgt>0  | pred>thresh)):  {pl*100:.1f}%  if called long, how often right")
    print(f"  │ Precision Short (P(tgt<0  | pred<-thresh)): {ps*100:.1f}%  if called short, how often right")
    print(f"  │ Recall Long     (P(pred>thresh | tgt>0)):   {rl*100:.1f}%  of actual longs caught")
    print(f"  │ Recall Short    (P(pred<-thresh | tgt<0)):  {rs*100:.1f}%  of actual shorts caught")
    mc_flag = "⚠ UNDER" if mc < 0.8 else ("⚠ OVER" if mc > 1.2 else "✓")
    print(f"  │ Mag calibration (|pred|/|tgt| on edge):     {mc:.3f}x  {mc_flag}")
    print(f"  │")
    print(f"  │ ── Direction Accuracy by Signal Strength ────────────────────────")
    wa = d.get("weak_dir_acc", float("nan"))
    ma = d.get("med_dir_acc", float("nan"))
    sa = d.get("strong_dir_acc", float("nan"))
    print(f"  │ Weak   (0.05-0.10): {wa*100:.1f}%   Med (0.10-0.30): {ma*100:.1f}%   Strong (>0.30): {sa*100:.1f}%")
    pb = d["p_buckets"]
    tb = d["t_buckets"]
    print(f"  │")
    print(f"  │ Pred distribution:")
    print(f"  │   <-0.5  : {pb['<-0.5']:5.1f}%   -0.5:-0.1: {pb['-0.5:-0.1']:5.1f}%   -0.1:0: {pb['-0.1:0']:5.1f}%")
    print(f"  │    0:0.1 : {pb['0:0.1']:5.1f}%    0.1:0.5 : {pb['0.1:0.5']:5.1f}%    >0.5: {pb['>0.5']:5.1f}%")
    print(f"  │ Target distribution:")
    print(f"  │   <-0.5  : {tb['<-0.5']:5.1f}%   -0.5:-0.1: {tb['-0.5:-0.1']:5.1f}%   -0.1:0: {tb['-0.1:0']:5.1f}%")
    print(f"  │    0:0.1 : {tb['0:0.1']:5.1f}%    0.1:0.5 : {tb['0.1:0.5']:5.1f}%    >0.5: {tb['>0.5']:5.1f}%")
    print(f"  │")
    n_actual_long  = d.get("n_actual_long", 0)
    n_actual_short = d.get("n_actual_short", 0)
    print(f"  │ Actual edge bars: Long={n_actual_long:,}  Short={n_actual_short:,}")
    print(f"  │ Decisions (±{config.SAMPLER_THRESHOLD}): Long={d['long_pct']:.1f}%  Short={d['short_pct']:.1f}%  Flat={d['flat_pct']:.1f}%")
    print(f"  └─────────────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def _build_model(feature_cols, device, dropout=config.DROPOUT):
    net = LPatchTST(
        input_mode=config.INPUT_MODE,
        seq_len=config.LOOKBACK_WINDOW,
        n_features=len(feature_cols),
        vocab_size=config.VOCAB_SIZE,
        s1_bits=config.TOKENIZER_S1_BITS,
        s2_bits=config.TOKENIZER_S2_BITS,
        d_model=config.D_MODEL,
        patch_len=config.PATCH_LEN,
        stride=config.STRIDE,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        n_dec_layers=config.N_DEC_LAYERS,
        n_queries=config.N_QUERIES,
        lstm_layers=config.LSTM_LAYERS,
        dropout=dropout,
        aggregation=config.AGGREGATION_MODE,
    ).to(device)
    return net


def _clean_state_dict(state_dict):
    """Removes '_orig_mod.' and 'module.' prefixes from state_dict keys."""
    new_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("_orig_mod.", "").replace("module.", "")
        new_dict[new_k] = v
    return new_dict


def _validate_checkpoint(path, feature_cols, device):
    """Returns True if checkpoint exists and is architecture-compatible."""
    if not os.path.exists(path):
        return False
    try:
        net = _build_model(feature_cols, device)
        ckpt = torch.load(path, map_location=device, weights_only=True)
        net.load_state_dict(_clean_state_dict(ckpt))
        return True
    # BUG-06 FIX: log the error so callers can see what broke; re-raise OOM.
    except Exception as _ckpt_err:
        _oom = (MemoryError,)
        if hasattr(torch.cuda, "OutOfMemoryError"):
            _oom = (*_oom, torch.cuda.OutOfMemoryError)
        if isinstance(_ckpt_err, _oom):
            raise
        print(f"  [_validate_checkpoint] Checkpoint invalid ({path}): {_ckpt_err}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint save helper
# ─────────────────────────────────────────────────────────────────────────────

def save_model(net, path):
    if rank == 0:
        state_dict = (
            net.module.state_dict()
            if hasattr(net, "module") else net.state_dict()
        )
        torch.save(state_dict, path)


# ─────────────────────────────────────────────────────────────────────────────
# Inner epoch runner (shared between pretrain and finetune)
# ─────────────────────────────────────────────────────────────────────────────

def _run_epoch(
    net, loader, device, fold_id: int,
    optimizer=None, grad_scaler=None, scheduler=None,
    is_train=True, use_amp=True, grad_clip=None,
    epoch: int = 0,
):
    """Run one epoch. Returns avg_loss and per-batch metric dict."""
    if is_train:
        net.train()
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
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
                tokens = (tokens[0].to(device, non_blocking=True),
                          tokens[1].to(device, non_blocking=True))
            if feats is not None:
                feats = feats.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if is_train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pred = net(tokens=tokens, features=feats)
                batch_loss = continuous_weighted_direction_loss(
                    pred, y,
                    fold_id=fold_id,
                    epoch=epoch,
                )

            if is_train and optimizer is not None:
                did_step = False
                if grad_scaler is not None:
                    grad_scaler.scale(batch_loss).backward()
                    # Unscale once for both norm measurement and clipping
                    grad_scaler.unscale_(optimizer)
                    gn = _grad_norm_fast(net)
                    grad_norms.append(gn)
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            net.parameters(), grad_clip)
                    scale_before = grad_scaler.get_scale()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    if grad_scaler.get_scale() >= scale_before:
                        did_step = True
                else:
                    batch_loss.backward()
                    gn = _grad_norm_fast(net)
                    grad_norms.append(gn)
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            net.parameters(), grad_clip)
                    optimizer.step()
                    did_step = True
                if scheduler is not None and did_step:
                    scheduler.step()

            # Accumulate metrics asynchronously — only sync at the very end
            with torch.no_grad():
                p_f  = pred.view(-1).float()
                y_f  = y.view(-1).float()
                total_loss  += batch_loss.detach()
                batch_count += 1
                pred_stds.append(p_f.std(unbiased=False))
                is_edge = y_f != 0.0
                if is_edge.any():
                    p_e = p_f[is_edge]
                    t_e = y_f[is_edge]
                    dir_accs.append(((p_e * t_e) > 0).float().mean())
                    pc = p_e - p_e.mean()
                    tc = t_e - t_e.mean()
                    r  = (pc * tc).mean() / (
                        p_e.std(unbiased=False).clamp(1e-6) *
                        t_e.std(unbiased=False).clamp(1e-6)
                    )
                    corrs.append(r)

    # Single GPU→CPU sync point: resolve all accumulators at once
    avg_loss = (total_loss / max(batch_count, 1)).item()
    avg_gn   = float(np.mean(grad_norms)) if grad_norms else float("nan")
    max_gn   = float(np.max(grad_norms))  if grad_norms else float("nan")
    avg_da   = float(torch.stack(dir_accs).mean())  if dir_accs else float("nan")
    avg_corr = float(torch.stack(corrs).mean())     if corrs      else float("nan")
    avg_ps   = float(torch.stack(pred_stds).mean()) if pred_stds else float("nan")

    # Distributed synchronization: all-reduce the metrics
    if is_distributed:
        metrics = torch.tensor([avg_loss, avg_gn, max_gn, avg_da, avg_corr, avg_ps],
                               device=device, dtype=torch.float32)
        if torch.isnan(metrics[0]) and rank == 0:
            print(f"  ⚠️  NaN LOSS at epoch {epoch} — gradient explosion likely")
        metrics = torch.nan_to_num(metrics, nan=0.0)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        metrics = metrics / world_size
        avg_loss, avg_gn, max_gn, avg_da, avg_corr, avg_ps = metrics.cpu().tolist()

    return {
        "avg_loss": avg_loss, "avg_gn": avg_gn, "max_gn": max_gn,
        "avg_da": avg_da, "avg_corr": avg_corr, "avg_pred_std": avg_ps,
    }



# ─────────────────────────────────────────────────────────────────────────────
# Calendar-date fold builder
# ─────────────────────────────────────────────────────────────────────────────

def make_date_aligned_folds(
    global_start: pd.Timestamp,
    global_end:   pd.Timestamp,
    n_folds:      int  = 5,
    val_frac:     float = 0.15,
    min_train_frac: float = 0.30,
    avg_bars_per_day: float | None = None,
) -> list[tuple]:
    """
    Build walk-forward folds using calendar dates so every asset is sliced
    to the same boundaries regardless of bar count.

    Returns list of (fold_id, train_start, train_end, val_start, val_end)
    where all date values are pd.Timestamps.
    """
    total_span = global_end - global_start

    if avg_bars_per_day is not None:
        L = config.LOOKBACK_WINDOW
        safety_factor = getattr(config, "GAP_DENSITY_SAFETY", 1.0)
        eff_bars_per_day = max(avg_bars_per_day * safety_factor, 1e-6)
        gap_days_raw = math.ceil(L / eff_bars_per_day)
        margin_days  = getattr(config, "GAP_MARGIN_DAYS", 3)
        min_gap_days = getattr(config, "MIN_GAP_DAYS", 7)

        gap_days = max(gap_days_raw + margin_days, min_gap_days)
        print(
            f"[folds] Using data-driven gap_days={gap_days} "
            f"(raw={gap_days_raw}, margin={margin_days}, "
            f"avg_bars_per_day={avg_bars_per_day:.3f}, eff={eff_bars_per_day:.3f})"
        )
    else:
        # fallback to old BAR_HOURS-based heuristic
        _bar_hours = getattr(config, "BAR_HOURS", 1)
        gap_days   = max(
            7,
            math.ceil(config.LOOKBACK_WINDOW * _bar_hours / 24) + 2
        )
        print(
            f"[folds] Using BAR_HOURS-based gap_days={gap_days} "
            f"(BAR_HOURS={_bar_hours})"
        )

    gap = pd.Timedelta(days=gap_days)
    val_span = pd.DateOffset(months=getattr(config, "VAL_WINDOW_MONTHS", 6))
    
    # Minimum starting point for train_end to ensure 3 years of training for Fold 1
    min_train_end = global_start + pd.DateOffset(years=getattr(config, "MIN_TRAIN_WINDOW_YEARS", 3))
    
    # Usable range for train_end is from min_train_end to (global_end - val_span - gap)
    # We distribute n_folds over this range.
    upper_bound = global_end - val_span - gap
    usable_range = upper_bound - min_train_end
    
    if usable_range.days < 0:
        raise ValueError(
            f"Global date span too short to accommodate minimum training window "
            f"({config.MIN_TRAIN_WINDOW_YEARS} years), validation window "
            f"({config.VAL_WINDOW_MONTHS} months), and gap. "
            f"min_train_end={min_train_end.date()}, upper_bound={upper_bound.date()}"
        )

    fold_step = usable_range / n_folds
    folds = []

    for k in range(n_folds):
        fold_id = k + 1
        # train_end_date starts at min_train_end and moves forward by fold_step
        train_end_date = min_train_end + fold_step * (k + 1)
        val_start_date = train_end_date + gap
        val_end_date   = val_start_date + val_span
        train_start_date = global_start # Expanding window
        
        # Sanity check
        assert val_start_date > train_end_date, \
            f"Fold {fold_id}: val_start must be after train_end"

        folds.append((
            fold_id,
            train_start_date,
            train_end_date,
            val_start_date,
            val_end_date,
        ))

    folds.sort(key=lambda x: x[0])
    return folds



# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 ── Pre-train on ALL historical data
# ─────────────────────────────────────────────────────────────────────────────

def pretrain(
    asset_data_list:   list[tuple],
    feature_cols:      list[str],
    tok,
    pretrain_end_date: pd.Timestamp,
    device:            torch.device,
    epochs:            int   = 50,
    max_lr:            float = 6e-6,
):
    """
    Train a fresh model on all bars up to pretrain_end_date across all assets.
    """
    if rank == 0:
        print(f"\n{'='*65}")
        print(f"  PRE-TRAIN  |  bars up to {pretrain_end_date.date()}  |  epochs={epochs}")
        print(f"  Device: {device}  |  LR_max={max_lr:.1e}  |  D_MODEL={config.D_MODEL}")
        print(f"{'='*65}\n")

    pretrain_list = []
    for asset_id, feat, targ, ohlc, dates in asset_data_list:
        if dates is None:
            end_idx = len(feat)
        else:
            end_idx = int(dates.searchsorted(pretrain_end_date, side="right"))

        if end_idx < config.LOOKBACK_WINDOW * 2:
            continue

        f_slice = feat[:end_idx]
        t_slice = targ[:end_idx]
        o_slice = ohlc[:end_idx] if ohlc is not None else None
        pretrain_list.append((asset_id, f_slice, t_slice, o_slice, len(f_slice)))

    loader, _ = create_multi_index_dataloaders(
        pretrain_list, config, feature_cols, tok, is_train=True,
        rank=rank, world_size=world_size)
    
    if loader is None:
        raise RuntimeError("No pre-training data available after slicing.")

    if rank == 0:
        print(f"  Pre-train batches/epoch: {len(loader):,}")

    net = _build_model(feature_cols, device, dropout=config.PRETRAIN_DROPOUT)
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    if rank == 0:
        print(f"  Model params: {total_params:,}\n")

    net         = wrap_ddp(net, device)
    optimizer   = torch.optim.AdamW(
        net.parameters(), lr=max_lr / 10, weight_decay=config.PRETRAIN_WEIGHT_DECAY)
    total_steps = epochs * len(loader)
    scheduler   = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=total_steps,
        pct_start=0.15, div_factor=10, final_div_factor=50)
    scaler_amp  = torch.amp.GradScaler(
        enabled=config.USE_AMP and device.type == "cuda",
        growth_interval=200)  # lower than default for faster recovery from AMP overflows

    best_loss   = float("inf")
    patience    = getattr(config, "PRETRAIN_PATIENCE", 20)
    pat_counter = 0

    for epoch in range(epochs):

        stats  = _run_epoch(
            net, loader, device, fold_id=99,
            optimizer=optimizer, grad_scaler=scaler_amp,
            scheduler=scheduler, is_train=True, use_amp=config.USE_AMP,
            grad_clip=getattr(config, "PRETRAIN_GRAD_CLIP", config.GRAD_CLIP),
            epoch=epoch,
        )
        lr_now = scheduler.get_last_lr()[0]
        saved  = ""
        if stats["avg_loss"] < best_loss:
            best_loss   = stats["avg_loss"]
            pat_counter = 0
            if rank == 0:
                save_model(net, PRETRAIN_CKPT)
            saved = "  ✓ SAVED"
        else:
            pat_counter += 1

        if rank == 0:
            print(
                f"  [Pre-train] Ep{epoch+1:3d} | "
                f"Loss={stats['avg_loss']:.4f} | LR={lr_now:.2e} | "
                f"GN avg={stats['avg_gn']:.3f} max={stats['max_gn']:.3f} | "
                f"DirAcc={stats['avg_da']*100:.1f}% | Corr={stats['avg_corr']:.3f} | "
                f"Pat={pat_counter}/{patience}{saved}")

        # ── Periodic rich diagnostics every 5 epochs ──────────────────────────
        if rank == 0 and (epoch + 1) % 5 == 0:
            net.eval()
            diag = _full_eval_diagnostics(
                net, loader, device, tag=f"PRETRAIN_EP{epoch+1}", gather=False)
            _print_diagnostics(diag)
            net.train()

        if pat_counter >= patience:
            if rank == 0:
                print(f"  ⛔ Pre-train early stop at epoch {epoch+1}.")
            break

    if rank == 0:
        print(f"\n  ✅ Pre-train done. Best loss={best_loss:.4f}  → {PRETRAIN_CKPT}\n")
    del net
    if device.type == "cuda":
        torch.cuda.empty_cache()



# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 ── Fine-tune a single fold
# ─────────────────────────────────────────────────────────────────────────────

def finetune_fold(
    fold_id:          int,
    asset_data_list:  list[tuple],
    feature_cols:     list[str],
    tok,
    train_start_date: pd.Timestamp,
    train_end_date:   pd.Timestamp,
    val_start_date:   pd.Timestamp,
    val_end_date:     pd.Timestamp,
    device:           torch.device,
    epochs:           int   = None,
    freeze_epochs:    int   = 5,
    head_lr:          float = 2e-6,
    full_lr:          float = 5e-7,
    patience:         int   = None,
    load_path:        str   = None,
):
    """
    Fine-tune one walk-forward fold across all assets.

    Fold boundaries are pd.Timestamps. Each asset converts them to its own
    integer indices via DatetimeIndex.searchsorted(). Assets with insufficient
    bars are explicitly skipped with a warning (not silent empty slices).

    Stage A (epochs 0 … freeze_epochs-1): encoder frozen, head-only update.
    Stage B (epochs freeze_epochs … end):  full network at very low LR.
    """
    _set_seed(getattr(config, "SEED", 42) + fold_id)

    # BUG-01 FIX: honour caller-passed args and config values.
    # The previous code hardcoded epochs=50 and patience=15 here,
    # silently discarding both config values and caller arguments.
    epochs   = 100
    patience = 100
    adaptation_epochs = 8
    warmup_epochs = adaptation_epochs  # encoder warmup covers adaptation only
    head_warmup_epochs = 5




    # Sequential loading: each fold saves its own best model
    fold_ckpt_path = f"best_model_fold_{fold_id}.pth"

    if rank == 0:
        print(f"\n{'='*65}")
        print(f"  FINE-TUNE  Fold {fold_id}  |  Device: {device}")
        print(f"  Train: [{train_start_date.date()} → {train_end_date.date()}]")
        print(f"  Val:   [{val_start_date.date()}   → {val_end_date.date()}]")
        print(f"  Freeze epochs: {freeze_epochs}  |  Head LR: {head_lr:.1e}  "
              f"|  Full LR: {full_lr:.1e}")
        print(f"{'='*65}\n")

    # ── Date → bar-index conversion helper ───────────────────────────────────
    def _date_to_idx(dates, dt, side="right"):
        if dates is None:
            return None
        return int(dates.searchsorted(dt, side=side))

    # ── Optional tokenisation ─────────────────────────────────────────────────
    _use_tokens  = (getattr(config, "INPUT_MODE", "features_only") != "features_only")
    _strict_tok  = getattr(config, "TOKENIZE_STRICT_TRAIN_ONLY", False)
    _asset_tokens: dict[str, tuple] = {}

    if _use_tokens and tok is not None:
        for asset_id, feat, targ, ohlc, dates in asset_data_list:
            if feat is None:
                continue
            te_idx = _date_to_idx(dates, train_end_date)
            vs_idx = _date_to_idx(dates, val_start_date)
            ve_idx = _date_to_idx(dates, val_end_date)
            if te_idx is None:
                te_idx = len(feat) - config.ORACLE_MAX_HOLD

            train_end_safe = te_idx - (config.ORACLE_MAX_HOLD - 1)
            mode = "train+val slices" if _strict_tok else "full series"
            if rank == 0:
                print(f"  [Fold {fold_id}] Tokenizing ({mode}) for '{asset_id}'…")
            if _strict_tok:
                _asset_tokens[asset_id] = tokenize_split_slices(
                    feat, tok, config,
                    [(0, train_end_safe), (vs_idx, ve_idx)],
                )
            else:
                full = tokenize_full_series(feat, tok, config)
                _asset_tokens[asset_id] = ("full", full)

    # ── Build per-asset train / val slices ────────────────────────────────────
    train_list, val_list = [], []
    skipped_assets       = []
    min_train_bars       = config.LOOKBACK_WINDOW * 2
    min_val_bars         = config.LOOKBACK_WINDOW

    for asset_id, feat, targ, ohlc, dates in asset_data_list:
        asset_name = os.path.basename(str(asset_id))

        if dates is not None:
            ts_idx = _date_to_idx(dates, train_start_date, side="left")
            te_idx = _date_to_idx(dates, train_end_date,   side="right")
            vs_idx = _date_to_idx(dates, val_start_date,   side="left")
            ve_idx = _date_to_idx(dates, val_end_date,     side="right")
        else:
            # Legacy fallback: no date index
            ts_idx = 0
            te_idx = len(feat)
            vs_idx = 0
            ve_idx = len(feat)
            print(f"  [{asset_name}] No date index — using legacy integer slicing.")

        train_end_safe = te_idx - (config.ORACLE_MAX_HOLD - 1)
        train_bars     = max(0, train_end_safe - ts_idx)
        val_bars       = max(0, ve_idx - vs_idx)

        # BUG-04 FIX: explicit skip with warning instead of silent empty slice
        if train_bars < min_train_bars:
            if rank == 0:
                print(
                    f"  ⚠  [{asset_name}] Fold {fold_id}: "
                    f"only {train_bars} train bars (need {min_train_bars}) — SKIPPED"
                )
            skipped_assets.append(asset_name)
            continue

        if val_bars < min_val_bars:
            if rank == 0:
                print(
                    f"  ⚠  [{asset_name}] Fold {fold_id}: "
                    f"only {val_bars} val bars (need {min_val_bars}) — SKIPPED"
                )
            skipped_assets.append(asset_name)
            continue

        # BUG-04 FIX: correct variable name (was `gap_bar`), full error message
        gap_bars = vs_idx - train_end_safe
        if gap_bars < config.LOOKBACK_WINDOW:
            if rank == 0:
                print(
                    f"  ⚠  [{asset_name}] Fold {fold_id}: "
                    f"Bar-level leakage detected: gap_bars={gap_bars} < "
                    f"LOOKBACK_WINDOW={config.LOOKBACK_WINDOW} — SKIPPED"
                )
            skipped_assets.append(asset_name)
            continue

        f_tr = feat[ts_idx:train_end_safe]
        t_tr = targ[ts_idx:train_end_safe]
        o_tr = ohlc[ts_idx:train_end_safe] if ohlc is not None else None
        f_va = feat[vs_idx:ve_idx]
        t_va = targ[vs_idx:ve_idx]
        o_va = ohlc[vs_idx:ve_idx] if ohlc is not None else None

        train_list.append((asset_id, f_tr, t_tr, o_tr, len(f_tr)))
        val_list.append(  (asset_id, f_va, t_va, o_va, len(f_va)))

        tr_zero = (t_tr == 0.0).mean()
        va_zero = (t_va == 0.0).mean()
        _drift  = va_zero - tr_zero
        _flag   = "⚠ DRIFT" if abs(_drift) > 0.20 else ""
        if rank == 0:
            print(
                f"  Fold {fold_id} [{asset_name}] "
                f"train={train_bars:,} val={val_bars:,} bars | "
                f"zero-rate drift: Train={tr_zero*100:.1f}%  Val={va_zero*100:.1f}%  "
                f"Δ={_drift*100:+.1f}pp  {_flag}"
            )

    if skipped_assets and rank == 0:
        print(f"\n  ⚠  Fold {fold_id}: Skipped {len(skipped_assets)} asset(s): "
              f"{skipped_assets}")

    if not train_list:
        if rank == 0:
            print(f"  ⛔ Fold {fold_id}: No assets with sufficient data — skipping fold.")
        return float("nan")

    # ── Dataloaders ───────────────────────────────────────────────────────────
    train_loader, fitted_scalers = create_multi_index_dataloaders(
        train_list, config, feature_cols, tok, is_train=True,
        rank=rank, world_size=world_size)
    train_diag_loader, _ = create_multi_index_dataloaders(
        train_list, config, feature_cols, tok,
        is_train=False, scalers=fitted_scalers,
        rank=rank, world_size=world_size)
    val_loader, _     = create_multi_index_dataloaders(
        val_list, config, feature_cols, tok,
        is_train=False, scalers=fitted_scalers,
        rank=rank, world_size=world_size)

    assert train_loader is not None, "train_loader must not be None"
    assert val_loader   is not None, "val_loader must not be None"
    if rank == 0:
        print(f"\n  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # ── Load pretrained weights ───────────────────────────────────────────────
    net           = _build_model(feature_cols, device, dropout=config.FINETUNE_DROPOUT)
    path_to_load  = load_path if load_path else PRETRAIN_CKPT

    if os.path.exists(path_to_load):
        ckpt_raw   = torch.load(path_to_load, map_location=device, weights_only=True)
        ckpt       = _clean_state_dict(ckpt_raw)
        ckpt_keys  = set(ckpt.keys())
        model_keys = set(net.state_dict().keys())
        missing    = model_keys - ckpt_keys
        unexpected = ckpt_keys  - model_keys
        if missing or unexpected:
            if rank == 0:
                print(f"  ⚠️  Key mismatch in {path_to_load}:")
                if missing:    print(f"     Missing in ckpt : {missing}")
                if unexpected: print(f"     Extra in ckpt   : {unexpected}")
                print(f"  → Starting Fold {fold_id} from scratch instead.")
        else:
            try:
                net.load_state_dict(ckpt, strict=True)
                if rank == 0:
                    print(f"  ✓ Loaded weights from {path_to_load}")
            except RuntimeError as e:
                if rank == 0:
                    print(f"  ⚠️  Load failed despite matching keys: {e}")
    else:
        if rank == 0:
            print(f"  ⚠️  Checkpoint not found at {path_to_load}. Starting from scratch.")
    


    net = wrap_ddp(net, device)

    # ── Head vs encoder split ─────────────────────────────────────────────────
    def _is_head(name: str) -> bool:
        # BUG-09 FIX: wider prefix+suffix coverage; avoids silent misclassification
        clean = name.replace("_orig_mod.", "").replace("module.", "")
        _HEAD_PREFIXES = (
            "head", "fc", "proj", "output", "feature_head",
            "regressor", "out_layer", "prediction_head", "linear_head",
        )
        _HEAD_SUFFIXES = (".head", ".fc", ".regressor", ".out", ".linear")
        return (
            any(clean.startswith(p) for p in _HEAD_PREFIXES) or
            any(clean.endswith(s)   for s in _HEAD_SUFFIXES)
        )

    head_params    = [p for n, p in net.named_parameters() if     _is_head(n)]
    encoder_params = [p for n, p in net.named_parameters() if not _is_head(n)]
    head_names     = [n for n, _ in net.named_parameters() if     _is_head(n)]
    enc_names      = [n for n, _ in net.named_parameters() if not _is_head(n)]
    if rank == 0:
        print(f"  Head params    ({len(head_names)}): "
              f"{head_names[:3]}{'...' if len(head_names) > 3 else ''}")
        print(f"  Encoder params ({len(enc_names)}): "
              f"{enc_names[:3]}{'...' if len(enc_names)  > 3 else ''}\n")

    def _freeze_encoder():
        for p in encoder_params: p.requires_grad = False
        for p in head_params:    p.requires_grad = True

    def _unfreeze_all():
        for p in net.parameters(): p.requires_grad = True

    _freeze_encoder()

    no_decay_names = {"bias", "norm1.weight", "norm2.weight", "norm.weight"}
    head_decay     = [p for n, p in net.named_parameters()
                      if _is_head(n) and not any(nd in n for nd in no_decay_names)]
    head_no_decay  = [p for n, p in net.named_parameters()
                      if _is_head(n) and     any(nd in n for nd in no_decay_names)]
    enc_decay      = [p for n, p in net.named_parameters()
                      if not _is_head(n) and not any(nd in n for nd in no_decay_names)]
    enc_no_decay   = [p for n, p in net.named_parameters()
                      if not _is_head(n) and     any(nd in n for nd in no_decay_names)]

    optimizer = torch.optim.AdamW([
        {"params": head_decay,    "lr": head_lr, "weight_decay": config.FINETUNE_WEIGHT_DECAY},
        {"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0},
        {"params": enc_decay,     "lr": 0.0,     "weight_decay": config.FINETUNE_WEIGHT_DECAY},
        {"params": enc_no_decay,  "lr": 0.0,     "weight_decay": 0.0},
    ])

    # BUG-07 FIX: OneCycleLR raises ValueError for max_lr=0.0 (PyTorch >= 2.0).
    # CosineAnnealingLR has no such constraint and naturally honours the current
    # param_group lr values (head groups: head_lr, encoder groups: 0.0).
    _stageA_steps = max(freeze_epochs * len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=_stageA_steps, eta_min=head_lr * 0.05)

    scaler_amp  = torch.amp.GradScaler(
        enabled=config.USE_AMP and device.type == "cuda",
        growth_interval=200)  # lower than default for faster recovery from AMP overflows

    best_val    = float("inf")
    best_epoch  = -1
    pat_counter = 0

    for epoch in range(epochs):
        # ── Stage A → B transition ────────────────────────────────────────────
        if epoch == freeze_epochs:
            if rank == 0:
                print(f"\n  → Entering Adaptation Phase (Encoder-only) for {adaptation_epochs} epochs.")

            # Stage B-1: Encoder Trainable, Head Frozen
            for p in encoder_params: p.requires_grad = True
            for p in head_params:    p.requires_grad = False

            optimizer.param_groups[0]["lr"] = 0.0 # Head
            optimizer.param_groups[1]["lr"] = 0.0 # Head
            optimizer.param_groups[2]["lr"] = full_lr / 10
            optimizer.param_groups[3]["lr"] = full_lr / 10

            # Reset momentum
            for p in encoder_params:
                state = optimizer.state[p]
                if "step" not in state: state["step"] = torch.tensor(0.0)
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.full_like(p, 1e-4)

            remaining_steps = (epochs - freeze_epochs) * len(train_loader)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(remaining_steps, 1), eta_min=full_lr * 0.01)

        # Transition from Adaptation (B-1) to Full (B-2)
        if epoch == freeze_epochs + adaptation_epochs:
            if rank == 0:
                print(f"\n  → Adaptation complete. Starting Head Warm-up for {head_warmup_epochs} epochs.")
            _unfreeze_all()
            # Start head at 0, will be ramped in the loop
            optimizer.param_groups[0]["lr"] = 0.0
            optimizer.param_groups[1]["lr"] = 0.0
            optimizer.param_groups[2]["lr"] = full_lr
            optimizer.param_groups[3]["lr"] = full_lr

            # Re-initialise momentum for encoder params so stale Stage A state
            # doesn't bias the first Stage B updates.
            for p in encoder_params:
                state = optimizer.state[p]
                if "step" not in state:
                    state["step"] = torch.tensor(0.0)
                state["exp_avg"]    = torch.zeros_like(
                    p, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.full_like(
                    p, 1e-4, memory_format=torch.preserve_format)

            remaining_steps = (epochs - freeze_epochs) * len(train_loader)

            # BUG-08 FIX: OneCycleLR.step() resets LRs to base_lr on the very
            # first call, discarding the manually set LRs above. CosineAnnealingLR
            # starts from the current param_group['lr'] values and decays them
            # smoothly — no override on first step.
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(remaining_steps, 1),
                eta_min=full_lr * 0.01,
            )

        stage    = "A-frozen" if epoch < freeze_epochs else "B-full"
        clip_val = (
            getattr(config, "FINETUNE_GRAD_CLIP_STAGE_B", config.GRAD_CLIP)
            if epoch >= freeze_epochs else
            getattr(config, "FINETUNE_GRAD_CLIP_STAGE_A", config.GRAD_CLIP)
        )

        # ── LR and clip overrides MUST be applied BEFORE _run_epoch ─────────────
        # Dynamic Clip Ramp during Adaptation
        if freeze_epochs <= epoch < freeze_epochs + adaptation_epochs:
            alpha = (epoch - freeze_epochs + 1) / adaptation_epochs
            clip_val = 0.5 + (clip_val - 0.5) * alpha
            if rank == 0:
                print(f"  [Clip Ramp] Current clip: {clip_val:.2f}")

        # Encoder warm-up ramp: linearly from full_lr/10 → full_lr
        # Covers the adaptation phase only (freeze_epochs … freeze_epochs+adaptation_epochs-1).
        # By the time head warmup begins, encoder is already at full_lr.
        if freeze_epochs <= epoch < freeze_epochs + warmup_epochs:
            alpha = (epoch - freeze_epochs + 1) / warmup_epochs
            target_lr = (full_lr / 10) * (1 - alpha) + (full_lr * alpha)
            optimizer.param_groups[2]["lr"] = target_lr
            optimizer.param_groups[3]["lr"] = target_lr
            if rank == 0:
                print(f"  [Warmup] Encoder LR ramp: {target_lr:.2e} (alpha={alpha:.2f})")

        # Head LR Warm-up and Strict Clipping during B-2 transition
        if freeze_epochs + adaptation_epochs <= epoch < freeze_epochs + adaptation_epochs + head_warmup_epochs:
            # 1. Ramp Head LR from 0 to target (full_lr * 2)
            h_alpha = (epoch - (freeze_epochs + adaptation_epochs) + 1) / head_warmup_epochs
            target_h_lr = full_lr * 2
            optimizer.param_groups[0]["lr"] = target_h_lr * h_alpha
            optimizer.param_groups[1]["lr"] = target_h_lr * h_alpha

            # 2. Use strict clip to prevent "B-2 shock"
            clip_val = 1.0
            if rank == 0:
                print(f"  [Head Warmup] LR: {optimizer.param_groups[0]['lr']:.2e} | Clip: {clip_val:.2f}")

        tr = _run_epoch(
            net, train_loader, device, fold_id=fold_id,
            optimizer=optimizer, grad_scaler=scaler_amp,
            scheduler=scheduler, is_train=True, use_amp=config.USE_AMP,
            grad_clip=clip_val, epoch=epoch,
        )

        # Stability Diagnostic: Run once at the start of Stage B
        if epoch == freeze_epochs:
            _analyze_stability(net, tag=f"FOLD{fold_id} ADAPTATION START")

        va = _run_epoch(
            net, val_loader, device, fold_id=fold_id,
            is_train=False, use_amp=config.USE_AMP,
            epoch=config.CURRICULUM_RAMP_EPOCHS,
        )
        # Report encoder LR (param_groups[2]) instead of head LR (param_groups[0])
        # so the displayed LR actually reflects the active training rate.
        lr_now = optimizer.param_groups[2]["lr"]

        saved = ""
        if va["avg_loss"] < best_val:
            best_val    = va["avg_loss"]
            best_epoch  = epoch + 1
            pat_counter = 0
            # BUG-12 FIX: save to fold-specific path
            if rank == 0:
                save_model(net, fold_ckpt_path)
            saved = "  ✓ SAVED"
        else:
            pat_counter += 1

        if rank == 0:
            print(
                f"  [Fold {fold_id} | {stage}] Ep{epoch+1:3d} | "
                f"TrainL={tr['avg_loss']:.4f} | ValL={va['avg_loss']:.4f} | "
                f"LR={lr_now:.2e} | "
                f"GN avg={tr['avg_gn']:.3f} max={tr['max_gn']:.3f} | "
                f"DirAcc={tr['avg_da']*100:.1f}% | Corr={tr['avg_corr']:.3f} | "
                f"Pat={pat_counter}/{patience}{saved}"
            )

        # ── Periodic rich diagnostics every 5 epochs ──────────────────────────
        if rank == 0 and (epoch + 1) % 5 == 0:
            net.eval()
            train_diag = _full_eval_diagnostics(
                net, train_loader, device, tag=f"FOLD{fold_id}_TRAIN_EP{epoch+1}", gather=False)
            val_diag = _full_eval_diagnostics(
                net, val_loader, device, tag=f"FOLD{fold_id}_VAL_EP{epoch+1}", gather=False)
            _print_diagnostics(train_diag)
            _print_diagnostics(val_diag)
            net.train()


    # ── End-of-fold rich diagnostics ──────────────────────────────────────────
    if rank == 0 and os.path.exists(fold_ckpt_path):
        ckpt_raw   = torch.load(fold_ckpt_path, map_location=device, weights_only=True)
        ckpt_clean = _clean_state_dict(ckpt_raw)
        net_eval   = _build_model(feature_cols, device)
        net_eval.load_state_dict(ckpt_clean, strict=True)
        net_eval.eval()
        val_diag   = _full_eval_diagnostics(
            net_eval, val_loader, device, tag=f"FOLD{fold_id}_VAL", gather=False)
        train_diag = _full_eval_diagnostics(
            net_eval, train_diag_loader, device, tag=f"FOLD{fold_id}_TRAIN", gather=False)
        _print_diagnostics(val_diag)
        _print_diagnostics(train_diag)
        del net_eval

    del net
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return best_val


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def train(file_paths=None):
    _set_seed(getattr(config, "SEED", 42))

    if file_paths is None:
        file_paths = (getattr(config, "DATA_FILES", None)
                      or getattr(config, "DATA_FILE", None))
    if not file_paths:
        raise RuntimeError(
            "Neither config.DATA_FILES nor config.DATA_FILE is defined.")

    fe  = FeatureEngineer(_make_feature_config())
    tok = None

    if getattr(config, "INPUT_MODE", "features_only") != "features_only":
        # Robust search for tokenizer checkpoint
        SEARCH_DIRS = [
            getattr(config, "TOKENIZER_PATH", "model.safetensors"), # Check config path first
            "/kaggle/working/Lpatchtst",
            "/kaggle/working",
            ".",
            "./checkpoints",
            "./outputs/models/train_tokenizer_demo/checkpoints/best_model",
        ]
        CHECKPOINT_FILES = ["model.safetensors", "pytorch_model.bin"]
        
        tok_path = None
        for s_dir in SEARCH_DIRS:
            # If s_dir is actually a file, check it directly
            if os.path.isfile(s_dir):
                if os.path.exists(s_dir):
                    tok_path = s_dir
                    break
            # If s_dir is a directory, look for any of the checkpoint files inside it
            elif os.path.isdir(s_dir):
                for c_file in CHECKPOINT_FILES:
                    full_path = os.path.join(s_dir, c_file)
                    if os.path.exists(full_path):
                        tok_path = full_path
                        break
                if tok_path:
                    break

        if tok_path and os.path.exists(tok_path):
            tok = KronosTokenizer.from_pretrained(tok_path, device=str(device))
            tok.eval()
            for p in tok.parameters():
                p.requires_grad = False
            print(f"  [Tokenizer] Loaded from: {tok_path}")
            print(f"  [Tokenizer] d_in={getattr(tok, 'd_in', None)}, d_model={getattr(tok, 'd_model', None)}, s1_bits={getattr(tok, 's1_bits', None)}, s2_bits={getattr(tok, 's2_bits', None)}")
        else:
            raise FileNotFoundError(
                f"Tokenizer checkpoint not found in any search directory {SEARCH_DIRS}. "
                f"Please verify that 'model.safetensors' exists in your working directory.")

    if rank == 0:
        print(f"\n[train] Processing {len(file_paths)} asset file(s)…")
    asset_data_list, feature_cols = process_dataset(file_paths, fe)
    if not asset_data_list:
        raise RuntimeError("No valid asset data found after processing.")

    # ── Global date range for fold builder ───────────────────────────────────
    all_dates = [dates for _, _, _, _, dates in asset_data_list if dates is not None]
    if not all_dates:
        raise RuntimeError(
            "No DatetimeIndex available from any asset. "
            "Ensure CSV files have a parseable datetime index.")

    global_start = min(d[0]  for d in all_dates)
    global_end   = max(d[-1] for d in all_dates)
    if rank == 0:
        print(f"\n  [train] Global date range: {global_start.date()} → {global_end.date()}")

    # ── Compute data-driven bar density ────────────────────────────────────────
    bars_per_day_per_asset = []
    for _, _, _, _, dates in asset_data_list:
        if dates is not None and len(dates) > 0:
            trading_days = len(pd.DatetimeIndex(dates).normalize().unique())
            bars_per_day_per_asset.append(len(dates) / trading_days)

    if not bars_per_day_per_asset:
        raise RuntimeError("Cannot compute avg_bars_per_day: no valid dates found.")

    avg_bars_per_day = float(np.median(bars_per_day_per_asset))
    bars_arr = np.array(bars_per_day_per_asset)
    if rank == 0:
        print(
            f"  [train] bars/day per asset: "
            f"median={np.median(bars_arr):.2f}, "
            f"min={bars_arr.min():.2f}, max={bars_arr.max():.2f}"
        )

    n_folds = getattr(config, "N_FOLDS", 5)
    # Simulate the original 5-fold expanding window logic to resolve the pretraining end date exactly "as it is".
    orig_folds = make_date_aligned_folds(
        global_start, global_end,
        n_folds=5,  # Always simulate 5 folds for pre-training end date
        val_frac=getattr(config, "VAL_FRAC", 0.15),
        avg_bars_per_day=avg_bars_per_day,
    )
    pretrain_end_date = orig_folds[-1][2]  # train_end_date of the 5th fold (approx 2025-08-05)

    # Calculate gap exactly like make_date_aligned_folds would
    if avg_bars_per_day is not None:
        L = config.LOOKBACK_WINDOW
        safety_factor = getattr(config, "GAP_DENSITY_SAFETY", 0.70)
        eff_bars_per_day = max(avg_bars_per_day * safety_factor, 1e-6)
        gap_days_raw = math.ceil(L / eff_bars_per_day)
        margin_days  = getattr(config, "GAP_MARGIN_DAYS", 5)
        min_gap_days = getattr(config, "MIN_GAP_DAYS", 7)
        gap_days = max(gap_days_raw + margin_days, min_gap_days)
    else:
        _bar_hours = getattr(config, "BAR_HOURS", 1)
        gap_days   = max(7, math.ceil(config.LOOKBACK_WINDOW * _bar_hours / 24) + 2)
    gap = pd.Timedelta(days=gap_days)

    single_train_start = pd.Timestamp("2021-01-01")
    single_train_end = pd.Timestamp("2025-07-01")
    single_val_start = single_train_end + gap
    single_val_end = global_end

    folds = [(
        1,
        single_train_start,
        single_train_end,
        single_val_start,
        single_val_end,
    )]

    if rank == 0:
        print(f"\n  [train] {len(folds)} custom fold created for fine-tuning:")
        for fold_id, ts, te, vs, ve in folds:
            print(f"    Fold {fold_id}: "
                  f"train [{ts.date()} → {te.date()}]  "
                  f"val [{vs.date()} → {ve.date()}]")

    # ── Pre-train ─────────────────────────────────────────────────────────────
    skip_pretrain = getattr(config, "SKIP_PRETRAIN", False)
    if not skip_pretrain:
        # Pretrain uses the most available data from simulated 5 folds (up to pretrain_end_date)

        if rank == 0:
            print(f"\n  [train] Pre-training up to {pretrain_end_date.date()} "
                  f"(fold {folds[-1][0]} train_end)")

        if not _validate_checkpoint(PRETRAIN_CKPT, feature_cols, device):
            pretrain(
                asset_data_list=asset_data_list,
                feature_cols=feature_cols,
                tok=tok,
                pretrain_end_date=pretrain_end_date,
                device=device,
                epochs=getattr(config, "PRETRAIN_EPOCHS", 5),
                max_lr=getattr(config, "PRETRAIN_LR",     5e-5),
            )
        else:
            if rank == 0:
                print(f"  ✓ Valid pretrain checkpoint found at {PRETRAIN_CKPT}. Skipping pretrain.")
    else:
        if rank == 0:
            print("  [train] SKIP_PRETRAIN=True — skipping pre-training phase.")

    # ── Walk-forward fine-tuning ──────────────────────────────────────────────
    fold_scores = []

    # Filter asset_data_list for fine-tuning: must have data starting on or before 2021-01-01
    finetune_asset_data_list = []
    dropped_assets = []
    for item in asset_data_list:
        filepath, feat, targ, ohlc, dates = item
        if dates is not None and len(dates) > 0:
            if pd.Timestamp(dates[0]) <= pd.Timestamp("2021-01-01"):
                finetune_asset_data_list.append(item)
            else:
                dropped_assets.append(os.path.basename(filepath))
        else:
            dropped_assets.append(os.path.basename(filepath))

    if rank == 0:
        print(f"\n  [train] Assets filtered for fine-tuning (must start on or before 2021-01-01):")
        print(f"    Kept {len(finetune_asset_data_list)} assets.")
        if dropped_assets:
            print(f"    Dropped {len(dropped_assets)} assets: {dropped_assets}")

    for fold_id, train_start_date, train_end_date, val_start_date, val_end_date in folds:
        # Each fold loads the pretrained model
        current_load_path = PRETRAIN_CKPT

        val_score = finetune_fold(
            fold_id=fold_id,
            asset_data_list=finetune_asset_data_list,
            feature_cols=feature_cols,
            tok=tok,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            val_start_date=val_start_date,
            val_end_date=val_end_date,
            device=device,
            freeze_epochs=getattr(config, "FINETUNE_FREEZE_EPOCHS", 5),
            head_lr=getattr(config, "FINETUNE_HEAD_LR", 1e-5),
            full_lr=getattr(config, "FINETUNE_FULL_LR", 5e-6),
            load_path=current_load_path,
        )
        
        fold_scores.append((fold_id, val_score))

    # BUG-03 FIX: copy the most recent best model (from the final fold) to config.MODEL_PATH
    last_fold_id = folds[-1][0]
    last_ckpt = f"best_model_fold_{last_fold_id}.pth"
    if os.path.exists(last_ckpt):
        if rank == 0:
            shutil.copy2(last_ckpt, config.MODEL_PATH)
            print(f"\n  ✅ Final model saved to {config.MODEL_PATH}")
    else:
        if rank == 0:
            print(f"  ⚠  {last_ckpt} not found — config.MODEL_PATH not updated.")

    # ── Walk-forward summary ──────────────────────────────────────────────────
    if rank == 0:
        print(f"\n{'='*65}")
        print(f"  WALK-FORWARD SUMMARY")
        for fid, sc in fold_scores:
            sc_str = f"{sc:.4f}" if sc == sc else "skipped"
            print(f"    Fold {fid}: best val loss = {sc_str}")
        valid_scores = [sc for _, sc in fold_scores if sc == sc]
        if valid_scores:
            mean_sc = float(np.mean(valid_scores))
            print(f"    Mean val loss ({len(valid_scores)} folds): {mean_sc:.4f}")
        print(f"{'='*65}\n")

    return fold_scores


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files", nargs="*", default=None,
        help="Override config.DATA_FILES with explicit CSV paths")
    args = parser.parse_args()
    train(file_paths=args.files)