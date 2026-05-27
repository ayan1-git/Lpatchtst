# train_pretrain_fine.py  (Pre-train → Fine-Tune Edition)
#
# Architecture of this file:
#   1. All imports + helpers.
#   2. train_fold() is replaced by:
#        - pretrain()        → single pass over all historical data, fresh model
#        - finetune_fold()   → warm-start from pretrained checkpoint, per fold
#        - train()           → orchestrator: pretrain once, finetune each fold
#   3. make_rolling_folds() uses expanding window.
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
#
# ── TARGET ALIGNMENT INVARIANT ──────────────────────────────────────────────
#   Oracle emits tanh-scaled continuous values.  Values in (-SAMPLER_THRESHOLD,
#   +SAMPLER_THRESHOLD) are ambiguous micro-signals that the loss function
#   treats as noise via the false_signal_loss component.  To remove this
#   contradiction at the source, process_dataset() ZEROS OUT all oracle targets
#   whose absolute value falls below config.SAMPLER_THRESHOLD immediately after
#   generate_targets() returns.  The resulting target array is always exactly
#   bimodal: 0.0 (flat/no-trade) or |tgt| ≥ SAMPLER_THRESHOLD (edge/trade).
#   All downstream masks (is_zero, is_edge, is_e) use == 0.0 / != 0.0 rather
#   than abs() comparisons to remain consistent.
# ────────────────────────────────────────────────────────────────────────────
#
# ── DATE-ALIGNMENT FIX ───────────────────────────────────────────────────────
#   Previously, folds were built using integer bar indices anchored to the
#   LONGEST asset.  Short assets silently produced empty slices for later
#   folds (val_start > len(short_asset)), causing:
#     • val loss computed on fewer assets than train loss (not comparable)
#     • different assets present in each fold with no accounting
#     • temporal misalignment: bar 0 = different calendar date per asset
#
#   Fix:
#     1. process_dataset() now preserves the datetime index alongside numpy
#        arrays: asset_data_list entries include a `dates` DatetimeIndex.
#     2. make_date_aligned_folds() builds fold boundaries as pd.Timestamps
#        (not integer bar counts) anchored to the same calendar dates across
#        all assets.
#     3. finetune_fold() converts calendar dates → per-asset integer indices
#        via DatetimeIndex.searchsorted(), so each asset contributes exactly
#        the bars it has for a given calendar window.
#     4. Assets with insufficient bars for a fold are EXPLICITLY SKIPPED with
#        a printed warning instead of silently contributing empty slices.
#     5. pretrain() similarly builds its slice from dates up to pretrain_end_date.
# ────────────────────────────────────────────────────────────────────────────

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
    tokenize_full_series,
    tokenize_split_slices,
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

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import builtins
import logging

# ── Distributed Setup ────────────────────────────────────────────────────────
is_distributed = "WORLD_SIZE" in os.environ
if is_distributed:
    try:
        dist.init_process_group(backend="nccl")
        _dist_backend = "nccl"
    except (RuntimeError, ValueError):
        dist.init_process_group(backend="gloo")
        _dist_backend = "gloo"
    local_rank = int(os.environ["LOCAL_RANK"])
    rank       = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    print(f"[dist] Backend={_dist_backend}  Device={device}  Rank={rank}/{world_size}")
    if rank != 0:
        logging.disable(logging.CRITICAL)
else:
    local_rank = 0
    rank       = 0
    world_size = 1
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

original_print = builtins.print
def print_rank0(*args, **kwargs):
    if rank == 0:
        original_print(*args, **kwargs)
builtins.print = print_rank0


# ── Distributed Weighted Sampler ─────────────────────────────────────────────
class DistributedWeightedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, weights, num_samples, num_replicas=None,
                 rank=None, replacement=True, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package")
            rank = dist.get_rank()
        self.dataset     = dataset
        self.weights     = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.num_replicas = num_replicas
        self.rank         = rank
        self.replacement  = replacement
        self.seed         = seed
        self.num_samples_per_replica = int(
            math.ceil(self.num_samples * 1.0 / self.num_replicas))
        self.total_size = self.num_samples_per_replica * self.num_replicas
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, self.total_size, self.replacement, generator=g).tolist()
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples_per_replica
        return iter(indices)

    def __len__(self):
        return self.num_samples_per_replica

    def set_epoch(self, epoch):
        self.epoch = epoch


def _make_distributed_loader(loader, is_train):
    if not is_distributed or loader is None:
        return loader
    ds = loader.dataset
    if is_train:
        orig_sampler = loader.sampler
        dist_sampler = DistributedWeightedSampler(
            dataset=ds,
            weights=orig_sampler.weights,
            num_samples=orig_sampler.num_samples,
            num_replicas=world_size,
            rank=rank,
            replacement=True,
            seed=42,
        )
        return _make_loader(ds, config, sampler=dist_sampler, drop_last=True)
    else:
        dist_sampler = DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=False)
        return _make_loader(ds, config, sampler=dist_sampler, drop_last=False)


def wrap_ddp_and_compile(net, device):
    if is_distributed:
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    if device.type == "cuda":
        try:
            net = torch.compile(net)
            print("  torch.compile: OK")
        except Exception as e:
            print(f"  torch.compile: Failed ({e})")
    return net


def save_model(net, path):
    if rank == 0:
        state_dict = net.module.state_dict() if is_distributed else net.state_dict()
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        torch.save(clean_state_dict, path)
    if is_distributed:
        dist.barrier()


def _gather_tensor(tensor, device):
    if not is_distributed:
        return tensor
    size = torch.tensor([tensor.numel()], dtype=torch.long, device=device)
    all_sizes = [torch.zeros(1, dtype=torch.long, device=device)
                 for _ in range(world_size)]
    dist.all_gather(all_sizes, size)
    max_size = max(s.item() for s in all_sizes)
    padded = torch.zeros(max_size, dtype=tensor.dtype, device=device)
    padded[:tensor.numel()] = tensor.to(device)
    gathered = [torch.zeros(max_size, dtype=tensor.dtype, device=device)
                for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    unpadded = [gathered[r][:all_sizes[r].item()] for r in range(world_size)]
    return torch.cat(unpadded).cpu()


PRETRAIN_CKPT = "pretrained_lpatchtst.pth"
MODEL_PATH    = "best_model_lpatchtst.pth"
OHLC_COLS     = ["open", "high", "low", "close", "volume"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    no_scale_cols += [
        "feat_efficiency", "feat_icp", "feat_momentum_rsi",
        "feat_vol_asymmetry", "feat_local_structure",
    ]
    robust_cols.append("feat_vol_squeeze")
    if fe_config.add_session_features:
        no_scale_cols += ["feat_session_sin", "feat_session_cos"]
    if fe_config.use_talib:
        try:
            from talib_features import TALIB_PASSTHROUGH, TALIB_SCALE
            no_scale_cols += TALIB_PASSTHROUGH
            no_scale_cols += TALIB_SCALE
        except ImportError:
            pass
    return no_scale_cols, robust_cols, robust_cols + no_scale_cols


def _build_features(df, fe):
    time_col = next((c for c in df.columns if c.lower() in ("date", "datetime")), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
    else:
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    df = df.sort_index()

    if getattr(config, "INPUT_MODE", "features_only") == "tokens_only":
        combined_df = df.copy()
        all_feat_cols = []
    else:
        feat_df = fe.build(df["close"], ohlc=df[OHLC_COLS],
                           include_target=False, dropna=False)
        combined_df = df.join(feat_df, how="inner")
        _, _, all_feat_cols = _build_feature_cols(fe.config)

    hl = combined_df["high"] - combined_df["low"]
    hc = (combined_df["high"] - combined_df["close"].shift()).abs()
    lc = (combined_df["low"]  - combined_df["close"].shift()).abs()
    combined_df["atr"] = (
        pd.concat([hl, hc, lc], axis=1).max(axis=1)
          .rolling(config.ATR_PERIOD).mean()
    )
    combined_df.dropna(inplace=True)
    return combined_df, all_feat_cols



def process_dataset(file_paths, fe):
    """
    Load raw CSV files, engineer features, call the Oracle, then ZERO OUT
    any oracle target whose absolute value is below config.SAMPLER_THRESHOLD.

    Rationale (TARGET ALIGNMENT):
        The Oracle emits tanh(net_R / σ) values in [-1, 1].  Values in
        (-SAMPLER_THRESHOLD, +SAMPLER_THRESHOLD) represent marginal trades
        that technically fired but produce net-R too small to distinguish from
        noise.  The loss function treats predictions in this band as false
        signals (false_signal_loss) regardless of oracle intent.  Retaining
        sub-threshold oracle values would place those bars in a contradictory
        state: the Oracle says "trade", the loss says "noise".  Zeroing them
        out makes the contract unambiguous: exactly 0.0 means no-trade, and
        any nonzero value is a real edge with |tgt| ≥ SAMPLER_THRESHOLD.

    DATE-ALIGNMENT:
        Each asset_data_list entry now includes `dates` (a DatetimeIndex of
        length valid_len), which aligns 1-to-1 with feat_vals and target_vals.
        This index is used by make_date_aligned_folds() and finetune_fold()
        to build calendar-anchored fold windows that are consistent across
        all assets, regardless of their individual bar counts.
    """
    asset_data_list, final_feature_cols = [], []
    thresh = config.SAMPLER_THRESHOLD  # e.g. 0.05

    for f in file_paths:
        if not os.path.exists(f):
            continue
        print(f"Processing {f}…")
        df_raw = pd.read_csv(f)
        df, feature_cols = _build_features(df_raw, fe)
        ohlc_returns = prepare_ohlc_features(df)
        df = df.iloc[1:]

        targets = generate_targets(
            df["open"].values, df["high"].values, df["low"].values,
            df["close"].values, df["atr"].values,
            max_hold=config.ORACLE_MAX_HOLD,
            fee_per_side=config.FEE_PER_SIDE,
            slippage=config.SLIPPAGE,
            sl_atr_mult=config.ORACLE_SL_ATR_MULT,
            tp_atr_mult=config.ORACLE_TP_ATR_MULT,
            saturation_factor=config.SATURATION_FACTOR,
            mae_penalty=config.MAE_PENALTY,
        )

        # ── TARGET ALIGNMENT FIX ─────────────────────────────────────────────
        targets = np.where(np.abs(targets) < thresh, 0.0, targets).astype(np.float32)
        # ─────────────────────────────────────────────────────────────────────

        valid_len = len(targets) - config.ORACLE_MAX_HOLD
        if valid_len <= 0:
            continue

        feat_vals   = np.asarray(df[feature_cols].values,  dtype=np.float32)[:valid_len]
        target_vals = targets[:valid_len]
        ohlc_vals   = ohlc_returns[:valid_len]

        # ── DATE-ALIGNMENT FIX: preserve datetime index ──────────────────────
        # Slice the DataFrame's DatetimeIndex to match the trimmed arrays so
        # that dates[i] corresponds exactly to feat_vals[i] and target_vals[i].
        dates = df.index[:valid_len]
        if not isinstance(dates, pd.DatetimeIndex):
            # Fallback: convert whatever index we have to DatetimeIndex
            try:
                dates = pd.DatetimeIndex(dates)
            except Exception:
                dates = None
        # ─────────────────────────────────────────────────────────────────────

        asset_data_list.append((f, feat_vals, target_vals, ohlc_vals, dates))
        final_feature_cols = feature_cols

        # Diagnostics use exact-zero mask — consistent with the loss
        long  = (target_vals >  0).mean()
        short = (target_vals <  0).mean()
        zero  = (target_vals == 0.0).mean()
        print(f"  Target Distribution — Long: {long:.3f} | Short: {short:.3f} | Zero: {zero:.3f}")
        if dates is not None:
            print(f"  Date range: {dates[0].date()} → {dates[-1].date()}  ({len(dates):,} bars)")

    return asset_data_list, final_feature_cols


def _grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return math.sqrt(total)


def _weight_norms(model):
    return {name: p.detach().norm(2).item()
            for name, p in model.named_parameters() if p.requires_grad}


@torch.no_grad()
def _full_eval_diagnostics(net, loader, device, tag="VAL"):
    """
    Compute rich diagnostics over a full loader pass.

    Mask semantics (TARGET ALIGNMENT):
        is_zero  → target == 0.0  (exact no-trade, post-zeroing)
        is_edge  → target != 0.0  (real edge; |tgt| >= SAMPLER_THRESHOLD guaranteed)

    false_sig_rate uses config.FALSE_SIGNAL_MARGIN (the same dead-zone boundary
    used inside false_signal_loss) so the metric directly reflects what the loss
    penalises, avoiding the [FALSE_SIGNAL_MARGIN, SAMPLER_THRESHOLD] discrepancy
    from the previous is_zero = tgts.abs() == SAMPLER_THRESHOLD implementation.
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

    if is_distributed:
        preds = _gather_tensor(preds, device)
        tgts  = _gather_tensor(tgts,  device)

    # ── TARGET ALIGNMENT: use exact-zero mask ────────────────────────────────
    is_zero = (tgts == 0.0)   # no-trade bars (exact, after process_dataset zeroing)
    is_edge = ~is_zero         # real edge bars (|tgt| >= SAMPLER_THRESHOLD guaranteed)
    # ─────────────────────────────────────────────────────────────────────────

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

    if is_zero.any():
        _fs_margin = getattr(config, "FALSE_SIGNAL_MARGIN", config.SAMPLER_THRESHOLD)
        false_sig_rate = (preds[is_zero].abs() > _fs_margin).float().mean().item()
    else:
        false_sig_rate = float("nan")

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

    p_buckets = _get_buckets(preds)
    t_buckets = _get_buckets(tgts)
    total_preds = len(preds)

    thresh  = config.SAMPLER_THRESHOLD
    n_long  = (preds >  thresh).sum().item()
    n_short = (preds < -thresh).sum().item()
    n_flat  = total_preds - n_long - n_short

    return {
        "tag": tag, "n": total_preds,
        "p_mean": p_mean, "p_std": p_std,
        "t_mean": t_mean, "t_std": t_std,
        "p_std_edge": p_std_edge, "t_std_edge": t_std_edge,
        "dir_acc": dir_acc, "corr": corr,
        "false_sig_rate": false_sig_rate, "mag_over": mag_over,
        "p_buckets": p_buckets,
        "t_buckets": t_buckets,
        "n_long": n_long, "n_short": n_short, "n_flat": n_flat,
        "long_pct":  100 * n_long  / total_preds,
        "short_pct": 100 * n_short / total_preds,
        "flat_pct":  100 * n_flat  / total_preds,
    }


def _print_diagnostics(d):
    _fs_margin = getattr(config, "FALSE_SIGNAL_MARGIN", config.SAMPLER_THRESHOLD)
    pse = d.get("p_std_edge", float("nan"))
    tse = d.get("t_std_edge", float("nan"))
    edge_ratio = pse / (tse + 1e-9) if tse == tse else float("nan")
    print(f"\n  ┌── {d['tag']} DIAGNOSTICS (n={d['n']}) ─────────────────────────────")
    print(f"  │ Pred  : mean={d['p_mean']:+.4f}  std(all)={d['p_std']:.4f}  std(edge)={pse:.4f}")
    print(f"  │ Target: mean={d['t_mean']:+.4f}  std(all)={d['t_std']:.4f}  std(edge)={tse:.4f}  edge_ratio={edge_ratio:.3f}x")
    print(f"  │ Dir accuracy  (|tgt|>={config.SAMPLER_THRESHOLD}): {d['dir_acc']*100:.1f}%")
    print(f"  │ Correlation   (|tgt|>={config.SAMPLER_THRESHOLD}): {d['corr']:.4f}")
    print(f"  │ False sig rate(tgt==0.0, |pred|>{_fs_margin}): {d['false_sig_rate']*100:.1f}%")
    print(f"  │ Pred magnitude > tgt magnitude (edge): {d['mag_over']*100:.1f}%")
    pb = d["p_buckets"]
    tb = d["t_buckets"]
    print(f"  │ Pred distribution:")
    print(f"  │   <-0.5  : {pb['<-0.5']:5.1f}%   -0.5:-0.1: {pb['-0.5:-0.1']:5.1f}%   -0.1:0: {pb['-0.1:0']:5.1f}%")
    print(f"  │    0:0.1 : {pb['0:0.1']:5.1f}%    0.1:0.5 : {pb['0.1:0.5']:5.1f}%    >0.5: {pb['>0.5']:5.1f}%")
    print(f"  │ Target distribution:")
    print(f"  │   <-0.5  : {tb['<-0.5']:5.1f}%   -0.5:-0.1: {tb['-0.5:-0.1']:5.1f}%   -0.1:0: {tb['-0.1:0']:5.1f}%")
    print(f"  │    0:0.1 : {tb['0:0.1']:5.1f}%    0.1:0.5 : {tb['0.1:0.5']:5.1f}%    >0.5: {tb['>0.5']:5.1f}%")
    print(f"  │ Decisions (±{config.SAMPLER_THRESHOLD}): Long={d['long_pct']:.1f}%  Short={d['short_pct']:.1f}%  Flat={d['flat_pct']:.1f}%")
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
    return net


def _validate_checkpoint(path, feature_cols, device):
    """Returns True if checkpoint exists and is architecture-compatible."""
    if not os.path.exists(path):
        return False
    try:
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
        net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Inner epoch runner (shared between pretrain and finetune)
# ─────────────────────────────────────────────────────────────────────────────

def _run_epoch(
    net, loader, device, fold_id: int,
    optimizer=None, grad_scaler=None, scheduler=None,
    is_train=True, use_amp=True, grad_clip=None,
    epoch: int = 0,
):
    """
    Run one epoch.  Returns avg_loss and per-batch metric dict.
    """
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
                tokens = (tokens[0].to(device), tokens[1].to(device))
            if feats is not None:
                feats = feats.to(device)
            y = y.to(device)

            if is_train and optimizer is not None:
                optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pred = net(tokens=tokens, features=feats)
                batch_loss = continuous_weighted_direction_loss(
                    pred, y,
                    fold_id=fold_id,
                    epoch=epoch,
                )

            if is_train:
                if grad_scaler is not None:
                    grad_scaler.scale(batch_loss).backward()
                    if optimizer is not None:
                        grad_scaler.unscale_(optimizer)
                else:
                    batch_loss.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(
                    net.parameters(),
                    grad_clip if grad_clip is not None else config.GRAD_CLIP,
                )
                if not torch.isfinite(total_norm):
                    total_norm = torch.tensor(0.0)
                grad_norms.append(total_norm.item())

                if optimizer is not None:
                    if grad_scaler is not None:
                        scale_before = grad_scaler.get_scale()
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                        if grad_scaler.get_scale() >= scale_before and scheduler is not None:
                            scheduler.step()
                    else:
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()

            total_loss  += batch_loss.item()
            batch_count += 1

            with torch.no_grad():
                p_f = pred.view(-1).float()
                t_f = y.view(-1).float()
                is_e = (t_f != 0.0)
                pred_stds.append(
                    p_f[is_e].std(unbiased=False).item() if is_e.sum() > 1 else 0.0)
                if is_e.sum() > 1:
                    da = ((p_f[is_e] * t_f[is_e]) > 0).float().mean().item()
                    dir_accs.append(da)
                    pc = p_f[is_e] - p_f[is_e].mean()
                    tc = t_f[is_e] - t_f[is_e].mean()
                    c  = (pc * tc).mean() / (
                        p_f[is_e].std(unbiased=False).clamp(1e-6) *
                        t_f[is_e].std(unbiased=False).clamp(1e-6)
                    )
                    corrs.append(c.item())

    if is_distributed:
        metrics = torch.tensor([
            total_loss, batch_count,
            sum(grad_norms), len(grad_norms),
            sum(pred_stds),  len(pred_stds),
            sum(dir_accs),   len(dir_accs),
            sum(corrs),      len(corrs),
        ], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        gl, gc = metrics[0].item(), metrics[1].item()
        gn_s,  gn_c = metrics[2].item(), metrics[3].item()
        ps_s,  ps_c = metrics[4].item(), metrics[5].item()
        da_s,  da_c = metrics[6].item(), metrics[7].item()
        co_s,  co_c = metrics[8].item(), metrics[9].item()
        max_gn_t = torch.tensor(
            [max(grad_norms) if grad_norms else 0.0], device=device)
        dist.all_reduce(max_gn_t, op=dist.ReduceOp.MAX)
        return {
            "avg_loss": gl / max(gc, 1),
            "avg_gn":   gn_s / max(gn_c, 1),
            "max_gn":   max_gn_t[0].item(),
            "avg_ps":   ps_s / max(ps_c, 1),
            "avg_da":   da_s / max(da_c, 1),
            "avg_corr": co_s / max(co_c, 1),
        }

    return {
        "avg_loss": total_loss / max(batch_count, 1),
        "avg_gn":   float(np.mean(grad_norms)) if grad_norms else 0.0,
        "max_gn":   float(np.max(grad_norms))  if grad_norms else 0.0,
        "avg_ps":   float(np.mean(pred_stds))  if pred_stds  else 0.0,
        "avg_da":   float(np.mean(dir_accs))   if dir_accs   else 0.0,
        "avg_corr": float(np.mean(corrs))      if corrs      else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DATE-ALIGNED FOLD BUILDER  (replaces make_rolling_folds)
# ─────────────────────────────────────────────────────────────────────────────

def make_date_aligned_folds(
    global_start: pd.Timestamp,
    global_end:   pd.Timestamp,
    n_folds:      int,
    val_frac:     float = 0.15,
    min_train_days: int = None,
):
    """
    Build expanding-window folds anchored to CALENDAR DATES rather than
    integer bar indices.

    All assets share the same fold date boundaries.  Each asset converts these
    boundaries to its own bar indices via DatetimeIndex.searchsorted() inside
    finetune_fold(), so every asset contributes exactly the bars it has within
    the calendar window.  Assets with insufficient bars for a given fold are
    explicitly skipped with a warning.

    Returns list of:
        (fold_id, train_start_date, train_end_date, val_start_date, val_end_date)
    where all dates are pd.Timestamps.

    Leakage guard:
        val_start_date = train_end_date + gap_days (at least 1 trading day).
        The per-asset bar-level gap check (gap >= LOOKBACK_WINDOW bars) is
        enforced separately inside finetune_fold().
    """
    total_days    = (global_end - global_start).days
    fold_days     = total_days // n_folds
    val_days      = max(int(fold_days * val_frac), 30)   # at least ~1 month
    # gap in calendar days — 2× LOOKBACK_WINDOW bars is a safe upper bound
    # since intraday bars are denser than 1/day.  finetune_fold() enforces
    # the precise bar-level gap assertion.
    gap_days      = 1   # calendar-level placeholder; bar-level check in finetune_fold

    if min_train_days is None:
        # Reasonable floor: 6 months of calendar days
        min_train_days = 180

    folds = []
    for k in range(n_folds):
        val_end_date   = global_end   - pd.Timedelta(days=k * fold_days)
        val_start_date = val_end_date - pd.Timedelta(days=val_days)
        train_end_date = val_start_date - pd.Timedelta(days=gap_days)
        train_start_date = global_start

        train_days = (train_end_date - train_start_date).days
        if train_days < min_train_days:
            print(
                f"  Fold {n_folds - k} skipped — insufficient train days "
                f"({train_days} < {min_train_days})"
            )
            continue

        folds.append((
            n_folds - k,
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
    asset_data_list: list[tuple],
    feature_cols:    list[str],
    tok,
    pretrain_end_date: pd.Timestamp,
    device:          torch.device,
    epochs:          int   = 50,
    max_lr:          float = 5e-5,
):
    """
    Train the model on bars up to pretrain_end_date across all assets.

    DATE-ALIGNMENT: Uses calendar-date boundary instead of a raw integer
    bar index.  Each asset's bars are sliced via searchsorted() on its
    preserved DatetimeIndex so every asset contributes exactly the bars
    it has up to that date.
    """
    print(f"\n{'='*65}")
    print(f"  PRE-TRAIN  |  bars up to {pretrain_end_date.date()}  |  epochs={epochs}")
    print(f"  Device: {device}  |  LR_max={max_lr:.1e}  |  D_MODEL={config.D_MODEL}")
    print(f"{'='*65}\n")

    pretrain_list = []
    for asset_id, feat, targ, ohlc, dates in asset_data_list:
        if dates is None:
            # Fallback: use all bars (legacy behaviour if dates unavailable)
            end_idx = len(feat)
            print(f"  [{os.path.basename(str(asset_id))}] No date index — using all {end_idx} bars for pretrain.")
        else:
            end_idx = int(dates.searchsorted(pretrain_end_date, side="right"))

        if end_idx < config.LOOKBACK_WINDOW * 2:
            print(f"  [{os.path.basename(str(asset_id))}] Only {end_idx} bars up to pretrain_end — skipping.")
            continue

        f_slice = feat[:end_idx]
        t_slice = targ[:end_idx]
        o_slice = ohlc[:end_idx] if ohlc is not None else None
        pretrain_list.append((asset_id, f_slice, t_slice, o_slice, len(f_slice)))
        print(f"  [{os.path.basename(str(asset_id))}] Pretrain bars: {end_idx:,}")

    loader, fitted_scalers = create_multi_index_dataloaders(
        pretrain_list, config, feature_cols, tok, is_train=True
    )
    loader = _make_distributed_loader(loader, is_train=True)

    if loader is None:
        raise RuntimeError("No pre-training data available after slicing.")

    print(f"  Pre-train batches/epoch: {len(loader):,}")

    net = _build_model(feature_cols, device)
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"  Model params: {total_params:,}\n")

    net = wrap_ddp_and_compile(net, device)

    optimizer   = torch.optim.AdamW(
        net.parameters(), lr=max_lr / 10, weight_decay=config.WEIGHT_DECAY)
    total_steps = epochs * len(loader)
    scheduler   = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=total_steps,
        pct_start=0.15, div_factor=10, final_div_factor=50)
    scaler_amp  = torch.amp.GradScaler(
        enabled=config.USE_AMP and device.type == "cuda", growth_interval=200)

    best_loss   = float("inf")
    patience    = 10
    pat_counter = 0

    for epoch in range(epochs):
        stats = _run_epoch(
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
            save_model(net, PRETRAIN_CKPT)
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
    del net
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 ── Fine-tune a single fold
# ─────────────────────────────────────────────────────────────────────────────

def finetune_fold(
    fold_id:            int,
    asset_data_list:    list[tuple],
    feature_cols:       list[str],
    tok,
    train_start_date:   pd.Timestamp,
    train_end_date:     pd.Timestamp,
    val_start_date:     pd.Timestamp,
    val_end_date:       pd.Timestamp,
    device:             torch.device,
    epochs:             int   = None,
    freeze_epochs:      int   = 5,
    head_lr:            float = 2e-6,
    full_lr:            float = 5e-7,
    patience:           int   = None,
    load_path:          str   = None,
):
    """
    Fine-tune one walk-forward fold across all assets.

    DATE-ALIGNMENT:
        Fold boundaries are now pd.Timestamps (not integer bar indices).
        Each asset converts these to its own integer indices via
        DatetimeIndex.searchsorted().  Assets with insufficient bars
        for a given fold are explicitly skipped with a warning instead
        of silently contributing empty slices.

    Stage A (epochs 0 … freeze_epochs-1): encoder frozen, head-only update.
    Stage B (epochs freeze_epochs … end): full network at very low LR.
    """
    _set_seed(getattr(config, "SEED", 42) + fold_id)
    epochs   = epochs   if epochs   is not None else config.EPOCHS
    patience = patience if patience is not None else config.WFV_PATIENCE
    epochs   = 50
    patience = 15

    print(f"\n{'='*65}")
    print(f"  FINE-TUNE  Fold {fold_id}  |  Device: {device}")
    print(f"  Train: [{train_start_date.date()} → {train_end_date.date()}]")
    print(f"  Val:   [{val_start_date.date()} → {val_end_date.date()}]")
    print(f"  Freeze epochs: {freeze_epochs}  |  Head LR: {head_lr:.1e}  |  Full LR: {full_lr:.1e}")
    print(f"{'='*65}\n")

    # ── Per-asset date → index conversion ────────────────────────────────────
    # DATE-ALIGNMENT FIX: each asset sliced to its own calendar-matching bars.
    # Assets without a DatetimeIndex fall back to all-bars (legacy behaviour).
    def _date_to_idx(dates, dt, side="right"):
        """Convert a Timestamp to an integer bar index via searchsorted."""
        if dates is None:
            return None
        return int(dates.searchsorted(dt, side=side))

    # ── Tokenise (optional) ──────────────────────────────────────────────────
    _use_tokens  = (getattr(config, "INPUT_MODE", "features_only") != "features_only")
    _strict_tok  = getattr(config, "TOKENIZE_STRICT_TRAIN_ONLY", False)
    _asset_tokens: dict[str, tuple] = {}

    if _use_tokens and tok is not None:
        for asset_id, feat, targ, ohlc, dates in asset_data_list:
            if ohlc is None:
                continue
            te_idx = _date_to_idx(dates, train_end_date)
            vs_idx = _date_to_idx(dates, val_start_date)
            ve_idx = _date_to_idx(dates, val_end_date)
            if te_idx is None:
                te_idx = len(feat) - config.ORACLE_MAX_HOLD

            train_end_safe = te_idx - (config.ORACLE_MAX_HOLD - 1)
            mode = "train+val slices" if _strict_tok else "full series"
            print(f"  [Fold {fold_id}] Tokenizing ({mode}) for '{asset_id}'…")
            if _strict_tok:
                _asset_tokens[asset_id] = tokenize_split_slices(
                    ohlc, tok, config,
                    [(0, train_end_safe), (vs_idx, ve_idx)],
                )
            else:
                full = tokenize_full_series(ohlc, tok, config)
                _asset_tokens[asset_id] = ("full", full)

    # ── Build train / val lists with per-asset date-indexed slicing ──────────
    train_list, val_list = [], []
    skipped_assets = []

    for asset_id, feat, targ, ohlc, dates in asset_data_list:
        asset_name = os.path.basename(str(asset_id))

        if dates is not None:
            ts_idx = _date_to_idx(dates, train_start_date, side="left")
            te_idx = _date_to_idx(dates, train_end_date,   side="right")
            vs_idx = _date_to_idx(dates, val_start_date,   side="left")
            ve_idx = _date_to_idx(dates, val_end_date,     side="right")
        else:
            # Legacy fallback: treat the whole array as one contiguous block
            # (behaviour identical to the old integer-index code path)
            ts_idx = 0
            te_idx = len(feat)
            vs_idx = 0
            ve_idx = len(feat)
            print(f"  [{asset_name}] No date index — using legacy integer slicing.")

        train_end_safe = te_idx - (config.ORACLE_MAX_HOLD - 1)
        train_bars = max(0, train_end_safe - ts_idx)
        val_bars   = max(0, ve_idx - vs_idx)

        # ── DATE-ALIGNMENT: explicit skip with warning (not silent empty slice) ──
        min_train_bars = config.LOOKBACK_WINDOW * 2
        min_val_bars   = config.LOOKBACK_WINDOW

        if train_bars < min_train_bars:
            print(
                f"  ⚠  [{asset_name}] Fold {fold_id}: "
                f"only {train_bars} train bars (need {min_train_bars}) — SKIPPED"
            )
            skipped_assets.append(asset_name)
            continue

        if val_bars < min_val_bars:
            print(
                f"  ⚠  [{asset_name}] Fold {fold_id}: "
                f"only {val_bars} val bars (need {min_val_bars}) — SKIPPED"
            )
            skipped_assets.append(asset_name)
            continue

        # ── Bar-level leakage assertion (calendar gap already enforced by fold builder) ──
        gap_bars = vs_idx - train_end_safe
        assert gap_bars >= config.LOOKBACK_WINDOW, (
            f"[{asset_name}] Fold {fold_id}: bar-level gap={gap_bars} < "
            f"LOOKBACK_WINDOW={config.LOOKBACK_WINDOW}. "
            "Val window overlaps with train — data leakage!"
        )

        f_tr = feat[ts_idx : train_end_safe]
        t_tr = targ[ts_idx : train_end_safe]
        f_va = feat[vs_idx : ve_idx]
        t_va = targ[vs_idx : ve_idx]

        tok_entry = _asset_tokens.get(asset_id)
        if tok_entry is None:
            c_tr = f_c_tr = c_va = f_c_va = None
        elif tok_entry[0] == "full":
            _c_full, _f_full = tok_entry[1]
            c_tr,  f_c_tr  = _c_full[ts_idx:train_end_safe], _f_full[ts_idx:train_end_safe]
            c_va,  f_c_va  = _c_full[vs_idx:ve_idx], _f_full[vs_idx:ve_idx]
        else:
            (c_tr, f_c_tr), (c_va, f_c_va) = tok_entry

        train_list.append((asset_id, f_tr, t_tr, None, len(f_tr), c_tr, f_c_tr))
        val_list.append(  (asset_id, f_va, t_va, None, None,      c_va, f_c_va))

        if dates is not None:
            tr_start_str = str(dates[ts_idx].date()) if ts_idx < len(dates) else "?"
            tr_end_str   = str(dates[min(train_end_safe, len(dates)-1)].date())
            va_start_str = str(dates[vs_idx].date()) if vs_idx < len(dates) else "?"
            va_end_str   = str(dates[min(ve_idx-1, len(dates)-1)].date())
            print(
                f"  [{asset_name}] "
                f"Train [{tr_start_str} → {tr_end_str}] {train_bars:,} bars  |  "
                f"Val   [{va_start_str} → {va_end_str}] {val_bars:,} bars"
            )

    if skipped_assets:
        print(f"\n  ⚠  Fold {fold_id}: {len(skipped_assets)} asset(s) skipped: {skipped_assets}")

    if not train_list:
        print(f"  ⛔ Fold {fold_id}: No assets with sufficient data — skipping fold.")
        return float("nan")

    # ── Zero-rate drift diagnostic ───────────────────────────────────────────
    all_train_targets = np.concatenate(
        [t_tr for _, _, t_tr, _, _, _, _ in train_list])
    all_val_targets   = np.concatenate(
        [t_va for _, _, t_va, _, _, _, _ in val_list])
    tr_zero = (all_train_targets == 0.0).mean()
    va_zero = (all_val_targets   == 0.0).mean()
    _drift  = va_zero - tr_zero
    _flag   = "⚠  HIGH DRIFT" if abs(_drift) > 0.10 else "OK"
    print(
        f"  Fold {fold_id} zero-rate drift: "
        f"Train={tr_zero*100:.1f}%  Val={va_zero*100:.1f}%  "
        f"Δ={_drift*100:+.1f}pp  {_flag}"
    )

    # ── Dataloaders ──────────────────────────────────────────────────────────
    train_loader, fitted_scalers = create_multi_index_dataloaders(
        train_list, config, feature_cols, tok, is_train=True
    )
    train_loader      = _make_distributed_loader(train_loader, is_train=True)
    train_diag_loader = _make_loader(train_loader.dataset, config, drop_last=False)
    train_diag_loader = _make_distributed_loader(train_diag_loader, is_train=False)
    val_loader, _     = create_multi_index_dataloaders(
        val_list, config, feature_cols, tok, is_train=False, scalers=fitted_scalers
    )
    val_loader = _make_distributed_loader(val_loader, is_train=False)

    assert train_loader is not None, "train_loader should not be None"
    assert val_loader   is not None, "val_loader should not be None"
    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # ── Load weights ─────────────────────────────────────────────────────────
    net = _build_model(feature_cols, device)
    path_to_load = load_path if load_path else PRETRAIN_CKPT

    if os.path.exists(path_to_load):
        ckpt       = torch.load(path_to_load, map_location=device, weights_only=True)
        ckpt_keys  = set(ckpt.keys())
        model_keys = set(net.state_dict().keys())
        missing    = model_keys - ckpt_keys
        unexpected = ckpt_keys  - model_keys
        if missing or unexpected:
            print(f"  ⚠️  Key mismatch in {path_to_load}:")
            if missing:    print(f"     Missing in ckpt : {missing}")
            if unexpected: print(f"     Extra in ckpt   : {unexpected}")
            print(f"  → Starting Fold {fold_id} from scratch instead.")
        else:
            try:
                net.load_state_dict(ckpt, strict=True)
                print(f"  ✓ Loaded weights from {path_to_load}")
            except RuntimeError as e:
                print(f"  ⚠️  Load failed despite matching keys: {e}")
    else:
        print(f"  ⚠️  Checkpoint not found at {path_to_load}. Starting from scratch.")

    net = wrap_ddp_and_compile(net, device)

    # ── Head vs. encoder split ────────────────────────────────────────────────
    def _is_head(name: str) -> bool:
        name = name.replace("_orig_mod.", "").replace("module.", "")
        return any(name.startswith(p)
                   for p in ("head", "fc", "proj", "output", "feature_head"))

    head_params    = [p for n, p in net.named_parameters() if     _is_head(n)]
    encoder_params = [p for n, p in net.named_parameters() if not _is_head(n)]
    head_names     = [n for n, _ in net.named_parameters() if     _is_head(n)]
    enc_names      = [n for n, _ in net.named_parameters() if not _is_head(n)]
    print(f"  Head params    ({len(head_names)}): {head_names[:3]}{'...' if len(head_names) > 3 else ''}")
    print(f"  Encoder params ({len(enc_names)}): {enc_names[:3]}{'...' if len(enc_names)  > 3 else ''}\n")

    def _freeze_encoder():
        for p in encoder_params: p.requires_grad = False
        for p in head_params:    p.requires_grad = True

    def _unfreeze_all():
        for p in net.parameters(): p.requires_grad = True

    _freeze_encoder()

    no_decay_names = {"bias", "norm1.weight", "norm2.weight", "norm.weight"}
    head_decay     = [p for n, p in net.named_parameters() if     _is_head(n) and not any(nd in n for nd in no_decay_names)]
    head_no_decay  = [p for n, p in net.named_parameters() if     _is_head(n) and     any(nd in n for nd in no_decay_names)]
    enc_decay      = [p for n, p in net.named_parameters() if not _is_head(n) and not any(nd in n for nd in no_decay_names)]
    enc_no_decay   = [p for n, p in net.named_parameters() if not _is_head(n) and     any(nd in n for nd in no_decay_names)]

    optimizer = torch.optim.AdamW([
        {"params": head_decay,    "lr": head_lr, "weight_decay": config.WEIGHT_DECAY},
        {"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0},
        {"params": enc_decay,     "lr": 0.0,     "weight_decay": config.WEIGHT_DECAY},
        {"params": enc_no_decay,  "lr": 0.0,     "weight_decay": 0.0},
    ])

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[head_lr, head_lr, 0.0, 0.0],
        total_steps=freeze_epochs * len(train_loader),
        pct_start=0.2, div_factor=5, final_div_factor=10,
    )
    scaler_amp  = torch.amp.GradScaler(
        enabled=config.USE_AMP and device.type == "cuda", growth_interval=200)

    best_val    = float("inf")
    best_epoch  = -1
    pat_counter = 0

    for epoch in range(epochs):
        # ── Stage A → B transition ────────────────────────────────────────────
        if epoch == freeze_epochs:
            print(f"\n  → Unfreezing encoder at epoch {epoch+1}. LR → {full_lr:.1e}")
            _unfreeze_all()
            optimizer.param_groups[0]["lr"] = head_lr / 5
            optimizer.param_groups[1]["lr"] = head_lr / 5
            optimizer.param_groups[2]["lr"] = full_lr / 10
            optimizer.param_groups[3]["lr"] = full_lr / 10
            for p in encoder_params:
                state = optimizer.state[p]
                if "step" not in state:
                    state["step"] = torch.tensor(0.0)
                state["exp_avg"]    = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.full_like(p, 1e-4,  memory_format=torch.preserve_format)
            remaining_steps = (epochs - freeze_epochs) * len(train_loader)
            pct_start       = 0.10 if remaining_steps > 400 else 0.05
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[head_lr / 10, head_lr / 10, full_lr / 5, full_lr / 5],
                total_steps=max(remaining_steps, 1),
                pct_start=pct_start,
                div_factor=5,
                final_div_factor=100,
            )

        stage    = "A-frozen" if epoch < freeze_epochs else "B-full"
        clip_val = (
            getattr(config, "FINETUNE_GRAD_CLIP_STAGE_B", config.GRAD_CLIP)
            if epoch >= freeze_epochs else
            getattr(config, "FINETUNE_GRAD_CLIP_STAGE_A", config.GRAD_CLIP)
        )

        tr = _run_epoch(
            net, train_loader, device, fold_id=fold_id,
            optimizer=optimizer, grad_scaler=scaler_amp,
            scheduler=scheduler, is_train=True, use_amp=config.USE_AMP,
            grad_clip=clip_val, epoch=epoch,
        )

        va = _run_epoch(
            net, val_loader, device, fold_id=fold_id,
            is_train=False, use_amp=config.USE_AMP,
            epoch=config.CURRICULUM_RAMP_EPOCHS,
        )
        lr_now = scheduler.get_last_lr()[0]

        saved = ""
        if va["avg_loss"] < best_val:
            best_val    = va["avg_loss"]
            best_epoch  = epoch + 1
            pat_counter = 0
            save_model(net, MODEL_PATH)
            saved = "  ✓ SAVED"
        else:
            pat_counter += 1

        print(
            f"  [Fold {fold_id} | {stage}] Ep{epoch+1:3d} | "
            f"TrainL={tr['avg_loss']:.4f} | ValL={va['avg_loss']:.4f} | "
            f"LR={lr_now:.2e} | "
            f"GN avg={tr['avg_gn']:.3f} max={tr['max_gn']:.3f} | "
            f"DirAcc={tr['avg_da']*100:.1f}% | "
            f"Corr={tr['avg_corr']:.3f} | "
            f"Pat={pat_counter}/{patience}"
            f"{saved}"
        )

        if pat_counter >= patience:
            print(f"  ⛔ Fold {fold_id} early stop at epoch {epoch+1}. "
                  f"Best val={best_val:.4f} @ ep{best_epoch}.")
            break

    # ── End-of-fold diagnostics ───────────────────────────────────────────────
    if os.path.exists(MODEL_PATH):
        ckpt_clean = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        net_eval   = _build_model(feature_cols, device)
        net_eval.load_state_dict(ckpt_clean, strict=True)
        net_eval.eval()
        val_diag   = _full_eval_diagnostics(net_eval, val_loader, device, tag=f"FOLD{fold_id}_VAL")
        train_diag = _full_eval_diagnostics(net_eval, train_diag_loader, device, tag=f"FOLD{fold_id}_TRAIN")
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
        file_paths = getattr(config, "DATA_FILES", None) or getattr(config, "DATA_FILE", None)
    if not file_paths:
        raise RuntimeError("Neither config.DATA_FILES nor config.DATA_FILE is defined. Set one in config.py.")

    fe = FeatureEngineer(_make_feature_config())
    tok = None
    if getattr(config, "INPUT_MODE", "features_only") != "features_only":
        tok_path = getattr(config, "TOKENIZER_PATH", "model.safetensors")
        if os.path.exists(tok_path):
            tok = KronosTokenizer.from_pretrained(tok_path, device=str(device))
            tok.eval()
            for p in tok.parameters():
                p.requires_grad = False
            print(f"  [Tokenizer] Loaded from checkpoint: {tok_path}")
        else:
            raise FileNotFoundError(
                f"Tokenizer checkpoint not found at '{tok_path}'. "
                f"Set config.TOKENIZER_PATH to the correct path."
            )

    print(f"\n[train] Processing {len(file_paths)} asset file(s)…")
    asset_data_list, feature_cols = process_dataset(file_paths, fe)
    if not asset_data_list:
        raise RuntimeError("No valid asset data found after processing.")

    # ── Date-alignment: report per-asset date ranges ─────────────────────────
    print(f"\n[train] Asset date ranges:")
    all_dates = []
    for asset_id, feat, targ, ohlc, dates in asset_data_list:
        name = os.path.basename(str(asset_id))
        if dates is not None:
            print(f"  {len(dates):>7,} bars  [{dates[0].date()} → {dates[-1].date()}]  ← {name}")
            all_dates.extend([dates[0], dates[-1]])
        else:
            print(f"  {len(feat):>7,} bars  [no date index]  ← {name}")

    if not all_dates:
        raise RuntimeError(
            "No DatetimeIndex found in any asset. "
            "Ensure your CSV files have a 'date' or 'datetime' column."
        )

    global_start = min(all_dates)
    global_end   = max(all_dates)
    print(f"\n[train] Global date range: {global_start.date()} → {global_end.date()}")
    print(f"[train] Fold boundaries based on CALENDAR DATES (all assets share the same window).\n")

    # ── Build date-aligned folds ──────────────────────────────────────────────
    n_folds  = getattr(config, "N_FOLDS", 5)
    val_frac = getattr(config, "VAL_FRAC", 0.15)
    folds    = make_date_aligned_folds(
        global_start=global_start,
        global_end=global_end,
        n_folds=n_folds,
        val_frac=val_frac,
    )

    if not folds:
        raise RuntimeError("make_date_aligned_folds() returned no valid folds. "
                           "Check your date range and N_FOLDS / VAL_FRAC config.")

    # Pretrain up to the train_end_date of the FIRST (earliest) fold
    pretrain_end_date = folds[0][2]   # train_end_date of fold 1

    # ── Pretrain ──────────────────────────────────────────────────────────────
    ckpt_valid = _validate_checkpoint(PRETRAIN_CKPT, feature_cols, device)
    if ckpt_valid:
        print(f"\n[train] Found valid pretrain checkpoint at {PRETRAIN_CKPT} — skipping pretrain.")
    else:
        pretrain(
            asset_data_list=asset_data_list,
            feature_cols=feature_cols,
            tok=tok,
            pretrain_end_date=pretrain_end_date,
            device=device,
            epochs=getattr(config, "PRETRAIN_EPOCHS", 50),
            max_lr=getattr(config, "PRETRAIN_LR", 5e-5),
        )

    # ── Walk-forward fine-tuning ──────────────────────────────────────────────
    fold_scores = []
    for fold_id, train_start_date, train_end_date, val_start_date, val_end_date in folds:
        val_score = finetune_fold(
            fold_id=fold_id,
            asset_data_list=asset_data_list,
            feature_cols=feature_cols,
            tok=tok,
            train_start_date=train_start_date,
            train_end_date=train_end_date,
            val_start_date=val_start_date,
            val_end_date=val_end_date,
            device=device,
            freeze_epochs=getattr(config, "FINETUNE_FREEZE_EPOCHS", 5),
            head_lr=getattr(config, "FINETUNE_HEAD_LR", 2e-6),
            full_lr=getattr(config, "FINETUNE_FULL_LR", 5e-7),
        )
        fold_scores.append((fold_id, val_score))

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
    parser.add_argument("--files", nargs="*", default=None,
                        help="Override config.DATA_FILES with explicit CSV paths")
    args = parser.parse_args()
    train(file_paths=args.files)
