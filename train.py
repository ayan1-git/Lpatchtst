# train.py  (Production — features.py fully integrated)
# %load_ext cudf.pandas
from __future__ import annotations

import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import importlib
import model
import features
import loss
import oracle # Added this import statement
import data_loader # Added this import statement
import config
importlib.reload(features)
importlib.reload(config)
importlib.reload(data_loader)

from oracle import generate_targets
from data_loader import create_multi_index_dataloaders
from model import PatchTST, LPatchTST
from loss import continuous_weighted_direction_loss

# ── features.py public API ────────────────────────────────────────────────────
from features import FeatureConfig, FeatureEngineer

import config

MODEL_PATH = "best_model_lpatchtst.pth" if config.USE_LPATCHTST else "best_model_patchtst.pth"
WARMUP_EPOCHS = 3  # skip checkpointing during OneCycleLR warmup ramp

OHLC_COLS = ["open", "high", "low", "close"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_aggregation_mode() -> str:
    return config.AGGREGATION_MODE


def _make_feature_config() -> FeatureConfig:
    """Map config.py hyperparameters → FeatureConfig."""
    return FeatureConfig(
        ewma_span=config.FE_VOL_LONG_PERIOD,
        return_horizons=config.FE_RETURN_HORIZONS,
        macd_pairs=config.FE_MACD_PAIRS,
        macd_price_std_window=config.FE_MACD_PRICE_STD_WIN,
        macd_signal_std_window=config.FE_MACD_SIGNAL_STD_WIN,
        target_clip=config.FE_TARGET_CLIP,
        # ── new OHLC fields ──
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
    )


def _build_feature_cols(
    fe_config: FeatureConfig,
) -> Tuple[list[str], list[str], list[str]]:
    """Derive the scaler-routing column lists from FeatureConfig.

    Column routing rationale
    ────────────────────────
    NO_SCALE  — ewma_vol   : small fraction ~0.003, tight band [0.002–0.004].
                             Centering to zero destroys its absolute meaning
                             (σ=0 is the natural reference; do not shift it).
              — ret_norm_* : vol-scaled returns, p1/p99 ≈ [-2.5, +2.5].
                             Already a z-score by construction.
              — macd_*_*   : 3-step normalised, empirical std ≈ 1.05.
                             Already unit-variance by construction.

    ROBUST    — vs_factor  : 1/σ.  Mean ~346, skew ~24, max can spike to
                             3000+ in low-vol regimes.  Will dominate gradients
                             if passed raw.  RobustScaler (median+IQR) centres
                             it without being destroyed by the right tail.

    Returns
    ───────
    no_scale_cols : identity scaler  (12 columns)
    robust_cols   : RobustScaler     (1 column)
    all_feat_cols : robust + no_scale, in model-input order
    """
    no_scale_cols: list[str] = []
    robust_cols:   list[str] = []

    # ewma_vol → NO_SCALE (already a tight small fraction, do not centre)
    no_scale_cols.append(f"ewma_vol_span{fe_config.ewma_span}")

    # multi-horizon normalised returns → NO_SCALE (~z-score)
    for h in fe_config.return_horizons:
        no_scale_cols.append(f"ret_norm_{h}d")

    # multi-scale MACD signals → NO_SCALE (~unit variance)
    for s, l in fe_config.macd_pairs:
        no_scale_cols.append(f"macd_{s}_{l}")

    # vs_factor → ROBUST (mean ~346, skew ~24, needs centering)
    robust_cols.append(f"vs_factor_span{fe_config.ewma_span}")

    # ── new OHLC features → NO_SCALE (all bounded [-1,+1]) ──
    no_scale_cols += [
        "feat_efficiency",
        "feat_icp",
        "feat_momentum_rsi",
        "feat_vol_asymmetry",
        "feat_local_structure",
    ]
    # session features only if configured
    if fe_config.add_session_features:
        no_scale_cols += ["feat_session_sin", "feat_session_cos"]

    # vol squeeze → ROBUST (right-skewed, unbounded above)
    robust_cols.append("feat_vol_squeeze")

    all_feat_cols = robust_cols + no_scale_cols
    return no_scale_cols, robust_cols, all_feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering per file
# ─────────────────────────────────────────────────────────────────────────────

def _build_features(
    df: pd.DataFrame,
    fe: FeatureEngineer,
) -> Tuple[pd.DataFrame, list[str]]:
    """Apply FeatureEngineer to a raw OHLC DataFrame.

    Steps
    ─────
    1. Ensure a DatetimeIndex (required for FeatureEngineer.build()).
    2. Compute all features from close price via FeatureEngineer.build().
    3. Join back to original OHLC so Oracle/backtest can access price data.
    4. Compute ATR (needed by Oracle and evaluate.py).
    5. Return combined DataFrame and the ordered model-input column list.

    Parameters
    ──────────
    df : raw OHLC DataFrame (must contain open, high, low, close columns).
    fe : FeatureEngineer with config already set.

    Returns
    ───────
    combined_df  : OHLC + engineered features + ATR, NaN rows dropped.
    feature_cols : ordered list of engineered columns for the DataLoader.
                   OHLC is kept in combined_df for Oracle but NOT in this list.
    """
    # 1. DatetimeIndex ─────────────────────────────────────────────────────────
    time_col = next(
        (c for c in df.columns if c.lower() in ("date", "datetime")), None
    )
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
    else:
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    df = df.sort_index()

    # 2. Feature engineering on close ─────────────────────────────────────────
    # include_target=False : we use Oracle targets, not next-day return.
    # dropna=False         : drop NaN globally after ATR is added (step 4).
    feat_df = fe.build(
        df["close"],
        ohlc=df[OHLC_COLS],          # enables features 6–13
        include_target=False,
        dropna=False,
    )

    # 3. Join OHLC + features ──────────────────────────────────────────────────
    combined_df = df.join(feat_df, how="inner")

    # 4. ATR (used by Oracle and backtest engine) ──────────────────────────────
    high_low   = combined_df["high"] - combined_df["low"]
    high_close = (combined_df["high"] - combined_df["close"].shift()).abs()
    low_close  = (combined_df["low"]  - combined_df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    combined_df["atr"] = true_range.rolling(config.ATR_PERIOD).mean()
    combined_df.dropna(inplace=True)

    # 5. Column list for the model (engineered only, no OHLC) ─────────────────
    _, _, all_feat_cols = _build_feature_cols(fe.config)
    return combined_df, all_feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# Multi-file dataset preparation
# ─────────────────────────────────────────────────────────────────────────────

def process_dataset(
    file_paths: list[str],
    fe: FeatureEngineer,
) -> Tuple[list[tuple[str, np.ndarray, np.ndarray]], list[str]]:
    """Process every CSV through the feature pipeline and Oracle.

    Returns
    ───────
    asset_data_list : list of (asset_id, features_array, targets_array) per valid file.
    feature_cols    : shared ordered column list (identical for all files).
    """
    asset_data_list:    list[tuple[str, np.ndarray, np.ndarray]] = []
    final_feature_cols: list[str] = []

    for f in file_paths:
        if not os.path.exists(f):
            print(f"Warning: {f} not found. Skipping.")
            continue

        print(f"Processing {f}…")
        df_raw = pd.read_csv(f)
        df, feature_cols = _build_features(df_raw, fe)

        targets = generate_targets(
            df["open"].values,
            df["high"].values,
            df["low"].values,
            df["close"].values,
            df["atr"].values,
            max_hold=config.ORACLE_MAX_HOLD,
            fee_per_side=config.FEE_PER_SIDE,
            slippage=config.SLIPPAGE,
            atr_mult=config.ATR_MULT,
            saturation_factor=config.SATURATION_FACTOR,
            mae_penalty=config.MAE_PENALTY,
        )

        valid_len = len(targets) - config.ORACLE_MAX_HOLD
        if valid_len <= 0:
            print(f"Warning: {f} too short after Oracle trim. Skipping.")
            continue

        feat_vals   = df[feature_cols].values.astype(np.float32)[:valid_len]
        target_vals = targets[:valid_len]

        asset_data_list.append((f, feat_vals, target_vals))
        final_feature_cols = feature_cols   # same for every file

    if not asset_data_list:
        raise ValueError("No valid data processed from the provided files.")

    return asset_data_list, final_feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_fold(
    fold_id: str,
    train_loader,
    val_loader,
    feature_cols: list[str],
    tokenizer=None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training Fold: {fold_id} ---")
    print(f"Device: {device} | Input features: {len(feature_cols)}")

    aggregation  = _get_aggregation_mode()
    num_features = 1 if config.USE_TOKENIZER else len(feature_cols)

    if config.USE_LPATCHTST:
        model = LPatchTST(
            seq_len=config.LOOKBACK_WINDOW,
            num_features=num_features,
            patch_len=config.PATCH_LEN,
            stride=config.STRIDE,
            d_model=config.D_MODEL,
            n_heads=config.N_HEADS,
            n_layers=config.N_LAYERS,
            lstm_layers=config.LSTM_LAYERS,
            dropout=config.DROPOUT,
        ).to(device)
        print(f"Model: LPatchTST | lstm_layers={config.LSTM_LAYERS}")
    else:
        model = PatchTST(
            seq_len=config.LOOKBACK_WINDOW,
            num_features=num_features,
            patch_len=config.PATCH_LEN,
            stride=config.STRIDE,
            d_model=config.D_MODEL,
            n_heads=config.N_HEADS,
            n_layers=config.N_LAYERS,
            dropout=config.DROPOUT,
            aggregation=aggregation,
            use_tokenizer=config.USE_TOKENIZER,
            vocab_size=config.VOCAB_SIZE,
        ).to(device)
        print(f"Model: PatchTST | aggregation={aggregation}")

    if device.type == "cuda":
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile.")
        except Exception as e:
            print(f"torch.compile skipped: {e}")
    else:
        print("Skipping torch.compile on CPU.")

    # AdamW with selective weight decay (biases and 1-D params excluded)
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or param.ndim == 1:
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay,    "weight_decay": config.WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.LEARNING_RATE,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        total_steps=config.EPOCHS * len(train_loader),
        pct_start=0.1,
        div_factor=10,
        final_div_factor=1e3,
    )

    # ── AMP (Automatic Mixed Precision) setup ────────────────────────────────
    #
    # CUDA (T4 / A100 / etc.):
    #   dtype  = float16  — 2× faster matmuls on Tensor Cores.
    #   scaler = GradScaler ENABLED.
    #       FP16 can only represent values ≥ ~6e-5.  During backprop through
    #       LSTM gates and attention softmax, gradients often fall below this
    #       threshold and "underflow" to zero, killing learning.
    #       GradScaler multiplies the loss by a large factor (e.g. 65536)
    #       before .backward(), pushing gradients into FP16's representable
    #       range.  After .backward(), scaler.unscale_() divides them back
    #       down so optimizer.step() sees correct magnitudes.
    #       If any gradient is Inf/NaN (overflow), scaler.step() SKIPS the
    #       optimizer update entirely and halves the scale factor — this is
    #       the automatic recovery mechanism.
    #
    # CPU:
    #   dtype  = bfloat16 — same exponent range as float32 (no underflow risk)
    #                        but reduced mantissa precision.
    #   scaler = DISABLED — bfloat16 doesn't need gradient scaling.
    #
    # ⚠️  LSTM + FP16 sensitivity:
    #   - LSTM hidden states are initialised in float32; autocast handles
    #     internal casting for matrix multiplications.
    #   - Gradient clipping (GRAD_CLIP) is applied AFTER scaler.unscale_()
    #     so the clipping threshold operates on true gradient magnitudes.
    #   - NaN loss watchdog: if loss becomes NaN, training halts immediately
    #     to prevent silent checkpoint corruption.
    # ───────────────────────────────────────────────────────────────────────────
    use_amp    = config.USE_AMP
    is_cuda    = device.type == "cuda"
    amp_dtype  = torch.float16 if is_cuda else torch.bfloat16
    scaler     = torch.amp.GradScaler(enabled=(use_amp and is_cuda))

    if use_amp:
        print(f"AMP enabled — device={device.type}, dtype={amp_dtype}.")
        if is_cuda:
            print(f"  GradScaler active | initial scale={scaler.get_scale():.0f}")
            print(f"  Grad clipping at {config.GRAD_CLIP} (applied after unscale_)")
        else:
            print("  GradScaler disabled (bfloat16 on CPU — no underflow risk).")
    else:
        print("AMP disabled — training in float32.")

    best_val          = float("inf")
    nan_count         = 0   # NaN loss watchdog

    for epoch in range(config.EPOCHS):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss      = 0.0
        grad_norm_accum = 0.0
        scaler_skips    = 0   # count optimizer steps skipped due to Inf/NaN grads

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred = model(x)
                loss = continuous_weighted_direction_loss(pred, y)

            # ── NaN loss watchdog (critical for LSTM + FP16) ──────────────────
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                if nan_count >= 5:
                    print(f"\n⚠️  FATAL: {nan_count} NaN/Inf losses detected — aborting.")
                    print("  Likely cause: FP16 overflow in LSTM gates or attention.")
                    print("  Try: reduce LR, increase GRAD_CLIP, or disable AMP.")
                    return
                print(f"  ⚠️  NaN/Inf loss at step (count={nan_count}) — skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            total_norm = 0.0
            if use_amp and is_cuda:
                # CUDA FP16 path: scale → backward → unscale → clip → step
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)       # ← MUST precede clip_grad_norm_
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                prev_scale = scaler.get_scale()
                scaler.step(optimizer)           # skips if grads contain Inf/NaN
                scaler.update()
                if scaler.get_scale() < prev_scale:
                    scaler_skips += 1            # scale was reduced → step was skipped
            else:
                # CPU bfloat16 or float32 path: no scaler needed
                loss.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                optimizer.step()

            scheduler.step()
            train_loss      += float(loss.item())
            grad_norm_accum += float(total_norm)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    pred = model(x)
                    loss = continuous_weighted_direction_loss(pred, y)
                val_loss += float(loss.item())

        avg_train     = train_loss      / max(1, len(train_loader))
        avg_val       = val_loss        / max(1, len(val_loader))
        avg_grad_norm = grad_norm_accum / max(1, len(train_loader))
        current_lr    = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1:3d}/{config.EPOCHS} | "
            f"Train={avg_train:.4f} | Val={avg_val:.4f} | "
            f"LR={current_lr:.2e} | GradNorm={avg_grad_norm:.4f}"
            + (f" | ScalerSkips={scaler_skips} Scale={scaler.get_scale():.0f}"
               if use_amp and is_cuda else "")
        )

        # Skip checkpointing during OneCycleLR warmup
        if epoch < WARMUP_EPOCHS:
            print(f"  (warmup {epoch+1}/{WARMUP_EPOCHS} — checkpoint skipped)")
            continue

        if avg_val < best_val:
            best_val          = avg_val
            save_path = (
                f"best_model_fold_{fold_id}.pth"
                if fold_id != "baseline"
                else MODEL_PATH
            )
            torch.save(model.state_dict(), save_path)
            print(f"  --> Best model saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def train() -> None:
    _set_seed(42)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("CUDA benchmark mode enabled.")

    aggregation = _get_aggregation_mode()
    print(f"Aggregation mode: '{aggregation}'")

    # 1. Build FeatureEngineer from config ────────────────────────────────────
    fe_config = _make_feature_config()
    fe        = FeatureEngineer(config=fe_config)
    print(
        f"FeatureEngineer ready | ewma_span={fe_config.ewma_span} | "
        f"horizons={fe_config.return_horizons} | macd_pairs={fe_config.macd_pairs}"
    )

    # 2. Process all CSVs through the unified feature pipeline ────────────────
    asset_data_list, feature_cols = process_dataset(config.DATA_FILE, fe)
    config.NUM_FEATURES = len(feature_cols)
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    # 3. Optional tokenizer ───────────────────────────────────────────────────
    tokenizer = None
    if config.USE_TOKENIZER:
        from tokenizer import KLineTokenizer
        print(f"Initializing KLineTokenizer (bits={config.TOKENIZER_BITS})…")
        tokenizer = KLineTokenizer(input_dim=len(feature_cols), n_bits=config.TOKENIZER_BITS)
        if os.path.exists("tokenizer.pth"):
            tokenizer.load_state_dict(torch.load("tokenizer.pth", map_location="cpu"))
            print("Pre-trained tokenizer weights loaded.")

    # 4. Split per asset into train / val ─────────────────────────────────────
    gap        = config.FORECAST_HORIZON + 50
    train_list: list[tuple[np.ndarray, np.ndarray]] = []
    val_list:   list[tuple[np.ndarray, np.ndarray]] = []

    for asset_id, feat, target in asset_data_list:
        total_len = len(feat)
        train_end  = int(total_len * config.TRAIN_RATIO)
        val_start  = train_end + gap

        # ── CLAMP val_end before it can bleed past total_len ──────────────
        val_end_raw = val_start + int(total_len * config.VAL_RATIO)
        val_end     = min(val_end_raw, total_len - gap - config.LOOKBACK_WINDOW)

        # Early, meaningful failure — pinpoints the root cause
        if val_end <= val_start:
            raise ValueError(
                f"Val split is degenerate after clamping: val_start={val_start}, "
                f"val_end={val_end}. total_len={total_len} is too small for "
                f"TRAIN_RATIO={config.TRAIN_RATIO}, VAL_RATIO={config.VAL_RATIO}, "
                f"gap={gap}. Minimum required rows ≈ "
                f"{int((config.TRAIN_RATIO + config.VAL_RATIO) * total_len) + 2*gap + 3*config.LOOKBACK_WINDOW}."
            )

        test_start = val_end + gap

        if train_end > config.LOOKBACK_WINDOW:
            train_list.append((asset_id, feat, target, train_end))
        if val_end > val_start + config.LOOKBACK_WINDOW:
            val_list.append((asset_id, feat[val_start:val_end], target[val_start:val_end], None))

    # feature_cols forwarded to ColumnSelectiveScaler inside DataLoader
    # so each column lands in the correct scaler bucket (NO_SCALE vs ROBUST).
    train_loader, fitted_scalers = create_multi_index_dataloaders(
        train_list, config,
        feature_cols=feature_cols,
        tokenizer=tokenizer,
        is_train=True,
    )
    val_loader, _ = create_multi_index_dataloaders(
        val_list, config,
        feature_cols=feature_cols,
        tokenizer=tokenizer,
        is_train=False,
        scalers=fitted_scalers,
    )

    train_fold("baseline", train_loader, val_loader, feature_cols, tokenizer)


if __name__ == "__main__":
    train()