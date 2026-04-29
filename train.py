# train.py (Production)

from __future__ import annotations

import os
import random
from typing import Tuple  # FIX: was missing, caused NameError on Tuple type hints

import numpy as np
import pandas as pd
import torch

from oracle import generate_targets
from data_loader import create_dataloaders
from model import PatchTST
from tokenizer import KLineTokenizer
from loss import continuous_weighted_direction_loss
from features import calculate_features, PASSTHROUGH_FEATURES, SCALE_FEATURES
import config


MODEL_PATH = "best_model_patchtst.pth"
DATA_FILE = config.DATA_FILE

# FIX: skip checkpointing for this many epochs to avoid saving a false-valley
# during the OneCycleLR warmup phase as the "best" model.
WARMUP_EPOCHS = 3


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_aggregation_mode() -> str:
    return config.AGGREGATION_MODE


def _build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    time_col = next((c for c in df.columns if c.lower() in ["date", "datetime"]), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)
    else:
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    feat_df = calculate_features(df)
    combined_df = df.join(feat_df, how="inner")

    high_low   = combined_df["high"] - combined_df["low"]
    high_close = np.abs(combined_df["high"] - combined_df["close"].shift())
    low_close  = np.abs(combined_df["low"]  - combined_df["close"].shift())
    ranges     = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    combined_df["atr"] = true_range.rolling(config.ATR_PERIOD).mean()
    combined_df.dropna(inplace=True)

    ohlc_cols    = ["open", "high", "low", "close"]
    feature_cols = ohlc_cols + PASSTHROUGH_FEATURES + SCALE_FEATURES

    return combined_df, feature_cols


def process_dataset(file_paths: list[str]) -> Tuple[list[tuple[np.ndarray, np.ndarray]], list[str]]:
    """Processes multiple files independently and returns a list of (features, targets)."""
    asset_data_list    = []
    final_feature_cols = []

    for f in file_paths:
        if not os.path.exists(f):
            print(f"Warning: File {f} not found. Skipping.")
            continue

        print(f"Processing {f}...")
        df_raw = pd.read_csv(f)
        df, feature_cols = _build_features(df_raw)

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
            print(f"Warning: File {f} too short. Skipping.")
            continue

        input_cols         = PASSTHROUGH_FEATURES + SCALE_FEATURES if config.USE_TOKENIZER else feature_cols
        final_feature_cols = input_cols

        feat_vals   = df[input_cols].values.astype(np.float32)[:valid_len]
        target_vals = targets[:valid_len]

        asset_data_list.append((feat_vals, target_vals))

    if not asset_data_list:
        raise ValueError("No valid data processed from the provided files.")

    return asset_data_list, final_feature_cols


def train() -> None:
    _set_seed(42)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("CUDA Benchmark mode enabled.")

    aggregation = _get_aggregation_mode()
    print(f"Training with aggregation='{aggregation}' (env PATCHTST_AGGREGATION).")

    asset_data_list, feature_cols = process_dataset(config.DATA_FILE)
    config.NUM_FEATURES = len(feature_cols)

    tokenizer = None
    if config.USE_TOKENIZER:
        print(f"Initializing Tokenizer (Hybrid Mode: 21 features, {config.TOKENIZER_BITS} bits)...")
        tokenizer = KLineTokenizer(input_dim=21, n_bits=config.TOKENIZER_BITS)
        if os.path.exists("tokenizer.pth"):
            print("Loading pre-trained tokenizer weights from tokenizer.pth...")
            tokenizer.load_state_dict(torch.load("tokenizer.pth", map_location="cpu"))

    from data_loader import create_multi_index_dataloaders

    train_list = []
    val_list   = []
    gap        = config.FORECAST_HORIZON + 50

    for feat, target in asset_data_list:
        total_len = len(feat)
        train_end = int(total_len * config.TRAIN_RATIO)
        val_start = train_end + gap
        val_end   = val_start + int(total_len * config.VAL_RATIO)

        if train_end > config.LOOKBACK_WINDOW:
            train_list.append((feat[:train_end], target[:train_end]))

        if val_end > val_start + config.LOOKBACK_WINDOW:
            val_list.append((feat[val_start:val_end], target[val_start:val_end]))

    train_loader = create_multi_index_dataloaders(
        train_list, config, tokenizer=tokenizer, feature_cols=feature_cols, is_train=True
    )
    val_loader = create_multi_index_dataloaders(
        val_list, config, tokenizer=tokenizer, feature_cols=feature_cols, is_train=False
    )

    train_fold(None, None, "baseline", train_loader, val_loader, tokenizer)


def train_fold(features, targets, fold_id, train_loader, val_loader, tokenizer=None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training Fold: {fold_id} ---")
    print(f"Using device: {device}")

    aggregation = _get_aggregation_mode()

    model = PatchTST(
        seq_len=config.LOOKBACK_WINDOW,
        num_features=1,
        patch_len=config.PATCH_LEN,
        stride=config.STRIDE,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        dropout=0.2,
        aggregation=aggregation,
        use_tokenizer=config.USE_TOKENIZER,
        vocab_size=config.VOCAB_SIZE,
    ).to(device)

    if device.type == "cuda":
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"torch.compile failed: {e}")
    else:
        print("Skipping torch.compile on CPU for stability.")

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
        pct_start=0.1,        # warmup over first 10% of steps
        div_factor=10,        # start_lr = max_lr / 10
        final_div_factor=1e3, # end_lr   = max_lr / 1000
    )

    # --- AMP Setup ---
    # FIX: bfloat16 on CPU does NOT use GradScaler. The old code called
    # scaler.step(optimizer) on CPU which silently skips weight updates when
    # the scaler is disabled (it checks for float16 inf/nan that don't apply
    # to bfloat16). We now branch: CUDA uses scaler, CPU uses optimizer directly.
    use_amp    = config.USE_AMP
    amp_device = device.type
    amp_dtype  = torch.bfloat16 if device.type == "cpu" else torch.float16

    scaler = torch.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    if use_amp:
        print(f"Using Automatic Mixed Precision (AMP) on {device.type} with {amp_dtype}")

    print(f"Starting training for fold {fold_id}...")
    best_val          = float("inf")
    epochs_no_improve = 0

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss      = 0.0
        # FIX: accumulate grad norm across batches, print once per epoch.
        # Printing ~196 lines/epoch inside the batch loop caused measurable
        # syscall overhead on Chromebook CPU (~20-30% slowdown per epoch).
        grad_norm_accum = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype, enabled=use_amp):
                pred = model(x)
                loss = continuous_weighted_direction_loss(pred, y)

            # FIX: total_norm declared before branch — always defined regardless of
            # device. clip_grad_norm_() return value reused in both paths, eliminating
            # ~9800 redundant per-param tensor traversals over 50 epochs.
            #
            # CUDA path: scale -> backward -> unscale_ -> clip -> step -> update
            #   scaler.unscale_() MUST come before clip_grad_norm_() so clipping
            #   operates on actual gradients, not the ~65536x scaled versions.
            #   Without unscale_(), GRAD_CLIP=2.0 never activates on GPU.
            #
            # CPU path: backward -> clip -> step  (no scaler involved)
            total_norm = 0.0
            if use_amp and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                optimizer.step()

            scheduler.step()
            train_loss      += float(loss.item())
            grad_norm_accum += float(total_norm)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype, enabled=use_amp):
                    pred = model(x)
                    loss = continuous_weighted_direction_loss(pred, y)
                val_loss += float(loss.item())

        avg_train     = train_loss      / max(1, len(train_loader))
        avg_val       = val_loss        / max(1, len(val_loader))
        avg_grad_norm = grad_norm_accum / max(1, len(train_loader))
        current_lr    = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | "
            f"Train={avg_train:.4f} | Val={avg_val:.4f} | "
            f"LR={current_lr:.2e} | AvgGradNorm={avg_grad_norm:.4f}"
        )

        # FIX: do not checkpoint during warmup epochs. The LR is still ramping
        # up and the model is not yet in a stable loss region. Saving here
        # captures a false-valley that makes early stopping fire prematurely.
        if epoch < WARMUP_EPOCHS:
            print(f"  (warmup epoch {epoch+1}/{WARMUP_EPOCHS}, skipping checkpoint)")
            continue

        if avg_val < best_val:
            best_val          = avg_val
            epochs_no_improve = 0
            save_path = f"best_model_fold_{fold_id}.pth" if fold_id != "baseline" else MODEL_PATH
            torch.save(model.state_dict(), save_path)
            print(f"--> Best model saved to: {save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.WFV_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break


if __name__ == "__main__":
    train()