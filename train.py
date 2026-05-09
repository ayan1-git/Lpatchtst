# train.py  (Production — Multi-Modal Kronos Integration)
import sys
import os
import random
from typing import Tuple
import numpy as np
import pandas as pd
import torch
import importlib

# Force Python to look in the workspace directory first
sys.path.insert(0, os.getcwd())

import config
import model
import features
import loss
import oracle
import data_loader
import tokenizer

importlib.reload(config)
importlib.reload(model)
importlib.reload(tokenizer)
importlib.reload(features)
importlib.reload(data_loader)
importlib.reload(loss)
importlib.reload(oracle)

from oracle import generate_targets
from data_loader import create_multi_index_dataloaders
from model import LPatchTST, InputMode
from loss import continuous_weighted_direction_loss
from features import FeatureConfig, FeatureEngineer
from tokenizer import prepare_ohlc_features, KronosTokenizer

MODEL_PATH = "best_model_lpatchtst.pth"
WARMUP_EPOCHS = 10
OHLC_COLS = ["open", "high", "low", "close"]

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _make_feature_config() -> FeatureConfig:
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
    )

def _build_feature_cols(fe_config: FeatureConfig) -> Tuple[list[str], list[str], list[str]]:
    no_scale_cols: list[str] = []
    robust_cols:   list[str] = []
    no_scale_cols.append(f"ewma_vol_span{fe_config.ewma_span}")
    for h in fe_config.return_horizons:
        no_scale_cols.append(f"ret_norm_{h}d")
    for s, l in fe_config.macd_pairs:
        no_scale_cols.append(f"macd_{s}_{l}")
    robust_cols.append(f"vs_factor_span{fe_config.ewma_span}")
    no_scale_cols += ["feat_efficiency", "feat_icp", "feat_momentum_rsi", "feat_vol_asymmetry", "feat_local_structure"]
    if fe_config.add_session_features:
        no_scale_cols += ["feat_session_sin", "feat_session_cos"]
    robust_cols.append("feat_vol_squeeze")
    all_feat_cols = robust_cols + no_scale_cols
    return no_scale_cols, robust_cols, all_feat_cols

def _build_features(df: pd.DataFrame, fe: FeatureEngineer) -> Tuple[pd.DataFrame, list[str]]:
    time_col = next((c for c in df.columns if c.lower() in ("date", "datetime")), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
    else:
        try: df.index = pd.to_datetime(df.index)
        except: pass
    df = df.sort_index()
    feat_df = fe.build(df["close"], ohlc=df[OHLC_COLS], include_target=False, dropna=False)
    combined_df = df.join(feat_df, how="inner")
    high_low   = combined_df["high"] - combined_df["low"]
    high_close = (combined_df["high"] - combined_df["close"].shift()).abs()
    low_close  = (combined_df["low"]  - combined_df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    combined_df["atr"] = true_range.rolling(config.ATR_PERIOD).mean()
    combined_df.dropna(inplace=True)
    _, _, all_feat_cols = _build_feature_cols(fe.config)
    return combined_df, all_feat_cols

def process_dataset(file_paths: list[str], fe: FeatureEngineer) -> Tuple[list[tuple[str, np.ndarray, np.ndarray, np.ndarray]], list[str]]:
    asset_data_list:    list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    final_feature_cols: list[str] = []
    for f in file_paths:
        if not os.path.exists(f): continue
        print(f"Processing {f}…")
        df_raw = pd.read_csv(f)
        df, feature_cols = _build_features(df_raw, fe)
        ohlc_returns = prepare_ohlc_features(df)
        df = df.iloc[1:]  # Shift df to align with log-returns (which drop the first row)
        
        targets = generate_targets(df["open"].values, df["high"].values, df["low"].values, df["close"].values, df["atr"].values,
                                   max_hold=config.ORACLE_MAX_HOLD, fee_per_side=config.FEE_PER_SIDE, slippage=config.SLIPPAGE,
                                   atr_mult=config.ATR_MULT, saturation_factor=config.SATURATION_FACTOR, mae_penalty=config.MAE_PENALTY)
        
        valid_len = len(targets) - config.ORACLE_MAX_HOLD
        if valid_len <= 0: continue
        
        feat_vals    = np.asarray(df[feature_cols].values, dtype=np.float32)[:valid_len]
        target_vals  = np.asarray(targets[:valid_len],      dtype=np.float32)
        ohlc_vals    = ohlc_returns[:valid_len]
        
        asset_data_list.append((f, feat_vals, target_vals, ohlc_vals))
        final_feature_cols = feature_cols
    return asset_data_list, final_feature_cols

def train_fold(fold_id: str, train_loader, val_loader, feature_cols: list[str]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training Fold: {fold_id} --- | Mode: {config.INPUT_MODE}")
    
    net = LPatchTST(
        input_mode=config.INPUT_MODE,
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
        try: net = torch.compile(net); print("Compiled.")
        except: pass

    optimizer = torch.optim.AdamW(net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.LEARNING_RATE, total_steps=config.EPOCHS * len(train_loader))
    grad_scaler = torch.amp.GradScaler(enabled=config.USE_AMP and device.type == "cuda")

    best_val = float("inf")
    for epoch in range(config.EPOCHS):
        net.train()
        train_loss = 0.0
        for tokens, features, y in train_loader:
            if tokens is not None: tokens = (tokens[0].to(device), tokens[1].to(device))
            if features is not None: features = features.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=config.USE_AMP):
                pred = net(tokens=tokens, features=features)
                loss = continuous_weighted_direction_loss(pred, y)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.GRAD_CLIP)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()
            train_loss += loss.item()

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for tokens, features, y in val_loader:
                if tokens is not None: tokens = (tokens[0].to(device), tokens[1].to(device))
                if features is not None: features = features.to(device)
                y = y.to(device)
                with torch.amp.autocast(device_type=device.type, enabled=config.USE_AMP):
                    pred = net(tokens=tokens, features=features)
                    loss = continuous_weighted_direction_loss(pred, y)
                val_loss += loss.item()
        
        avg_train, avg_val = train_loss/len(train_loader), val_loss/len(val_loader)
        print(f"Epoch {epoch+1:2d} | Train={avg_train:.4f} | Val={avg_val:.4f}")
        if avg_val < best_val and epoch >= WARMUP_EPOCHS:
            best_val = avg_val
            torch.save(net.state_dict(), MODEL_PATH)

def train() -> None:
    _set_seed(42)
    fe = FeatureEngineer(config=_make_feature_config())
    asset_data_list, feature_cols = process_dataset(config.DATA_FILE, fe)
    
    tokenizer = None
    if config.INPUT_MODE in (InputMode.TOKENS_ONLY, InputMode.COMBINED):
        tokenizer = KronosTokenizer(
            d_in=4, 
            d_model=config.TOKENIZER_D_MODEL, 
            n_heads=config.TOKENIZER_N_HEADS,
            ff_dim=config.TOKENIZER_FF_DIM,
            s1_bits=config.TOKENIZER_S1_BITS, 
            s2_bits=config.TOKENIZER_S2_BITS
        )
        if os.path.exists("tokenizer.pt"):
            tokenizer.load_state_dict(torch.load("tokenizer.pt", map_location="cpu"))
            print("Loaded tokenizer.pt")
        tokenizer.eval()

    gap = config.FORECAST_HORIZON + 50
    train_list, val_list = [], []
    for asset_id, feat, target, ohlc in asset_data_list:
        train_end = int(len(feat) * config.TRAIN_RATIO)
        val_start = train_end + gap
        val_end = min(val_start + int(len(feat) * config.VAL_RATIO), len(feat) - gap - config.LOOKBACK_WINDOW)
        if train_end > config.LOOKBACK_WINDOW:
            train_list.append((asset_id, feat, target, ohlc, train_end))
        if val_end > val_start + config.LOOKBACK_WINDOW:
            val_list.append((asset_id, feat[val_start:val_end], target[val_start:val_end], ohlc[val_start:val_end], None))

    train_loader, fitted_scalers = create_multi_index_dataloaders(train_list, config, feature_cols, tokenizer, is_train=True)
    val_loader, _ = create_multi_index_dataloaders(val_list, config, feature_cols, tokenizer, is_train=False, scalers=fitted_scalers)
    train_fold("baseline", train_loader, val_loader, feature_cols)

if __name__ == "__main__":
    train()