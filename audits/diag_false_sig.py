
import torch
import pandas as pd
import numpy as np
import os
import config
from model import LPatchTST
from oracle import generate_targets
from features import FeatureEngineer, FeatureConfig
from tokenizer import KronosTokenizer, prepare_ohlc_features
from evaluate import run_inference, _build_features

# Force the correct model path since config.py might be outdated relative to filesystem
MODEL_PATH_OVERRIDE = "models/best_model_lpatchtst.pth"

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

def run_diag():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    data_path = "data/NIFTY 50_30minute.csv"
    df_raw = pd.read_csv(data_path)
    if "date" in df_raw.columns:
        df_raw = df_raw.set_index("date")
    df_raw.index = pd.to_datetime(df_raw.index)
    df_raw = df_raw.sort_index()

    fe_config = _make_feature_config()
    fe = FeatureEngineer(config=fe_config)
    df, feature_cols = _build_features(df_raw, fe)
    
    targets = generate_targets(
        df["open"].values, df["high"].values, df["low"].values, df["close"].values, df["atr"].values,
        max_hold=config.ORACLE_MAX_HOLD, fee_per_side=config.FEE_PER_SIDE, slippage=config.SLIPPAGE,
        sl_atr_mult=config.ORACLE_SL_ATR_MULT, tp_atr_mult=config.ORACLE_TP_ATR_MULT,
        saturation_factor=config.SATURATION_FACTOR, mae_penalty=config.MAE_PENALTY,
    )
    
    valid_len = len(targets) - config.ORACLE_MAX_HOLD
    df = df.iloc[:valid_len].copy()
    targets = targets[:valid_len]
    
    features = df[feature_cols].values.astype(np.float32) if feature_cols else np.zeros((len(df), 0), dtype=np.float32)
    
    tokenizer = None
    ohlc_returns = None
    if config.INPUT_MODE in ("tokens_only", "combined"):
        tokenizer = KronosTokenizer(
            d_in=config.TOKENIZER_D_IN, d_model=config.TOKENIZER_D_MODEL, n_heads=config.N_HEADS,
            ff_dim=config.TOKENIZER_FF_DIM, n_enc_layers=config.TOKENIZER_N_ENC, n_dec_layers=config.TOKENIZER_N_DEC,
            s1_bits=config.TOKENIZER_S1_BITS, s2_bits=config.TOKENIZER_S2_BITS, group_size=config.TOKENIZER_GROUP_SIZE,
            beta=config.TOKENIZER_BETA, gamma0=config.TOKENIZER_GAMMA0, gamma=config.TOKENIZER_GAMMA,
            zeta=config.TOKENIZER_ZETA, attn_dropout_p=config.TOKENIZER_ATTN_DROPOUT,
            ffn_dropout_p=config.TOKENIZER_FFN_DROPOUT, resid_dropout_p=config.TOKENIZER_RESID_DROPOUT,
        )
        tok_path = "models/model.safetensors"
        if os.path.exists(tok_path):
            tokenizer.load_pretrained(tok_path, device="cpu")
            print(f"Tokenizer loaded from {tok_path}")
        tokenizer.eval()
        
        ohlc_returns = prepare_ohlc_features(df)
        df = df.iloc[1:].copy()
        targets = targets[1:]
        features = df[feature_cols].values.astype(np.float32) if feature_cols else np.zeros((len(df), 0), dtype=np.float32)

    # LIMIT DATA for speed
    max_diag_samples = 5000
    if len(df) > max_diag_samples:
        df = df.iloc[:max_diag_samples].copy()
        targets = targets[:max_diag_samples]
        features = features[:max_diag_samples] if features.size > 0 else np.zeros((max_diag_samples, 0), dtype=np.float32)
        if ohlc_returns is not None:
            ohlc_returns = ohlc_returns[:max_diag_samples]

    from data_loader import create_fold_dataloaders
    loader = create_fold_dataloaders(
        features, targets,
        train_indices=(0, min(100, len(features))), val_indices=(0, len(df)), test_indices=(0, 0),
        config=config, feature_cols=feature_cols, tokenizer=tokenizer, ohlc_returns=ohlc_returns
    )[1]
    
    model = LPatchTST(
        input_mode=config.INPUT_MODE, seq_len=config.LOOKBACK_WINDOW, n_features=len(feature_cols),
        s1_bits=config.TOKENIZER_S1_BITS, s2_bits=config.TOKENIZER_S2_BITS, d_model=config.D_MODEL,
        patch_len=config.PATCH_LEN, stride=config.STRIDE, n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS, lstm_layers=config.LSTM_LAYERS, dropout=config.DROPOUT,
        aggregation=config.AGGREGATION_MODE, vocab_size=config.VOCAB_SIZE
    ).to(device)
    
    if os.path.exists(MODEL_PATH_OVERRIDE):
        state = torch.load(MODEL_PATH_OVERRIDE, map_location=device)
        if any(k.startswith("_orig_mod.") for k in state.keys()):
            state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        print(f"Model weights loaded from {MODEL_PATH_OVERRIDE}")
    else:
        print(f"Warning: {MODEL_PATH_OVERRIDE} not found!")
        
    model.eval()
    
    preds = run_inference(model, loader, device)
    
    tgt_start = config.LOOKBACK_WINDOW - 1
    tgts_aligned = targets[tgt_start:]
    
    min_len = min(len(preds), len(tgts_aligned))
    preds = preds[:min_len]
    tgts_aligned = tgts_aligned[:min_len]
    
    is_zero = (np.abs(tgts_aligned) < config.SAMPLER_THRESHOLD) | (tgts_aligned == 0.0)
    p_zero = preds[is_zero]
    
    print(f"Number of zero-target samples: {len(p_zero)}")
    if len(p_zero) > 0:
        print(f"Mean abs pred on zero targets: {np.mean(np.abs(p_zero)):.6f}")
        print(f"Min abs pred on zero targets: {np.min(np.abs(p_zero)):.6f}")
        print(f"Max abs pred on zero targets: {np.max(np.abs(p_zero)):.6f}")
        
        for margin in [0.0, 0.01, 0.05]:
            rate = (np.abs(p_zero) > margin).mean()
            print(f"False sig rate (margin={margin}): {rate*100:.2f}%")
    else:
        print("No zero-target samples found.")

if __name__ == "__main__":
    run_diag()
