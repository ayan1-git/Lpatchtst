#!/usr/bin/env python
# inference.py
# Production-Grade Live Inference Integration for LPatchTST

import sys
import os
import argparse
import time
import json
import subprocess
import traceback
import builtins
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime

# Add current working directory to path to resolve local imports
sys.path.insert(0, os.getcwd())

import config
from model import LPatchTST, PatchTST
from features import FeatureConfig, FeatureEngineer
from data_loader import ColumnSelectiveScaler, fit_scaler
from tokenizer import KronosTokenizer, prepare_ohlc_features

# Save the original print to allow raw output writing and ensure unbuffered flushing
ORIGINAL_PRINT = builtins.print
def print_always_flush(*print_args, **print_kwargs):
    print_kwargs['flush'] = True
    ORIGINAL_PRINT(*print_args, **print_kwargs)
builtins.print = print_always_flush

def log_msg(msg):
    sys.stderr.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    sys.stderr.flush()

# ─────────────────────────────────────────────────────────────────────────────
# Model Building Helpers (Mirrors evaluate.py)
# ─────────────────────────────────────────────────────────────────────────────

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
    if config.INPUT_MODE == "tokens_only":
        return [], [], []
    no_scale_cols, robust_cols = [], []
    no_scale_cols.append(f"ewma_vol_span{fe_config.ewma_span}")
    for h in fe_config.return_horizons:
        no_scale_cols.append(f"ret_norm_{h}d")
    for s, l in fe_config.macd_pairs:
        no_scale_cols.append(f"macd_{s}_{l}")
    no_scale_cols += ["feat_efficiency","feat_icp","feat_momentum_rsi","feat_vol_asymmetry","feat_local_structure"]
    robust_cols.append("feat_vol_squeeze")
    if fe_config.add_session_features:
        no_scale_cols += ["feat_session_sin","feat_session_cos"]

    if fe_config.use_talib:
        try:
            from talib_features import TALIB_PASSTHROUGH, TALIB_SCALE
            no_scale_cols += TALIB_PASSTHROUGH
            no_scale_cols += TALIB_SCALE
        except ImportError:
            pass

    return no_scale_cols, robust_cols, robust_cols + no_scale_cols

def _build_model(aggregation: str, num_features: int) -> PatchTST:
    if config.USE_LPATCHTST:
        return LPatchTST(
            input_mode=config.INPUT_MODE,
            seq_len=config.LOOKBACK_WINDOW,
            n_features=num_features,
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
            dropout=config.FINETUNE_DROPOUT,
            aggregation=aggregation,
            vocab_size=config.VOCAB_SIZE,
        )
    return PatchTST(
        seq_len=config.LOOKBACK_WINDOW,
        num_features=num_features,
        patch_len=config.PATCH_LEN,
        stride=config.STRIDE,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        n_dec_layers=config.N_DEC_LAYERS,
        n_queries=config.N_QUERIES,
        lstm_layers=config.LSTM_LAYERS,
        dropout=config.DROPOUT,
        aggregation=aggregation,
        input_mode=config.INPUT_MODE,
        vocab_size=config.VOCAB_SIZE,
        s1_bits=config.TOKENIZER_S1_BITS,
        s2_bits=config.TOKENIZER_S2_BITS,
    )

def _load_model(model_path: str, device: torch.device, num_features: int) -> PatchTST:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model checkpoint not found at: {model_path}")
        
    state = torch.load(model_path, map_location=device)

    # Clean compiled prefix if present
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}

    preferred = config.AGGREGATION_MODE
    fallback  = "mean" if preferred == "mixing" else "mixing"

    for agg in (preferred, fallback):
        try:
            model = _build_model(agg, num_features).to(device)
            model.load_state_dict(state, strict=False)
            model.eval()
            log_msg(f"Successfully loaded model from {model_path} (aggregation='{agg}')")
            return model
        except (RuntimeError, ValueError) as e:
            log_msg(f"Failed loading model with aggregation='{agg}': {e}")

    raise RuntimeError(f"Could not load {model_path} with either aggregation mode.")

# ─────────────────────────────────────────────────────────────────────────────
# Data Fetching Logic (TradingView MCP CLI integration)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_live_bars(mcp_cli_path="tradingview-mcp/src/cli/index.js", count=3800):
    cmd = ["node", mcp_cli_path, "ohlcv", "--count", str(count)]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"TradingView MCP CLI failed with code {result.returncode}: {result.stderr.strip()}")
        
    try:
        data = json.loads(result.stdout)
    except Exception as e:
        raise RuntimeError(f"Failed to parse TradingView MCP output as JSON: {e}\nRaw output: {result.stdout}")
        
    if not data.get("success"):
        raise RuntimeError(f"TradingView MCP reported failure: {data.get('error', 'Unknown error')}")
        
    bars = data.get("bars", [])
    if not bars:
        raise RuntimeError("TradingView MCP returned an empty bars list. The chart may still be loading.")
        
    df = pd.DataFrame(bars)
    
    # Ensure correct data types
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
        
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.sort_values("time").reset_index(drop=True)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Inference Execution Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    df, 
    model, 
    tokenizer, 
    fe_config, 
    feature_cols, 
    device, 
    policy_params
):
    input_mode = config.INPUT_MODE
    seq_len = config.LOOKBACK_WINDOW
    smoothing = config.INFERENCE_SMOOTHING
    
    idx_coarse = None
    idx_fine = None
    scaled_features = None
    
    # ── 1. Process Tokenizer Inputs ──
    if input_mode in ("tokens_only", "combined"):
        if tokenizer is None:
            raise ValueError("Tokenizer instance is required for token modes.")
            
        # Extract raw OHLCV to match Kronos_finetune and Lpatchtst training
        cols = {c.lower(): c for c in df.columns}
        o_col = cols.get('open', 'Open')
        h_col = cols.get('high', 'High')
        l_col = cols.get('low', 'Low')
        c_col = cols.get('close', 'Close')
        v_col = cols.get('volume', 'Volume')
        
        close  = df[c_col].values.astype(np.float32)
        volume = df[v_col].values.astype(np.float32) if v_col in df.columns else np.zeros_like(close)
        amount = close * volume
        
        ohlc_returns = np.stack([
            df[o_col].values.astype(np.float32),
            df[h_col].values.astype(np.float32),
            df[l_col].values.astype(np.float32),
            df[c_col].values.astype(np.float32),
            volume,
            amount
        ], axis=1)
        
        T_ret = len(ohlc_returns)
        S = config.LOOKBACK_WINDOW  # Use LOOKBACK_WINDOW for normalization and sequence length
        
        # Guard minimum returned bars size
        if T_ret < seq_len + smoothing - 1:
            raise ValueError(
                f"Insufficient bars ({T_ret}) to form lookback window ({seq_len}) "
                f"with smoothing ({smoothing}). Need at least {seq_len + smoothing} bars."
            )
            
        # Pad start using first bar to avoid zero-shock
        pad = np.tile(ohlc_returns[0:1], (S-1, 1))
        padded = np.concatenate([pad, ohlc_returns], axis=0)  # (T_ret + S - 1, 6)
        
        # Build sliding windows
        from numpy.lib.stride_tricks import as_strided
        C = ohlc_returns.shape[1]
        shape = (T_ret, S, C)
        strides = (padded.strides[0], padded.strides[0], padded.strides[1])
        windows = as_strided(padded, shape=shape, strides=strides)
        
        chunk_size = getattr(config, "TOKENIZER_CHUNK_SIZE", 2048)
        c_list, f_list = [], []
        
        tokenizer.eval()
        with torch.no_grad():
            for i in range(0, T_ret, chunk_size):
                batch = torch.from_numpy(windows[i : i + chunk_size]).to(device).float()
                
                # Per-window normalization matching training
                w_mean = batch.mean(dim=1, keepdim=True)
                w_std  = batch.std(dim=1, keepdim=True) + 1e-5
                batch  = (batch - w_mean) / w_std
                batch  = torch.clamp(batch, -5.0, 5.0)
                
                idx_c, idx_f = tokenizer.encode(batch, half=True)
                c_list.append(idx_c[:, -1].cpu())
                f_list.append(idx_f[:, -1].cpu())
                
        idx_coarse = torch.cat(c_list)  # (T_ret,)
        idx_fine = torch.cat(f_list)    # (T_ret,)
        
    # ── 2. Process Continuous Feature Inputs ──
    if input_mode in ("features_only", "combined"):
        fe = FeatureEngineer(fe_config)
        
        # Clean copy and sort index
        df_feats = df.copy()
        df_feats.index = pd.to_datetime(df_feats["time"])
        df_feats = df_feats.sort_index()
        
        feat_df = fe.build(df_feats["close"], ohlc=df_feats[["open", "high", "low", "close", "volume"]], include_target=False, dropna=False)
        combined_df = df_feats.join(feat_df, how="inner")
        
        hl = combined_df["high"] - combined_df["low"]
        hc = (combined_df["high"] - combined_df["close"].shift()).abs()
        lc = (combined_df["low"]  - combined_df["close"].shift()).abs()
        combined_df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(config.ATR_PERIOD).mean()
        combined_df.dropna(inplace=True)
        
        # Extract features array
        features = combined_df[feature_cols].values.astype(np.float32)
        
        if len(features) < seq_len + smoothing - 1:
            raise ValueError(
                f"Insufficient valid feature rows ({len(features)}) after NaN dropping. "
                f"Check that enough bars were fetched (FE_MACD_SIGNAL_STD_WIN={config.FE_MACD_SIGNAL_STD_WIN})."
            )
            
        # Fit a fresh ColumnSelectiveScaler dynamically on the series
        scaler = ColumnSelectiveScaler(
            feature_cols,
            clip_bounds=config.ROBUST_CLIP_BOUNDS,
            default_clip_bound=config.ROBUST_CLIP_BOUND_DEFAULT,
        )
        scaler.fit(features)
        scaled_features = torch.from_numpy(scaler.transform(features))
        
    # ── 3. Slice and Construct Inference Batches ──
    if input_mode in ("tokens_only", "combined"):
        active_len = len(idx_coarse)
    else:
        active_len = len(scaled_features)
        
    batch_tokens = None
    batch_features = None
    
    t_coarse_list, t_fine_list = [], []
    feat_list = []
    
    for offset in range(smoothing):
        start_idx = active_len - seq_len - smoothing + 1 + offset
        end_idx = start_idx + seq_len
        
        if input_mode in ("tokens_only", "combined"):
            t_coarse_list.append(idx_coarse[start_idx:end_idx].unsqueeze(0))
            t_fine_list.append(idx_fine[start_idx:end_idx].unsqueeze(0))
            
        if input_mode in ("features_only", "combined"):
            feat_list.append(scaled_features[start_idx:end_idx].unsqueeze(0))
            
    if input_mode in ("tokens_only", "combined"):
        batch_tokens = (
            torch.cat(t_coarse_list, dim=0).to(device),
            torch.cat(t_fine_list, dim=0).to(device)
        )
        
    if input_mode in ("features_only", "combined"):
        batch_features = torch.cat(feat_list, dim=0).to(device)
        
    # ── 4. Forward Pass & Signal Generation ──
    model.eval()
    use_amp = config.USE_AMP and device.type == "cuda"
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
                raw_preds = model(tokens=batch_tokens, features=batch_features)
        else:
            raw_preds = model(tokens=batch_tokens, features=batch_features)

    raw_preds_cpu = raw_preds.detach().float().cpu().numpy()
    levels = list(getattr(config, "QUANTILE_LEVELS", []))
    if getattr(config, "QUANTILE_HEAD", False) and raw_preds_cpu.ndim == 2 \
            and raw_preds_cpu.shape[-1] > 1 and 0.50 in levels:
        med_i = levels.index(0.50)
        i_lo = int(np.argmin(levels))
        i_hi = int(np.argmax(levels))
        spread = float(np.mean(raw_preds_cpu[:, i_hi] - raw_preds_cpu[:, i_lo]))
        scores = raw_preds_cpu[:, med_i]
    else:
        scores = raw_preds_cpu.reshape(-1)
        spread = float("nan")

    smoothed_pred = float(np.mean(scores))
    
    threshold, bias = policy_params
    adjusted_score = smoothed_pred + bias
    
    if adjusted_score > threshold:
        signal = 1
        label = "LONG"
    elif adjusted_score < -threshold:
        signal = -1
        label = "SHORT"
    else:
        signal = 0
        label = "FLAT"
        
    latest_bar = df.iloc[-1]
    
    return {
        "timestamp": datetime.now().isoformat(),
        "bar_time": latest_bar["time"].isoformat(),
        "close": float(latest_bar["close"]),
        "raw_predictions": [float(p) for p in scores],
        "smoothed_prediction": smoothed_pred,
        "adjusted_score": adjusted_score,
        "spread_qhi_qlo": spread,
        "signal": signal,
        "signal_label": label,
        "threshold": threshold,
        "bias": bias
    }

# ─────────────────────────────────────────────────────────────────────────────
# Helper to Load Policy JSON
# ─────────────────────────────────────────────────────────────────────────────

def load_policy(policy_path="models/best_policy.json"):
    try:
        with open(policy_path, "r") as f:
            policy = json.load(f)
            threshold = policy.get("threshold", 0.25)
            bias = policy.get("bias", -0.16)
            log_msg(f"Loaded policy from {policy_path}: threshold={threshold}, bias={bias}")
            return threshold, bias
    except Exception as e:
        log_msg(f"Warning: Could not load policy from {policy_path} ({e}). Using default threshold=0.25, bias=-0.16")
        return 0.25, -0.16

# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Live Inference Engine for LPatchTST")
    parser.add_argument("--model-path", default="models/best_model_lpatchtst.pth", help="Path to trained PyTorch model weights")
    parser.add_argument("--tokenizer-path", default="models/model.safetensors", help="Path to pre-trained tokenizer safetensors")
    parser.add_argument("--policy-path", default="models/best_policy.json", help="Path to policy json")
    parser.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds for daemon mode")
    parser.add_argument("--loop", action="store_true", help="Run continuously as a daemon polling for new bars")
    parser.add_argument("--device", default=None, help="Target device (cuda or cpu)")
    parser.add_argument("--jsonl", action="store_true", help="Format output as JSONL for pipe-friendliness (logs to stderr)")
    
    args = parser.parse_args()
    
    # ── 1. Configure print redirection ──
    if args.jsonl:
        def print_stderr(*print_args, **print_kwargs):
            print_kwargs['file'] = sys.stderr
            print_kwargs['flush'] = True
            ORIGINAL_PRINT(*print_args, **print_kwargs)
        builtins.print = print_stderr
        
    # ── 2. Device configuration ──
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_msg(f"Target device: {device}")
    
    # ── 3. Load feature definitions ──
    fe_config = _make_feature_config()
    _, _, feature_cols = _build_feature_cols(fe_config)
    num_features = len(feature_cols)
    log_msg(f"Feature configuration resolved. Features={num_features}, Mode={config.INPUT_MODE}")
    
    # ── 4. Load weights & policy ──
    try:
        model = _load_model(args.model_path, device, num_features)
    except Exception as e:
        log_msg(f"CRITICAL ERROR: Failed to load model weights: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
        
    tokenizer = None
    if config.INPUT_MODE in ("tokens_only", "combined"):
        log_msg("Initializing KronosTokenizer...")
        tokenizer = KronosTokenizer(
            d_in=config.TOKENIZER_D_IN,
            d_model=config.TOKENIZER_D_MODEL,
            n_heads=config.TOKENIZER_N_HEADS,
            ff_dim=config.TOKENIZER_FF_DIM,
            n_enc_layers=config.TOKENIZER_N_ENC,
            n_dec_layers=config.TOKENIZER_N_DEC,
            s1_bits=config.TOKENIZER_S1_BITS,
            s2_bits=config.TOKENIZER_S2_BITS,
            group_size=config.TOKENIZER_GROUP_SIZE,
            beta=config.TOKENIZER_BETA,
            gamma0=config.TOKENIZER_GAMMA0,
            gamma=config.TOKENIZER_GAMMA,
            zeta=config.TOKENIZER_ZETA,
            attn_dropout_p=config.TOKENIZER_ATTN_DROPOUT,
            ffn_dropout_p=config.TOKENIZER_FFN_DROPOUT,
            resid_dropout_p=config.TOKENIZER_RESID_DROPOUT,
        )
        if os.path.exists(args.tokenizer_path):
            tokenizer.load_pretrained(args.tokenizer_path, device="cpu")
            log_msg(f"Successfully loaded tokenizer from {args.tokenizer_path}")
        else:
            alt_path = "tokenizer.pt"
            if os.path.exists(alt_path):
                tokenizer.load_pretrained(alt_path, device="cpu")
                log_msg(f"Successfully loaded tokenizer from fallback {alt_path}")
            else:
                log_msg(f"WARNING: Tokenizer weights not found at {args.tokenizer_path} or {alt_path}!")
                
        for p in tokenizer.parameters():
            p.requires_grad = False
        tokenizer.eval()
        tokenizer.to(device)
        
    policy_params = load_policy(args.policy_path)
    
    # ── 5. Run single-shot or daemon loop ──
    warmup_bars = 50
    if config.INPUT_MODE in ("features_only", "combined"):
        # We need the macd std signal window buffer plus lookback
        warmup_bars = config.FE_MACD_SIGNAL_STD_WIN + 200
        
    count = config.LOOKBACK_WINDOW + warmup_bars + config.INFERENCE_SMOOTHING
    log_msg(f"Target bar fetch count calculated: {count}")
    
    if not args.loop:
        # Single-shot execution
        try:
            df = fetch_live_bars(count=count)
            output = run_inference(
                df=df,
                model=model,
                tokenizer=tokenizer,
                fe_config=fe_config,
                feature_cols=feature_cols,
                device=device,
                policy_params=policy_params
            )
            
            # Print clean JSON strictly to stdout
            ORIGINAL_PRINT(json.dumps(output, indent=2))
        except Exception as e:
            log_msg(f"CRITICAL ERROR during single-shot inference: {e}")
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)
    else:
        # Daemon looping mode
        log_msg("Entering live inference daemon polling loop...")
        last_bar_time = None
        consecutive_errors = 0
        
        while True:
            try:
                # Fetch bars and compare latest timestamp to skip redundant inferences
                df = fetch_live_bars(count=count)
                latest_bar_time = df.iloc[-1]["time"]
                
                if last_bar_time is None or latest_bar_time != last_bar_time:
                    if last_bar_time is not None:
                        log_msg(f"New bar detected closing at {latest_bar_time}. Running inference...")
                    else:
                        log_msg(f"Initial inference loop starting. Current bar closed at {latest_bar_time}.")
                        
                    output = run_inference(
                        df=df,
                        model=model,
                        tokenizer=tokenizer,
                        fe_config=fe_config,
                        feature_cols=feature_cols,
                        device=device,
                        policy_params=policy_params
                    )
                    
                    # Print clean single-line JSONL strictly to stdout
                    ORIGINAL_PRINT(json.dumps(output))
                    last_bar_time = latest_bar_time
                    consecutive_errors = 0
                
            except Exception as e:
                consecutive_errors += 1
                backoff_time = min(args.interval * (2 ** (consecutive_errors - 1)), 60.0)
                log_msg(f"ERROR: [Attempt {consecutive_errors}] Inference cycle failed: {e}")
                log_msg(f"Retrying in {backoff_time:.1f} seconds...")
                time.sleep(backoff_time)
                continue
                
            time.sleep(args.interval)

if __name__ == "__main__":
    main()
