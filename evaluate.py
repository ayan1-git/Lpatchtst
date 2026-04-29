# evaluate.py (Production v3: Parameterized Oracle + V2 Backtest)

from __future__ import annotations

import os
import json
import itertools
import numpy as np
import pandas as pd
import torch

from data_loader import create_dataloaders
from model import PatchTST
from tokenizer import KLineTokenizer
from oracle import generate_targets
from backtest_engine import backtest_one_position
from features import calculate_features, PASSTHROUGH_FEATURES, SCALE_FEATURES
import config


MODEL_PATH = "best_model_patchtst.pth"
DATA_FILE = config.DATA_FILE


def _get_aggregation_mode() -> str:
    # Use config-defined mode
    return config.AGGREGATION_MODE


def _build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    # Set datetime index
    # Set datetime index
    time_col = next((c for c in df.columns if c.lower() in ["date", "datetime"]), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)
    else:
        # Try to parse index if it's already strings
        try:
            df.index = pd.to_datetime(df.index)
        except:
            pass
    
    # Generate engineered features
    feat_df = calculate_features(df)
    
    # Merge engineered features back to original df to keep OHLC
    combined_df = df.join(feat_df, how="inner")
    
    # Calculate ATR (needed for oracle and backtesting)
    high_low = combined_df["high"] - combined_df["low"]
    high_close = np.abs(combined_df["high"] - combined_df["close"].shift())
    low_close = np.abs(combined_df["low"] - combined_df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    combined_df["atr"] = true_range.rolling(config.ATR_PERIOD).mean()
    combined_df.dropna(inplace=True)
    
    # Inclusion contract: OHLC + Engineered features
    ohlc_cols = ["open", "high", "low", "close"]
    feature_cols = ohlc_cols + PASSTHROUGH_FEATURES + SCALE_FEATURES
    
    return combined_df, feature_cols


def compute_split_indices(total_len: int, cfg) -> tuple[int, int, int, int]:
    gap = cfg.FORECAST_HORIZON + 50
    train_end = int(total_len * cfg.TRAIN_RATIO)
    val_start = train_end + gap
    val_end = val_start + int(total_len * cfg.VAL_RATIO)
    test_start = val_end + gap
    return train_end, val_start, val_end, test_start


def expected_num_windows(split_start: int, split_end: int, seq_len: int) -> int:
    span = split_end - split_start
    return span - seq_len + 1


def run_inference(model: torch.nn.Module, loader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    use_amp = config.USE_AMP and device.type == "cuda"
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                p = model(x).detach().cpu().numpy().reshape(-1)
            preds.append(p)
    if not preds:
        return np.array([], dtype=np.float32)
    
    raw_preds = np.concatenate(preds).astype(np.float32)
    
    # Apply Signal Smoothing to reduce jitter
    if config.INFERENCE_SMOOTHING > 1:
        return pd.Series(raw_preds).rolling(window=config.INFERENCE_SMOOTHING, min_periods=1).mean().values
    
    return raw_preds
 

def make_signals(preds: np.ndarray, threshold: float, bias: float) -> np.ndarray:
    adj = preds + bias
    signals = np.zeros(adj.shape[0], dtype=np.int8)
    signals[adj > threshold] = 1
    signals[adj < -threshold] = -1
    return signals


def get_pf_and_return(pnl: np.ndarray, executed_mask: np.ndarray) -> dict:
    active = pnl[executed_mask]
    if active.size == 0:
        return {
            "profit_factor": 0.0,
            "net_return": 0.0,
            "num_trades": 0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "win_rate": 0.0,
        }

    winners = active[active > 0]
    losers = active[active < 0]
    gp = float(winners.sum()) if winners.size else 0.0
    gl = float(-losers.sum()) if losers.size else 0.0

    if gl == 0.0:
        pf = float("inf") if gp > 0.0 else 0.0
    else:
        pf = gp / gl

    win_rate = float((active > 0).mean())
    compounded = np.prod(1 + active) - 1

    return {
        "profit_factor": float(pf),
        "net_return": float(compounded),  # Compounded by default
        "net_return_compounded": float(compounded),
        "net_return_additive": float(active.sum()),
        "avg_return_per_trade": float(active.mean()),
        "num_trades": int(active.size),
        "gross_profit": float(gp),
        "gross_loss": float(gl),
        "win_rate": float(win_rate),
    }
 

def tune_policy_on_val(preds_val: np.ndarray, ohlc: dict, first_signal_bar_idx: int, cfg) -> tuple[float, float, dict]:
    thresholds = np.round(np.arange(0.05, 0.51, 0.05), 2)  # 0.05..0.50
    biases = np.round(np.arange(-0.20, 0.21, 0.05), 2)      # -0.20..+0.20

    best = None

    print("\n" + "=" * 60)
    print("VALIDATION POLICY TUNING - Grid Search")
    print("=" * 60)

    for th, b in itertools.product(thresholds, biases):
        signals = make_signals(preds_val, th, b)

        pnl, executed_mask, skipped, stopped_early = backtest_one_position(
            signals,
            ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"], ohlc["atr"],
            first_signal_bar_idx=first_signal_bar_idx,
            max_hold=cfg.ORACLE_MAX_HOLD,
            fee=cfg.FEE_PER_SIDE,
            slippage=cfg.SLIPPAGE,
            atr_mult=cfg.ATR_MULT,
        )

        metrics = get_pf_and_return(pnl, executed_mask)

        if metrics["num_trades"] < cfg.MIN_TRADES_TUNE:
            continue
        if metrics["net_return"] <= -0.05:  # Allow 5% loss to see "almost" strategies
            continue

        candidate = (metrics["net_return"], metrics["profit_factor"], metrics["num_trades"], th, b, metrics)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
            print(
                f"✓ New best: th={th:.2f} bias={b:+.2f} | PF={metrics['profit_factor']:.2f} "
                f"Net={metrics['net_return']:.4f} Trades={metrics['num_trades']} WinRate={metrics['win_rate']*100:.1f}%"
            )

    if best is None:
        print("No valid policy found. Using fallback.")
        return 0.20, 0.0, {
            "profit_factor": 0.0,
            "net_return": 0.0,
            "num_trades": 0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "win_rate": 0.0,
            "note": "No valid policy met constraints",
        }

    th, b, metrics = best[3], best[4], best[5]
    return th, b, metrics


def _build_model(aggregation: str) -> PatchTST:
    return PatchTST(
        seq_len=config.LOOKBACK_WINDOW,
        num_features=config.NUM_FEATURES,
        patch_len=config.PATCH_LEN,
        stride=config.STRIDE,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        dropout=0.2,
        aggregation=aggregation,
        use_tokenizer=config.USE_TOKENIZER,
        vocab_size=config.VOCAB_SIZE,
    )


def _load_model_with_fallback(device: torch.device) -> PatchTST:
    state = torch.load(MODEL_PATH, map_location=device)

    preferred = _get_aggregation_mode()
    fallback = "mean" if preferred == "mixing" else "mixing"

    for agg in (preferred, fallback):
        model = _build_model(agg).to(device)
        try:
            model.load_state_dict(state, strict=True)
            print(f"Model loaded successfully: {MODEL_PATH} (aggregation='{agg}')")
            return model
        except RuntimeError as e:
            print(f"Load failed for aggregation='{agg}': {e}")

    raise RuntimeError(
        f"Could not load checkpoint {MODEL_PATH} with either aggregation mode."
    )


def evaluate() -> None:
    # Handle list of files (e.g. from config.DATA_FILE)
    if isinstance(DATA_FILE, list):
        if not DATA_FILE:
            raise ValueError("DATA_FILE list is empty in config.py")
        data_path = DATA_FILE[0]
        if len(DATA_FILE) > 1:
            print(f"Note: Multiple files in DATA_FILE. Evaluating the first one: {data_path}")
    else:
        data_path = DATA_FILE

    print(f"Loading data from {data_path} for evaluation...")
    df = pd.read_csv(data_path)
    df, feature_cols = _build_features(df)

    print("Generating oracle targets...")
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
        raise ValueError(f"Dataset too short after trimming. valid_len={valid_len}")

    df = df.iloc[:valid_len].copy()
    targets = targets[:valid_len]

    # Selection of columns for data loader
    input_cols = PASSTHROUGH_FEATURES + SCALE_FEATURES if config.USE_TOKENIZER else feature_cols
    config.NUM_FEATURES = len(input_cols)

    features = df[input_cols].values.astype(np.float32)

    tokenizer = None
    if config.USE_TOKENIZER:
        print(f"Initializing Tokenizer (Hybrid Mode: 21 features, {config.TOKENIZER_BITS} bits)...")
        tokenizer = KLineTokenizer(input_dim=21, n_bits=config.TOKENIZER_BITS)
        if os.path.exists("tokenizer.pth"):
            print("Loading pre-trained tokenizer weights from tokenizer.pth...")
            tokenizer.load_state_dict(torch.load("tokenizer.pth", map_location="cpu"))
        else:
            print("Warning: No pre-trained tokenizer found. Inference may be inconsistent.")

    # Loaders
    _, val_loader, test_loader = create_dataloaders(features, targets, config, tokenizer=tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = _load_model_with_fallback(device)

    # Backtest setup
    ohlc = {
        "open": df["open"].values,
        "high": df["high"].values,
        "low": df["low"].values,
        "close": df["close"].values,
        "atr": df["atr"].values,
    }

    total_len = len(df)
    train_end, val_start, val_end, test_start = compute_split_indices(total_len, config)

    seq = config.LOOKBACK_WINDOW
    exp_val = expected_num_windows(val_start, val_end, seq)
    exp_test = expected_num_windows(test_start, total_len, seq)

    preds_val = run_inference(model, val_loader, device)
    preds_test = run_inference(model, test_loader, device)

    if len(preds_val) != exp_val:
        print(f"Warning: val preds length {len(preds_val)} != expected {exp_val}")
    if len(preds_test) != exp_test:
        print(f"Warning: test preds length {len(preds_test)} != expected {exp_test}")

    first_val_signal_bar = val_start + seq - 1
    first_test_signal_bar = test_start + seq - 1

    # Tune on validation
    print("\n--- Tuning policy on validation ---")
    best_th, best_bias, val_metrics = tune_policy_on_val(preds_val, ohlc, first_val_signal_bar, config)

    policy = {"threshold": best_th, "bias": best_bias, "val_metrics": val_metrics}

    def json_serial(obj):
        if np.isinf(obj):
            return "inf"
        if np.isnan(obj):
            return "nan"
        return obj

    with open("best_policy.json", "w") as f:
        json.dump(policy, f, indent=2, default=json_serial)

    # Final test
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    signals_test = make_signals(preds_test, best_th, best_bias)

    pnl_test, executed_mask_test, skipped_test, stopped_early_test = backtest_one_position(
        signals_test,
        ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"], ohlc["atr"],
        first_signal_bar_idx=first_test_signal_bar,
        max_hold=config.ORACLE_MAX_HOLD,
        fee=config.FEE_PER_SIDE,
        slippage=config.SLIPPAGE,
        atr_mult=config.ATR_MULT,
    )

    test_metrics = get_pf_and_return(pnl_test, executed_mask_test)

    print("\nExecution Analysis:")
    print(f" Signals generated: {int(np.count_nonzero(signals_test))}")
    print(f" Trades executed: {test_metrics['num_trades']}")
    print(f" Skipped (position open): {skipped_test}")
    print(f" Skipped (insufficient bars): {stopped_early_test}")

    print("\nPerformance Metrics:")
    print(f" Profit Factor: {test_metrics['profit_factor']:.3f}")
    print(f" Net Return (Compounded): {test_metrics['net_return_compounded']:.4f}")
    print(f" Net Return (Additive): {test_metrics['net_return_additive']:.4f}")
    print(f" Avg Return/Trade: {test_metrics['avg_return_per_trade']:.4f}")
    print(f" Win Rate: {test_metrics['win_rate']:.1%}")

    # Export test results to CSV
    results_df = pd.DataFrame({
        "Prediction": preds_test,
        "Signal": signals_test,
        "Target_PnL": targets[first_test_signal_bar:],
        "Strategy_PnL": pnl_test
    })
    results_df.to_csv("backtest_results.csv", index=False)
    print(f"\n[Fix 1] Regenerated backtest_results.csv from backtest_one_position()")

    # Save test metrics separately
    with open("test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2, default=json_serial)
    print(f"[Fix 2] Saved test metrics to test_metrics.json")


if __name__ == "__main__":
    evaluate()
