# evaluate.py  (Production — features.py fully integrated)

from __future__ import annotations

import os
import json
import itertools
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from data_loader import create_dataloaders
from model import PatchTST
from oracle import generate_targets
from backtest_engine import backtest_one_position

# ── features.py public API ────────────────────────────────────────────────────
from features import FeatureConfig, FeatureEngineer

import config


MODEL_PATH = "best_model_patchtst.pth"


# ─────────────────────────────────────────────────────────────────────────────
# FeatureEngineer bootstrap  (mirrors train.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

def _make_feature_config() -> FeatureConfig:
    """Map config.py hyperparameters → FeatureConfig.
    Must stay in sync with train.py._make_feature_config().
    """
    return FeatureConfig(
        ewma_span=config.FE_VOL_LONG_PERIOD,
        return_horizons=config.FE_RETURN_HORIZONS,
        macd_pairs=config.FE_MACD_PAIRS,
        macd_price_std_window=config.FE_MACD_PRICE_STD_WIN,
        macd_signal_std_window=config.FE_MACD_SIGNAL_STD_WIN,
        target_clip=config.FE_TARGET_CLIP,
    )


def _build_feature_cols(
    fe_config: FeatureConfig,
) -> Tuple[list[str], list[str], list[str]]:
    """Derive scaler-routing column lists from FeatureConfig.
    Must stay in sync with train.py._build_feature_cols().

    Returns
    -------
    no_scale_cols, robust_cols, all_feat_cols
    """
    no_scale_cols: list[str] = []
    robust_cols:   list[str] = []

    no_scale_cols.append(f"ewma_vol_span{fe_config.ewma_span}")
    for h in fe_config.return_horizons:
        no_scale_cols.append(f"ret_norm_{h}d")
    for s, l in fe_config.macd_pairs:
        no_scale_cols.append(f"macd_{s}_{l}")
    robust_cols.append(f"vs_factor_span{fe_config.ewma_span}")

    all_feat_cols = robust_cols + no_scale_cols
    return no_scale_cols, robust_cols, all_feat_cols


def _build_features(
    df: pd.DataFrame,
    fe: FeatureEngineer,
) -> Tuple[pd.DataFrame, list[str]]:
    """Apply FeatureEngineer to a raw OHLC DataFrame.

    Identical contract to train.py._build_features():
    - include_target=False  (Oracle targets are used, not next-day return)
    - dropna=False          (global dropna after ATR is added)
    - OHLC kept in combined_df for Oracle/backtest but NOT in feature_cols

    Returns
    -------
    combined_df  : OHLC + engineered features + ATR, NaN rows dropped.
    feature_cols : ordered model-input column list (no OHLC).
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
    feat_df = fe.build(df["close"], include_target=False, dropna=False)

    # 3. Join OHLC + features ──────────────────────────────────────────────────
    combined_df = df.join(feat_df, how="inner")

    # 4. ATR (Oracle + backtest engine) ────────────────────────────────────────
    high_low   = combined_df["high"] - combined_df["low"]
    high_close = (combined_df["high"] - combined_df["close"].shift()).abs()
    low_close  = (combined_df["low"]  - combined_df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    combined_df["atr"] = true_range.rolling(config.ATR_PERIOD).mean()
    combined_df.dropna(inplace=True)

    # 5. Feature column list (engineered only, no OHLC) ───────────────────────
    _, _, all_feat_cols = _build_feature_cols(fe.config)
    return combined_df, all_feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# Split geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_split_indices(
    total_len: int, cfg
) -> Tuple[int, int, int, int]:
    """Return (train_end, val_start, val_end, test_start)."""
    gap       = cfg.FORECAST_HORIZON + 50
    train_end = int(total_len * cfg.TRAIN_RATIO)
    val_start = train_end  + gap
    val_end   = val_start  + int(total_len * cfg.VAL_RATIO)
    test_start = val_end   + gap
    return train_end, val_start, val_end, test_start


def expected_num_windows(split_start: int, split_end: int, seq_len: int) -> int:
    return max(0, (split_end - split_start) - seq_len + 1)


# ─────────────────────────────────────────────────────────────────────────────
# Model helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_model(aggregation: str, num_features: int) -> PatchTST:
    return PatchTST(
        seq_len=config.LOOKBACK_WINDOW,
        num_features=num_features,
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


def _load_model(device: torch.device, num_features: int) -> PatchTST:
    """Load checkpoint with aggregation-mode fallback.

    Tries config.AGGREGATION_MODE first, then the other mode, so a
    checkpoint saved under a different mode is still usable.
    """
    state     = torch.load(MODEL_PATH, map_location=device)
    preferred = config.AGGREGATION_MODE
    fallback  = "mean" if preferred == "mixing" else "mixing"

    for agg in (preferred, fallback):
        model = _build_model(agg, num_features).to(device)
        try:
            model.load_state_dict(state, strict=True)
            print(f"Model loaded: {MODEL_PATH}  (aggregation='{agg}')")
            return model
        except RuntimeError as e:
            print(f"Load failed for aggregation='{agg}': {e}")

    raise RuntimeError(
        f"Could not load {MODEL_PATH} with either aggregation mode."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    use_amp   = config.USE_AMP and device.type == "cuda"
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    preds: list[np.ndarray] = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                p = model(x).detach().cpu().numpy().reshape(-1)
            preds.append(p)

    if not preds:
        return np.array([], dtype=np.float32)

    raw = np.concatenate(preds).astype(np.float32)

    if config.INFERENCE_SMOOTHING > 1:
        return (
            pd.Series(raw)
            .rolling(window=config.INFERENCE_SMOOTHING, min_periods=1)
            .mean()
            .to_numpy(dtype=np.float32)
        )
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Signal + backtest helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_signals(
    preds: np.ndarray, threshold: float, bias: float
) -> np.ndarray:
    adj     = preds + bias
    signals = np.zeros(len(adj), dtype=np.int8)
    signals[adj >  threshold] =  1
    signals[adj < -threshold] = -1
    return signals


def get_metrics(pnl: np.ndarray, executed_mask: np.ndarray) -> dict:
    active = pnl[executed_mask]
    if active.size == 0:
        return {
            "profit_factor": 0.0, "net_return": 0.0,
            "net_return_compounded": 0.0, "net_return_additive": 0.0,
            "avg_return_per_trade": 0.0, "num_trades": 0,
            "gross_profit": 0.0, "gross_loss": 0.0, "win_rate": 0.0,
        }

    winners = active[active > 0]
    losers  = active[active < 0]
    gp = float(winners.sum()) if winners.size else 0.0
    gl = float(-losers.sum()) if losers.size else 0.0
    pf = (float("inf") if gp > 0.0 else 0.0) if gl == 0.0 else gp / gl

    compounded = float(np.prod(1 + active) - 1)
    return {
        "profit_factor":         float(pf),
        "net_return":            compounded,
        "net_return_compounded": compounded,
        "net_return_additive":   float(active.sum()),
        "avg_return_per_trade":  float(active.mean()),
        "num_trades":            int(active.size),
        "gross_profit":          float(gp),
        "gross_loss":            float(gl),
        "win_rate":              float((active > 0).mean()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Policy tuning (val grid search)
# ─────────────────────────────────────────────────────────────────────────────

def tune_policy_on_val(
    preds_val: np.ndarray,
    ohlc: dict,
    first_signal_bar_idx: int,
    cfg,
) -> Tuple[float, float, dict]:
    """Grid-search threshold × bias on the validation split.

    Returns
    -------
    best_threshold, best_bias, val_metrics
    """
    thresholds = np.round(np.arange(0.05, 0.51, 0.05), 2)
    biases     = np.round(np.arange(-0.20, 0.21, 0.05), 2)
    best       = None

    print("\n" + "=" * 60)
    print("VALIDATION POLICY TUNING — Grid Search")
    print("=" * 60)

    for th, b in itertools.product(thresholds, biases):
        signals = make_signals(preds_val, th, b)
        pnl, executed_mask, _, _ = backtest_one_position(
            signals,
            ohlc["open"], ohlc["high"], ohlc["low"],
            ohlc["close"], ohlc["atr"],
            first_signal_bar_idx=first_signal_bar_idx,
            max_hold=cfg.ORACLE_MAX_HOLD,
            fee=cfg.FEE_PER_SIDE,
            slippage=cfg.SLIPPAGE,
            atr_mult=cfg.ATR_MULT,
        )
        m = get_metrics(pnl, executed_mask)

        if m["num_trades"] < cfg.MIN_TRADES_TUNE:
            continue
        if m["net_return"] <= -0.05:
            continue

        candidate = (m["net_return"], m["profit_factor"], m["num_trades"], th, b, m)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
            print(
                f"✓  th={th:.2f}  bias={b:+.2f}  |  "
                f"PF={m['profit_factor']:.2f}  "
                f"Net={m['net_return']:.4f}  "
                f"Trades={m['num_trades']}  "
                f"WR={m['win_rate']*100:.1f}%"
            )

    if best is None:
        print("No valid policy found — using fallback (th=0.20, bias=0.0).")
        return 0.20, 0.0, {
            "profit_factor": 0.0, "net_return": 0.0, "num_trades": 0,
            "gross_profit": 0.0, "gross_loss": 0.0, "win_rate": 0.0,
            "note": "No valid policy met constraints",
        }

    return best[3], best[4], best[5]


# ─────────────────────────────────────────────────────────────────────────────
# JSON serialiser (handles inf / nan from profit_factor)
# ─────────────────────────────────────────────────────────────────────────────

def _json_serial(obj):
    if isinstance(obj, float):
        if np.isinf(obj): return "inf"
        if np.isnan(obj): return "nan"
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def evaluate() -> None:
    # ── 1. Resolve data file ──────────────────────────────────────────────────
    data_files = config.DATA_FILE
    if isinstance(data_files, list):
        if not data_files:
            raise ValueError("config.DATA_FILE list is empty.")
        data_path = data_files[0]
        if len(data_files) > 1:
            print(f"Note: evaluating first file only: {data_path}")
    else:
        data_path = data_files

    print(f"Loading data from {data_path}…")
    df_raw = pd.read_csv(data_path)

    # ── 2. Build FeatureEngineer (identical config to train.py) ───────────────
    fe_config = _make_feature_config()
    fe        = FeatureEngineer(config=fe_config)
    print(
        f"FeatureEngineer | ewma_span={fe_config.ewma_span} | "
        f"horizons={fe_config.return_horizons} | "
        f"macd_pairs={fe_config.macd_pairs}"
    )

    df, feature_cols = _build_features(df_raw, fe)
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    # ── 3. Oracle targets ─────────────────────────────────────────────────────
    print("Generating Oracle targets…")
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
        raise ValueError(f"Dataset too short after Oracle trim: valid_len={valid_len}")

    df      = df.iloc[:valid_len].copy()
    targets = targets[:valid_len]

    # ── 4. Feature array + config.NUM_FEATURES ────────────────────────────────
    features = df[feature_cols].values.astype(np.float32)
    config.NUM_FEATURES = 1 if config.USE_TOKENIZER else len(feature_cols)

    # ── 5. Optional tokenizer ─────────────────────────────────────────────────
    tokenizer = None
    if config.USE_TOKENIZER:
        from tokenizer import KLineTokenizer
        print(f"Initializing KLineTokenizer (bits={config.TOKENIZER_BITS})…")
        tokenizer = KLineTokenizer(
            input_dim=len(feature_cols),   # ← dynamic, not hardcoded 21
            n_bits=config.TOKENIZER_BITS,
        )
        if os.path.exists("tokenizer.pth"):
            tokenizer.load_state_dict(
                torch.load("tokenizer.pth", map_location="cpu")
            )
            print("Pre-trained tokenizer weights loaded.")
        else:
            print("Warning: tokenizer.pth not found — inference may be inconsistent.")

    # ── 6. DataLoaders ────────────────────────────────────────────────────────
    # feature_cols forwarded so ColumnSelectiveScaler routes each column
    # to the correct bucket (NO_SCALE vs ROBUST) — same as training.
    _, val_loader, test_loader = create_dataloaders(
        features, targets, config,
        feature_cols=feature_cols,
        tokenizer=tokenizer,
    )

    # ── 7. Load model ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = _load_model(device, num_features=config.NUM_FEATURES)

    # ── 8. OHLC dict for backtest ─────────────────────────────────────────────
    ohlc = {
        "open":  df["open"].values,
        "high":  df["high"].values,
        "low":   df["low"].values,
        "close": df["close"].values,
        "atr":   df["atr"].values,
    }

    # ── 9. Split geometry ─────────────────────────────────────────────────────
    total_len = len(df)
    train_end, val_start, val_end, test_start = compute_split_indices(
        total_len, config
    )
    seq = config.LOOKBACK_WINDOW

    exp_val  = expected_num_windows(val_start,  val_end,   seq)
    exp_test = expected_num_windows(test_start, total_len, seq)

    # ── 10. Inference ─────────────────────────────────────────────────────────
    preds_val  = run_inference(model, val_loader,  device)
    preds_test = run_inference(model, test_loader, device)

    if len(preds_val) != exp_val:
        print(f"Warning: val preds {len(preds_val)} ≠ expected {exp_val}")
    if len(preds_test) != exp_test:
        print(f"Warning: test preds {len(preds_test)} ≠ expected {exp_test}")

    first_val_bar  = val_start  + seq - 1
    first_test_bar = test_start + seq - 1

    # ── 11. Policy tuning on val ──────────────────────────────────────────────
    print("\n--- Tuning policy on validation split ---")
    best_th, best_bias, val_metrics = tune_policy_on_val(
        preds_val, ohlc, first_val_bar, config
    )

    policy = {"threshold": best_th, "bias": best_bias, "val_metrics": val_metrics}
    with open("best_policy.json", "w") as f:
        json.dump(policy, f, indent=2, default=_json_serial)
    print(f"Policy saved → best_policy.json  (th={best_th}, bias={best_bias:+.2f})")

    # ── 12. Final test evaluation ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    signals_test = make_signals(preds_test, best_th, best_bias)
    pnl_test, executed_mask_test, skipped, stopped_early = backtest_one_position(
        signals_test,
        ohlc["open"], ohlc["high"], ohlc["low"],
        ohlc["close"], ohlc["atr"],
        first_signal_bar_idx=first_test_bar,
        max_hold=config.ORACLE_MAX_HOLD,
        fee=config.FEE_PER_SIDE,
        slippage=config.SLIPPAGE,
        atr_mult=config.ATR_MULT,
    )

    test_metrics = get_metrics(pnl_test, executed_mask_test)

    # ── 13. Print results ─────────────────────────────────────────────────────
    print("\nExecution:")
    print(f"  Signals generated       : {int(np.count_nonzero(signals_test))}")
    print(f"  Trades executed         : {test_metrics['num_trades']}")
    print(f"  Skipped (pos. open)     : {skipped}")
    print(f"  Skipped (insuff. bars)  : {stopped_early}")

    print("\nPerformance:")
    print(f"  Profit Factor           : {test_metrics['profit_factor']:.3f}")
    print(f"  Net Return (compounded) : {test_metrics['net_return_compounded']:.4f}")
    print(f"  Net Return (additive)   : {test_metrics['net_return_additive']:.4f}")
    print(f"  Avg Return / Trade      : {test_metrics['avg_return_per_trade']:.4f}")
    print(f"  Win Rate                : {test_metrics['win_rate']:.1%}")

    # ── 14. Export ────────────────────────────────────────────────────────────
    pd.DataFrame({
        "Prediction":   preds_test,
        "Signal":       signals_test,
        "Target_PnL":   targets[first_test_bar:],
        "Strategy_PnL": pnl_test,
    }).to_csv("backtest_results.csv", index=False)
    print("\nbacktest_results.csv saved.")

    with open("test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2, default=_json_serial)
    print("test_metrics.json saved.")


if __name__ == "__main__":
    evaluate()