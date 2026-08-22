
import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Mocking the environment to match audit_training.py
REPO_ROOT = Path("/home/ayanmarvin124/Lpatchtst")
sys.path.insert(0, str(REPO_ROOT))

try:
    import config as CFG
    from data_loader import tokenize_full_series
    from oracle import generate_targets
    from tokenizer import prepare_ohlc_features, KronosTokenizer
    from features import FeatureEngineer, FeatureConfig
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

OHLC_COLS = ["open", "high", "low", "close", "volume"]

def _load_fe():
    return FeatureEngineer(FeatureConfig(
        ewma_span             = CFG.FE_VOL_LONG_PERIOD,
        return_horizons       = CFG.FE_RETURN_HORIZONS,
        macd_pairs            = CFG.FE_MACD_PAIRS,
        macd_price_std_window = CFG.FE_MACD_PRICE_STD_WIN,
        macd_signal_std_window= CFG.FE_MACD_SIGNAL_STD_WIN,
        target_clip           = CFG.FE_TARGET_CLIP,
        momentum_period       = CFG.FE_MOMENTUM_PERIOD,
        rsi_period            = CFG.FE_RSI_PERIOD,
        vol_asym_window       = CFG.FE_VOL_ASYM_WINDOW,
        icp_period            = CFG.FE_ICP_PERIOD,
        local_structure_bars  = CFG.FE_LOCAL_STRUCTURE_BARS,
        vol_squeeze_fast      = CFG.FE_VOL_SQUEEZE_FAST,
        vol_squeeze_slow      = CFG.FE_VOL_SQUEEZE_SLOW,
        atr_period            = CFG.ATR_PERIOD,
        session_open          = CFG.FE_SESSION_OPEN,
        session_close         = CFG.FE_SESSION_CLOSE,
        session_tz            = CFG.FE_SESSION_TZ,
        add_session_features  = CFG.FE_ADD_SESSION,
        use_talib             = getattr(CFG, "USE_TALIB", False),
    ))

def _build_one_asset_with_filter(csv_path: Path, fe):
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.lower().strip() for c in df.columns]
        time_col = next((c for c in df.columns if c in ("date","datetime","time")), None)
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col).sort_index()
            # THE FILTER
            df = df[(df.index >= '2015-01-09') & (df.index <= '2019-12-12')]
        
        print(f"DF size after filter: {len(df)}")
        
        feat_df = fe.build(df["close"], ohlc=df[OHLC_COLS], include_target=False, dropna=False)
        combined = df.join(feat_df, how="inner")
        hl = combined["high"] - combined["low"]
        hc = (combined["high"] - combined["close"].shift()).abs()
        lc = (combined["low"]  - combined["close"].shift()).abs()
        combined["atr"] = pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(CFG.ATR_PERIOD).mean()
        combined.dropna(inplace=True)

        targets = generate_targets(
            combined["open"].values, combined["high"].values,
            combined["low"].values,  combined["close"].values,
            combined["atr"].values,
            max_hold          = CFG.ORACLE_MAX_HOLD,
            fee_per_side      = CFG.FEE_PER_SIDE,
            slippage          = CFG.SLIPPAGE,
            sl_atr_mult       = CFG.ORACLE_SL_ATR_MULT,
            tp_atr_mult       = CFG.ORACLE_TP_ATR_MULT,
            saturation_factor = CFG.SATURATION_FACTOR,
            mae_penalty       = CFG.MAE_PENALTY,
        )
        return True
    except Exception as e:
        print(f"Failed with filter: {e}")
        return False

def _build_one_asset_no_filter(csv_path: Path, fe):
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.lower().strip() for c in df.columns]
        time_col = next((c for c in df.columns if c in ("date","datetime","time")), None)
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col).sort_index()
            # NO FILTER
        
        print(f"DF size without filter: {len(df)}")
        
        feat_df = fe.build(df["close"], ohlc=df[OHLC_COLS], include_target=False, dropna=False)
        combined = df.join(feat_df, how="inner")
        hl = combined["high"] - combined["low"]
        hc = (combined["high"] - combined["close"].shift()).abs()
        lc = (combined["low"]  - combined["close"].shift()).abs()
        combined["atr"] = pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(CFG.ATR_PERIOD).mean()
        combined.dropna(inplace=True)

        targets = generate_targets(
            combined["open"].values, combined["high"].values,
            combined["low"].values,  combined["close"].values,
            combined["atr"].values,
            max_hold          = CFG.ORACLE_MAX_HOLD,
            fee_per_side      = CFG.FEE_PER_SIDE,
            slippage          = CFG.SLIPPAGE,
            sl_atr_mult       = CFG.ORACLE_SL_ATR_MULT,
            tp_atr_mult       = CFG.ORACLE_TP_ATR_MULT,
            saturation_factor = CFG.SATURATION_FACTOR,
            mae_penalty       = CFG.MAE_PENALTY,
        )
        return True
    except Exception as e:
        print(f"Failed without filter: {e}")
        return False

    csv_path = Path("/home/ayanmarvin124/Lpatchtst/data/20MICRONS_complete_data.csv")
fe = _load_fe()

print("Testing with filter...")
success_with = _build_one_asset_with_filter(csv_path, fe)
print(f"Success with filter: {success_with}")

print("\nTesting without filter...")
success_without = _build_one_asset_no_filter(csv_path, fe)
print(f"Success without filter: {success_without}")
