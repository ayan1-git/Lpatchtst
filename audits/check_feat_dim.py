import pandas as pd
import numpy as np
from features import FeatureEngineer, FeatureConfig
import config as CFG

# Mock data
N = 1000
idx = pd.date_range("2020-01-01", periods=N, freq="30min", tz="Asia/Kolkata")
close = np.cumsum(np.random.randn(N) * 0.1) + 100
df = pd.DataFrame({
    "open": close + np.random.randn(N) * 0.01,
    "high": close + np.abs(np.random.randn(N) * 0.05),
    "low": close - np.abs(np.random.randn(N) * 0.05),
    "close": close,
    "volume": np.random.randn(N) + 1000
}, index=idx)

cfg = FeatureConfig(
    ewma_span=CFG.FE_VOL_LONG_PERIOD,
    return_horizons=CFG.FE_RETURN_HORIZONS,
    macd_pairs=CFG.FE_MACD_PAIRS,
    macd_price_std_window=CFG.FE_MACD_PRICE_STD_WIN,
    macd_signal_std_window=CFG.FE_MACD_SIGNAL_STD_WIN,
    momentum_period=CFG.FE_MOMENTUM_PERIOD,
    rsi_period=CFG.FE_RSI_PERIOD,
    vol_asym_window=CFG.FE_VOL_ASYM_WINDOW,
    icp_period=CFG.FE_ICP_PERIOD,
    local_structure_bars=CFG.FE_LOCAL_STRUCTURE_BARS,
    vol_squeeze_fast=CFG.FE_VOL_SQUEEZE_FAST,
    vol_squeeze_slow=CFG.FE_VOL_SQUEEZE_SLOW,
    session_open=CFG.FE_SESSION_OPEN,
    session_close=CFG.FE_SESSION_CLOSE,
    session_tz=CFG.FE_SESSION_TZ,
    add_session_features=CFG.FE_ADD_SESSION,
)

fe = FeatureEngineer(config=cfg)
feats = fe.build(df["close"], ohlc=df, include_target=False, dropna=False)
print(f"Feature shape: {feats.shape}")
print(f"Columns: {feats.columns.tolist()}")
