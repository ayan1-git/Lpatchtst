# audit_leakage.py
import pandas as pd, numpy as np, glob, os, sys
sys.path.insert(0, os.getcwd())
import config
from features import FeatureEngineer, FeatureConfig

fe = FeatureEngineer(FeatureConfig(
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
    use_talib=False,
))

f = config.DATA_FILE[0]
df = pd.read_csv(f)
time_col = next((c for c in df.columns if c.lower() in ("date","datetime")), None)
if time_col:
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)
df = df.sort_index()

OHLC = ["open","high","low","close","volume"]
feat_df = fe.build(df["close"], ohlc=df[OHLC], include_target=False, dropna=False)
df = df.join(feat_df, how="inner").dropna()

# Future return at t+1 and t+5 (what the oracle is predicting)
fut1 = df["close"].pct_change().shift(-1)
fut5 = df["close"].pct_change(5).shift(-5)

print(f"\n{'Col':<35} {'corr_fut1':>10} {'corr_fut5':>10}  FLAG")
print("-" * 65)
suspicious = []
for col in feat_df.columns:
    if col not in df.columns:
        continue
    c1 = df[col].corr(fut1)
    c5 = df[col].corr(fut5)
    flag = "*** LEAKAGE ***" if abs(c1) > 0.10 or abs(c5) > 0.10 else ""
    if flag:
        suspicious.append((col, c1, c5))
    print(f"{col:<35} {c1:>10.4f} {c5:>10.4f}  {flag}")

print(f"\nSuspicious features (|corr| > 0.10): {len(suspicious)}")
for col, c1, c5 in suspicious:
    print(f"  {col}: fut1={c1:.4f}, fut5={c5:.4f}")