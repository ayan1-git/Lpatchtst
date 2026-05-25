# audit_all_folds.py
import pandas as pd, numpy as np, sys, os
sys.path.insert(0, os.getcwd())
import config
from oracle import generate_targets

f = config.DATA_FILE[0]
df = pd.read_csv(f)
time_col = next((c for c in df.columns if c.lower() in ("date","datetime")), None)
if time_col:
    df[time_col] = pd.to_datetime(df[time_col]); df = df.set_index(time_col)
df = df.sort_index()

hl = df["high"] - df["low"]
hc = (df["high"] - df["close"].shift()).abs()
lc = (df["low"]  - df["close"].shift()).abs()
df["atr"] = pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(config.ATR_PERIOD).mean()
df.dropna(inplace=True)

targets = generate_targets(
    df["open"].values, df["high"].values, df["low"].values,
    df["close"].values, df["atr"].values,
    max_hold=config.ORACLE_MAX_HOLD, fee_per_side=config.FEE_PER_SIDE,
    slippage=config.SLIPPAGE, sl_atr_mult=config.ORACLE_SL_ATR_MULT,
    tp_atr_mult=config.ORACLE_TP_ATR_MULT, saturation_factor=config.SATURATION_FACTOR,
    mae_penalty=config.MAE_PENALTY)
valid_len = len(targets) - config.ORACLE_MAX_HOLD
targets = targets[:valid_len]

N = valid_len
train_end = config.WFV_TRAIN_BARS
GAP = config.LOOKBACK_WINDOW
VAL = config.WFV_VAL_BARS
STEP = config.WFV_STEP_BARS

print(f"\n{'Fold':<6} {'Train End':<12} {'Val Start':<12} {'Val End':<12} "
      f"{'Tr Mean':>8} {'Va Mean':>8} {'Tr Std':>8} {'Va Std':>8} "
      f"{'Tr Zero%':>9} {'Va Zero%':>9} {'Val Date Range'}")
print("-" * 130)

fold_id = 0
while True:
    val_start = train_end + GAP
    val_end   = val_start + VAL
    if val_end > N: break

    tr = targets[:train_end]
    va = targets[val_start:val_end]
    
    val_dates = ""
    if isinstance(df.index, pd.DatetimeIndex):
        vs_date = df.index[min(val_start, len(df.index)-1)].strftime("%Y-%m")
        ve_date = df.index[min(val_end-1, len(df.index)-1)].strftime("%Y-%m")
        val_dates = f"{vs_date} → {ve_date}"

    print(f"{fold_id:<6} {train_end:<12} {val_start:<12} {val_end:<12} "
          f"{tr.mean():>8.4f} {va.mean():>8.4f} {tr.std():>8.4f} {va.std():>8.4f} "
          f"{(np.abs(tr)<0.1).mean()*100:>8.1f}% {(np.abs(va)<0.1).mean()*100:>8.1f}%  {val_dates}")
    
    fold_id  += 1
    train_end += STEP