# prepare_nifty50_noklib.py
import os, pickle
import pandas as pd

OUTPUT_DIR = "./data/processed_datasets"
LOOKBACK, PREDICT = 90, 10

# --- Load your data however you have it ---
# Option A: one CSV per symbol (columns: datetime, open, high, low, close)
def load_from_csvs(data_dir):
    all_data = {}
    for f in os.listdir(data_dir):
        if not f.endswith(".csv"): continue
        sym = f.replace(".csv", "")
        df = pd.read_csv(f"{data_dir}/{f}")
        df["date"] = pd.to_datetime(df["date"])
        df = df.rename(columns={"date": "datetime"})
        df = df.set_index("datetime").sort_index()
        df = df.rename(columns={"volume": "vol"})
        df["amt"] = 0.0
        df = df[["open", "high", "low", "close", "vol", "amt"]].dropna()
        if len(df) >= LOOKBACK + PREDICT + 1:
            all_data[sym] = df
    return all_data

all_data = load_from_csvs("Data ")

# --- Split by date (overlap for lookback buffer) ---
splits = {
    "train": (None,         "2024-06-30"),
    "val":   ("2024-04-01", "2025-06-30"),  # overlap
    "test":  ("2025-04-01", None),           # overlap
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
for name, (start, end) in splits.items():
    split = {}
    for sym, df in all_data.items():
        mask = pd.Series(True, index=df.index)
        if start: mask &= df.index >= start
        if end:   mask &= df.index <= end
        sliced = df[mask]
        if len(sliced) >= LOOKBACK + PREDICT + 1:
            split[sym] = sliced
    with open(f"{OUTPUT_DIR}/{name}_data.pkl", "wb") as f:
        pickle.dump(split, f)
    print(f"{name}: {len(split)} symbols saved")