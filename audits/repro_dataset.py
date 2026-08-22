import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from Kronos_finetune.features import FeatureEngineer, FeatureConfig

# Mock Config
class MockConfig:
    def __init__(self):
        self.seed = 42
        self.dataset_path = "./data/processed_datasets"
        self.n_train_iter = 1000
        self.n_val_iter = 100
        self.train_time_range = [datetime(2000, 1, 1), datetime(2025, 1, 1)]
        self.val_time_range = [datetime(2025, 1, 1), datetime(2026, 1, 1)]
        self.lookback_window = 63
        self.predict_window = 5
        self.feature_list = ["ewma_vol_span260", "ret_norm_1d", "macd_8_24"]
        self.time_feature_list = ["minute", "hour", "weekday", "day", "month"]

def test_dataset_logic():
    config = MockConfig()
    window = config.lookback_window + config.predict_window + 1
    
    # Create dummy data for 2 symbols
    symbols = ["AAPL", "MSFT"]
    data = {}
    for sym in symbols:
        idx = pd.date_range("2020-01-01", periods=5000, freq="30min")
        close = np.abs(np.random.randn(5000).cumsum()) + 100
        high = close + np.abs(np.random.randn(5000))
        low = close - np.abs(np.random.randn(5000))
        open_p = close + np.random.randn(5000) * 0.5
        df = pd.DataFrame({
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
        }, index=idx)
        df.index.name = "datetime"
        data[sym] = df

    fe = FeatureEngineer()
    indices = []
    
    # Mimic the problematic loop in dataset.py
    num_samples = 0
    last_symbol = ""
    for symbol in symbols:
        df = data[symbol]
        # Line 115: Cut off first 3276 bars
        df = df.iloc[3276:]
        
        series_len = len(df)
        num_samples = series_len - window + 1
        
        prices = df['close']
        ohlc = df[['open', 'high', 'low', 'close']]
        eng_feats = fe.build(prices, ohlc=ohlc, include_target=False, dropna=False)
        df = df.join(eng_feats)
        
        # Time features
        df['minute'] = df.index.minute
        df['hour'] = df.index.hour
        df['weekday'] = df.index.weekday
        df['day'] = df.index.day
        df['month'] = df.index.month
        
        data[symbol] = df[config.feature_list + config.time_feature_list]
        last_symbol = symbol

    # Line 148: Loop OUTSIDE the symbol loop
    for i in range(num_samples):
        indices.append((last_symbol, i))

    print(f"Total indices: {len(indices)}")
    print(f"Symbols in indices: {set([s for s, i in indices])}")
    
    # Check for NaNs in the first sample of the first symbol (if it were there)
    # Since only last_symbol is there, let's check last_symbol's first sample
    sym = last_symbol
    df = data[sym]
    first_sample = df.iloc[0:window]
    nan_count = first_sample.isna().sum().sum()
    print(f"NaNs in first sample of {sym}: {nan_count}")

if __name__ == "__main__":
    test_dataset_logic()
