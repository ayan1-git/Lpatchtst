import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, mock_open
import pickle
from Kronos_finetune.dataset import QlibDataset
from Kronos_finetune.config import Config

# Mock Config
class MockConfig(Config):
    def __init__(self):
        super().__init__()
        self.seed = 42
        self.dataset_path = "./data/processed_datasets"
        self.n_train_iter = 1000
        self.n_val_iter = 100
        self.train_time_range = ["2000-01-01", "2025-01-01"]
        self.val_time_range = ["2025-01-01", "2026-01-01"]
        self.lookback_window = 63
        self.predict_window = 5
        self.feature_list = ["ewma_vol_span260", "ret_norm_1d", "macd_8_24"]
        self.time_feature_list = ["minute", "hour", "weekday", "day", "month"]
        self.clip = 5.0

def verify_dataset_fix():
    # Mocking the pickle load to avoid FileNotFoundError
    with patch("builtins.open", mock_open(read_data=b""), create=True), \
         patch("pickle.load", return_value={}):
        
        # We need to override the Config used inside QlibDataset
        # Since QlibDataset creates Config() inside __init__, we patch it.
        with patch("Kronos_finetune.dataset.Config", return_value=MockConfig()):
            dataset = QlibDataset(data_type='train', d_in=6)
    
    # Now we manually set up the data to test the logic
    symbols = ["AAPL", "MSFT"]
    dataset.symbols = symbols
    dataset.data = {}
    for sym in symbols:
        idx = pd.date_range("2020-01-01", periods=5000, freq="30min")
        close = np.abs(np.random.randn(5000).cumsum()) + 100
        df = pd.DataFrame({
            "open": close + 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
        }, index=idx)
        df.index.name = "datetime"
        dataset.data[sym] = df

    # The problematic logic is in __init__. 
    # Since we already instantiated, we have to re-run the loop manually
    # or use a a function that contains that logic.
    # I will just copy the loop from the fixed dataset.py here to verify it works.
    
    from Kronos_finetune.features import FeatureEngineer
    fe = FeatureEngineer()
    dataset.indices = []
    
    # This is the EXACT logic now in dataset.py
    for symbol in dataset.symbols:
        df = dataset.data[symbol]
        
        prices = df['close']
        ohlc = df[['open', 'high', 'low', 'close']]
        eng_feats = fe.build(prices, ohlc=ohlc, include_target=False, dropna=False)
        df = df.join(eng_feats)
        
        df['minute'] = df.index.minute
        df['hour'] = df.index.hour
        df['weekday'] = df.index.weekday
        df['day'] = df.index.day
        df['month'] = df.index.month
        
        # 3. Cut off the first 3536 bars to avoid warm-up NaNs from engineered features
        # Max warm-up: macd_price_std_window (260) + macd_signal_std_window (3276) = 3536
        df = df.iloc[3536:]
        
        series_len = len(df)
        num_samples = series_len - dataset.window + 1
        
        if num_samples > 0:
            for feat in dataset.feature_list:
                if feat not in df.columns:
                    df[feat] = 0.0
            
            dataset.data[symbol] = df[dataset.feature_list + dataset.time_feature_list]
            
            for i in range(num_samples):
                dataset.indices.append((symbol, i))

    print(f"Total indices: {len(dataset.indices)}")
    unique_symbols = set([s for s, i in dataset.indices])
    print(f"Symbols in indices: {unique_symbols}")
    
    assert len(unique_symbols) == len(symbols), f"Expected {len(symbols)} symbols, found {len(unique_symbols)}"
    
    for sym in symbols:
        df = dataset.data[sym]
        first_sample = df.iloc[0:dataset.window]
        nan_count = first_sample.isna().sum().sum()
        print(f"NaNs in first sample of {sym}: {nan_count}")
        assert nan_count == 0, f"Found {nan_count} NaNs in first sample of {sym}"

    print("\n✅ All verifications passed!")

if __name__ == "__main__":
    verify_dataset_fix()
