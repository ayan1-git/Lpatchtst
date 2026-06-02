import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from config import Config


class QlibDataset(Dataset):
    """
    A PyTorch Dataset for handling Qlib financial time series data.

    This dataset pre-computes all possible start indices for sliding windows
    and then randomly samples from them during training/validation.

    Args:
        data_type (str): The type of dataset to load, either 'train' or 'val'.

    Raises:
        ValueError: If `data_type` is not 'train' or 'val'.
    """

    def __init__(self, data_type: str = 'train'):
        self.config = Config()
        if data_type not in ['train', 'val']:
            raise ValueError("data_type must be 'train' or 'val'")
        self.data_type = data_type

        # Use a dedicated random number generator for sampling to avoid
        # interfering with other random processes (e.g., in model initialization).
        self.py_rng = random.Random(self.config.seed)

        # Set paths and number of samples based on the data type.
        if data_type == 'train':
            self.data_path = f"{self.config.dataset_path}/train_data.pkl"
            self.n_samples = self.config.n_train_iter
        else:
            self.data_path = f"{self.config.dataset_path}/val_data.pkl"
            self.n_samples = self.config.n_val_iter

        # Try to load from raw CSV directories if they exist.
        # This allows training on all raw files of different lengths.
        raw_data_dirs = ["./Data ", "./data"]
        self.data = {}
        found_csvs = False
        
        for raw_data_dir in raw_data_dirs:
            if os.path.exists(raw_data_dir):
                csv_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
                if csv_files:
                    print(f"[{data_type.upper()}] Loading raw CSV files from {raw_data_dir}...")
                    found_csvs = True
                    for f in csv_files:
                        sym = f.replace(".csv", "")
                        if sym in self.data: continue
                        
                        df = pd.read_csv(f"{raw_data_dir}/{f}")
                        
                        # Basic data cleaning and renaming
                        if 'date' in df.columns:
                            df["date"] = pd.to_datetime(df["date"])
                            df = df.rename(columns={"date": "datetime"})
                        elif 'datetime' in df.columns:
                            df["datetime"] = pd.to_datetime(df["datetime"])
                        
                        if 'volume' in df.columns:
                            df = df.rename(columns={"volume": "vol"})
                        if 'amt' not in df.columns:
                            df["amt"] = 0.0
                        
                        # Split based on data_type if needed
                        df = df.set_index("datetime").sort_index()
                        if data_type == 'train':
                            mask = (df.index >= self.config.train_time_range[0]) & (df.index <= self.config.train_time_range[1])
                        elif data_type == 'val':
                            mask = (df.index >= self.config.val_time_range[0]) & (df.index <= self.config.val_time_range[1])
                        else:
                            mask = pd.Series(True, index=df.index)
                        
                        df = df[mask]
                        if len(df) >= (self.config.lookback_window + self.config.predict_window + 1):
                            self.data[sym] = df
        
        if not found_csvs or not self.data:
            if not found_csvs:
                print(f"[{data_type.upper()}] No raw CSV directories found. Loading from pickle: {self.data_path}")
            else:
                print(f"[{data_type.upper()}] No valid samples found in CSVs. Falling back to pickle: {self.data_path}")
            
            with open(self.data_path, 'rb') as f:
                self.data = pickle.load(f)

        self.window = self.config.lookback_window + self.config.predict_window + 1

        self.symbols = list(self.data.keys())
        self.feature_list = self.config.feature_list
        self.time_feature_list = self.config.time_feature_list

        # Pre-compute all possible (symbol, start_index) pairs.
        self.indices = []
        print(f"[{data_type.upper()}] Pre-computing sample indices...")
        for symbol in self.symbols:
            df = self.data[symbol]
            # If 'datetime' was index, reset it to access it for time features
            if df.index.name == 'datetime':
                df = df.reset_index()
            
            series_len = len(df)
            num_samples = series_len - self.window + 1

            if num_samples > 0:
                # Generate time features and store them directly in the dataframe.
                if 'datetime' in df.columns:
                    df['minute'] = df['datetime'].dt.minute
                    df['hour'] = df['datetime'].dt.hour
                    df['weekday'] = df['datetime'].dt.weekday
                    df['day'] = df['datetime'].dt.day
                    df['month'] = df['datetime'].dt.month
                
                # Ensure all features exist
                for feat in self.feature_list:
                    if feat not in df.columns:
                        df[feat] = 0.0
                
                # Keep only necessary columns to save memory.
                self.data[symbol] = df[self.feature_list + self.time_feature_list]

                # Add all valid starting indices for this symbol to the global list.
                # Use a step to avoid too much redundancy if needed, but here we use all.
                for i in range(num_samples):
                    self.indices.append((symbol, i))

        # The effective dataset size is the minimum of the configured iterations
        # and the total number of available samples.
        self.n_samples = min(self.n_samples, len(self.indices))
        print(f"[{data_type.upper()}] Found {len(self.indices)} possible samples. Using {self.n_samples} per epoch.")

    def set_epoch_seed(self, epoch: int):
        """
        Sets a new seed for the random sampler for each epoch. This is crucial
        for reproducibility in distributed training.

        Args:
            epoch (int): The current epoch number.
        """
        epoch_seed = self.config.seed + epoch
        self.py_rng.seed(epoch_seed)

    def __len__(self) -> int:
        """Returns the number of samples per epoch."""
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        
        # Select a random sample from the entire pool of indices.
        random_idx = self.py_rng.randint(0, len(self.indices) - 1)
        symbol, start_idx = self.indices[random_idx]

        # Extract the sliding window from the dataframe.
        df = self.data[symbol]
        end_idx = start_idx + self.window
        win_df = df.iloc[start_idx:end_idx]

        # Separate main features and time features.
        x = win_df[self.feature_list].values.astype(np.float32)
        x_stamp = win_df[self.time_feature_list].values.astype(np.float32)

        # Normalize the window. Mean and std are calculated strictly on the
        # lookback window (past data) to prevent future data leakage.
        past_len = self.config.lookback_window
        past_x = x[:past_len]

        x_mean = np.mean(past_x, axis=0)
        x_std  = np.std(past_x, axis=0)

        # Apply normalization and robust clipping to the entire sequence
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)

        # Ensure we have exactly 6 features (padding with zeros if necessary)
        # to match the pretrained model's expectation.
        if x.shape[1] < 6:
            padding = np.zeros((x.shape[0], 6 - x.shape[1]), dtype=np.float32)
            x = np.concatenate([x, padding], axis=1)
        elif x.shape[1] > 6:
            x = x[:, :6]

        # Convert to PyTorch tensors.
        x_tensor = torch.from_numpy(x)
        x_stamp_tensor = torch.from_numpy(x_stamp)

        return x_tensor, x_stamp_tensor


if __name__ == '__main__':
    # Example usage and verification.
    print("Creating training dataset instance...")
    train_dataset = QlibDataset(data_type='train')

    print(f"Dataset length: {len(train_dataset)}")

    if len(train_dataset) > 0:
        try_x, try_x_stamp = train_dataset[100]  # Index 100 is ignored.
        print(f"Sample feature shape: {try_x.shape}")
        print(f"Sample time feature shape: {try_x_stamp.shape}")
    else:
        print("Dataset is empty.")