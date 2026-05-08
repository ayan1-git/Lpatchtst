import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from tokenizer import KLineTokenizer
from features import calculate_features, PASSTHROUGH_FEATURES, SCALE_FEATURES
import config
#11
TOKENIZER_MODEL_PATH = "tokenizer.pth"
DATA_FILE = config.DATA_FILE

def train_tokenizer(save_best_only=True):
    print(f"Loading data for tokenizer training: {DATA_FILE}")
    if isinstance(DATA_FILE, str):
        files = [DATA_FILE]
    else:
        files = DATA_FILE

    all_features = []
    for f in files:
        if not os.path.exists(f):
            print(f"Warning: File {f} not found, skipping.")
            continue
            
        df_raw = pd.read_csv(f)
        
        # Set datetime index
        time_col = next((c for c in df_raw.columns if c.lower() in ["date", "datetime"]), None)
        if time_col:
            df_raw[time_col] = pd.to_datetime(df_raw[time_col])
            df_raw.set_index(time_col, inplace=True)
        
        feat_df = calculate_features(df_raw)
        
        # We only care about the 21 hybrid features
        input_cols = PASSTHROUGH_FEATURES + SCALE_FEATURES
        asset_features = feat_df[input_cols].values.astype(np.float32)
        
        # Prevent data leakage: only train the tokenizer on the training split
        train_end = int(len(asset_features) * config.TRAIN_RATIO)
        asset_features = asset_features[:train_end]
        
        # Handle NaNs/Infs and Outliers
        # Most engineered features are z-scores or bounded [-1, 1]
        # Clipping to [-3, 3] ensures outliers don't skew the BSQ latents
        asset_features = np.nan_to_num(asset_features, nan=0.0, posinf=3.0, neginf=-3.0)
        asset_features = np.clip(asset_features, -3.0, 3.0)
        all_features.append(asset_features)
    
    if not all_features:
        raise ValueError("No data loaded for tokenizer training.")
        
    features = np.concatenate(all_features, axis=0)
    
    # Convert to Tensor
    x_train = torch.FloatTensor(features)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # Initialize Tokenizer
    # input_dim=21, n_bits=8
    model = KLineTokenizer(input_dim=21, n_bits=config.TOKENIZER_BITS).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    batch_size = 128
    epochs = 150
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, total_steps=epochs * (len(x_train) // batch_size + 1)
    )
    
    print(f"Starting Tokenizer Training ({len(x_train)} samples)...")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(x_train.size(0))
        epoch_loss = 0.0
        
        for i in range(0, x_train.size(0), batch_size):
            indices = permutation[i : i + batch_size]
            batch_x = x_train[indices].to(device)
            
            # Forward pass: reconstruct the features
            # model(x) returns (x_recon, indices, z_q)
            x_recon, _, _ = model(batch_x)
            
            loss = criterion(x_recon, batch_x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / (len(x_train)/batch_size)
        print(f"Epoch {epoch+1}/{epochs} | Reconstruction Loss: {avg_loss:.6f}")
        
        if save_best_only:
            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f"New best loss: {best_loss:.6f}. Saving tokenizer...")
                torch.save(model.state_dict(), TOKENIZER_MODEL_PATH)

    if not save_best_only:
        # Save the trained tokenizer
        torch.save(model.state_dict(), TOKENIZER_MODEL_PATH)
        print(f"Tokenizer saved to {TOKENIZER_MODEL_PATH}")
    else:
        print(f"Best Tokenizer is saved to {TOKENIZER_MODEL_PATH} with loss {best_loss:.6f}")
        # Load the best model for verification
        model.load_state_dict(torch.load(TOKENIZER_MODEL_PATH))

    # Verification: Check a few tokens
    model.eval()
    with torch.no_grad():
        test_x = x_train[:5].to(device)
        _, tokens, _ = model(test_x)
        print(f"Sample Tokens: {tokens.cpu().numpy().tolist()}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the Tokenizer")
    parser.add_argument("--save-all", action="store_true", help="Save the last tokenizer instead of only the best one")
    args = parser.parse_args()
    
    train_tokenizer(save_best_only=not args.save_all)