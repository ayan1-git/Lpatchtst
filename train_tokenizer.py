# train_tokenizer.py
import os
import sys
# Force local project directory to priority in path to avoid /content/ shadow imports
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tokenizer import KronosTokenizer

# ── Config ────────────────────────────────────────────────────
D_IN        = 4          # log_ret_open, log_ret_high, log_ret_low, log_ret_close
SEQ_LEN     = 64         # sequence length fed to transformer encoder
D_MODEL     = 64
N_HEADS     = 4
FF_DIM      = 128
N_ENC       = 3          # enc_layers (transformer blocks = N_ENC - 1)
N_DEC       = 3
S1_BITS     = 6          # coarse token: 2^6 = 64 codes
S2_BITS     = 6          # fine token:   2^6 = 64 codes
# BSQ loss hyperparams — directly from official Kronos config
BETA        = 0.25       # commit loss weight
GAMMA0      = 0.1        # per-sample entropy weight (pushes confident bits)
GAMMA       = 1.0        # codebook entropy weight  (pushes utilization) ← CRITICAL
ZETA        = 0.1        # overall entropy scale
GROUP_SIZE  = 6          # 12 bits / 6 = 2 groups

EPOCHS      = 300
BATCH_SIZE  = 256
LR          = 3e-4
SAVE_PATH   = "tokenizer.pt"
DATA_PATH   = "data/features_4col.npy"  # (N, 4) array of log returns
# ─────────────────────────────────────────────────────────────

def build_windows(data, seq_len, stride=1):
    """Slice (N, F) array into (num_windows, seq_len, F) windows."""
    windows = []
    for i in range(0, len(data) - seq_len, stride):
        windows.append(data[i:i + seq_len])
    return np.stack(windows, axis=0).astype(np.float32)


def main():
    # Load OHLC data from CSVs as defined in config or fallback to local 'Data ' directory
    import pandas as pd
    import config
    from tokenizer import prepare_ohlc_features

    data_files = config.DATA_FILE
    if isinstance(data_files, str): data_files = [data_files]
    
    all_features = []
    for f in data_files:
        # Fallback for local workspace structure
        if not os.path.exists(f):
            local_f = os.path.join("Data ", os.path.basename(f))
            if os.path.exists(local_f):
                f = local_f
            else:
                print(f"Warning: {f} not found. Skipping.")
                continue
        
        print(f"Loading {f}...")
        df = pd.read_csv(f)
        feats = prepare_ohlc_features(df)
        all_features.append(feats)
    
    if not all_features:
        raise ValueError("No data files found. Check config.DATA_FILE or 'Data ' directory.")
        
    data = np.concatenate(all_features, axis=0)
    print(f"Total rows loaded: {len(data):,}")

    assert data.shape[1] == D_IN, f"Expected {D_IN} features, got {data.shape[1]}"

    # Normalize per-feature to zero mean, unit std
    mu = data.mean(0, keepdims=True)
    sd = data.std(0, keepdims=True) + 1e-8
    data = (data - mu) / sd
    data = np.clip(data, -5, 5)

    windows = build_windows(data, SEQ_LEN, stride=1)
    print(f"Training windows: {windows.shape}")  # (N_windows, SEQ_LEN, 4)

    dataset = TensorDataset(torch.from_numpy(windows))
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = KronosTokenizer(
        d_in=D_IN, d_model=D_MODEL, n_heads=N_HEADS, ff_dim=FF_DIM,
        n_enc_layers=N_ENC, n_dec_layers=N_DEC,
        s1_bits=S1_BITS, s2_bits=S2_BITS,
        beta=BETA, gamma0=GAMMA0, gamma=GAMMA, zeta=ZETA,
        group_size=GROUP_SIZE
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_recon_c = total_recon_f = total_bsq = total = 0.0
        n_batches = 0

        for (x,) in loader:
            x = x.to(device)                          # (B, SEQ_LEN, 4)

            (z_pre, z_full), bsq_loss, quantized, z_indices = model(x)

            # Reconstruction losses — predict next timestep (shift by 1)
            # Official Kronos trains as a reconstruction autoencoder on same x
            loss_c = F.mse_loss(z_pre, x)
            loss_f = F.mse_loss(z_full, x)

            # BSQ loss already contains commit + entropy penalty (from official BSQuantizer)
            loss = 0.7 * loss_c + 0.3 * loss_f + bsq_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_recon_c += loss_c.item()
            total_recon_f += loss_f.item()
            total_bsq     += bsq_loss.item()
            total         += loss.item()
            n_batches     += 1

        scheduler.step()
        avg = lambda v: v / n_batches

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"total={avg(total):.5f}  "
              f"recon_c={avg(total_recon_c):.4f}  "
              f"recon_f={avg(total_recon_f):.4f}  "
              f"bsq={avg(total_bsq):.4f}")

        if avg(total) < best_loss:
            best_loss = avg(total)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✓ Saved (loss={best_loss:.5f})")

    print("Done.")


if __name__ == "__main__":
    import torch.nn.functional as F
    main()