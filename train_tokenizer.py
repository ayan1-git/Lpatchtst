# train_tokenizer.py
import os
import sys
# Force local project directory to priority in path to avoid /content/ shadow imports
sys.path.insert(0, os.getcwd())

# train_tokenizer.py — corrected

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizer import KronosTokenizer, prepare_ohlc_features
import pandas as pd
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
EPOCHS      = 100
LR          = 3e-4
BATCH_SIZE  = 256
SEQ_LEN     = 64       # short context window for tokenizer training
S1_BITS     = 6
S2_BITS     = 6
D_MODEL     = 128
N_HEADS     = 4
FF_DIM      = 512
N_LAYERS    = 3        # n_enc_layers = n_dec_layers = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("/content/NIFTY 50_30minute.csv")
time_col = next((c for c in df.columns if c.lower() in ("date","datetime")), None)
if time_col:
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()

ohlc = prepare_ohlc_features(df)  # (N, 4) log-returns
N = len(ohlc)
train_end = int(N * 0.85)
ohlc_train = ohlc[:train_end]

# Build sliding windows: (num_windows, SEQ_LEN, 4)
def make_windows(arr, seq_len):
    idx = np.arange(len(arr) - seq_len + 1)
    return np.stack([arr[i:i+seq_len] for i in idx])

windows = make_windows(ohlc_train, SEQ_LEN)
windows_t = torch.from_numpy(windows)  # (W, SEQ_LEN, 4)
dataset = torch.utils.data.TensorDataset(windows_t)
loader  = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)

# ── Model ─────────────────────────────────────────────────────────────────────
tok = KronosTokenizer(
    d_in=4, d_model=D_MODEL, n_heads=N_HEADS, ff_dim=FF_DIM,
    n_enc_layers=N_LAYERS, n_dec_layers=N_LAYERS,
    s1_bits=S1_BITS, s2_bits=S2_BITS,
    beta=0.25,      # commitment loss
    gamma0=0.0,     # disable per-sample entropy (prevents bit collapse)
    gamma=0.1,      # small codebook entropy bonus
    zeta=0.1,       # entropy penalty scale (nudge, not objective)
    group_size=6,
).to(device)

optimizer = torch.optim.AdamW(tok.parameters(), lr=LR, weight_decay=1e-4)
steps_per_epoch = len(loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR,
    total_steps=EPOCHS * steps_per_epoch,
    pct_start=0.1,
)

# ── Training ──────────────────────────────────────────────────────────────────
best_loss = float("inf")
for epoch in range(EPOCHS):
    tok.train()
    total_loss = total_recon = total_bsq = total_codes = 0.0

    for (x,) in loader:
        x = x.to(device)

        # Forward
        (z_pre, z_full), bsq_loss, quantized, z_indices = tok(x)

        # ── Reconstruction losses ───────────────────────────────────────────
        # Coarse: predict from first s1_bits only
        loss_coarse = F.mse_loss(z_pre,  x)
        # Fine: predict from all 12 bits
        loss_fine   = F.mse_loss(z_full, x)
        recon_loss  = 0.7 * loss_coarse + 0.3 * loss_fine

        # ── Total Loss ────────────────────────────────────────────────────
        recon_weight = 10.0   # prioritize reconstruction
        total = recon_weight * recon_loss + bsq_loss

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(tok.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss  += total.item()
        total_recon += recon_loss.item()
        total_bsq   += bsq_loss.item()

        # Track codebook utilization
        if isinstance(z_indices, list):
            codes_used = z_indices[0].unique().numel()
        else:
            codes_used = z_indices.unique().numel()
        total_codes += codes_used

    avg_loss   = total_loss  / steps_per_epoch
    avg_recon  = total_recon / steps_per_epoch
    avg_bsq    = total_bsq   / steps_per_epoch
    avg_codes  = total_codes / steps_per_epoch

    print(f"Epoch {epoch+1:3d} | loss={avg_loss:.4f} | "
          f"recon={avg_recon:.4f} | bsq={avg_bsq:.4f} | "
          f"codes_used={avg_codes:.1f}/64")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(tok.state_dict(), "tokenizer.pt")
        print(f"  ✓ Saved (loss={best_loss:.4f})")