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
ohlc_val   = ohlc[train_end:]

# Build sliding windows: (num_windows, SEQ_LEN, 4)
def make_windows(arr, seq_len, clip=10.0):
    starts = np.arange(len(arr) - seq_len + 1)
    windows = []
    for i in starts:
        w = arr[i : i + seq_len].copy()          # (SEQ_LEN, 4)
        # Per-window normalization
        mean = w.mean(axis=0)
        std  = w.std(axis=0)
        w    = (w - mean) / (std + 1e-5)
        w    = np.clip(w, -clip, clip)
        windows.append(w)
    return np.stack(windows)                      # (N, SEQ_LEN, 4)

train_windows = make_windows(ohlc_train, SEQ_LEN)
val_windows   = make_windows(ohlc_val, SEQ_LEN)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(train_windows).float()),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(val_windows).float()),
    batch_size=BATCH_SIZE, shuffle=False
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
steps_per_epoch = len(train_loader)
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

    # ── Warm-up: ramp bsq weight from 0 → 1 over first 20 epochs
    bsq_warmup = min(1.0, epoch / 20.0)

    for (x,) in train_loader:
        x = x.to(device)

        # Forward
        (z_pre, z_full), bsq_loss, quantized, z_indices = tok(x)

        # ── Official Reconstruction Loss ────────────────────────────────────
        recon_loss = F.mse_loss(z_pre,  x) + F.mse_loss(z_full, x)

        # ── Total Loss ────────────────────────────────────────────────────
        # (recon + bsq) / 2 as per official formula
        total = (recon_loss + bsq_warmup * bsq_loss) / 2

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(tok.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss  += total.item()
        total_recon += recon_loss.item()
        total_bsq   += bsq_loss.item()

        # Track codebook utilization (Global 12-bit indices)
        codes_used = z_indices.unique().numel()
        total_codes += codes_used

    # ── Validation (Save criterion: avg_val_loss = F.mse_loss(z_full, x))
    tok.eval()
    val_recon_sum = 0.0
    with torch.no_grad():
        for (vx,) in val_loader:
            vx = vx.to(device)
            (_, vz_full), _, _, _ = tok(vx)
            val_recon_sum += F.mse_loss(vz_full, vx).item()
    
    avg_val_loss = val_recon_sum / len(val_loader)
    avg_loss     = total_loss  / steps_per_epoch
    avg_codes    = total_codes / steps_per_epoch

    print(f"Epoch {epoch+1:3d} | Train={avg_loss:.4f} | Val_Recon={avg_val_loss:.4f} | codes={avg_codes:.1f}/4096")

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(tok.state_dict(), "tokenizer.pt")
        print(f"  ✓ Saved (val_recon={best_loss:.4f})")