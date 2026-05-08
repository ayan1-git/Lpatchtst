# train_tokenizer.py — full replacement

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import pandas as pd
import numpy as np
import os
import config
from tokenizer import KLineTokenizer
from features import FeatureConfig, FeatureEngineer

TOKENIZER_MODEL_PATH = "tokenizer.pth"
SEQ_LEN   = 64    # context window during tokenizer training
           # Using 64 (not 512) here: tokenizer learns local context,
           # PatchTST handles long-range. Keeps batch size large.
STRIDE    = 16    # hop between windows — 75% overlap → rich training signal
BATCH_SIZE = 128  # fits 2x T4 comfortably; increase to 256 if no OOM
EPOCHS    = 150
LR        = 1e-3
LAMBDA_Q  = 0.01  # commitment loss weight
LAMBDA_ENT = 0.10  # entropy/diversity loss — push bits toward 50% ON-rate


def build_sequences(features: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    """Slice (N, 21) into overlapping (M, seq_len, 21) windows."""
    indices = range(0, len(features) - seq_len + 1, stride)
    return np.stack([features[i:i + seq_len] for i in indices], axis=0)


def train_tokenizer(rank=0, world_size=1):
    # ── DDP setup ───────────────────────────────────────────────────────────
    use_ddp = world_size > 1
    if use_ddp:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    use_cuda = device.type == "cuda"

    # ── Data loading ─────────────────────────────────────────────────────────
    files = [config.DATA_FILE] if isinstance(config.DATA_FILE, str) else config.DATA_FILE
    all_seqs = []

    for f in files:
        if not os.path.exists(f):
            print(f"Warning: {f} not found, skipping.")
            continue

        df_raw = pd.read_csv(f)
        time_col = next((c for c in df_raw.columns if c.lower() in ["date", "datetime"]), None)
        if time_col:
            df_raw[time_col] = pd.to_datetime(df_raw[time_col])
            df_raw.set_index(time_col, inplace=True)

        fe = FeatureEngineer(config=FeatureConfig())
        feat_df = fe.build(df_raw["close"], ohlc=df_raw, dropna=True)

        all_cols  = feat_df.columns.tolist()
        robust    = [c for c in all_cols if "vs_factor" in c or "squeeze" in c]
        no_scale  = [c for c in all_cols if c not in robust]
        input_cols = robust + no_scale

        arr = feat_df[input_cols].values.astype(np.float32)

        # Train split only — no leakage
        train_end = int(len(arr) * config.TRAIN_RATIO)
        arr = arr[:train_end]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Slice into overlapping sequences of length SEQ_LEN
        seqs = build_sequences(arr, SEQ_LEN, STRIDE)  # (M, 64, 21)
        all_seqs.append(seqs)
        if rank == 0:
            print(f"{f}: {len(arr)} bars → {len(seqs)} sequences")

    sequences = np.concatenate(all_seqs, axis=0)   # (M_total, 64, 21)
    x_train   = torch.FloatTensor(sequences)

    # Per-feature std for normalized MSE loss (computed on training data once)
    feat_std = x_train.std(dim=(0, 1), keepdim=True) + 1e-6  # (1, 1, 21)

    dataset = TensorDataset(x_train)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if use_ddp else None
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=2 if use_cuda else 0,
        pin_memory=use_cuda,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = KLineTokenizer(
        input_dim=len(input_cols),
        n_bits=config.TOKENIZER_BITS,
        d_enc=64,
        n_heads=4,
        n_enc_layers=2,
        seq_len=SEQ_LEN,
    ).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR,
        steps_per_epoch=len(loader),
        epochs=EPOCHS,
        pct_start=0.1,
    )
    criterion  = nn.MSELoss()
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_cuda)

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        if use_ddp:
            sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0

        for (batch_x,) in loader:
            batch_x = batch_x.to(device, non_blocking=True)  # (B, 64, 21)

            with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
                raw_model = model.module if use_ddp else model

                # Forward — gets continuous latents via _encode_latent
                z = raw_model._encode_latent(batch_x)          # (B, L, 12)
                zc_cont = z[..., :raw_model.half_bits]
                zf_cont = z[..., raw_model.half_bits:]

                import torch.nn.functional as F_
                zc_norm = F_.normalize(zc_cont, dim=-1)           # (B, L, 6)
                zf_norm = F_.normalize(zf_cont, dim=-1)

                from tokenizer import StraightThroughEstimator
                zq_c = StraightThroughEstimator.apply(zc_norm)     # (B, L, 6) ∈ {±1}
                zq_f = StraightThroughEstimator.apply(zf_norm)

                x_recon_c = raw_model.decoder_coarse(zq_c)
                x_recon_f = raw_model.decoder_fine(
                    torch.cat([zq_c, zq_f], dim=-1)
                )

                # 1. Normalized reconstruction loss (vs_factor no longer dominates)
                fstd      = feat_std.to(device)
                loss_coarse = ((x_recon_c - batch_x) / fstd).pow(2).mean()
                loss_fine   = ((x_recon_f - batch_x) / fstd).pow(2).mean()

                # 2. Commitment loss — CORRECT: normalized vs binary-of-normalized
                loss_commit = (zc_norm - zq_c.detach()).pow(2).mean() \
                            + (zf_norm - zq_f.detach()).pow(2).mean()

                # 3. Entropy/diversity loss — push each bit toward 50% ON-rate
                bits_c   = (zq_c + 1) / 2                          # {0,1}
                bits_f   = (zq_f + 1) / 2
                prob_c   = bits_c.mean(dim=(0, 1))                  # (6,)
                prob_f   = bits_f.mean(dim=(0, 1))                  # (6,)
                loss_ent = F_.binary_cross_entropy(prob_c, torch.full_like(prob_c, 0.5)) \
                         + F_.binary_cross_entropy(prob_f, torch.full_like(prob_f, 0.5))

                loss = 0.7 * loss_coarse + 0.3 * loss_fine \
                     + LAMBDA_Q   * loss_commit \
                     + LAMBDA_ENT * loss_ent

            optimizer.zero_grad()
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            scheduler.step()

            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        if rank == 0:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | loss={avg:.6f} "
                  f"(recon_c={loss_coarse.item():.4f} "
                  f"recon_f={loss_fine.item():.4f} "
                  f"commit={loss_commit.item():.4f} "
                  f"ent={loss_ent.item():.4f})")
            if avg < best_loss:
                best_loss = avg
                state = model.module.state_dict() if use_ddp else model.state_dict()
                torch.save(state, TOKENIZER_MODEL_PATH)
                print(f"  ✓ Saved best tokenizer (loss={best_loss:.6f})")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    # Launch with: torchrun --nproc_per_node=2 train_tokenizer.py
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    train_tokenizer(rank=local_rank, world_size=world_size)