# train_tokenizer.py — Kronos-style 4-feature log-return tokenizer
#
# Architecture:
#   Input     : 4 log-return features (O, H, L, C) via prepare_ohlc_features()
#   Encoder   : d_enc=64, 2-layer causal Transformer → 12-bit BSQ
#   Training  : MSE reconstruction + commitment + diversity (bit-balance) loss
#
# This matches the architecture expected by check_tokenizer.py and model.py.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import pandas as pd
import numpy as np
import os
import config
from tokenizer import KLineTokenizer, StraightThroughEstimator, prepare_ohlc_features

TOKENIZER_MODEL_PATH = "tokenizer.pth"
SEQ_LEN    = 512
STRIDE     = 64
BATCH_SIZE = 128
EPOCHS     = 150
LR         = 3e-4   # lowered from 1e-3 — OneCycleLR was overshooting
LAMBDA_Q   = 0.01   # commitment loss weight
LAMBDA_ENT = 0.80   # entropy/diversity loss weight (increased)


def build_sequences(features: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    indices = range(0, len(features) - seq_len + 1, stride)
    if not indices: return np.array([])
    return np.stack([features[i:i + seq_len] for i in indices], axis=0)


def train_tokenizer(rank=0, world_size=1):
    use_ddp = world_size > 1
    if use_ddp:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files    = [config.DATA_FILE] if isinstance(config.DATA_FILE, str) else config.DATA_FILE
    all_seqs = []
    
    # ── Kronos-style 4-feature log-return preprocessing ───────────────────────
    for f in files:
        if not os.path.exists(f):
            print(f"Warning: {f} not found, skipping.")
            continue

        df_raw   = pd.read_csv(f)
        time_col = next((c for c in df_raw.columns if c.lower() in ["date", "datetime"]), None)
        if time_col:
            df_raw[time_col] = pd.to_datetime(df_raw[time_col])
            df_raw.set_index(time_col, inplace=True)

        # 4-feature log-returns: no scaler needed (already ~[-0.05, +0.05])
        arr = prepare_ohlc_features(df_raw)   # (N, 4)
        arr = arr[:int(len(arr) * config.TRAIN_RATIO)]

        seqs = build_sequences(arr, SEQ_LEN, STRIDE)
        if len(seqs) > 0:
            all_seqs.append(seqs)
            if rank == 0:
                print(f"{f}: {len(arr)} bars → {len(seqs)} sequences")

    if not all_seqs:
        raise ValueError("No sequences could be built. Check data paths and TRAIN_RATIO.")

    sequences = np.concatenate(all_seqs, axis=0)
    x_train   = torch.FloatTensor(sequences)

    dataset = TensorDataset(x_train)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if use_ddp else None
    loader  = DataLoader(
        dataset, batch_size=BATCH_SIZE,
        sampler=sampler, shuffle=(sampler is None),
        num_workers=2, pin_memory=True,
    )

    model = KLineTokenizer(
        input_dim=4,
        n_bits=config.TOKENIZER_BITS,
        d_enc=64, n_heads=4, n_enc_layers=2,
        seq_len=SEQ_LEN,
    ).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR,
        steps_per_epoch=len(loader), epochs=EPOCHS, pct_start=0.1,
    )
    scaler_amp = torch.cuda.amp.GradScaler()
    best_loss  = float("inf")

    for epoch in range(EPOCHS):
        if use_ddp:
            sampler.set_epoch(epoch)

        model.train()
        totals = dict(total=0, rc=0, rf=0, cmt=0, ent=0)

        for (batch_x,) in loader:
            batch_x   = batch_x.to(device, non_blocking=True)
            raw_model = model.module if use_ddp else model

            with torch.cuda.amp.autocast():
                # Encode
                z       = raw_model._encode_latent(batch_x)        # (B, L, 12)
                zc_cont = z[..., :raw_model.half_bits]
                zf_cont = z[..., raw_model.half_bits:]

                zc_norm = F.normalize(zc_cont, dim=-1)              # unit sphere
                zf_norm = F.normalize(zf_cont, dim=-1)

                zq_c = StraightThroughEstimator.apply(zc_norm)      # {±1}
                zq_f = StraightThroughEstimator.apply(zf_norm)

                # Decode
                x_recon_c = raw_model.decoder_coarse(zq_c)
                x_recon_f = raw_model.decoder_fine(torch.cat([zq_c, zq_f], dim=-1))

                # ── Loss 1: Reconstruction ──────────────────────────────────
                loss_c = (x_recon_c - batch_x).pow(2).mean()
                loss_f = (x_recon_f - batch_x).pow(2).mean()

                # ── Loss 2: Commitment ──────────────────────────────────────
                loss_commit = (zc_norm - zq_c.detach()).pow(2).mean() \
                            + (zf_norm - zq_f.detach()).pow(2).mean()

                # ── Loss 3: Diversity (on QUANTIZED bits) ───────────────────
                mean_c   = zq_c.mean(dim=(0, 1))                 # (6,)
                mean_f   = zq_f.mean(dim=(0, 1))                 # (6,)
                loss_ent = mean_c.pow(2).mean() + mean_f.pow(2).mean()

                loss = (0.7  * loss_c
                      + 0.3  * loss_f
                      + 0.01 * loss_commit
                      + LAMBDA_ENT * loss_ent)

            optimizer.zero_grad()
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            scheduler.step()

            totals["total"] += loss.item()
            totals["rc"]    += loss_c.item()
            totals["rf"]    += loss_f.item()
            totals["cmt"]   += loss_commit.item()
            totals["ent"]   += loss_ent.item()

        n = len(loader)
        if rank == 0:
            print(
                f"Epoch {epoch+1:3d}/{EPOCHS} | "
                f"total={totals['total']/n:.5f}  "
                f"recon_c={totals['rc']/n:.4f}  "
                f"recon_f={totals['rf']/n:.4f}  "
                f"commit={totals['cmt']/n:.4f}  "
                f"entropy={totals['ent']/n:.4f}"
            )
            avg = totals["total"] / n
            if avg < best_loss:
                best_loss = avg
                state = model.module.state_dict() if use_ddp else model.state_dict()
                torch.save(state, TOKENIZER_MODEL_PATH)
                print(f"  ✓ Saved (loss={best_loss:.5f})")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    train_tokenizer(rank=local_rank, world_size=world_size)