# train_tokenizer.py — codebook-collapse fix
#
# Three root causes fixed vs previous version:
#
# 1. NORMALIZED MSE LOSS
#    vs_factor_span260 (mean~346, spikes to 3000+) dominated raw MSELoss,
#    contributing ~99% of gradient signal. Encoder collapsed to 3 tokens
#    representing low/medium/high vs_factor. Fix: divide residuals by
#    per-feature std before squaring — every feature contributes equally.
#
# 2. CORRECT COMMITMENT LOSS
#    Previous code compared zc_cont (raw, unbounded) against zq_c (±1 binary
#    of the *normalized* vector) — two different spaces. This caused the encoder
#    to shrink zc_cont toward zero to minimize the loss, which makes F.normalize
#    numerically unstable and worsens collapse. Fix: commitment loss must be
#    computed in the normalized space: (zc_norm - zq_c.detach()).pow(2).
#
# 3. ENTROPY / DIVERSITY LOSS
#    Pure MSE + commitment gives no incentive to spread tokens across the
#    vocabulary. Fix: penalize each bit's marginal ON-rate deviating from 0.5
#    (maximum per-bit entropy). This directly drives bit utilization.

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
from tokenizer import KLineTokenizer, StraightThroughEstimator
from features import FeatureConfig, FeatureEngineer

TOKENIZER_MODEL_PATH = "tokenizer.pth"
SEQ_LEN    = 64
STRIDE     = 16
BATCH_SIZE = 128
EPOCHS     = 150
LR         = 3e-4   # lowered from 1e-3 — OneCycleLR was overshooting
LAMBDA_Q   = 0.01   # commitment loss weight
LAMBDA_ENT = 0.10   # entropy/diversity loss weight


def build_sequences(features: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    indices = range(0, len(features) - seq_len + 1, stride)
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

    for f in files:
        if not os.path.exists(f):
            print(f"Warning: {f} not found, skipping.")
            continue

        df_raw   = pd.read_csv(f)
        time_col = next((c for c in df_raw.columns if c.lower() in ["date", "datetime"]), None)
        if time_col:
            df_raw[time_col] = pd.to_datetime(df_raw[time_col])
            df_raw.set_index(time_col, inplace=True)

        fe      = FeatureEngineer(config=FeatureConfig())
        feat_df = fe.build(df_raw["close"], ohlc=df_raw, dropna=True)

        all_cols   = feat_df.columns.tolist()
        robust     = [c for c in all_cols if "vs_factor" in c or "squeeze" in c]
        no_scale   = [c for c in all_cols if c not in robust]
        input_cols = robust + no_scale

        arr = feat_df[input_cols].values.astype(np.float32)
        arr = arr[:int(len(arr) * config.TRAIN_RATIO)]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        seqs = build_sequences(arr, SEQ_LEN, STRIDE)
        all_seqs.append(seqs)
        if rank == 0:
            print(f"{f}: {len(arr)} bars → {len(seqs)} sequences")

    sequences = np.concatenate(all_seqs, axis=0)
    x_train   = torch.FloatTensor(sequences)

    # ── KEY FIX 1: per-feature std, computed once on training data ──────────
    # Shape (1, 1, 21) — broadcast against (B, L, 21) batch tensors.
    # This makes every feature contribute equally to reconstruction loss.
    feat_std = x_train.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
    if rank == 0:
        print(f"Feature std range: [{feat_std.min():.4f}, {feat_std.max():.4f}]")
    feat_std = feat_std.to(device)

    dataset = TensorDataset(x_train)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if use_ddp else None
    loader  = DataLoader(
        dataset, batch_size=BATCH_SIZE,
        sampler=sampler, shuffle=(sampler is None),
        num_workers=2, pin_memory=True,
    )

    model = KLineTokenizer(
        input_dim=len(input_cols),
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

                # ── Loss 1: Normalized reconstruction ───────────────────────
                # (residual / feat_std)² gives each feature equal weight.
                # vs_factor's large std is absorbed — it no longer dominates.
                loss_c = ((x_recon_c - batch_x) / feat_std).pow(2).mean()
                loss_f = ((x_recon_f - batch_x) / feat_std).pow(2).mean()

                # ── Loss 2: Commitment in the CORRECT space ──────────────────
                # Both tensors are on the unit sphere — comparable magnitudes.
                # Pulls the normalized latent toward its binary quantization.
                loss_commit = (zc_norm - zq_c.detach()).pow(2).mean() \
                            + (zf_norm - zq_f.detach()).pow(2).mean()

                # ── Loss 3: Diversity — act on RAW latents before F.normalize ──
                # Penalize non-zero batch mean per bit-dimension.
                # If mean[k] != 0, bit k fires more on one side → bias → collapse.
                # Gradient flows directly into encoder without normalization
                # Jacobian blocking it.
                mean_c   = zc_cont.mean(dim=(0, 1))              # (6,)
                mean_f   = zf_cont.mean(dim=(0, 1))              # (6,)
                loss_ent = mean_c.pow(2).mean() + mean_f.pow(2).mean()

                loss = (0.7  * loss_c
                      + 0.3  * loss_f
                      + 0.01 * loss_commit
                      + 0.50 * loss_ent)   # higher weight — simpler signal

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