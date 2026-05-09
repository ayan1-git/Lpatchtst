# train_tokenizer_debug.py
# ─────────────────────────────────────────────────────────────────────────────
# FULLY INSTRUMENTED tokenizer training script.
# Every signal that matters is logged so we NEVER guess what is happening.
# ─────────────────────────────────────────────────────────────────────────────
import os, sys
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tokenizer import KronosTokenizer, prepare_ohlc_features
from loss import bit_balance_loss


# ══════════════════════════════════════════════════════════════════════════════
# 0. ANSI colours (makes the log much easier to scan)
# ══════════════════════════════════════════════════════════════════════════════
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def ok(msg):   print(f"{GREEN}  ✔ {msg}{RESET}")
def warn(msg): print(f"{YELLOW}  ⚠ {msg}{RESET}")
def bad(msg):  print(f"{RED}  ✘ {msg}{RESET}")
def info(msg): print(f"{CYAN}  ℹ {msg}{RESET}")
def dim(msg):  print(f"{DIM}    {msg}{RESET}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIG  ── change ONLY these values when tuning
# ══════════════════════════════════════════════════════════════════════════════
EPOCHS      = 200
LR          = 3e-5
BATCH_SIZE  = 256
SEQ_LEN     = 64
S1_BITS     = 6
S2_BITS     = 6
D_MODEL     = 128
N_HEADS     = 4
FF_DIM      = 512
N_LAYERS    = 3

# ── BSQ hyperparams ──────────────────────────────────────────────────────────
BETA        = 0.25    # commit loss weight
GAMMA0      = 5.0     # 5× stronger per-sample diversity pressure
GAMMA       = 0.5     # proportionally increase codebook reward
ZETA        = 2.0     # scale up overall entropy signal
GROUP_SIZE  = 6       # 12 / 6 = 2 groups

# ── Loss weights ─────────────────────────────────────────────────────────────
# Set MANUAL_BSQ_WEIGHT = None to use AUTO-CALIBRATION (recommended)
MANUAL_BSQ_WEIGHT = None
BIT_BALANCE_WEIGHT = 2.0   # targeted bit regularization

# ── Collapse thresholds ───────────────────────────────────────────────────────
MIN_CODES_FRACTION  = 0.05   # for CUMULATIVE epoch-wide tracking
BATCH_CAPACITY      = BATCH_SIZE * SEQ_LEN
MIN_CODES_PER_BATCH = 200   # raise alarm only below 200 (was 819)

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 2 ** (S1_BITS + S2_BITS)
info(f"device={device}  vocab_size={VOCAB_SIZE}  codebook_dim={S1_BITS+S2_BITS}")
assert (S1_BITS + S2_BITS) % GROUP_SIZE == 0


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n── DATA " + "─"*60)
df = pd.read_csv("/content/NIFTY 50_30minute.csv")
time_col = next((c for c in df.columns if c.lower() in ("date","datetime")), None)
if time_col:
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()
info(f"Loaded CSV: {len(df)} rows, columns={list(df.columns)}")

ohlc = prepare_ohlc_features(df)   # (N, 4) log-returns
N    = len(ohlc)
info(f"Log-returns shape: {ohlc.shape}")
info(f"Log-returns stats: mean={ohlc.mean(0).round(5)}  "
     f"std={ohlc.std(0).round(5)}  "
     f"min={ohlc.min(0).round(4)}  max={ohlc.max(0).round(4)}")

# ── Global normalisation (NOT per-window) ────────────────────────────────────
GLOBAL_MEAN = ohlc.mean(axis=0)
GLOBAL_STD  = ohlc.std(axis=0)
info(f"Global mean={GLOBAL_MEAN.round(6)}  std={GLOBAL_STD.round(6)}")

def make_windows(arr, seq_len, clip=5.0):
    arr_norm = np.clip((arr - GLOBAL_MEAN) / (GLOBAL_STD + 1e-5), -clip, clip)
    return np.stack([arr_norm[i:i+seq_len]
                     for i in range(len(arr_norm) - seq_len + 1)]).astype(np.float32)

train_end     = int(N * 0.85)
train_windows = make_windows(ohlc[:train_end], SEQ_LEN)
val_windows   = make_windows(ohlc[train_end:], SEQ_LEN)
info(f"Train windows: {train_windows.shape}  val: {val_windows.shape}")
info(f"Window range after norm+clip: [{train_windows.min():.3f}, {train_windows.max():.3f}]")

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(train_windows)),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(val_windows)),
    batch_size=BATCH_SIZE, shuffle=False)


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n── MODEL " + "─"*60)
tok = KronosTokenizer(
    d_in=4, d_model=D_MODEL, n_heads=N_HEADS, ff_dim=FF_DIM,
    n_enc_layers=N_LAYERS, n_dec_layers=N_LAYERS,
    s1_bits=S1_BITS, s2_bits=S2_BITS,
    beta=BETA, gamma0=GAMMA0, gamma=GAMMA, zeta=ZETA,
    group_size=GROUP_SIZE,
).to(device)
info(f"Params: {sum(p.numel() for p in tok.parameters()):,}")
info(f"BSQ: beta={BETA} gamma0={GAMMA0} gamma={GAMMA} zeta={ZETA} group={GROUP_SIZE}")

if os.path.exists("tokenizer.pt"):
    try:
        tok.load_state_dict(torch.load("tokenizer.pt", map_location=device))
        ok("Loaded existing tokenizer.pt for fine-tuning")
    except Exception as e:
        warn(f"Could not load tokenizer.pt: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════
optimizer = torch.optim.AdamW(tok.parameters(), lr=LR, weight_decay=1e-4)
steps_per_epoch = len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR,
    total_steps=EPOCHS * steps_per_epoch, pct_start=0.1)


# ══════════════════════════════════════════════════════════════════════════════
# 5. AUTO-CALIBRATE BSQ_WEIGHT
#    One forward pass → measure scale of recon vs bsq → set weight so
#    bsq contributes 20% of recon at the start. No hand-tuning needed.
# ══════════════════════════════════════════════════════════════════════════════
print("\n── AUTO-CALIBRATION " + "─"*50)
tok.eval()
with torch.no_grad():
    x_calib = next(iter(train_loader))[0].to(device)
    (z_pre_c, z_full_c), bsq_c, _, _ = tok(x_calib)
    recon_c = (F.mse_loss(z_pre_c, x_calib) + F.mse_loss(z_full_c, x_calib)).item()
    bsq_c   = bsq_c.item()
info(f"Calibration → recon={recon_c:.5f}  bsq={bsq_c:.5f}")

if MANUAL_BSQ_WEIGHT is not None:
    BSQ_WEIGHT = MANUAL_BSQ_WEIGHT
    info(f"Using MANUAL_BSQ_WEIGHT = {BSQ_WEIGHT}")
elif abs(bsq_c) < 1e-8:
    BSQ_WEIGHT = 1.0
    warn("bsq ~ 0 at init, defaulting BSQ_WEIGHT=1.0")
elif bsq_c < 0:
    BSQ_WEIGHT = 0.0
    bad("bsq NEGATIVE at init → gamma/zeta too high. Lower them. Setting BSQ_WEIGHT=0.")
else:
    BSQ_WEIGHT = round(0.40 * recon_c / bsq_c, 4)
    ok(f"Auto-calibrated BSQ_WEIGHT={BSQ_WEIGHT}  "
       f"(weighted_bsq={BSQ_WEIGHT*bsq_c:.5f} = 40% of recon={recon_c:.5f})")
tok.train()

# ── Per-bit activation at init ───────────────────────────────────────────────
print()
info("Per-bit mean activation at init (near 0.0 = healthy, near ±1 = dead):")
with torch.no_grad():
    (_, _), _, q_init, _ = tok(x_calib)
    init_bit_means = q_init.mean(dim=[0,1]).cpu().numpy()
for i, bm in enumerate(init_bit_means):
    flag = "ok  " if abs(bm) < 0.3 else ("WARN" if abs(bm) < 0.7 else "DEAD")
    print(f"    bit[{i:2d}] mean={bm:+.3f}  {flag}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def get_codes_used(z_indices):
    if isinstance(z_indices, list):
        flat = torch.cat([z_indices[0].flatten(), z_indices[1].flatten()])
    else:
        flat = z_indices.flatten()
    return flat.unique().numel()

def reset_dead_codes(model, all_codes_used, device):
    """Re-initialize embeddings for codes never used this epoch."""
    # BSQ has no explicit embedding table, but we can perturb
    # the quant_embed projection weights for unused bit patterns
    used_fraction = len(all_codes_used) / VOCAB_SIZE
    if used_fraction < 0.4:
        with torch.no_grad():
            # Add small noise to quant_embed to escape current local minimum
            noise_scale = 0.01 * (0.4 - used_fraction)
            for name, param in model.named_parameters():
                if 'quant_embed' in name:
                    param.add_(torch.randn_like(param) * noise_scale)
        print(f"    ↻ Perturbed quant_embed (used={used_fraction:.1%}, noise={noise_scale:.4f})")


def gradient_norms(model):
    groups = {"embed": 0.0, "encoder": 0.0, "decoder": 0.0,
              "quant_proj": 0.0, "post_quant": 0.0}
    for name, p in model.named_parameters():
        if p.grad is None: continue
        g = p.grad.norm().item()
        if   "post_quant_embed" in name: groups["post_quant"]  += g
        elif "quant_embed"      in name: groups["quant_proj"]  += g
        elif "encoder"          in name: groups["encoder"]     += g
        elif "decoder"          in name: groups["decoder"]     += g
        elif "embed"            in name: groups["embed"]       += g
    return groups


def fix_stuck_bits(model, bit_onrates, threshold_high=0.85, threshold_low=0.15):
    """
    Reinitialize only the stuck rows of quant_embed weight.
    bit_onrates: list of ON-rates from your health check
    """
    stuck_bits = [i for i, r in enumerate(bit_onrates)
                  if r > threshold_high or r < threshold_low]
    
    if not stuck_bits:
        info("No stuck bits to fix.")
        return
    
    warn(f"Fixing stuck bits: {stuck_bits}")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'quant_embed' in name and 'weight' in name:
                for bit_idx in stuck_bits:
                    # Reinit this row with small random values
                    nn.init.normal_(param[bit_idx], mean=0.0, std=0.02)
                    dim(f"↻ Reset quant_embed row {bit_idx}")
            if 'quant_embed' in name and 'bias' in name:
                # Should not exist if bias=False, but here for completeness
                for bit_idx in stuck_bits:
                    param[bit_idx].zero_()
                    dim(f"↻ Zeroed quant_embed bias {bit_idx}")


# ── SURGICAL BIT RESET ───────────────────────────────────────────────────────
bit_onrates = [0.709, 0.276, 0.598, 0.308, 0.312,
               0.123, 0.908, 0.389, 0.540, 0.696, 0.405, 0.439]
fix_stuck_bits(tok, bit_onrates)


# ══════════════════════════════════════════════════════════════════════════════
# 7. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print(f"  TRAINING  BSQ_WEIGHT={BSQ_WEIGHT}  LR={LR}  BATCH={BATCH_SIZE}")
print("═"*72)

history  = []
best_val = float("inf")

for epoch in range(EPOCHS):
    tok.train()
    acc_total = acc_recon = acc_bsq = acc_codes = acc_bitbal = 0.0
    acc_bsq_neg = 0
    all_codes_epoch = set()

    for step, (x,) in enumerate(train_loader):
        x = x.to(device)
        (z_pre, z_full), bsq_loss, quantized, z_indices = tok(x)
        
        recon_loss   = F.mse_loss(z_pre, x) + F.mse_loss(z_full, x)
        z_presign_approx = quantized * (S1_BITS + S2_BITS) ** 0.5
        bit_bal_loss = bit_balance_loss(z_presign_approx)
        total        = recon_loss + BSQ_WEIGHT * bsq_loss + BIT_BALANCE_WEIGHT * bit_bal_loss

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(tok.parameters(), 1.0)
        gnorms = gradient_norms(tok)
        optimizer.step()
        scheduler.step()

        # ── TRACK CUMULATIVE UTILISATION ────────────────────────────────────
        if isinstance(z_indices, list):
            flat = torch.cat([z_indices[0].flatten(), z_indices[1].flatten()])
        else:
            flat = z_indices.flatten()
        all_codes_epoch.update(flat.cpu().numpy().tolist())

        codes_used    = flat.unique().numel()
        acc_total    += total.item()
        acc_recon    += recon_loss.item()
        acc_bsq      += bsq_loss.item()
        acc_codes    += codes_used
        acc_bitbal   += bit_bal_loss.item()
        if bsq_loss.item() < 0: acc_bsq_neg += 1

        # ── Step-0 deep dump every epoch ────────────────────────────────────
        if step == 0:
            with torch.no_grad():
                z_raw = tok.embed(x)
                for layer in tok.encoder: z_raw = layer(z_raw)
                z_raw = tok.quant_embed(z_raw)
                z_norm = F.normalize(z_raw, dim=-1)
                commit = BETA * ((quantized.detach() - z_norm)**2).sum(dim=-1).mean()
                bm     = quantized.mean(dim=[0,1]).cpu()

            dim(f"E{epoch+1} s0 | recon={recon_loss.item():.5f}  "
                f"bsq={bsq_loss.item():.5f}  "
                f"commit={commit.item():.5f}  "
                f"entropy_part={(bsq_loss.item()-commit.item()):.5f}  "
                f"codes={codes_used}")
            dim(f"       grads | embed={gnorms['embed']:.4f}  "
                f"enc={gnorms['encoder']:.4f}  "
                f"dec={gnorms['decoder']:.4f}  "
                f"q_proj={gnorms['quant_proj']:.4f}  "
                f"post_q={gnorms['post_quant']:.4f}")
            dead  = (bm.abs() > 0.9).sum().item()
            risky = (bm.abs() > 0.5).sum().item()
            tag   = ok if dead == 0 and risky == 0 else (warn if dead == 0 else bad)
            tag(f"       bits  | dead={dead}/{len(bm)}  risky={risky}/{len(bm)}  "
                f"means={bm.numpy().round(2).tolist()}")

    # ── End-of-epoch averages ─────────────────────────────────────────────────
    avg_total  = acc_total  / steps_per_epoch
    avg_recon  = acc_recon  / steps_per_epoch
    avg_bsq    = acc_bsq    / steps_per_epoch
    avg_codes  = acc_codes  / steps_per_epoch
    avg_bitbal = acc_bitbal / steps_per_epoch
    
    epoch_unique_codes = len(all_codes_epoch)
    epoch_codes_pct    = epoch_unique_codes / VOCAB_SIZE * 100
    cur_lr             = scheduler.get_last_lr()[0]

    # ── Validation ────────────────────────────────────────────────────────────
    tok.eval()
    val_recon_sum = val_codes_sum = 0.0
    with torch.no_grad():
        for (vx,) in val_loader:
            vx = vx.to(device)
            (_, vz), _, _, vi = tok(vx)
            val_recon_sum += F.mse_loss(vz, vx).item()
            val_codes_sum += get_codes_used(vi)
    tok.train()
    avg_val_recon = val_recon_sum / len(val_loader)
    avg_val_codes = val_codes_sum / len(val_loader)

    # ── Main epoch line ───────────────────────────────────────────────────────
    print(f"\nEpoch {epoch+1:3d}/{EPOCHS} │ "
          f"Train={avg_total:.4f} │ Recon={avg_recon:.4f} │ "
          f"BSQ={avg_bsq:+.5f} │ BitBal={avg_bitbal:.5f} │ "
          f"Val={avg_val_recon:.4f} │ "
          f"codes(ep)={epoch_unique_codes}/{VOCAB_SIZE}({epoch_codes_pct:.1f}%) │ "
          f"codes(batch_avg)={avg_codes:.0f} │ lr={cur_lr:.2e}")

    # ── Reset / Perturb dead codes ────────────────────────────────────────────
    reset_dead_codes(tok, all_codes_epoch, device)

    # ── Health checks ─────────────────────────────────────────────────────────
    if avg_bsq < 0:
        bad(f"  BSQ NEGATIVE (avg={avg_bsq:.5f}) → reduce gamma or zeta")
    if acc_bsq_neg > 0:
        warn(f"  bsq_loss negative in {acc_bsq_neg}/{steps_per_epoch} steps")
    
    # 1. Batch health (is the model learning ANYTHING per batch?)
    if avg_codes < MIN_CODES_PER_BATCH:
        bad(f"  BATCH COLLAPSE: Avg {avg_codes:.0f} unique codes < {MIN_CODES_PER_BATCH} threshold")
    
    # 2. Vocabulary health (is the model exploring the whole codebook?)
    if epoch_codes_pct < MIN_CODES_FRACTION * 100:
        bad(f"  VOCAB COLLAPSE: {epoch_codes_pct:.1f}% vocab used cumulative")
    elif epoch_codes_pct < 15:
        warn(f"  Low total utilisation: {epoch_codes_pct:.1f}%")
    elif epoch_codes_pct > 40:
        ok(f"  Good total utilisation: {epoch_codes_pct:.1f}%")
    if epoch >= 2 and abs(history[-1]["avg_bsq"] - avg_bsq) < 0.001 and avg_bsq > 0.1:
        warn(f"  BSQ stagnating (Δ<0.001) — gradient not flowing through quantizer")

    # ── Save best ─────────────────────────────────────────────────────────────
    if avg_val_recon < best_val:
        best_val = avg_val_recon
        torch.save(tok.state_dict(), "tokenizer.pt")
        ok(f"  Saved tokenizer.pt (val={best_val:.4f})")

    history.append(dict(epoch=epoch+1, train=avg_total, recon=avg_recon,
                        avg_bsq=avg_bsq, val_recon=avg_val_recon,
                        codes_pct=epoch_codes_pct, bitbal=avg_bitbal))

    # ── Full per-bit dump every 10 epochs ─────────────────────────────────────
    if (epoch + 1) % 10 == 0:
        print(f"\n  {'─'*64}")
        print(f"  Epoch {epoch+1} per-bit means (▓=saturation, near 0 = healthy):")
        with torch.no_grad():
            xs = next(iter(train_loader))[0].to(device)
            (_, _), _, qs, _ = tok(xs)
            bm_all = qs.mean(dim=[0,1]).cpu().numpy()
        for i, bm in enumerate(bm_all):
            bar = ("█" * int(abs(bm)*20)).ljust(20)
            flag = "ok  " if abs(bm) < 0.3 else ("WARN" if abs(bm) < 0.7 else "DEAD")
            print(f"    bit[{i:2d}] {bar} {bm:+.3f}  {flag}")
        dead = sum(1 for bm in bm_all if abs(bm) > 0.9)
        (ok if dead == 0 else bad)(f"  {dead}/{len(bm_all)} dead bits")
        print(f"  {'─'*64}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 8. FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═"*72)
print("  TRAINING COMPLETE")
print("═"*72)
best_r = min(history, key=lambda h: h["val_recon"])
best_c = max(history, key=lambda h: h["codes_pct"])
ok(f"Best val_recon = {best_r['val_recon']:.4f}  at epoch {best_r['epoch']}")
ok(f"Peak vocab use = {best_c['codes_pct']:.1f}%  at epoch {best_c['epoch']}")

final_codes = history[-1]["codes_pct"]
if   final_codes < 5:  bad(f"FINAL: COLLAPSED ({final_codes:.1f}%)")
elif final_codes < 15: warn(f"FINAL: UNDERUTILISED ({final_codes:.1f}%)")
else:                  ok(f"FINAL: HEALTHY ({final_codes:.1f}%)")

neg_epochs = [h["epoch"] for h in history if h["avg_bsq"] < 0]
if neg_epochs: bad(f"BSQ went negative in epochs: {neg_epochs}")
else:          ok("BSQ stayed positive throughout.")

print("\n  Utilisation trend:")
for h in history[::10]:
    bar = int(h["codes_pct"] / 2)
    print(f"    ep{h['epoch']:3d}: {'█'*bar}{'░'*(50-bar)} {h['codes_pct']:.1f}%")
print("═"*72)