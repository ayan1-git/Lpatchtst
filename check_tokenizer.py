
import os, sys
sys.path.insert(0, os.getcwd())

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import Counter

import config
from tokenizer import KronosTokenizer, prepare_ohlc_features

# ── Must match train_tokenizer.py exactly ─────────────────────────────────────
SEQ_LEN    = 64
S1_BITS    = 6
S2_BITS    = 6
D_IN       = 4
D_MODEL    = 128
N_HEADS    = 4
FF_DIM     = 512
N_ENC      = 3
N_DEC      = 3
GROUP_SIZE = 4
TOKENIZER_PATH = "tokenizer.pt"
ALL_FEATURES = ["log_ret_open", "log_ret_high", "log_ret_low", "log_ret_close"]

GREEN  = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"
BOLD   = "\033[1m";  RESET  = "\033[0m"

def ok(m):     print(f"  {GREEN}✅ {m}{RESET}")
def warn(m):   print(f"  {YELLOW}⚠️  {m}{RESET}")
def fail(m):   print(f"  {RED}❌ {m}{RESET}")
def header(m): print(f"\n{BOLD}{'─'*55}\n  {m}\n{'─'*55}{RESET}")


# ── Data loading (identical normalization as train_tokenizer.py) ───────────────
def load_data():
    files = config.DATA_FILE
    if isinstance(files, str): files = [files]
    all_feats = []
    for f in files:
        if not os.path.exists(f):
            local_f = os.path.join("Data ", os.path.basename(f))
            if os.path.exists(local_f): f = local_f
            else: continue
        print(f"  Loading {f}...")
        df = pd.read_csv(f)
        feats = prepare_ohlc_features(df)
        all_feats.append(feats)
    data = np.concatenate(all_feats, axis=0).astype(np.float32)
    train_end = int(len(data) * config.TRAIN_RATIO)
    # Return raw data; build_windows will handle per-window normalization
    return torch.FloatTensor(data[:train_end])


def load_model():
    model = KronosTokenizer(
        d_in=D_IN, d_model=D_MODEL, n_heads=N_HEADS, ff_dim=FF_DIM,
        n_enc_layers=N_ENC, n_dec_layers=N_DEC,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        s1_bits=S1_BITS, s2_bits=S2_BITS,
        beta=0.25, gamma0=0.0, gamma=0.1, zeta=0.1, group_size=GROUP_SIZE
    )
    model.load_state_dict(torch.load(TOKENIZER_PATH, map_location="cpu"))
    model.eval()
    return model


def build_windows(data: torch.Tensor, seq_len: int, stride: int, max_w: int):
    """Returns (N, seq_len, C) — with per-window normalization (matches train_tokenizer.py)."""
    T = len(data)
    starts = list(range(0, T - seq_len, stride))[:max_w]
    idx = torch.tensor(starts).unsqueeze(1) + torch.arange(seq_len).unsqueeze(0)
    windows = data[idx]   # (N, seq_len, C)

    # Global normalization instead of per-window
    mean = data.mean(dim=0, keepdim=True).unsqueeze(0) # (1, 1, C)
    std  = data.std(dim=0, keepdim=True).unsqueeze(0)
    windows = (windows - mean) / (std + 1e-5)
    windows = windows.clamp(-5.0, 5.0)
    
    return windows


# ── CHECK 1: Codebook utilization ─────────────────────────────────────────────
def check_utilization(model, data):
    header("CHECK 1 · Codebook Utilization")
    issues = []
    windows = build_windows(data, SEQ_LEN, stride=4, max_w=2000)
    print(f"  Windows built    : {len(windows):,}  (stride=4, seq_len={SEQ_LEN})")

    s1_all, s2_all = [], []
    with torch.no_grad():
        for i in range(0, len(windows), 64):
            b = windows[i:i+64]
            idx_s1, idx_s2 = model.encode(b, half=True)   # (B, SEQ_LEN)
            # take last timestep token — most informative
            s1_all.append(idx_s1[:, -1])
            s2_all.append(idx_s2[:, -1])

    s1 = torch.cat(s1_all); s2 = torch.cat(s2_all)
    vocab = 2 ** S1_BITS  # 64
    u1 = len(torch.unique(s1)) / vocab * 100
    u2 = len(torch.unique(s2)) / vocab * 100
    avg = (u1 + u2) / 2

    print(f"  Stream S1 Utilization: {u1:.1f}% ({int(u1*vocab/100)}/{vocab})")
    print(f"  Stream S2 Utilization: {u2:.1f}% ({int(u2*vocab/100)}/{vocab})")

    if avg >= 80:   ok(f"Excellent utilization ({avg:.1f}%)")
    elif avg >= 40: warn(f"Moderate utilization ({avg:.1f}%)"); issues.append("moderate_util")
    else:           fail(f"CRITICAL: Low utilization ({avg:.1f}%) — codebook collapse"); issues.append("CRITICAL_collapse")

    return (s1, s2), issues


# ── CHECK 2: Bit collapse ──────────────────────────────────────────────────────
def check_bits(model, data):
    header("CHECK 2 · Bit Collapse Detection")
    issues = []
    windows = build_windows(data, SEQ_LEN, stride=8, max_w=1000)
    print(f"  Windows built    : {len(windows):,}  (stride=8, seq_len={SEQ_LEN})")

    bits_all = []
    with torch.no_grad():
        for i in range(0, len(windows), 64):
            b = windows[i:i+64]
            # encode returns list [idx_s1, idx_s2] each (B, SEQ_LEN)
            idx_s1, idx_s2 = model.encode(b, half=True)
            # Reconstruct 12-bit binary from indices
            basis_s1 = (2 ** torch.arange(S1_BITS, device=idx_s1.device)).long()
            basis_s2 = (2 ** torch.arange(S2_BITS, device=idx_s2.device)).long()
            b1 = ((idx_s1[:, -1].unsqueeze(-1) & basis_s1) != 0).float()  # (B, 6)
            b2 = ((idx_s2[:, -1].unsqueeze(-1) & basis_s2) != 0).float()  # (B, 6)
            bits_all.append(torch.cat([b1, b2], dim=-1))  # (B, 12)

    bits = torch.cat(bits_all, dim=0)   # (N, 12)
    rates = bits.mean(0).tolist()

    collapsed = 0
    print(f"  {'Bit':>4}  {'ON-rate':>8}  Status")
    print(f"  {'─'*4}  {'─'*8}  {'─'*20}")
    for i, r in enumerate(rates):
        if 0.20 <= r <= 0.80:
            status = f"{GREEN}✅ healthy{RESET}"
        elif 0.10 <= r < 0.20 or 0.80 < r <= 0.90:
            status = f"{YELLOW}⚠️  marginal{RESET}"; collapsed += 1
        else:
            status = f"{RED}❌ COLLAPSED{RESET}"; collapsed += 1
        print(f"  {i:>4}  {r:>8.3f}  {status}")

    print()
    n_bits = S1_BITS + S2_BITS
    if collapsed == 0:       ok(f"All {n_bits} bits healthy")
    elif collapsed <= 3:     warn(f"{collapsed}/{n_bits} bits marginal"); issues.append("marginal_bits")
    else:                    fail(f"{collapsed}/{n_bits} bits collapsed"); issues.append("CRITICAL_bits")

    return issues


# ── CHECK 3: Reconstruction quality ───────────────────────────────────────────
def check_reconstruction(model, data):
    header("CHECK 3 · Reconstruction MSE (autoencoder)")
    issues = []
    # Autoencoder: reconstruct the input, not predict next step
    # Correct: pass window x → get z_full → compare to x
    windows = build_windows(data, SEQ_LEN, stride=4, max_w=500)
    print(f"  Windows built    : {len(windows):,}  (seq_len={SEQ_LEN})")

    recon_list = []
    with torch.no_grad():
        for i in range(0, len(windows), 32):
            b = windows[i:i+32]
            (z_pre, z_full), _, _, _ = model(b)   # z_full: (B, SEQ_LEN, D_IN)
            recon_list.append(z_full)

    x_recon = torch.cat(recon_list, dim=0)    # (N, SEQ_LEN, D_IN)
    raw_mse = ((x_recon - windows[:len(x_recon)]) ** 2).mean(dim=(0, 1)).tolist()
    var     = windows.var(dim=(0, 1)).clamp(min=1e-9).tolist()
    nmse    = [m / v for m, v in zip(raw_mse, var)]

    print(f"  {'Feature':<35}  {'Raw MSE':>10}  {'NMSE':>8}  Status")
    print(f"  {'─'*35}  {'─'*10}  {'─'*8}  {'─'*15}")
    for name, mse, nm in zip(ALL_FEATURES, raw_mse, nmse):
        if nm < 0.5:    status = f"{GREEN}✅ good{RESET}"
        elif nm < 0.85: status = f"{YELLOW}⚠️  marginal{RESET}"
        else:           status = f"{RED}❌ poor{RESET}"
        print(f"  {name:<35}  {mse:>10.6f}  {nm:>8.4f}  {status}")

    avg_nmse = np.mean(nmse)
    print(f"\n  Overall NMSE: {avg_nmse:.4f}")
    if avg_nmse < 0.5:    ok(f"Excellent reconstruction (NMSE={avg_nmse:.3f})")
    elif avg_nmse < 0.85: warn(f"Moderate reconstruction (NMSE={avg_nmse:.3f})"); issues.append("moderate_recon")
    else:                 fail(f"Poor reconstruction (NMSE={avg_nmse:.3f})"); issues.append("CRITICAL_recon")

    return issues


# ── CHECK 4: Token entropy ─────────────────────────────────────────────────────
def check_entropy(indices_tuple):
    header("CHECK 4 · Token Entropy & Distribution")
    issues = []
    s1, s2 = indices_tuple

    def stats(idx):
        cnt = Counter(idx.tolist())
        vals = np.array(list(cnt.values()), dtype=np.float64)
        p = vals / vals.sum()
        ent = float(-np.sum(p * np.log2(p + 1e-12)))
        top10 = sum(v for _, v in sorted(cnt.items(), key=lambda x: -x[1])[:10]) / len(idx) * 100
        return ent, top10

    e1, t1 = stats(s1); e2, t2 = stats(s2)
    max_ent = float(S1_BITS)
    eff1, eff2 = e1 / max_ent * 100, e2 / max_ent * 100
    avg_eff = (eff1 + eff2) / 2

    print(f"  Stream S1 Entropy: {e1:.2f} bits (Eff: {eff1:.1f}%)")
    print(f"  Stream S2 Entropy: {e2:.2f} bits (Eff: {eff2:.1f}%)")
    print(f"  Top-10 coverage  : S1={t1:.1f}%, S2={t2:.1f}%")

    if avg_eff >= 70:   ok(f"Good distribution ({avg_eff:.1f}% efficiency)")
    elif avg_eff >= 40: warn(f"Moderate clustering ({avg_eff:.1f}%)"); issues.append("moderate_entropy")
    else:               fail(f"Heavy clustering ({avg_eff:.1f}%)"); issues.append("CRITICAL_entropy")

    if t1 > 50 or t2 > 50:
        warn("Top-10 codes dominate — distribution skewed"); issues.append("skewed_dist")

    return issues


# ── VERDICT ────────────────────────────────────────────────────────────────────
def verdict(all_issues):
    header("FINAL VERDICT")
    critical = [i for i in all_issues if i.startswith("CRITICAL")]
    warnings = [i for i in all_issues if not i.startswith("CRITICAL")]

    if not all_issues:
        print(f"  {GREEN}{BOLD}🚀 TOKENIZER READY — safe to start model training{RESET}")
    elif not critical:
        print(f"  {YELLOW}{BOLD}⚠️  PROCEED WITH CAUTION — {len(warnings)} warning(s){RESET}")
        for w in warnings: print(f"    • {w}")
    else:
        print(f"  {RED}{BOLD}🔴 DO NOT TRAIN — fix tokenizer first{RESET}")
        for c in critical: print(f"    • {c}")
        print("\n  Action plan:")
        if "CRITICAL_collapse" in critical or "CRITICAL_bits" in critical:
            print("    1. Increase GAMMA in train_tokenizer.py (try 2.0 or 5.0)")
            print("    2. Decrease ZETA (try 0.05) to reduce commit loss dominance")
            print("    3. Train longer (500+ epochs) or reduce SEQ_LEN to 32")
        if "CRITICAL_recon" in critical:
            print("    4. Increase D_MODEL to 128 or FF_DIM to 256")
    print()


def main():
    print(f"\n{BOLD}{'═'*55}")
    print(f"  TOKENIZER HEALTH CHECK")
    print(f"  Hierarchical BSQ: {S1_BITS} coarse + {S2_BITS} fine bits (12 total)")
    print(f"  Vocab Size      : {2**S1_BITS} coarse + {2**S2_BITS} fine codes")
    print(f"  SEQ_LEN         : {SEQ_LEN}   ← must match train_tokenizer.py")
    print(f"{'═'*55}{RESET}")

    print("\nLoading data and tokenizer...")
    data  = load_data()
    model = load_model()
    print(f"  Training samples : {len(data):,}")
    print(f"  Features         : {data.shape[1]}")

    all_issues = []
    indices, issues = check_utilization(model, data); all_issues += issues
    all_issues += check_bits(model, data)
    all_issues += check_reconstruction(model, data)
    all_issues += check_entropy(indices)
    verdict(all_issues)


if __name__ == "__main__":
    main()