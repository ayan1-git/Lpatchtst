#!/usr/bin/env python3
"""
check_tokenizer.py
==================
Run this after train_tokenizer.py completes to get a full health report
before starting PatchTST training.

Usage:
    python check_tokenizer.py
"""

import os
import sys
# Force local project directory to priority in path to avoid /content/ shadow imports
sys.path.insert(0, os.getcwd())

import numpy as np
import os
import sys
import math
import torch
import torch.nn.functional as F
import pandas as pd
from collections import Counter

# ── project imports ───────────────────────────────────────────────────────────
import config
from tokenizer import KronosTokenizer, prepare_ohlc_features
from features import FeatureConfig, FeatureEngineer

# ── constants ──────────────────────────────────────────────────────────────────
TOKENIZER_PATH = "tokenizer.pt"
# ── derive feature columns ──────────────────────────────────────────────
ALL_FEATURES = ["log_ret_open", "log_ret_high", "log_ret_low", "log_ret_close"]

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):     print(f"  {GREEN}✅ {msg}{RESET}")
def warn(msg):   print(f"  {YELLOW}⚠️  {msg}{RESET}")
def fail(msg):   print(f"  {RED}❌ {msg}{RESET}")
def header(msg): print(f"\n{BOLD}{'─'*55}\n  {msg}\n{'─'*55}{RESET}")

from data_loader import fit_scaler

# ── load data (same logic as train_tokenizer.py) ──────────────────────────────
def load_train_features():
    data_file = config.DATA_FILE
    files = [data_file] if isinstance(data_file, str) else data_file

    all_features = []
    for f in files:
        if not os.path.exists(f):
            # Fallback for local workspace structure
            local_f = os.path.join("Data ", os.path.basename(f))
            if os.path.exists(local_f):
                f = local_f
            else:
                print(f"  Warning: {f} not found, skipping.")
                continue
        
        print(f"  Loading {f}...")
        df_raw = pd.read_csv(f)
        time_col = next((c for c in df_raw.columns if c.lower() in ["date", "datetime"]), None)
        if time_col:
            df_raw[time_col] = pd.to_datetime(df_raw[time_col])
            df_raw.set_index(time_col, inplace=True)

        from tokenizer import prepare_ohlc_features
        asset_features = prepare_ohlc_features(df_raw)

        train_end = int(len(asset_features) * config.TRAIN_RATIO)
        asset_features = asset_features[:train_end]
        # No scaler needed for Kronos log-returns (already in tight [-0.05, +0.05] range)
        
        all_features.append(asset_features)

    if not all_features:
        raise ValueError("No data loaded.")
    return torch.FloatTensor(np.concatenate(all_features, axis=0))

# ── load tokenizer ─────────────────────────────────────────────────────────────
def load_tokenizer():
    if not os.path.exists(TOKENIZER_PATH):
        print(f"{RED}ERROR: {TOKENIZER_PATH} not found. Run train_tokenizer.py first.{RESET}")
        sys.exit(1)
    model = KronosTokenizer(
        d_in=4, d_model=64, n_heads=4, ff_dim=128,
        n_enc_layers=3, n_dec_layers=3,
        s1_bits=6, s2_bits=6,
    )
    if os.path.exists(TOKENIZER_PATH):
        try:
            model.load_state_dict(torch.load(TOKENIZER_PATH, map_location="cpu"))
        except Exception as e:
            print(f"{YELLOW}Warning: Could not load weights from {TOKENIZER_PATH}: {e}{RESET}")
            print(f"Health check will continue with randomly initialized weights.{RESET}")
    model.eval()
    return model

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER — efficient window building (zero-copy via unfold)
# ═══════════════════════════════════════════════════════════════════════════════
def _build_windows(flat: torch.Tensor, seq_len: int, stride: int = 1,
                   max_windows: int = 2000) -> torch.Tensor:
    """
    Build (N, seq_len, C) windows from a flat (T, C) tensor.
    Uses torch.unfold for zero-copy strided views — no list comprehension.
    """
    T, C = flat.shape
    if T < seq_len:
        # pad front so we get at least 1 window
        pad = flat[:1].expand(seq_len - T, -1)
        flat = torch.cat([pad, flat], dim=0)
        T = flat.shape[0]

    n_possible = (T - seq_len) // stride + 1
    actual_stride = stride
    if n_possible > max_windows:
        actual_stride = max(1, (T - seq_len) // max_windows)
        n_possible = (T - seq_len) // actual_stride + 1

    # unfold gives (T, seq_len) view along dim-0, then we gather per-feature
    # More memory-efficient: build index tensor once
    starts = torch.arange(0, min(n_possible, max_windows)) * actual_stride
    idx = starts.unsqueeze(1) + torch.arange(seq_len).unsqueeze(0)  # (N, L)
    return flat[idx]   # (N, L, C)  — uses advanced indexing, still efficient


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 1 — Codebook Utilization
# ═══════════════════════════════════════════════════════════════════════════════
def check_codebook_utilization(model, x_train):
    header("CHECK 1 · Codebook Utilization")
    issues = []
    SEQ_LEN_TOK = 512

    # Use a moderate sample with strided windows to keep RAM low
    sample = x_train[:5000] if len(x_train) > 5000 else x_train
    windows = _build_windows(sample, SEQ_LEN_TOK, stride=4, max_windows=2000)
    print(f"  Windows built    : {len(windows):,}  (stride=4)")

    with torch.no_grad():
        batch_size = 64
        s1_list, s2_list = [], []
        for i in range(0, len(windows), batch_size):
            batch = windows[i : i + batch_size]
            [idx_s1, idx_s2] = model.encode(batch, half=True)
            s1_list.append(idx_s1[:, -1])
            s2_list.append(idx_s2[:, -1])
        
        indices_s1 = torch.cat(s1_list, dim=0)
        indices_s2 = torch.cat(s2_list, dim=0)

    n_unique_s1 = len(torch.unique(indices_s1))
    n_unique_s2 = len(torch.unique(indices_s2))
    vocab_per_stream = 2**6 # 64
    util_s1 = n_unique_s1 / vocab_per_stream * 100
    util_s2 = n_unique_s2 / vocab_per_stream * 100

    print(f"  Stream S1 Utilization: {util_s1:.1f}% ({n_unique_s1}/{vocab_per_stream})")
    print(f"  Stream S2 Utilization: {util_s2:.1f}% ({n_unique_s2}/{vocab_per_stream})")

    avg_util = (util_s1 + util_s2) / 2

    if avg_util >= 80:
        ok(f"Excellent utilization ({avg_util:.1f}%)")
    elif avg_util >= 50:
        warn(f"Moderate utilization ({avg_util:.1f}%)")
        issues.append("moderate_util")
    else:
        fail(f"CRITICAL: Low utilization ({avg_util:.1f}%) — codebook collapse")
        issues.append("CRITICAL_collapse")

    return (indices_s1, indices_s2), issues

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 2 — Bit Collapse
# ═══════════════════════════════════════════════════════════════════════════════
def check_bit_collapse(model, x_train):
    header("CHECK 2 · Bit Collapse Detection")
    issues = []

    sample_raw = x_train[:3000] if len(x_train) > 3000 else x_train
    SEQ_LEN_TOK = 512
    windows = _build_windows(sample_raw, SEQ_LEN_TOK, stride=8, max_windows=1000)
    print(f"  Windows built    : {len(windows):,}  (stride=8)")

    with torch.no_grad():
        batch_size = 64
        all_q = []
        for i in range(0, len(windows), batch_size):
            batch = windows[i : i + batch_size]
            _, _, quantized, _ = model(batch)   # (B, L, 12)
            all_q.append(quantized[:, -1, :])    # last bar latent
        
        # quantized is in {-scale, +scale}. Convert to binary for ON-rate check
        q_tensor = torch.cat(all_q, dim=0)
        bits = (torch.sign(q_tensor) + 1) / 2
        bit_means = bits.mean(dim=0)            # (12,)

    collapsed = 0
    print(f"  {'Bit':>4}  {'ON-rate':>8}  Status")
    print(f"  {'─'*4}  {'─'*8}  {'─'*20}")
    for i, mean in enumerate(bit_means.tolist()):
        if 0.20 <= mean <= 0.80:
            status = f"{GREEN}✅ healthy{RESET}"
        elif 0.10 <= mean < 0.20 or 0.80 < mean <= 0.90:
            status = f"{YELLOW}⚠️  marginal{RESET}"
            collapsed += 1
        else:
            status = f"{RED}❌ COLLAPSED{RESET}"
            collapsed += 1
        print(f"  {i:>4}  {mean:>8.3f}  {status}")

    print()
    n_bits = 12
    if collapsed == 0:
        ok(f"All {n_bits} bits are healthy (0.2–0.8 ON-rate)")
    elif collapsed <= 3:
        warn(f"{collapsed}/{n_bits} bits marginal")
        issues.append("marginal_bits")
    else:
        fail(f"{collapsed}/{n_bits} bits collapsed")
        issues.append("CRITICAL_bits")

    return issues

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 3 — Per-Feature Reconstruction MSE
# ═══════════════════════════════════════════════════════════════════════════════
def check_reconstruction(model, x_train):
    header("CHECK 3 · Per-Feature Predictive MSE")
    issues = []

    # To check predictive health, we need window[0:T] to predict sample[T+1]
    sample_raw = x_train[:2000] if len(x_train) > 2000 else x_train
    SEQ_LEN_TOK = 512
    
    # Use windows ending at T to predict T+1
    # So if we have N bars, we can build windows from bars 0..N-2 to predict bars 1..N-1
    windows = _build_windows(sample_raw[:-1], SEQ_LEN_TOK, stride=4, max_windows=500)
    targets = sample_raw[1:][-len(windows):] # align targets to the end of windows
    
    print(f"  Windows built    : {len(windows):,}  (predictive task)")

    feat_std = x_train.std(dim=0, keepdim=True) + 1e-6

    with torch.no_grad():
        batch_size = 32
        recon_list = []
        for i in range(0, len(windows), batch_size):
            batch = windows[i : i + batch_size]
            (z_pre, z_full), _, _, _ = model(batch)
            recon_list.append(z_full[:, -1, :])  # Prediction for T+1 from full codebook
        x_recon = torch.cat(recon_list, dim=0)

    # Calculate Raw MSE (matches your training logs)
    raw_mse_feat = ((x_recon - targets) ** 2).mean(dim=0).tolist()
    overall_mse  = float(np.mean(raw_mse_feat))

    # Calculate NMSE internally for health status (1.0 = baseline/random)
    f_var = (x_train.var(dim=0) + 1e-9).to(x_recon.device)
    nmse_feat = (((x_recon - targets) ** 2).mean(dim=0) / f_var).tolist()

    print(f"  {'Feature':<35}  {'Raw MSE':>10}  Status")
    print(f"  {'─'*35}  {'─'*10}  {'─'*15}")
    
    for name, mse, nmse in zip(ALL_FEATURES, raw_mse_feat, nmse_feat):
        if nmse < 0.85:
            status = f"{GREEN}✅ predictive{RESET}"
        elif nmse < 0.98:
            status = f"{YELLOW}⚠️  marginal{RESET}"
        else:
            status = f"{RED}❌ random{RESET}"
        print(f"  {name:<35}  {mse:>10.6f}  {status}")

    print(f"\n  Overall Raw MSE: {overall_mse:.7f}")
    avg_nmse = np.mean(nmse_feat)
    if avg_nmse < 0.90:
        ok(f"Model is learning (NMSE={avg_nmse:.3f})")
    elif avg_nmse < 1.0:
        warn(f"Weak signal (NMSE={avg_nmse:.3f})")
    else:
        fail(f"No signal (NMSE={avg_nmse:.3f}) — model is random")

    return issues

# CHECK 4 — Token Entropy
# ═══════════════════════════════════════════════════════════════════════════════
def check_token_entropy(indices_tuple):
    header("CHECK 4 · Token Entropy & Distribution")
    issues = []
    idx_s1, idx_s2 = indices_tuple

    def get_stats(idx):
        counts = np.array(list(Counter(idx.tolist()).values()), dtype=np.float64)
        probs = counts / counts.sum()
        ent = float(-np.sum(probs * np.log2(probs + 1e-12)))
        top10 = sorted(Counter(idx.tolist()).items(), key=lambda x: -x[1])[:10]
        top10_pct = sum(v for _, v in top10) / len(idx) * 100
        return ent, top10_pct

    ent_s1, top10_s1 = get_stats(idx_s1)
    ent_s2, top10_s2 = get_stats(idx_s2)
    max_ent = 6.0 # log2(64)
    eff_s1, eff_s2 = ent_s1/max_ent*100, ent_s2/max_ent*100
    avg_eff = (eff_s1 + eff_s2) / 2

    print(f"  Stream S1 Entropy: {ent_s1:.2f} bits (Eff: {eff_s1:.1f}%)")
    print(f"  Stream S2 Entropy: {ent_s2:.2f} bits (Eff: {eff_s2:.1f}%)")
    print(f"  Top-10 coverage  : S1={top10_s1:.1f}%, S2={top10_s2:.1f}%")

    if avg_eff >= 70:
        ok(f"Good token distribution (avg efficiency={avg_eff:.1f}%)")
    elif avg_eff >= 50:
        warn(f"Moderate clustering (avg efficiency={avg_eff:.1f}%)")
        issues.append("moderate_entropy")
    else:
        fail(f"Heavy token clustering (avg efficiency={avg_eff:.1f}%)")
        issues.append("CRITICAL_entropy")

    if top10_s1 > 50 or top10_s2 > 50:
        warn(f"High coverage in top-10 codes — distribution is skewed")
        issues.append("skewed_dist")

    return issues

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════════════════════════════════════════
def final_verdict(all_issues):
    header("FINAL VERDICT")
    critical = [i for i in all_issues if i.startswith("CRITICAL")]
    warnings = [i for i in all_issues if not i.startswith("CRITICAL")]

    if not all_issues:
        print(f"  {GREEN}{BOLD}🚀 TOKENIZER READY — safe to start PatchTST training{RESET}")
    elif not critical:
        print(f"  {YELLOW}{BOLD}⚠️  PROCEED WITH CAUTION — {len(warnings)} warning(s){RESET}")
        for w in warnings:
            print(f"    • {w}")
        print(f"\n  Quick fix — add to train_tokenizer.py loss:")
        print(f"    loss = mse_loss + 0.1 * ((bit_means - 0.5)**2).mean()")
    else:
        print(f"  {RED}{BOLD}🔴 DO NOT TRAIN — fix tokenizer first{RESET}")
        for c in critical:
            print(f"    • {c}")
        print(f"\n  Action plan:")
        if any("collapse" in i or "bits" in i for i in critical):
            print(f"    1. Add diversity loss. n_bits={config.TOKENIZER_BITS} — try lowering to {config.TOKENIZER_BITS - 4} if collapse persists")
        if any("recon" in i for i in critical):
            print(f"    2. Increase hidden_dim (try 256)")
        print(f"    3. Re-run train_tokenizer.py then this script")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"\n{BOLD}{'═'*55}")
    print(f"  TOKENIZER HEALTH CHECK")
    print(f"  n_bits={config.TOKENIZER_BITS}  vocab={config.VOCAB_SIZE:,}  features={len(ALL_FEATURES)}")
    print(f"{'═'*55}{RESET}")

    print("\nLoading data and tokenizer...")
    x_train = load_train_features()
    model   = load_tokenizer()
    print(f"  Training samples : {len(x_train):,}")
    print(f"  Features         : {x_train.shape[1]}")

    all_issues = []
    indices, issues = check_codebook_utilization(model, x_train); all_issues += issues
    all_issues += check_bit_collapse(model, x_train)
    all_issues += check_reconstruction(model, x_train)
    all_issues += check_token_entropy(indices)
    final_verdict(all_issues)

if __name__ == "__main__":
    main()