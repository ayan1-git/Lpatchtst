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
import math
import numpy as np
import pandas as pd
import torch
from collections import Counter

# ── project imports ───────────────────────────────────────────────────────────
import config
from tokenizer import KLineTokenizer
from features import FeatureConfig, FeatureEngineer

# ── constants ──────────────────────────────────────────────────────────────────
TOKENIZER_PATH = "tokenizer.pth"

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):     print(f"  {GREEN}✅ {msg}{RESET}")
def warn(msg):   print(f"  {YELLOW}⚠️  {msg}{RESET}")
def fail(msg):   print(f"  {RED}❌ {msg}{RESET}")
def header(msg): print(f"\n{BOLD}{'─'*55}\n  {msg}\n{'─'*55}{RESET}")

# ── load data (same logic as train_tokenizer.py) ──────────────────────────────
def load_train_features():
    data_file = config.DATA_FILE
    files = [data_file] if isinstance(data_file, str) else data_file

    all_features = []
    for f in files:
        if not os.path.exists(f):
            print(f"  Warning: {f} not found, skipping.")
            continue
        df_raw = pd.read_csv(f)
        time_col = next((c for c in df_raw.columns if c.lower() in ["date", "datetime"]), None)
        if time_col:
            df_raw[time_col] = pd.to_datetime(df_raw[time_col])
            df_raw.set_index(time_col, inplace=True)

        fe = FeatureEngineer(config=FeatureConfig())
        feat_df = fe.build(df_raw["close"], ohlc=df_raw, dropna=True)
        
        # Identify robust vs no_scale for consistency with train.py
        all_cols = feat_df.columns.tolist()
        robust = [c for c in all_cols if "vs_factor" in c or "squeeze" in c]
        no_scale = [c for c in all_cols if c not in robust]
        ordered_cols = robust + no_scale
        
        asset_features = feat_df[ordered_cols].values.astype(np.float32)

        train_end = int(len(asset_features) * config.TRAIN_RATIO)
        asset_features = asset_features[:train_end]
        asset_features = np.nan_to_num(asset_features, nan=0.0, posinf=3.0, neginf=-3.0)
        asset_features = np.clip(asset_features, -3.0, 3.0)
        all_features.append(asset_features)

    if not all_features:
        raise ValueError("No data loaded.")
    return torch.FloatTensor(np.concatenate(all_features, axis=0))

# ── load tokenizer ─────────────────────────────────────────────────────────────
def load_tokenizer():
    if not os.path.exists(TOKENIZER_PATH):
        print(f"{RED}ERROR: {TOKENIZER_PATH} not found. Run train_tokenizer.py first.{RESET}")
        sys.exit(1)
    model = KLineTokenizer(
        input_dim=config.NUM_FEATURES if config.NUM_FEATURES else 21,
        n_bits=config.TOKENIZER_BITS,
        hidden_dim=256
    )
    model.load_state_dict(torch.load(TOKENIZER_PATH, map_location="cpu"))
    model.eval()
    return model

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 1 — Codebook Utilization
# ═══════════════════════════════════════════════════════════════════════════════
def check_codebook_utilization(model, x_train):
    header("CHECK 1 · Codebook Utilization")
    issues = []

    with torch.no_grad():
        indices = model.encode(x_train)   # (N,) direct encoding

    n_unique    = len(torch.unique(indices))
    vocab_size  = config.VOCAB_SIZE
    utilization = n_unique / vocab_size * 100

    print(f"  Vocab size       : {vocab_size:,}  (2^{config.TOKENIZER_BITS})")
    print(f"  Unique codes used: {n_unique:,}")
    print(f"  Utilization      : {utilization:.1f}%")

    if utilization >= 80:
        ok(f"Excellent utilization ({utilization:.1f}%)")
    elif utilization >= 50:
        warn(f"Moderate utilization ({utilization:.1f}%) — consider adding diversity loss")
        issues.append("moderate_util")
    elif utilization >= 20:
        warn(f"Low utilization ({utilization:.1f}%) — add diversity loss before retraining")
        issues.append("low_util")
    else:
        fail(f"CRITICAL: only {utilization:.1f}% of codebook used — codebook collapse")
        issues.append("CRITICAL_collapse")

    return indices, issues

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 2 — Bit Collapse
# ═══════════════════════════════════════════════════════════════════════════════
def check_bit_collapse(model, x_train):
    header("CHECK 2 · Bit Collapse Detection")
    issues = []

    sample = x_train[:5000] if len(x_train) > 5000 else x_train

    with torch.no_grad():
        z         = model.encoder(sample)       # (N, n_bits)
        bits      = (torch.sign(z) + 1) / 2    # binary [0,1]
        bit_means = bits.mean(dim=0)            # (n_bits,)

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
    n_bits = config.TOKENIZER_BITS
    if collapsed == 0:
        ok(f"All {n_bits} bits are healthy (0.2–0.8 ON-rate)")
    elif collapsed <= 2:
        warn(f"{collapsed}/{n_bits} bits marginal — add diversity loss for next run")
        issues.append("marginal_bits")
    else:
        fail(f"{collapsed}/{n_bits} bits collapsed — tokenizer is severely limited")
        issues.append("CRITICAL_bits")

    return issues

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 3 — Per-Feature Reconstruction MSE
# ═══════════════════════════════════════════════════════════════════════════════
def check_reconstruction(model, x_train):
    header("CHECK 3 · Per-Feature Reconstruction MSE")
    issues = []

    sample = x_train[:2000] if len(x_train) > 2000 else x_train

    with torch.no_grad():
        # Check reconstruction from the 'fine' decoder (full capacity)
        _, x_recon, _, _, _ = model(sample)

    per_feat_mse = ((x_recon - sample) ** 2).mean(dim=0).tolist()
    overall_mse  = float(np.mean(per_feat_mse))

    print(f"  {'Feature':<35}  {'MSE':>8}  Status")
    print(f"  {'─'*35}  {'─'*8}  {'─'*12}")
    bad_features = []
    for name, mse in sorted(zip(ALL_FEATURES, per_feat_mse), key=lambda x: -x[1]):
        if mse < 0.05:
            status = f"{GREEN}✅{RESET}"
        elif mse < 0.15:
            status = f"{YELLOW}⚠️ {RESET}"; bad_features.append(name)
        else:
            status = f"{RED}❌{RESET}";   bad_features.append(name)
        print(f"  {name:<35}  {mse:>8.4f}  {status}")

    print(f"\n  Overall MSE: {overall_mse:.4f}")
    if overall_mse < 0.05:
        ok(f"Excellent reconstruction (MSE={overall_mse:.4f})")
    elif overall_mse < 0.10:
        warn(f"Acceptable reconstruction (MSE={overall_mse:.4f})")
        issues.append("moderate_recon")
    else:
        fail(f"Poor reconstruction (MSE={overall_mse:.4f}) — model capacity may be too low")
        issues.append("CRITICAL_recon")

    if bad_features:
        warn(f"Poorly reconstructed features (MSE>0.05): {bad_features}")

    return issues

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 4 — Token Entropy
# ═══════════════════════════════════════════════════════════════════════════════
def check_token_entropy(indices):
    header("CHECK 4 · Token Entropy & Distribution")
    issues = []

    counts  = np.array(list(Counter(indices.tolist()).values()), dtype=np.float64)
    probs   = counts / counts.sum()
    entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
    max_ent = math.log2(config.VOCAB_SIZE)
    eff     = entropy / max_ent * 100

    top10     = sorted(Counter(indices.tolist()).items(), key=lambda x: -x[1])[:10]
    top10_pct = sum(v for _, v in top10) / len(indices) * 100

    print(f"  Token entropy     : {entropy:.2f} bits")
    print(f"  Max entropy       : {max_ent:.2f} bits  (perfectly uniform)")
    print(f"  Efficiency        : {eff:.1f}%")
    print(f"  Top-10 codes cover: {top10_pct:.1f}% of all tokens")
    print(f"  Top-10 codes      : {[t for t, _ in top10]}")

    if eff >= 70:
        ok(f"Good token distribution (efficiency={eff:.1f}%)")
    elif eff >= 50:
        warn(f"Moderate clustering (efficiency={eff:.1f}%) — add diversity loss")
        issues.append("moderate_entropy")
    else:
        fail(f"Heavy token clustering (efficiency={eff:.1f}%) — tokenizer near-useless")
        issues.append("CRITICAL_entropy")

    if top10_pct > 50:
        warn(f"Top-10 codes represent {top10_pct:.1f}% of tokens — very skewed")
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