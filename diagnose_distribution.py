"""
diagnose_distribution.py
════════════════════════════════════════════════════════════════════════════════
Full forensic audit of the train/val distribution mismatch.

What this checks (in order):
  STAGE 0 — Raw OHLC feature statistics (mean, std, skew) train vs val
  STAGE 1 — Post-normalization input to tokenizer  (after prepare_ohlc_features)
  STAGE 2 — Latent z before F.normalize / BSQ       (quant_embed output)
  STAGE 3 — Latent z AFTER F.normalize              (what BSQ actually sees)
  STAGE 4 — Token ID distribution                   (coarse + fine vocab usage)
  STAGE 5 — Token embedding cosine similarity       (train vs val centroids)
  STAGE 6 — Prediction vs target correlation        (Pearson, DirAcc, magnitude)

Run:
  python diagnose_distribution.py \
      --data_dir  Data/ \
      --tokenizer_path model.safetensors \
      --train_frac 0.7 \
      --window 128 \
      --stride 1

All outputs are printed AND written to diagnose_output.txt
════════════════════════════════════════════════════════════════════════════════
"""

import argparse
import os
import sys
import warnings
import traceback
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ── Logging helper ─────────────────────────────────────────────────────────────
LOG_LINES = []

def log(msg=""):
    print(msg)
    LOG_LINES.append(str(msg))

def save_log(path="diagnose_output.txt"):
    with open(path, "w") as f:
        f.write("\n".join(LOG_LINES))
    print(f"\n✓ Full output saved → {path}")

# ── Utility ────────────────────────────────────────────────────────────────────

def describe(arr, name, indent="  "):
    """Print mean/std/min/max/skew for a 2-D array (samples × features)."""
    if arr.ndim == 1:
        arr = arr[:, None]
    mean  = arr.mean(axis=0)
    std   = arr.std(axis=0)
    mn    = arr.min(axis=0)
    mx    = arr.max(axis=0)
    # pearson skewness approximation
    skew  = ((arr - mean) ** 3).mean(axis=0) / (std ** 3 + 1e-9)
    log(f"{indent}{name}")
    log(f"{indent}  mean : {np.round(mean, 4)}")
    log(f"{indent}  std  : {np.round(std,  4)}")
    log(f"{indent}  min  : {np.round(mn,   4)}")
    log(f"{indent}  max  : {np.round(mx,   4)}")
    log(f"{indent}  skew : {np.round(skew, 4)}")


def kl_div_bins(a, b, n_bins=50):
    """Symmetric KL divergence between two 1-D arrays using histogram bins."""
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    bins = np.linspace(lo, hi, n_bins + 1)
    pa, _ = np.histogram(a, bins=bins, density=True)
    pb, _ = np.histogram(b, bins=bins, density=True)
    pa = pa / (pa.sum() + 1e-12)
    pb = pb / (pb.sum() + 1e-12)
    eps = 1e-10
    kl_ab = np.sum(np.where(pa > eps, pa * np.log((pa + eps) / (pb + eps)), 0))
    kl_ba = np.sum(np.where(pb > eps, pb * np.log((pb + eps) / (pa + eps)), 0))
    return 0.5 * (kl_ab + kl_ba)


def pearson_corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.std(a) * np.std(b) + 1e-12)
    return float(np.mean(a * b) / denom)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_csv_files(data_dir):
    """Return a list of DataFrames, one per CSV found in data_dir."""
    dfs = []
    for p in sorted(Path(data_dir).glob("**/*.csv")):
        try:
            df = pd.read_csv(p)
            df.columns = [c.strip().title() for c in df.columns]
            required = {"Open", "High", "Low", "Close"}
            if not required.issubset(df.columns):
                log(f"  ⚠ Skipping {p.name}: missing OHLC columns")
                continue
            for col in ["Open", "High", "Low", "Close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            if "Volume" not in df.columns:
                df["Volume"] = 1.0
            df = df.dropna(subset=["Open", "High", "Low", "Close"])
            if len(df) < 600:
                log(f"  ⚠ Skipping {p.name}: only {len(df)} rows (<600)")
                continue
            df["__source__"] = p.name
            dfs.append(df.reset_index(drop=True))
            log(f"  ✓ Loaded {p.name} → {len(df)} rows")
        except Exception as e:
            log(f"  ✗ Error loading {p.name}: {e}")
    return dfs


# ── OHLC feature extraction (verbatim from tokenizer.py) ──────────────────────

def prepare_ohlc_features_RAW(df):
    """
    Verbatim copy of prepare_ohlc_features WITHOUT the rolling z-score.
    Used to measure the raw log-return distribution.
    """
    cols = {c.lower(): c for c in df.columns}
    o_col = cols.get("open", "Open")
    h_col = cols.get("high", "High")
    l_col = cols.get("low", "Low")
    c_col = cols.get("close", "Close")
    v_col = cols.get("volume", "Volume")

    close = df[c_col].values.astype(np.float64)
    prev_close = np.roll(close, 1)

    if v_col in df.columns:
        volume = df[v_col].values.astype(np.float64)
        amount = close * volume
    else:
        volume = np.zeros_like(close)
        amount = np.zeros_like(close)

    prev_volume = np.roll(volume, 1)
    prev_amount = np.roll(amount, 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        o = np.log(df[o_col].values.astype(np.float64) / prev_close)
        h = np.log(df[h_col].values.astype(np.float64) / prev_close)
        l = np.log(df[l_col].values.astype(np.float64) / prev_close)
        c = np.log(df[c_col].values.astype(np.float64) / prev_close)
        v = np.log((volume + 1e-6) / (prev_volume + 1e-6))
        a = np.log((amount + 1e-6) / (prev_amount + 1e-6))

    out = np.stack([o, h, l, c, v, a], axis=1)[1:]
    out = np.nan_to_num(out).astype(np.float32)
    return out


def prepare_ohlc_features_NORMED(df, window=500):
    """
    prepare_ohlc_features WITH rolling z-score.
    This is what the fixed tokenizer.py produces.
    """
    out = prepare_ohlc_features_RAW(df)
    df_out    = pd.DataFrame(out)
    roll_mean = df_out.rolling(window, min_periods=50).mean().bfill().values
    roll_std  = df_out.rolling(window, min_periods=50).std().bfill().values
    out = ((out - roll_mean) / (roll_std + 1e-8)).astype(np.float32)
    return out


# ── Token-level audit (no tokenizer weights needed) ───────────────────────────

def audit_sign_patterns(z_train, z_val, name="latent z"):
    """
    z shape: (N, D) — check how many bit positions systematically flip between
    train and val, which is what causes token ID drift in BSQ.
    """
    # fraction of positive values per dimension
    pos_train = (z_train > 0).mean(axis=0)  # (D,)
    pos_val   = (z_val   > 0).mean(axis=0)

    bit_drift = np.abs(pos_train - pos_val)
    log(f"  Sign-pattern drift ({name})")
    log(f"    mean |Δpos_fraction| per dim : {bit_drift.mean():.4f}")
    log(f"    max  |Δpos_fraction| per dim : {bit_drift.max():.4f}")
    log(f"    dims with |Δ| > 0.10         : {(bit_drift > 0.10).sum()} / {len(bit_drift)}")
    log(f"    dims with |Δ| > 0.20         : {(bit_drift > 0.20).sum()} / {len(bit_drift)}")

    # Expected token overlap: treat each dim as independent Bernoulli
    # P(same token) ≈ prod_d [p_t*p_v + (1-p_t)*(1-p_v)]
    p_same = pos_train * pos_val + (1 - pos_train) * (1 - pos_val)
    expected_overlap = p_same.prod()
    log(f"    Expected token ID overlap (independent approx): {expected_overlap:.4f}")
    if expected_overlap < 0.3:
        log(f"    ✗ CRITICAL: < 30% of val tokens match train token space")
    elif expected_overlap < 0.6:
        log(f"    ⚠ WARNING:  < 60% overlap — significant OOD token production")
    else:
        log(f"    ✓ Overlap looks acceptable")
    return bit_drift


# ── Tokenizer-based audit (requires model weights) ────────────────────────────

def load_tokenizer(path, device):
    """Load KronosTokenizer from .safetensors or .pt file."""
    from tokenizer import KronosTokenizer
    tok = KronosTokenizer()
    tok.load_pretrained(path, device=str(device))
    tok.eval()
    for p in tok.parameters():
        p.requires_grad_(False)
    return tok.to(device)


@torch.no_grad()
def extract_latents(tokenizer, feat_tensor, device, batch_size=512):
    """
    Run encoder + quant_embed on feat_tensor (N, T, 6).
    Returns:
      z_pre_norm : (N*T, D)  quant_embed output, before F.normalize
      z_post_norm: (N*T, D)  after F.normalize
      token_ids  : (N*T,)    BSQ token indices
    """
    tokenizer.eval()
    all_z_pre, all_z_post, all_ids = [], [], []

    for i in range(0, len(feat_tensor), batch_size):
        x = feat_tensor[i : i + batch_size].to(device)
        z = tokenizer.embed(x)
        for layer in tokenizer.encoder:
            z = layer(z)
        z = tokenizer.quant_embed(z)               # (B, T, D) — pre normalize

        z_flat = z.reshape(-1, z.shape[-1])
        z_norm = F.normalize(z_flat, dim=-1)        # post normalize

        # get token IDs via BSQ quantize → sign → bits_to_indices
        zq = tokenizer.tokenizer.bsq.quantize(z_norm)
        ids = tokenizer.tokenizer.bsq.codes_to_indexes(zq)

        all_z_pre.append(z_flat.cpu().float().numpy())
        all_z_post.append(z_norm.cpu().float().numpy())
        all_ids.append(ids.reshape(-1).cpu().numpy())

    return (
        np.concatenate(all_z_pre,  axis=0),
        np.concatenate(all_z_post, axis=0),
        np.concatenate(all_ids,    axis=0),
    )


def windows_from_feats(feats, window, stride=1):
    """Return (N, window, C) windows from (T, C) feature array."""
    T, C = feats.shape
    starts = range(0, T - window + 1, stride)
    return np.stack([feats[s : s + window] for s in starts], axis=0)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DIAGNOSTIC
# ═══════════════════════════════════════════════════════════════════════════════

def run_diagnostics(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    log(f"Args  : {vars(args)}\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    log("=" * 70)
    log("LOADING DATA")
    log("=" * 70)
    dfs = load_csv_files(args.data_dir)
    if not dfs:
        log("✗ No valid CSV files found. Check --data_dir")
        return

    # Combine all data; use chronological split
    combined_raw    = []  # raw log-returns
    combined_normed = []  # post rolling z-score
    for df in dfs:
        raw_   = prepare_ohlc_features_RAW(df)
        normed = prepare_ohlc_features_NORMED(df, window=args.norm_window)
        combined_raw.append(raw_)
        combined_normed.append(normed)

    raw_all    = np.concatenate(combined_raw,    axis=0)
    normed_all = np.concatenate(combined_normed, axis=0)

    N = len(raw_all)
    split_idx = int(N * args.train_frac)

    raw_train    = raw_all[:split_idx]
    raw_val      = raw_all[split_idx:]
    normed_train = normed_all[:split_idx]
    normed_val   = normed_all[split_idx:]

    log(f"\nTotal bars : {N}")
    log(f"Train      : {len(raw_train)} bars (0 → {split_idx})")
    log(f"Val        : {len(raw_val)} bars ({split_idx} → {N})")

    feat_names = ["log_ret_O", "log_ret_H", "log_ret_L", "log_ret_C", "log_ret_V", "log_ret_A"]

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 0 — Raw feature distribution
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("STAGE 0 — Raw log-return distribution (before any normalization)")
    log("=" * 70)
    describe(raw_train, "TRAIN raw", indent="  ")
    log()
    describe(raw_val,   "VAL   raw", indent="  ")

    log("\n  Per-feature symmetric KL divergence (train ‖ val):")
    kl_raw = []
    for i, fn in enumerate(feat_names):
        kl = kl_div_bins(raw_train[:, i], raw_val[:, i])
        kl_raw.append(kl)
        flag = "✗ HIGH" if kl > 0.3 else ("⚠ MED" if kl > 0.1 else "✓ OK")
        log(f"    {fn:14s}: KL = {kl:.4f}  {flag}")
    log(f"  Mean KL (raw): {np.mean(kl_raw):.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Post-normalization distribution
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("STAGE 1 — Post rolling z-score (what tokenizer.encode() receives)")
    log("=" * 70)
    describe(normed_train, "TRAIN normed", indent="  ")
    log()
    describe(normed_val,   "VAL   normed", indent="  ")

    log("\n  Per-feature KL (normed train ‖ normed val):")
    kl_normed = []
    for i, fn in enumerate(feat_names):
        kl = kl_div_bins(normed_train[:, i], normed_val[:, i])
        kl_normed.append(kl)
        flag = "✗ HIGH" if kl > 0.3 else ("⚠ MED" if kl > 0.1 else "✓ OK")
        log(f"    {fn:14s}: KL = {kl:.4f}  {flag}")
    log(f"  Mean KL (normed): {np.mean(kl_normed):.4f}")
    log(f"\n  KL improvement (raw → normed): {np.mean(kl_raw):.4f} → {np.mean(kl_normed):.4f}")
    reduction = (np.mean(kl_raw) - np.mean(kl_normed)) / (np.mean(kl_raw) + 1e-9)
    log(f"  Distribution gap reduced by: {reduction*100:.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2 & 3 — Latent space & token IDs (requires tokenizer weights)
    # ══════════════════════════════════════════════════════════════════════════
    if args.tokenizer_path and os.path.exists(args.tokenizer_path):
        log("\n" + "=" * 70)
        log("STAGE 2&3 — Latent z & Token ID audit (frozen tokenizer)")
        log("=" * 70)

        try:
            tokenizer = load_tokenizer(args.tokenizer_path, device)

            # Build windowed tensors — use a subset to keep it fast
            max_windows = 4000
            wt = windows_from_feats(normed_train, args.window, stride=max(1, len(normed_train) // max_windows))
            wv = windows_from_feats(normed_val,   args.window, stride=max(1, len(normed_val)   // max_windows))

            # Truncate to max_windows
            wt = wt[:max_windows]
            wv = wv[:max_windows]

            log(f"\n  Using {len(wt)} train windows, {len(wv)} val windows (len={args.window})")

            tt = torch.from_numpy(wt).float()
            tv = torch.from_numpy(wv).float()

            log("\n  Extracting train latents...")
            z_pre_train, z_post_train, ids_train = extract_latents(tokenizer, tt, device)
            log("  Extracting val latents...")
            z_pre_val,   z_post_val,   ids_val   = extract_latents(tokenizer, tv, device)

            # STAGE 2: pre-normalize latent
            log("\n--- STAGE 2: quant_embed output (pre F.normalize) ---")
            describe(z_pre_train, "TRAIN z_pre", indent="  ")
            log()
            describe(z_pre_val,   "VAL   z_pre", indent="  ")
            log()
            kl_pre = np.mean([kl_div_bins(z_pre_train[:, d], z_pre_val[:, d])
                              for d in range(min(z_pre_train.shape[1], 12))])
            log(f"  Mean KL over first 12 dims (pre-norm z): {kl_pre:.4f}")

            # STAGE 3: post-normalize latent (what BSQ sign() sees)
            log("\n--- STAGE 3: After F.normalize (what BSQ sees) ---")
            describe(z_post_train, "TRAIN z_post", indent="  ")
            log()
            describe(z_post_val,   "VAL   z_post", indent="  ")
            log()
            kl_post = np.mean([kl_div_bins(z_post_train[:, d], z_post_val[:, d])
                               for d in range(min(z_post_train.shape[1], 12))])
            log(f"  Mean KL over first 12 dims (post-norm z): {kl_post:.4f}")

            # Bit-flip audit (THE KEY DIAGNOSTIC)
            log("\n--- Bit-sign-pattern drift (post-normalize) ---")
            drift = audit_sign_patterns(z_post_train, z_post_val, name="z_post_norm")

            # STAGE 4: Token ID statistics
            log("\n--- STAGE 4: Token ID distribution ---")
            vocab_size = 2 ** tokenizer.codebook_dim
            train_vocab = np.unique(ids_train)
            val_vocab   = np.unique(ids_val)
            overlap_ids = np.intersect1d(train_vocab, val_vocab)

            log(f"  Vocab size (2^{tokenizer.codebook_dim})  : {vocab_size}")
            log(f"  Unique IDs used (train) : {len(train_vocab)} ({100*len(train_vocab)/vocab_size:.1f}%)")
            log(f"  Unique IDs used (val)   : {len(val_vocab)} ({100*len(val_vocab)/vocab_size:.1f}%)")
            log(f"  Token ID overlap        : {len(overlap_ids)} ({100*len(overlap_ids)/max(len(val_vocab),1):.1f}% of val vocab in train)")

            if len(overlap_ids) / max(len(val_vocab), 1) < 0.5:
                log(f"  ✗ CRITICAL: >50% of val token IDs were NEVER seen in train windows!")
            elif len(overlap_ids) / max(len(val_vocab), 1) < 0.8:
                log(f"  ⚠ WARNING:  >20% of val token IDs unseen in train")
            else:
                log(f"  ✓ Token ID overlap is acceptable")

            # Frequency KL on tokens
            train_counts = np.bincount(ids_train % vocab_size, minlength=vocab_size).astype(np.float64)
            val_counts   = np.bincount(ids_val   % vocab_size, minlength=vocab_size).astype(np.float64)
            train_p = train_counts / train_counts.sum()
            val_p   = val_counts   / val_counts.sum()
            eps = 1e-10
            kl_token = np.sum(np.where(val_p > eps, val_p * np.log((val_p + eps) / (train_p + eps)), 0))
            log(f"\n  Token distribution KL(val ‖ train): {kl_token:.4f}")
            if kl_token > 2.0:
                log(f"  ✗ CRITICAL: Token distributions are very divergent")
            elif kl_token > 0.5:
                log(f"  ⚠ WARNING:  Moderate token distribution shift")
            else:
                log(f"  ✓ Token distributions are close")

            # STAGE 5: Embedding centroid cosine similarity
            log("\n--- STAGE 5: Embedding centroid cosine similarity ---")
            # post_quant_embed is the embedding matrix
            with torch.no_grad():
                # Build all possible token IDs for the codebook_dim
                # (too many for large dims, so sample from observed IDs)
                sample_ids_train = torch.from_numpy(ids_train[:10000]).long().to(device)
                sample_ids_val   = torch.from_numpy(ids_val[:10000]).long().to(device)

                # Use indices_to_bits to get embeddings
                bits_t = tokenizer.tokenizer.bsq.get_codebook_entry(sample_ids_train)  # (N, D)
                bits_v = tokenizer.tokenizer.bsq.get_codebook_entry(sample_ids_val)

                # Project through post_quant_embed
                emb_t = tokenizer.post_quant_embed(bits_t).mean(0)  # centroid
                emb_v = tokenizer.post_quant_embed(bits_v).mean(0)

                cos_sim = F.cosine_similarity(emb_t.unsqueeze(0), emb_v.unsqueeze(0)).item()
            log(f"  Embedding centroid cosine similarity (train vs val): {cos_sim:.4f}")
            if cos_sim < 0.7:
                log(f"  ✗ CRITICAL: Embedding centroids differ substantially — "
                    f"model sees a different embedding 'universe' on val")
            elif cos_sim < 0.9:
                log(f"  ⚠ WARNING:  Some embedding shift — partial OOD")
            else:
                log(f"  ✓ Embedding centroids are close")

        except Exception as e:
            log(f"\n  ✗ Tokenizer audit failed: {e}")
            traceback.print_exc()
    else:
        log(f"\n  Skipping STAGE 2-5 (no tokenizer weights at '{args.tokenizer_path}')")
        log("  Pass --tokenizer_path model.safetensors to enable tokenizer audit")

        # Still do the sign-pattern audit on the raw normed features
        # (approximation: sign of feature directly — not exact but informative)
        log("\n  --- Approximate bit-sign-pattern drift (on normed features) ---")
        audit_sign_patterns(normed_train, normed_val, name="normed features (proxy)")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 6 — Simple linear probe: can we predict next-bar return at all?
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("STAGE 6 — Linear predictability check (no model needed)")
    log("  Tests whether the FEATURE SPACE itself has train/val correlation")
    log("=" * 70)

    def linear_probe(feat, horizon=1, name=""):
        """
        Fit OLS on train using lag-1 window, predict on val.
        Target = next bar close log-return.
        """
        X = feat[:-horizon]
        y = feat[horizon:, 3]  # log_ret_C is index 3
        split = int(len(X) * args.train_frac)
        Xtr, ytr = X[:split], y[:split]
        Xva, yva = X[split:], y[split:]

        # OLS via least-squares
        try:
            coeffs, _, _, _ = np.linalg.lstsq(
                np.hstack([Xtr, np.ones((len(Xtr), 1))]),
                ytr, rcond=None
            )
            y_pred_tr = Xtr @ coeffs[:-1] + coeffs[-1]
            y_pred_va = Xva @ coeffs[:-1] + coeffs[-1]

            c_tr = pearson_corr(y_pred_tr, ytr)
            c_va = pearson_corr(y_pred_va, yva)
            dir_tr = np.mean(np.sign(y_pred_tr) == np.sign(ytr))
            dir_va = np.mean(np.sign(y_pred_va) == np.sign(yva))
            log(f"  {name:20s}  Train Corr={c_tr:.4f}  DirAcc={dir_tr:.3f} | "
                f"Val Corr={c_va:.4f}  DirAcc={dir_va:.3f}")
        except Exception as e:
            log(f"  {name}: probe failed — {e}")

    linear_probe(raw_all,    name="Raw log-returns")
    linear_probe(normed_all, name="Normed log-returns")

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    log("\n" + "=" * 70)
    log("SUMMARY & VERDICT")
    log("=" * 70)

    mean_kl_raw    = np.mean(kl_raw)
    mean_kl_normed = np.mean(kl_normed)

    log(f"""
  Raw feature KL (train ‖ val)   : {mean_kl_raw:.4f}
  Normed feature KL (train ‖ val): {mean_kl_normed:.4f}

  Interpretation:
    • If KL(raw) is HIGH (>0.2) and KL(normed) is LOW (<0.1):
        → Rolling z-score normalization FIXES the distribution mismatch.
          The current code (which does apply rolling z-score) is correct.
          Root cause is in train.py / data_loader.py not using
          prepare_ohlc_features correctly, OR the tokenizer was being
          invoked on un-normalised features.

    • If KL(normed) is STILL HIGH (>0.2):
        → Even after normalization, train/val see different distributions.
          This means the val period has a structurally different regime
          (e.g., a major vol regime change) and no simple normalization
          will fully fix it. Consider:
            1. Increasing norm_window to 1000+
            2. Using WalkForward / expanding window training
            3. Unfreezing tokenizer on the full combined dataset

    • If Stage 4 shows <50% token overlap:
        → Token IDs themselves are OOD. Fix = normalize inputs BEFORE
          calling tokenizer.encode(). The rolling z-score in
          prepare_ohlc_features (already in tokenizer.py) is the fix.
          Confirm your data_loader calls prepare_ohlc_features and does
          NOT re-scale the features downstream.

  What to do next:
    1. Run this script with --tokenizer_path model.safetensors to get
       Stage 2-5 results (need the actual .safetensors weights).
    2. If token overlap <70%: Apply the rolling z-score fix and
       re-run this script to confirm KL dropped.
    3. Re-run training with the fixed tokenizer.py (already has the fix).
    4. If val Corr is still <0.1 after fix: the problem is in data_loader.py
       — check whether it calls prepare_ohlc_features() at all.
""")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose train/val distribution mismatch")
    parser.add_argument("--data_dir",        default="Data/",         help="Directory with CSV files")
    parser.add_argument("--tokenizer_path",  default="model.safetensors", help="Path to frozen tokenizer weights")
    parser.add_argument("--train_frac",      default=0.7,  type=float, help="Train fraction (chronological)")
    parser.add_argument("--window",          default=128,  type=int,   help="Window size for latent extraction")
    parser.add_argument("--stride",          default=10,   type=int,   help="Stride for window sampling")
    parser.add_argument("--norm_window",     default=500,  type=int,   help="Rolling z-score window size")
    parser.add_argument("--output",          default="diagnose_output.txt", help="Output log file")
    args = parser.parse_args()

    try:
        run_diagnostics(args)
    finally:
        save_log(args.output)
