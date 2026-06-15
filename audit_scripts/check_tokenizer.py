"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TOKENIZER DEEP HEALTH AUDIT v2.0                        ║
║                                                                            ║
║  Diagnoses EVERY aspect of tokenizer quality:                              ║
║    1. Data / Feature Diagnostics    – are inputs healthy?                  ║
║    2. Codebook Utilization          – per-stream, per-group, per-level     ║
║    3. Bit-Level Analysis            – collapse, correlation, entropy       ║
║    4. Reconstruction Quality        – per-feature, per-stream, per-pos     ║
║    5. Token Distribution            – entropy, Gini, Zipf, coverage        ║
║    6. Hierarchical S1/S2 Analysis   – conditional dependency               ║
║    7. Encoder Output Analysis       – norm distribution, saturation       ║
║    8. Loss Component Estimation     – recon vs commit vs entropy           ║
║    9. Group Quantization Analysis   – per-group utilization                ║
║   10. Architecture Sanity           – config mismatch detection            ║
║   11. Actionable Recommendations    – specific hyperparameter changes     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, math
sys.path.insert(0, os.getcwd())

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from datetime import datetime

import config
from tokenizer import KronosTokenizer, prepare_ohlc_features
from features import FeatureEngineer, FeatureConfig

# ── Hyperparameters (must match train_tokenizer.py) ─────────────────────────
SEQ_LEN      = 64
S1_BITS      = config.TOKENIZER_S1_BITS    # 10
S2_BITS      = config.TOKENIZER_S2_BITS    # 10
D_IN         = config.TOKENIZER_D_IN       # 24
D_MODEL      = config.TOKENIZER_D_MODEL    # 256
N_HEADS      = config.TOKENIZER_N_HEADS    # 4
FF_DIM       = config.TOKENIZER_FF_DIM     # 512
N_ENC        = config.TOKENIZER_N_ENC       # 4
N_DEC        = config.TOKENIZER_N_DEC       # 4
GROUP_SIZE   = config.TOKENIZER_GROUP_SIZE # 4
CODEBOOK_DIM = S1_BITS + S2_BITS            # 20
VOCAB_SIZE   = 2 ** S1_BITS                 # 1024
TOKENIZER_PATH = getattr(config, "TOKENIZER_PATH", "model.safetensors")
BETA   = getattr(config, "TOKENIZER_BETA", 0.05)
GAMMA0 = getattr(config, "TOKENIZER_GAMMA0", 1.0)
GAMMA  = getattr(config, "TOKENIZER_GAMMA", 1.5)
ZETA   = getattr(config, "TOKENIZER_ZETA", 0.25)

ALL_FEATURES = [
    "log_ret_open", "log_ret_high", "log_ret_low",
    "log_ret_close", "log_ret_vol", "log_ret_amount"
]

DEFAULT_FEATURE_LIST = [
    'open', 'high', 'low', 'close',
    'ewma_vol_span260',
    'ret_norm_1d', 'ret_norm_3d', 'ret_norm_6d', 'ret_norm_13d',
    'ret_norm_26d', 'ret_norm_65d', 'ret_norm_130d', 'ret_norm_260d',
    'macd_8_24', 'macd_26_78', 'macd_52_156',
    'feat_efficiency', 'feat_icp', 'feat_momentum_rsi',
    'feat_vol_asymmetry', 'feat_local_structure',
    'feat_session_sin', 'feat_session_cos', 'feat_vol_squeeze'
]

# ── ANSI colors ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"; CYAN = "\033[96m"
BOLD   = "\033[1m";  RESET  = "\033[0m"; DIM = "\033[2m"
MAGENTA = "\033[95m"

def ok(m):      print(f"  {GREEN}✅ {m}{RESET}")
def warn(m):    print(f"  {YELLOW}⚠️  {m}{RESET}")
def fail(m):    print(f"  {RED}❌ {m}{RESET}")
def info(m):    print(f"  {CYAN}ℹ️  {m}{RESET}")
def header(m):
    w = 70
    print(f"\n{BOLD}{'═' * w}")
    print(f"  {m}")
    print(f"{'═' * w}{RESET}")
def subheader(m):
    print(f"\n{BOLD}{'─' * 60}")
    print(f"  {m}")
    print(f"{'─' * 60}{RESET}")


def _make_feature_config():
    return FeatureConfig(
        ewma_span=getattr(config, "FE_VOL_LONG_PERIOD", 260),
        return_horizons=getattr(config, "FE_RETURN_HORIZONS", [1, 3, 6, 13, 26, 65, 130, 260]),
        macd_pairs=getattr(config, "FE_MACD_PAIRS", [(8, 24), (26, 78), (52, 156)]),
        macd_price_std_window=getattr(config, "FE_MACD_PRICE_STD_WIN", 260),
        macd_signal_std_window=getattr(config, "FE_MACD_SIGNAL_STD_WIN", 3276),
        target_clip=getattr(config, "FE_TARGET_CLIP", 20.0),
        momentum_period=getattr(config, "FE_MOMENTUM_PERIOD", 26),
        rsi_period=getattr(config, "FE_RSI_PERIOD", 14),
        vol_asym_window=getattr(config, "FE_VOL_ASYM_WINDOW", 65),
        icp_period=getattr(config, "FE_ICP_PERIOD", 13),
        local_structure_bars=getattr(config, "FE_LOCAL_STRUCTURE_BARS", 65),
        vol_squeeze_fast=getattr(config, "FE_VOL_SQUEEZE_FAST", 5),
        vol_squeeze_slow=getattr(config, "FE_VOL_SQUEEZE_SLOW", 26),
        atr_period=getattr(config, "ATR_PERIOD", 1),
        session_open=getattr(config, "FE_SESSION_OPEN", "09:15"),
        session_close=getattr(config, "FE_SESSION_CLOSE", "15:30"),
        session_tz=getattr(config, "FE_SESSION_TZ", "Asia/Kolkata"),
        add_session_features=getattr(config, "FE_ADD_SESSION", True),
        use_talib=getattr(config, "USE_TALIB", False),
    )


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load features exactly as training pipeline does — NO global normalization.
    Per-window z-score normalization happens inside build_windows() via
    tokenize_full_series() logic, matching the training pipeline exactly."""
    files = sorted(config.DATA_FILE)
    if isinstance(files, str): files = [files]
    fe = FeatureEngineer(_make_feature_config())
    all_feats = []
    for f in files:
        if not os.path.exists(f):
            continue
        print(f"  Loading {f}...")
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        cols = {c.lower(): c for c in df.columns}
        ohlc_df = df[[cols.get('open','Open'), cols.get('high','High'),
                       cols.get('low','Low'), cols.get('close','Close')]]
        prices = df[cols.get('close', 'Close')]
        eng_feats = fe.build(prices, ohlc=ohlc_df, include_target=False, dropna=False)
        df2 = df.join(eng_feats)
        for col in DEFAULT_FEATURE_LIST:
            if col not in df2.columns:
                df2[col] = 0.0
        feat_vals = df2[DEFAULT_FEATURE_LIST].values.astype(np.float32)
        feat_vals = np.nan_to_num(feat_vals, nan=0.0)
        # NO global normalization — training pipeline doesn't do this
        # Per-window z-score happens in build_windows()
        all_feats.append(feat_vals)
    data = np.concatenate(all_feats, axis=0).astype(np.float32)
    train_end = int(len(data) * config.TRAIN_RATIO)
    return torch.FloatTensor(data[:train_end])


def load_model():
    model = KronosTokenizer(
        d_in=D_IN, d_model=D_MODEL, n_heads=N_HEADS, ff_dim=FF_DIM,
        n_enc_layers=N_ENC, n_dec_layers=N_DEC,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        s1_bits=S1_BITS, s2_bits=S2_BITS,
        beta=BETA, gamma0=GAMMA0, gamma=GAMMA, zeta=ZETA,
        group_size=GROUP_SIZE
    )
    model.load_pretrained(TOKENIZER_PATH)
    model.eval()
    return model


def build_windows(data, seq_len, stride, max_w):
    T = len(data)
    starts = list(range(0, T - seq_len, stride))[:max_w]
    idx = torch.tensor(starts).unsqueeze(1) + torch.arange(seq_len).unsqueeze(0)
    windows = data[idx]  # (N, seq_len, C)
    # Per-window normalization — exactly matches QlibDataset.__getitem__
    # Each window gets its own mean/std computed over all timesteps in that window
    mean = windows.mean(dim=1, keepdim=True)   # (N, 1, C)
    std  = windows.std(dim=1, keepdim=True)    # (N, 1, C)
    windows = (windows - mean) / (std + 1e-5)
    windows = windows.clamp(-5.0, 5.0)
    return windows


def encode_full(model, windows, batch_size=64):
    """Run full model forward pass, returning all intermediates."""
    all_z_pre, all_z_full, all_quantized = [], [], []
    all_indices_s1, all_indices_s2 = [], []
    all_encoder_out = []

    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            b = windows[i:i+batch_size]
            # Manual forward to capture intermediates
            z = model.embed(b)
            for layer in model.encoder:
                z = layer(z)
            enc_out = z.clone()
            all_encoder_out.append(enc_out)

            z = model.quant_embed(z)
            bsq_loss, quantized, z_indices, metrics = model.tokenizer(z, half=True)
            all_quantized.append(quantized)
            all_indices_s1.append(z_indices[0])
            all_indices_s2.append(z_indices[1])

            # Decoder pass
            quantized_pre = quantized[:, :, :S1_BITS]
            z_pre = model.post_quant_embed_pre(quantized_pre)
            for layer in model.decoder_pre:
                z_pre = layer(z_pre)
            z_pre = model.head(z_pre)
            all_z_pre.append(z_pre)

            z_full = model.post_quant_embed(quantized)
            for layer in model.decoder_full:
                z_full = layer(z_full)
            z_full = model.head(z_full)
            all_z_full.append(z_full)

    return {
        'z_pre':      torch.cat(all_z_pre),
        'z_full':     torch.cat(all_z_full),
        'quantized':  torch.cat(all_quantized),
        'indices_s1': torch.cat(all_indices_s1),
        'indices_s2': torch.cat(all_indices_s2),
        'encoder_out': torch.cat(all_encoder_out),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 1: DATA / FEATURE INPUT DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def check_data_diagnostics(data):
    """Are the input features healthy? Detects dead features, saturation, scale mismatches."""
    header("CHECK 1 · Input Feature Health")
    issues = []

    T, C = data.shape
    feature_names = DEFAULT_FEATURE_LIST
    print(f"  Samples: {T:,}  |  Features: {C}")

    # Per-feature statistics
    dead_features = []
    saturated_features = []
    low_variance_features = []
    scale_issues = []

    stats_rows = []
    for i, name in enumerate(feature_names):
        col = data[:, i]
        valid = col[~torch.isnan(col)] if col.dtype.is_floating_point else col
        if len(valid) == 0:
            dead_features.append(name)
            continue

        mean = valid.mean().item()
        std  = valid.std().item()
        mn   = valid.min().item()
        mx   = valid.max().item()
        # Fraction at clamp bounds
        at_low  = (valid < -4.5).float().mean().item()
        at_high = (valid > 4.5).float().mean().item()
        sat_frac = at_low + at_high
        # Fraction exactly zero
        zero_frac = (valid.abs() < 1e-8).float().mean().item()

        if std < 1e-6:
            dead_features.append(name)
        elif sat_frac > 0.1:
            saturated_features.append((name, sat_frac))
        elif std < 0.01:
            low_variance_features.append((name, std))

        # Scale check: after normalization, most features should be in [-3, 3]
        # Features with |mean| > 1 or std > 3 are likely not normalized properly
        if abs(mean) > 2.0 or std > 5.0:
            scale_issues.append((name, mean, std))

        stats_rows.append((name, mean, std, mn, mx, zero_frac, sat_frac))

    # Print table
    print(f"\n  {'Feature':<28} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Zero%':>7} {'Sat%':>7}")
    print(f"  {'─'*28} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*7} {'─'*7}")
    for name, mean, std, mn, mx, zf, sf in stats_rows:
        flags = ""
        if name in dead_features:
            flags = f"{RED}DEAD{RESET}"
        elif sf > 0.1:
            flags = f"{YELLOW}SAT{RESET}"
        elif std < 0.01:
            flags = f"{YELLOW}LOW-VAR{RESET}"
        elif any(n == name for n, _, _ in scale_issues):
            flags = f"{MAGENTA}SCALE?{RESET}"
        print(f"  {name:<28} {mean:>8.3f} {std:>8.3f} {mn:>8.2f} {mx:>8.2f} {zf*100:>6.1f}% {sf*100:>6.1f}% {flags}")

    print()
    if dead_features:
        fail(f"DEAD features (std≈0): {', '.join(dead_features)}")
        issues.append(f"dead_features={dead_features}")
    else:
        ok("All features have non-zero variance")

    if saturated_features:
        names = [f"{n} ({f*100:.1f}%)" for n, f in saturated_features]
        warn(f"Saturated (>10% at ±5 clamp): {', '.join(names)}")
        issues.append(f"saturated={saturated_features}")
    else:
        ok("No features heavily saturated at clamp bounds")

    if low_variance_features:
        names = [f"{n} (σ={s:.4f})" for n, s in low_variance_features]
        warn(f"Low-variance features: {', '.join(names)}")
        issues.append(f"low_var={low_variance_features}")
    else:
        ok("All features have reasonable variance")

    if scale_issues:
        names = [f"{n} (μ={m:.2f}, σ={s:.2f})" for n, m, s in scale_issues]
        warn(f"Scale issues (possible mis-normalization): {', '.join(names)}")
        issues.append(f"scale_issues={scale_issues}")

    # Feature correlation check — detect redundant features
    subheader("Feature Correlation Matrix (absolute, >0.95)")
    corr = torch.corrcoef(data.T.float())
    high_corr = []
    for i in range(C):
        for j in range(i+1, C):
            if abs(corr[i, j].item()) > 0.95:
                high_corr.append((feature_names[i], feature_names[j], corr[i, j].item()))

    if high_corr:
        warn(f"Found {len(high_corr)} highly correlated feature pairs (|r|>0.95):")
        for a, b, r in high_corr[:10]:
            print(f"    {a:<28} ↔ {b:<28}  r={r:>+.4f}")
        issues.append(f"high_correlation={len(high_corr)}_pairs")
    else:
        ok("No highly correlated feature pairs")

    return issues


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 2: ARCHITECTURE / CONFIG SANITY
# ══════════════════════════════════════════════════════════════════════════════

def check_architecture(model):
    """Verify model architecture matches config, detect common misconfigurations."""
    header("CHECK 2 · Architecture & Config Sanity")
    issues = []

    configs = {
        'd_in': model.d_in, 'd_model': D_MODEL, 's1_bits': model.s1_bits,
        's2_bits': model.s2_bits, 'codebook_dim': model.codebook_dim,
        'n_enc_layers': len(model.encoder) + 1,
        'n_dec_layers': len(model.decoder_full) + 1,
    }
    print(f"  MODEL SPECS:")
    for k, v in configs.items():
        print(f"    {k:<20s} = {v}")
    print(f"  LOSS HYPERPARAMETERS:")
    print(f"    {'beta':<20s} = {BETA}")
    print(f"    {'gamma0':<20s} = {GAMMA0}")
    print(f"    {'gamma':<20s} = {GAMMA}")
    print(f"    {'zeta':<20s} = {ZETA}")
    print(f"    {'group_size':<20s} = {GROUP_SIZE}")

    # Check total parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters: {total:,}  |  Trainable: {trainable:,}")

    # Architecture warnings
    issues_list = []

    # Codebook size vs data ratio
    cb_size = 2 ** CODEBOOK_DIM  # 2^20 = ~1M
    if cb_size > 100_000:
        warn(f"Huge codebook: 2^{CODEBOOK_DIM} = {cb_size:,} codes — very hard to fully utilize")
        issues_list.append(f"huge_codebook_{cb_size}")

    # d_model vs codebook_dim ratio
    ratio = D_MODEL / CODEBOOK_DIM
    if ratio < 4:
        warn(f"d_model/codebook_dim = {ratio:.1f} — encoder may not have enough capacity to produce diverse codes")
        issues_list.append(f"low_capacity_ratio_{ratio:.1f}")

    # Bit balance
    if abs(S1_BITS - S2_BITS) > 2:
        warn(f"S1/S2 bit imbalance: {S1_BITS}/{S2_BITS} — may cause one stream to dominate")
        issues_list.append(f"bit_imbalance_{S1_BITS}_{S2_BITS}")

    # Group size divisibility
    if CODEBOOK_DIM % GROUP_SIZE != 0:
        fail(f"CODEBOOK_DIM ({CODEBOOK_DIM}) not divisible by GROUP_SIZE ({GROUP_SIZE})!")
        issues_list.append("group_not_divisible")

    n_groups = CODEBOOK_DIM // GROUP_SIZE
    if n_groups < 2:
        warn(f"Only {n_groups} quantization group(s) — limits codebook diversity")
        issues_list.append(f"few_groups_{n_groups}")

    # Loss weight balance check
    effective_commit_weight = BETA
    effective_entropy_weight = ZETA * GAMMA
    if effective_commit_weight > 10 * effective_entropy_weight:
        warn(f"Beta ({BETA}) >> Zeta*Gamma ({effective_entropy_weight:.3f}) — commit loss may overpower entropy")
        issues_list.append(f"commit_dominant_beta={BETA}")
    elif effective_entropy_weight > 10 * effective_commit_weight:
        warn(f"Zeta*Gamma ({effective_entropy_weight:.3f}) >> Beta ({BETA}) — entropy may dominate, codebook under-trained")
        issues_list.append(f"entropy_dominant")

    print()
    if not issues_list:
        ok("Architecture looks sound")

    return issues_list


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 3: CODEBOOK UTILIZATION (DETAILED)
# ══════════════════════════════════════════════════════════════════════════════

def check_utilization_detailed(results):
    """Deep codebook utilization analysis — per-stream, per-position, per-timestep."""
    header("CHECK 3 · Codebook Utilization (Detailed)")
    issues = []
    s1 = results['indices_s1']
    s2 = results['indices_s2']
    B, T = s1.shape

    vocab = VOCAB_SIZE  # 2^S1_BITS

    # ── Overall utilization ─────────────────────────────────────────────────
    u1_total = len(torch.unique(s1))
    u2_total = len(torch.unique(s2))
    pct1 = u1_total / vocab * 100
    pct2 = u2_total / vocab * 100

    print(f"  UNIQUE CODES USED:")
    print(f"    Stream S1 (coarse, {S1_BITS}-bit): {u1_total:>5,} / {vocab:,}  ({pct1:.1f}%)")
    print(f"    Stream S2 (fine,   {S2_BITS}-bit): {u2_total:>5,} / {vocab:,}  ({pct2:.1f}%)")

    # Last-timestep only (most informative — this is what the encoder chose)
    u1_last = len(torch.unique(s1[:, -1]))
    u2_last = len(torch.unique(s2[:, -1]))
    pct1_last = u1_last / vocab * 100
    pct2_last = u2_last / vocab * 100
    print(f"  LAST TIMESTEP ONLY:")
    print(f"    Stream S1: {u1_last:>5,} / {vocab:,}  ({pct1_last:.1f}%)")
    print(f"    Stream S2: {u2_last:>5,} / {vocab:,}  ({pct2_last:.1f}%)")

    avg_util = (pct1 + pct2) / 2
    if avg_util >= 80:
        ok(f"Excellent utilization ({avg_util:.1f}%)")
    elif avg_util >= 40:
        warn(f"Moderate utilization ({avg_util:.1f}%)")
        issues.append(f"moderate_utilization_{avg_util:.1f}")
    else:
        fail(f"CRITICAL: Low utilization ({avg_util:.1f}%) — codebook collapse")
        issues.append(f"CRITICAL_collapse_{avg_util:.1f}")

    # ── Per-timestep utilization ────────────────────────────────────────────
    subheader("Per-Timestep Utilization (S1)")
    per_step_s1 = []
    for t in range(T):
        n = len(torch.unique(s1[:, t]))
        per_step_s1.append(n)
    per_step_s2 = []
    for t in range(T):
        n = len(torch.unique(s2[:, t]))
        per_step_s2.append(n)

    print(f"  {'Pos':>4}  {'S1 Used':>8}  {'S2 Used':>8}  {'S1 %':>8}  {'S2 %':>8}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    for t in range(T):
        s1p = per_step_s1[t] / vocab * 100
        s2p = per_step_s2[t] / vocab * 100
        flag = ""
        if s1p < 10 and s2p < 10:
            flag = f"{RED}BOTH LOW{RESET}"
        elif s1p < 10 or s2p < 10:
            flag = f"{YELLOW}LOW{RESET}"
        print(f"  {t:>4}  {per_step_s1[t]:>8,}  {per_step_s2[t]:>8,}  {s1p:>7.1f}%  {s2p:>7.1f}%  {flag}")

    # Check for positional bias — earlier positions should have higher utilization
    first_5_avg = np.mean(per_step_s1[:5]) / vocab * 100
    last_5_avg  = np.mean(per_step_s1[-5:]) / vocab * 100
    if first_5_avg > last_5_avg * 1.5:
        warn(f"Early positions use {first_5_avg:.1f}% codes vs last positions {last_5_avg:.1f}% — positional collapse toward end")
        issues.append("positional_collapse")

    return issues, s1, s2


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 4: BIT-LEVEL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def check_bit_analysis(results):
    """Detailed bit analysis — collapse, correlation, mutual information proxy."""
    header("CHECK 4 · Bit-Level Analysis")
    issues = []
    s1 = results['indices_s1'][:, -1]  # last timestep
    s2 = results['indices_s2'][:, -1]
    N = len(s1)
    device = s1.device

    # Convert indices to bit arrays
    basis_s1 = (2 ** torch.arange(S1_BITS, device=device)).long()
    basis_s2 = (2 ** torch.arange(S2_BITS, device=device)).long()
    bits_s1 = ((s1.unsqueeze(-1) & basis_s1) != 0).float()
    bits_s2 = ((s2.unsqueeze(-1) & basis_s2) != 0).float()
    all_bits = torch.cat([bits_s1, bits_s2], dim=-1)  # (N, 20)
    n_bits = S1_BITS + S2_BITS

    # ── Bit activation rates ────────────────────────────────────────────────
    rates = all_bits.mean(0).tolist()
    print(f"  {'Stream':>6} {'Bit':>4}  {'ON-rate':>8}  {'OFF-rate':>8}  {'Bias':>8}  Status")
    print(f"  {'─'*6} {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*15}")

    collapsed_bits = []
    marginal_bits = []
    healthy_bits = []

    for i in range(n_bits):
        stream = "S1" if i < S1_BITS else "S2"
        bit_idx = i if i < S1_BITS else i - S1_BITS
        r = rates[i]
        bias = abs(r - 0.5)
        if 0.20 <= r <= 0.80:
            status = f"{GREEN}✅ healthy{RESET}"
            healthy_bits.append(i)
        elif 0.10 <= r < 0.20 or 0.80 < r <= 0.90:
            status = f"{YELLOW}⚠️  marginal{RESET}"
            marginal_bits.append((i, r))
        else:
            status = f"{RED}❌ COLLAPSED{RESET}"
            collapsed_bits.append((i, r))
        print(f"  {stream:>6} {bit_idx:>4}  {r:>8.4f}  {1-r:>8.4f}  {bias:>8.4f}  {status}")

    print(f"\n  Summary: {len(healthy_bits)} healthy, {len(marginal_bits)} marginal, {len(collapsed_bits)} collapsed")

    # ── Bit bias direction analysis ─────────────────────────────────────────
    if collapsed_bits:
        subheader("Collapsed Bit Direction Analysis")
        for i, r in collapsed_bits:
            stream = "S1" if i < S1_BITS else "S2"
            bit_idx = i if i < S1_BITS else i - S1_BITS
            direction = "always-1" if r > 0.5 else "always-0"
            print(f"    {stream}[{bit_idx}] → {direction} (rate={r:.4f})")

    # ── Bit correlation matrix ───────────────────────────────────────────────
    subheader("Bit Correlation Analysis (Pearson r, |r|>0.5)")
    bit_corr = torch.corrcoef(all_bits.T.float())
    correlated_pairs = []
    for i in range(n_bits):
        for j in range(i+1, n_bits):
            if abs(bit_corr[i, j].item()) > 0.5:
                correlated_pairs.append((i, j, bit_corr[i, j].item()))

    if correlated_pairs:
        warn(f"Found {len(correlated_pairs)} correlated bit pairs (|r|>0.5):")
        for i, j, r in correlated_pairs[:15]:
            si = f"S1[{i}]" if i < S1_BITS else f"S2[{i-S1_BITS}]"
            sj = f"S1[{j}]" if j < S1_BITS else f"S2[{j-S1_BITS}]"
            print(f"    {si:>6} ↔ {sj:>6}  r={r:>+.4f}")
            issues.append(f"bit_correlation_{si}_{sj}_{r:.2f}")
    else:
        ok("No highly correlated bit pairs (|r|>0.5)")

    # ── Effective bits ───────────────────────────────────────────────────────
    # Shannon entropy per bit
    entropies = []
    for i in range(n_bits):
        r = rates[i]
        if r < 1e-8 or r > 1 - 1e-8:
            entropies.append(0.0)
        else:
            h = -(r * math.log2(r) + (1-r) * math.log2(1-r))
            entropies.append(h)

    total_entropy = sum(entropies)
    max_possible   = n_bits  # 1 bit per dimension
    effective_bits = total_entropy / max_possible * 100

    print(f"\n  Bit-level Shannon entropy:")
    print(f"    Total effective bits: {total_entropy:.2f} / {n_bits} ({effective_bits:.1f}%)")
    for i in range(n_bits):
        stream = "S1" if i < S1_BITS else "S2"
        bit_idx = i if i < S1_BITS else i - S1_BITS
        print(f"    {stream}[{bit_idx}]: H={entropies[i]:.4f} bits")

    if effective_bits < 50:
        fail(f"Only {effective_bits:.1f}% effective bits — severe bit collapse")
        issues.append(f"CRITICAL_low_effective_bits_{effective_bits:.1f}")
    elif effective_bits < 75:
        warn(f"Moderate effective bits: {effective_bits:.1f}%")
        issues.append(f"moderate_effective_bits_{effective_bits:.1f}")
    else:
        ok(f"Good effective bit usage: {effective_bits:.1f}%")

    # ── Verdict ─────────────────────────────────────────────────────────────
    n_collapsed = len(collapsed_bits)
    n_marginal  = len(marginal_bits)
    if n_collapsed == 0 and n_marginal == 0:
        ok(f"All {n_bits} bits healthy")
    elif n_collapsed + n_marginal <= 3:
        warn(f"{n_collapsed} collapsed + {n_marginal} marginal bits")
        issues.append(f"some_bit_collapse_{n_collapsed}_{n_marginal}")
    else:
        fail(f"{n_collapsed} collapsed + {n_marginal} marginal bits — severe")
        issues.append(f"CRITICAL_bits_{n_collapsed}_{n_marginal}")

    return issues


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 5: RECONSTRUCTION QUALITY (DETAILED)
# ══════════════════════════════════════════════════════════════════════════════

def check_reconstruction_detailed(results, windows):
    """Deep reconstruction analysis — per-feature, per-stream, per-position."""
    header("CHECK 5 · Reconstruction Quality (Detailed)")
    issues = []

    x_true    = windows[:len(results['z_full'])]
    z_pre     = results['z_pre']
    z_full    = results['z_full']
    N, T, C   = x_true.shape

    # ── S1-only (coarse) reconstruction ─────────────────────────────────────
    mse_pre = ((z_pre - x_true) ** 2).mean(dim=(0, 1))
    mse_full = ((z_full - x_true) ** 2).mean(dim=(0, 1))
    var_per_feat = x_true.var(dim=(0, 1)).clamp(min=1e-9)

    nmse_pre  = (mse_pre / var_per_feat).tolist()
    nmse_full = (mse_full / var_per_feat).tolist()

    print(f"  {'Feature':<28} {'NMSE(S1)':>10} {'NMSE(S1+S2)':>12} {'Improve':>10}  Status")
    print(f"  {'─'*28} {'─'*10} {'─'*12} {'─'*10} {'─'*12}")

    feature_names_local = DEFAULT_FEATURE_LIST[:C] if C <= len(DEFAULT_FEATURE_LIST) else [f"feat_{i}" for i in range(C)]
    recon_issues = []
    for i in range(C):
        name = feature_names_local[i] if i < len(feature_names_local) else f"feat_{i}"
        imp = (1 - nmse_full[i] / max(nmse_pre[i], 1e-9)) * 100 if nmse_pre[i] > 0 else 0
        if nmse_full[i] < 0.5:
            status = f"{GREEN}✅ good{RESET}"
        elif nmse_full[i] < 0.85:
            status = f"{YELLOW}⚠️  marginal{RESET}"
            recon_issues.append((name, nmse_full[i]))
        else:
            status = f"{RED}❌ poor{RESET}"
            recon_issues.append((name, nmse_full[i]))
        print(f"  {name:<28} {nmse_pre[i]:>10.4f} {nmse_full[i]:>12.4f} {imp:>9.1f}%  {status}")

    avg_nmse_pre  = np.mean(nmse_pre)
    avg_nmse_full = np.mean(nmse_full)
    print(f"\n  Overall NMSE: S1-only={avg_nmse_pre:.4f}  |  S1+S2={avg_nmse_full:.4f}")
    print(f"  S2 improvement: {(1 - avg_nmse_full/max(avg_nmse_pre,1e-9))*100:.1f}%")

    if avg_nmse_full < 0.5:
        ok(f"Excellent reconstruction (NMSE={avg_nmse_full:.3f})")
    elif avg_nmse_full < 0.85:
        warn(f"Moderate reconstruction (NMSE={avg_nmse_full:.3f})")
        issues.append(f"moderate_recon_{avg_nmse_full:.3f}")
    else:
        fail(f"Poor reconstruction (NMSE={avg_nmse_full:.3f})")
        issues.append(f"CRITICAL_recon_{avg_nmse_full:.3f}")

    # Which features benefit most from S2?
    subheader("S2 (Fine Stream) Contribution Per Feature")
    s2_contrib = []
    for i in range(C):
        name = feature_names_local[i] if i < len(feature_names_local) else f"feat_{i}"
        if nmse_pre[i] > 1e-9:
            reduction = (nmse_pre[i] - nmse_full[i]) / nmse_pre[i] * 100
        else:
            reduction = 0
        s2_contrib.append((name, reduction, nmse_pre[i], nmse_full[i]))
    s2_contrib.sort(key=lambda x: -x[1])

    print(f"  {'Feature':<28} {'S2 Reduction':>14}  {'S1 NMSE':>10}  {'S1+S2 NMSE':>12}")
    print(f"  {'─'*28} {'─'*14}  {'─'*10}  {'─'*12}")
    for name, red, pre, full in s2_contrib:
        flag = f"{GREEN}+{RESET}" if red > 10 else (f"{YELLOW}~{RESET}" if red > 0 else f"{RED}-{RESET}")
        print(f"  {name:<28}  {flag}{red:>12.1f}%  {pre:>10.4f}  {full:>12.4f}")

    # Check if S2 provides meaningful improvement at all
    s2_benefit_count = sum(1 for _, r, _, _ in s2_contrib if r > 10)
    total_feat = len(s2_contrib)
    if s2_benefit_count == 0:
        fail("S2 stream provides NO meaningful improvement — stream is dead")
        issues.append("CRITICAL_dead_s2_stream")
    elif s2_benefit_count < total_feat // 2:
        warn(f"S2 only helps {s2_benefit_count}/{total_feat} features — underutilized")
        issues.append(f"underutilized_s2_{s2_benefit_count}_{total_feat}")
    else:
        ok(f"S2 helps {s2_benefit_count}/{total_feat} features")

    # Per-position reconstruction quality
    subheader("Per-Position NMSE (last 10 positions)")
    pos_nmse = []
    for t in range(max(0, T-10), T):
        mse_t = ((z_full[:, t, :] - x_true[:, t, :]) ** 2).mean(0)
        nmse_t = (mse_t / var_per_feat.clamp(min=1e-9)).mean().item()
        pos_nmse.append((t, nmse_t))
        bar = "█" * int(nmse_t * 50) if nmse_t < 2 else "█" * 100
        print(f"  Pos {t:>3}: NMSE={nmse_t:.4f}  {DIM}{bar}{RESET}")

    return issues


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 6: TOKEN DISTRIBUTION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def check_token_distribution(s1, s2):
    """Entropy, Gini coefficient, Zipf's law, coverage curves."""
    header("CHECK 6 · Token Distribution Analysis")
    issues = []

    device = s1.device

    def analyze_stream(indices, name, n_bits):
        """indices: (N, T) — analyze distribution."""
        flat = indices.flatten()
        N_total = len(flat)
        vocab = 2 ** n_bits

        # Use numpy for frequency counting
        counts_np = torch.bincount(flat, minlength=vocab).cpu().numpy()
        nonzero = counts_np[counts_np > 0]

        # Shannon entropy
        probs = nonzero / N_total
        entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
        efficiency = entropy / n_bits * 100

        # Gini coefficient
        sorted_counts = np.sort(nonzero)
        n = len(sorted_counts)
        cumulative = np.cumsum(sorted_counts)
        gini = (2 * np.sum((np.arange(1, n+1) * sorted_counts)) / (n * cumulative[-1]) - (n + 1) / n) if n > 0 and cumulative[-1] > 0 else 0.0

        # Top-K coverage
        sorted_desc = np.sort(nonzero)[::-1]
        top1_counts  = sorted_desc[:1].sum() / N_total * 100 if len(sorted_desc) >= 1 else 0
        top5_counts  = sorted_desc[:5].sum() / N_total * 100 if len(sorted_desc) >= 5 else 0
        top10_counts = sorted_desc[:10].sum() / N_total * 100 if len(sorted_desc) >= 10 else 0
        top50_counts = sorted_desc[:50].sum() / N_total * 100 if len(sorted_desc) >= 50 else 0
        top100_counts = sorted_desc[:100].sum() / N_total * 100 if len(sorted_desc) >= 100 else 0

        # Codes needed for 50%, 80%, 95% coverage
        cum_frac = np.cumsum(sorted_desc) / N_total
        codes_50 = int(np.searchsorted(cum_frac, 0.50)) + 1
        codes_80 = int(np.searchsorted(cum_frac, 0.80)) + 1
        codes_95 = int(np.searchsorted(cum_frac, 0.95)) + 1

        # Zipf's law test: log-log slope
        freqs = sorted_desc[:min(500, len(sorted_desc))]
        if len(freqs) >= 10:
            log_ranks = np.log1p(np.arange(len(freqs)))
            log_freqs = np.log1p(freqs)
            slope, intercept = np.polyfit(log_ranks, log_freqs, 1)
        else:
            slope = 0

        # Dead codes (never used)
        n_used = int((counts_np > 0).sum())
        dead_pct = (vocab - n_used) / vocab * 100

        print(f"\n  ── Stream {name} ({n_bits}-bit, vocab={vocab:,}) ──")
        print(f"    Unique codes used:  {n_used:>6,} / {vocab:,}  ({n_used/vocab*100:.1f}%)")
        print(f"    Dead codes:         {vocab-n_used:>6,} ({dead_pct:.1f}%)")
        print(f"    Shannon entropy:    {entropy:.3f} bits  (efficiency: {efficiency:.1f}%)")
        print(f"    Gini coefficient:   {gini:.4f}  (0=perfectly uniform, 1=single code dominates)")
        print(f"    Zipf slope:         {slope:.3f}  (ideal ≈ -1.0)")
        print(f"    Top-1 coverage:     {top1_counts:.1f}%")
        print(f"    Top-5 coverage:     {top5_counts:.1f}%")
        print(f"    Top-10 coverage:    {top10_counts:.1f}%")
        print(f"    Top-50 coverage:    {top50_counts:.1f}%")
        print(f"    Top-100 coverage:   {top10_counts:.1f}%")
        print(f"    Codes for 50% cov:  {codes_50:,} ({codes_50/vocab*100:.1f}% of vocab)")
        print(f"    Codes for 80% cov:  {codes_80:,} ({codes_80/vocab*100:.1f}% of vocab)")
        print(f"    Codes for 95% cov:  {codes_95:,} ({codes_95/vocab*100:.1f}% of vocab)")

        stream_issues = []
        if efficiency >= 80:
            ok(f"{name}: Excellent distribution ({efficiency:.1f}% entropy efficiency)")
        elif efficiency >= 60:
            warn(f"{name}: Moderate distribution ({efficiency:.1f}% efficiency)")
            stream_issues.append(f"moderate_entropy_{name}_{efficiency:.1f}")
        elif efficiency >= 40:
            warn(f"{name}: Clustered distribution ({efficiency:.1f}% efficiency)")
            stream_issues.append(f"clustered_entropy_{name}_{efficiency:.1f}")
        else:
            fail(f"{name}: Severe clustering ({efficiency:.1f}% efficiency)")
            stream_issues.append(f"CRITICAL_entropy_{name}_{efficiency:.1f}")

        if gini > 0.8:
            fail(f"{name}: Gini={gini:.3f} — extremely skewed (near-collapse)")
            stream_issues.append(f"CRITICAL_gini_{name}_{gini:.3f}")
        elif gini > 0.6:
            warn(f"{name}: Gini={gini:.3f} — moderately skewed distribution")
            stream_issues.append(f"moderate_gini_{name}_{gini:.3f}")
        else:
            ok(f"{name}: Gini={gini:.3f} — acceptable uniformity")

        if top1_counts > 30:
            fail(f"{name}: Top-1 code covers {top1_counts:.1f}% — massive collapse to single code")
            stream_issues.append(f"CRITICAL_top1_{name}_{top1_counts:.1f}")
        elif top1_counts > 10:
            warn(f"{name}: Top-1 code covers {top1_counts:.1f}% — skewed")

        if dead_pct > 90:
            fail(f"{name}: {dead_pct:.1f}% dead codes — severe underutilization")
            stream_issues.append(f"CRITICAL_dead_pct_{name}_{dead_pct:.1f}")

        return stream_issues, entropy, gini, n_used

    s1_issues, e1, g1, u1 = analyze_stream(s1, "S1 (coarse)", S1_BITS)
    s2_issues, e2, g2, u2 = analyze_stream(s2, "S2 (fine)", S2_BITS)
    issues = s1_issues + s2_issues

    # Cross-stream analysis
    subheader("Cross-Stream Token Analysis")
    s1_last = s1[:, -1]
    s2_last = s2[:, -1]

    # How many unique S2 codes per S1 code?
    s1_vals = s1_last.cpu().numpy()
    s2_vals = s2_last.cpu().numpy()
    cross_map = defaultdict(set)
    for a, b in zip(s1_vals, s2_vals):
        cross_map[a].add(b)

    mapping_sizes = [len(v) for v in cross_map.values()]
    avg_s2_per_s1 = np.mean(mapping_sizes) if mapping_sizes else 0
    max_s2_per_s1 = np.max(mapping_sizes) if mapping_sizes else 0
    s1_codes_with_multiple_s2 = sum(1 for v in mapping_sizes if v > 1)

    print(f"    Unique (S1,S2) pairs seen: {len(s1_vals):,}")
    print(f"    Avg S2 codes per S1 code:   {avg_s2_per_s1:.2f}")
    print(f"    Max S2 codes per S1 code:   {max_s2_per_s1}")
    print(f"    S1 codes with >1 S2:       {s1_codes_with_multiple_s2} / {len(cross_map)}")

    if avg_s2_per_s1 < 1.1 and u2 < 100:
        fail("S2 is nearly a deterministic function of S1 — no hierarchical benefit")
        issues.append("CRITICAL_deterministic_s2")
    elif s1_codes_with_multiple_s2 < len(cross_map) * 0.3:
        warn(f"Only {s1_codes_with_multiple_s2}/{len(cross_map)} S1 codes have multiple S2 children")
        issues.append("limited_hierarchical_branching")
    else:
        ok(f"Good hierarchical branching: {s1_codes_with_multiple_s2} S1 codes branch to multiple S2")

    return issues


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 7: ENCODER OUTPUT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def check_encoder_analysis(results, model):
    """Analyze encoder outputs before quantization — detect saturation, norm issues."""
    header("CHECK 7 · Encoder Output Analysis")
    issues = []

    enc_out = results['encoder_out']    # (N, T, D_MODEL)
    quantized = results['quantized']    # (N, T, CODEBOOK_DIM)

    # ── Encoder norm distribution ───────────────────────────────────────────
    norms = enc_out.norm(dim=-1)  # (N, T)
    print(f"  Encoder output norm distribution:")
    print(f"    Mean:   {norms.mean().item():.3f}")
    print(f"    Std:    {norms.std().item():.3f}")
    print(f"    Min:    {norms.min().item():.3f}")
    print(f"    Max:    {norms.max().item():.3f}")
    print(f"    Median: {norms.median().item():.3f}")

    # Check for norm collapse
    if norms.mean().item() < 0.1:
        fail(f"Encoder output norm near zero ({norms.mean().item():.4f}) — encoder collapsed")
        issues.append("CRITICAL_encoder_norm_collapse")
    elif norms.std().item() < 0.01:
        warn(f"Encoder outputs have very low variance (std={norms.std().item():.4f}) — encoder may be saturated")
        issues.append(f"encoder_low_variance_{norms.std().item():.4f}")
    else:
        ok(f"Encoder output norms healthy (μ={norms.mean().item():.3f}, σ={norms.std().item():.3f})")

    # After quant_embed
    with torch.no_grad():
        z = model.quant_embed(enc_out)
    z_norms = z.norm(dim=-1)
    print(f"\n  After quant_embed (pre-quantization) norm distribution:")
    print(f"    Mean:   {z_norms.mean().item():.3f}")
    print(f"    Std:    {z_norms.std().item():.3f}")

    # Expected norm for random unit vector in CODEBOOK_DIM dimensions ≈ sqrt(CODEBOOK_DIM/3)
    expected_norm = math.sqrt(CODEBOOK_DIM / 3)
    actual_norm = z_norms.mean().item()
    print(f"    Expected for random unit vec: ~{expected_norm:.3f}")

    # ── Quantization saturation ──────────────────────────────────────────────
    # Each bit is ±1 after quantization. Before quantization, values should ideally
    # be spread around 0. Very large or very small pre-quantization = saturation
    z_flat = z.flatten().cpu().numpy()
    frac_near_zero = (np.abs(z_flat) < 0.01).mean() * 100
    frac_large    = (np.abs(z_flat) > 3.0).mean() * 100

    print(f"\n  Pre-quantization value distribution:")
    print(f"    Fraction |z| < 0.01: {frac_near_zero:.1f}% (too small → gradient vanishing)")
    print(f"    Fraction |z| > 3.0:  {frac_large:.1f}% (too large → hard quantization)")

    if frac_near_zero > 50:
        fail(f"{frac_near_zero:.1f}% of pre-quant values near zero — encoder not producing strong signals")
        issues.append(f"CRITICAL_weak_encoder_signals_{frac_near_zero:.1f}")
    elif frac_near_zero > 20:
        warn(f"{frac_near_zero:.1f}% of pre-quant values are weak")

    # ── Bit entropy from pre-quant values ────────────────────────────────────
    # Before hard quantization, check the "soft" bit probabilities
    # sigmoid-based soft probability would be related to z values
    probs_pre = torch.sigmoid(z * 10)  # sharp sigmoid as proxy for hard quant
    bit_entropy_pre = -(probs_pre * torch.log(probs_pre + 1e-8) + (1-probs_pre) * torch.log(1-probs_pre + 1e-8))
    avg_bit_entropy_pre = bit_entropy_pre.mean().item()
    print(f"\n  Pre-quantization soft bit entropy: {avg_bit_entropy_pre:.4f} bits")
    print(f"    (max = 1.0, low = bits are already decided before quant)")

    if avg_bit_entropy_pre < 0.3:
        fail(f"Very low pre-quant bit entropy ({avg_bit_entropy_pre:.4f}) — quantization adds almost nothing")
        issues.append(f"CRITICAL_low_pre_quant_entropy_{avg_bit_entropy_pre:.4f}")
    elif avg_bit_entropy_pre > 0.9:
        ok(f"Pre-quant entropy high ({avg_bit_entropy_pre:.4f}) — quantization is deciding bits")
    else:
        info(f"Pre-quant entropy moderate ({avg_bit_entropy_pre:.4f})")

    return issues


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 8: GROUP QUANTIZATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def check_group_analysis(results):
    """Per-group utilization in the group quantization scheme."""
    header("CHECK 8 · Group Quantization Analysis")
    issues = []

    s1 = results['indices_s1'][:, -1]
    s2 = results['indices_s2'][:, -1]
    N = len(s1)

    # Group-based indices from the model
    n_groups_per_stream = S1_BITS // GROUP_SIZE  # S1 group count
    # For a 10-bit stream with group_size=4: 10/4=2.5 — not clean!
    # Actually the groups span (s1_bits + s2_bits) / group_size total groups
    total_groups = CODEBOOK_DIM // GROUP_SIZE  # 20/4 = 5 groups

    print(f"  Total quantization groups: {total_groups} (CODEBOOK_DIM={CODEBOOK_DIM}, GROUP_SIZE={GROUP_SIZE})")
    print(f"  Group vocab size: 2^{GROUP_SIZE} = {2**GROUP_SIZE} codes per group")

    # Reconstruct per-group indices from the full indices
    # Full index maps to bits, which map to groups
    device = s1.device
    combined = (s1.long() << S2_BITS) | s2.long()  # full 20-bit index

    # Extract group indices
    group_masks = []
    for g in range(total_groups):
        shift = g * GROUP_SIZE
        mask = (combined >> shift) & ((1 << GROUP_SIZE) - 1)
        group_masks.append(mask)

    group_vocabs = 2 ** GROUP_SIZE
    print(f"\n  {'Group':>6}  {'Bits':>10}  {'Used':>8}  {'Util%':>8}  {'Entropy':>8}  Status")
    print(f"  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*10}")

    for g in range(total_groups):
        shift = g * GROUP_SIZE
        bits_range = f"[{shift}:{shift+GROUP_SIZE}]"
        grp_idx = group_masks[g]
        n_used = len(torch.unique(grp_idx))
        util = n_used / group_vocabs * 100

        counts = torch.bincount(grp_idx, minlength=group_vocabs).float()
        probs = counts / counts.sum()
        entropy = float(-(probs * torch.log2(probs + 1e-8)).sum())
        max_ent = GROUP_SIZE  # max = group_size bits

        if util >= 90:
            status = f"{GREEN}✅{RESET}"
        elif util >= 50:
            status = f"{YELLOW}⚠️{RESET}"
        else:
            status = f"{RED}❌{RESET}"
            issues.append(f"group_{g}_low_util_{util:.1f}")

        print(f"  {g:>6}  {bits_range:>10}  {n_used:>8,}  {util:>7.1f}%  {entropy:>7.2f}  {status}")

    # Cross-group correlation
    subheader("Cross-Group Dependency")
    grp_tensors = [group_masks[g].float() for g in range(total_groups)]
    grp_matrix = torch.stack(grp_tensors, dim=1)  # (N, total_groups)
    grp_corr = torch.corrcoef(grp_matrix.T.float())
    high_grp_corr = []
    for i in range(total_groups):
        for j in range(i+1, total_groups):
            if abs(grp_corr[i, j].item()) > 0.3:
                high_grp_corr.append((i, j, grp_corr[i, j].item()))

    if high_grp_corr:
        warn(f"Found {len(high_grp_corr)} correlated group pairs (|r|>0.3):")
        for i, j, r in high_grp_corr:
            print(f"    Group {i} ↔ Group {j}: r={r:+.4f}")
    else:
        ok("Groups are reasonably independent")

    return issues


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 9: QUANTIZATION LOSS COMPONENT ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════

def check_loss_components(results, windows):
    """Estimate the loss components during training — recon, commit, entropy."""
    header("CHECK 9 · Loss Component Analysis (Training-Time Estimate)")
    issues = []

    x_true = windows[:len(results['z_full'])]
    N = len(x_true)

    # Run model in training mode to get loss metrics
    model = results.get('_model', None)
    if model is None:
        warn("Model not passed to this check — skipping loss component analysis")
        return issues

    model.train()  # Need training mode for EMA, etc.

    all_recon_pre, all_recon_full = [], []
    all_estimates = defaultdict(list)

    with torch.no_grad():
        for i in range(0, min(N, 2000), 64):
            b = x_true[i:i+64]
            (z_pre, z_full), bsq_loss, quantized, z_indices, metrics = model(b)

            recon_pre  = F.mse_loss(z_pre, b, reduction='none').mean()
            recon_full = F.mse_loss(z_full, b, reduction='none').mean()

            all_recon_pre.append(recon_pre.item())
            all_recon_full.append(recon_full.item())

            # Metrics from BSQ
            if 'H' in metrics:
                h_val = metrics['H'].item() if torch.is_tensor(metrics['H']) else metrics['H']
                all_estimates['codebook_entropy'].append(h_val)
            if 'utilization' in metrics:
                util = metrics['utilization'].item() if torch.is_tensor(metrics['utilization']) else metrics['utilization']
                all_estimates['utilization'].append(util)

    model.eval()

    avg_recon_pre  = np.mean(all_recon_pre)
    avg_recon_full = np.mean(all_recon_full)
    avg_recon_avg  = (avg_recon_pre + avg_recon_full) / 2

    print(f"  Reconstruction loss (S1-only):    {avg_recon_pre:.6f}")
    print(f"  Reconstruction loss (S1+S2):      {avg_recon_full:.6f}")
    print(f"  Average reconstruction loss:      {avg_recon_avg:.6f}")

    # Estimate commit loss
    with torch.no_grad():
        enc_out = model.embed(x_true[:64])
        for layer in model.encoder:
            enc_out = layer(enc_out)
        z = model.quant_embed(enc_out)
        z_norm = F.normalize(z, dim=-1)
        q_scale = 1. / (CODEBOOK_DIM ** 0.5)
        zq_unscaled = torch.sign(z_norm)
        zq = zq_unscaled * q_scale
        commit_per_dim = ((zq.detach() - z_norm) ** 2).sum(dim=-1).mean()
        commit_loss_est = BETA * commit_per_dim.item()

    print(f"  Estimated commit loss (β·E[‖zq-z‖²]): {commit_loss_est:.6f}")

    if all_estimates['codebook_entropy']:
        avg_cb_entropy = np.mean(all_estimates['codebook_entropy'])
        entropy_penalty = GAMMA0 * 0  # persample unknown
        print(f"  Codebook entropy (H):            {avg_cb_entropy:.4f}")
        print(f"  Estimated entropy penalty (γ·H): {GAMMA * avg_cb_entropy:.4f}")

    if all_estimates['utilization']:
        avg_util = np.mean(all_estimates['utilization'])
        print(f"  Batch utilization:               {avg_util:.4f} ({avg_util * 100:.1f}%)")

    # Ratio analysis
    print(f"\n  Loss ratios (relative to total recon):")
    if avg_recon_avg > 0:
        commit_ratio = commit_loss_est / avg_recon_avg
        print(f"    Commit/Recon:  {commit_ratio:.4f}")
        if commit_ratio > 10:
            fail(f"Commit loss {commit_ratio:.1f}x recon loss — commit dominates, codebook undertrained")
            issues.append(f"CRITICAL_commit_dominant_{commit_ratio:.1f}")
        elif commit_ratio < 0.01:
            warn(f"Commit loss negligible ({commit_ratio:.4f}x) — may undertrain codebook")
        else:
            ok(f"Commit/Recon ratio: {commit_ratio:.4f} — balanced")

    # Estimate total loss composition
    est_total = avg_recon_avg + commit_loss_est
    if all_estimates['codebook_entropy']:
        est_entropy_term = ZETA * GAMMA * avg_cb_entropy
        est_total += est_entropy_term
        print(f"\n    Recon:       {avg_recon_avg:.6f}  ({avg_recon_avg/est_total*100:.1f}%)")
        print(f"    Commit:      {commit_loss_est:.6f}  ({commit_loss_est/est_total*100:.1f}%)")
        print(f"    Entropy:     {est_entropy_term:.6f}  ({est_entropy_term/est_total*100:.1f}%)")

    return issues


# ══════════════════════════════════════════════════════════════════════════════
# CHECK 10: TEMPORAL COHERENCE
# ══════════════════════════════════════════════════════════════════════════════

def check_temporal_coherence(results):
    """Are token transitions sensible over time? Detect stuck/oscillating tokens."""
    header("CHECK 10 · Temporal Coherence")
    issues = []

    s1 = results['indices_s1']
    s2 = results['indices_s2']
    N, T = s1.shape

    # ── Token transition analysis ───────────────────────────────────────────
    subheader("Token Transition Analysis")

    for name, idx in [("S1", s1), ("S2", s2)]:
        # Same-token probability (consecutive timesteps)
        same_count = (idx[:, 1:] == idx[:, :-1]).float().mean().item()
        print(f"  {name}: P(token[t] == token[t-1]) = {same_count:.4f} ({same_count*100:.1f}%)")

        # 2-step same
        same_2step = (idx[:, 2:] == idx[:, :-2]).float().mean().item()
        print(f"  {name}: P(token[t] == token[t-2]) = {same_2step:.4f} ({same_2step*100:.1f}%)")

        # Expected for random uniform: 1/vocab
        expected_same = 1.0 / VOCAB_SIZE
        print(f"  {name}: Expected for random uniform = {expected_same:.4f}")

        if same_count > 0.5:
            fail(f"{name}: {same_count*100:.1f}% same-token transitions — tokens are stuck")
            issues.append(f"CRITICAL_stuck_tokens_{name}_{same_count:.3f}")
        elif same_count > 0.2:
            warn(f"{name}: {same_count*100:.1f}% same-token transitions — slightly sticky")
        else:
            ok(f"{name}: {same_count*100:.1f}% same-token transitions — temporally diverse")

    # ── Run-length analysis ──────────────────────────────────────────────────
    subheader("Token Run-Length Distribution (S1)")
    for sample_idx in [0, N//4, N//2, 3*N//4]:
        tokens = s1[sample_idx].cpu().tolist()
        runs = []
        current = tokens[0]
        run_len = 1
        for t in tokens[1:]:
            if t == current:
                run_len += 1
            else:
                runs.append((current, run_len))
                current = t
                run_len = 1
        runs.append((current, run_len))

        max_run = max(r[1] for r in runs)
        avg_run = np.mean([r[1] for r in runs])
        n_runs = len(runs)
        print(f"  Sample {sample_idx:>5}: {n_runs:>3} runs, avg_len={avg_run:.1f}, max_len={max_run}")

        if max_run > T * 0.8:
            fail(f"  Sample {sample_idx}: Single token dominates {max_run}/{T} positions")
            issues.append(f"CRITICAL_long_run_{sample_idx}")

    return issues


# ══════════════════════════════════════════════════════════════════════════════
# FINAL VERDICT & RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════

def verdict(all_issues, data_shape):
    header("FININAL VERDICT & ACTIONABLE RECOMMENDATIONS")

    critical = [i for i in all_issues if i.startswith("CRITICAL")]
    warnings = [i for i in all_issues if not i.startswith("CRITICAL")]

    if not all_issues:
        print(f"  {GREEN}{BOLD}🚀 TOKENIZER HEALTHY — safe to start model training{RESET}")
        print()
        return

    if critical:
        print(f"  {RED}{BOLD}🔴 DO NOT TRAIN — fix tokenizer first{RESET}")
    else:
        print(f"  {YELLOW}{BOLD}⚠️  PROCEED WITH CAUTION — {len(warnings)} warning(s){RESET}")

    for i in sorted(set(all_issues)):
        prefix = f"{RED}❌{RESET}" if i.startswith("CRITICAL") else f"{YELLOW}⚠️{RESET}"
        desc = i.replace("CRITICAL_", "").replace("_", " ")
        print(f"    {prefix} {desc}")

    # ── Specific recommendations ─────────────────────────────────────────────
    print(f"\n  {BOLD}═══════════════════════════════════════════════════════════════")
    print(f"  SPECIFIC RECOMMENDATIONS (Copy-Paste Ready)")
    print(f"  ═══════════════════════════════════════════════════════════════{RESET}\n")

    rec_num = 0

    # Codebook collapse
    if any("collapse" in i or "utilization" in i for i in critical):
        rec_num += 1
        print(f"  {BOLD}{rec_num}. FIX CODEBOOK COLLAPSE{RESET}")
        print(f"     Current: S1_BITS={S1_BITS}, S2_BITS={S2_BITS}, total={CODEBOOK_DIM}-bit")
        print(f"     Current hyperparams: β={BETA}, γ₀={GAMMA0}, γ={GAMMA}, ζ={ZETA}")

        ideal_bits = max(6, min(S1_BITS, int(np.log2(data_shape[0] * 0.001))))
        print(f"")
        print(f"     Option A — Reduce codebook size (safest):")
        print(f"       S1_BITS = {ideal_bits}, S2_BITS = {ideal_bits}")
        print(f"       (Reduces vocab from {VOCAB_SIZE:,} to {2**ideal_bits:,} codes)")
        print(f"")
        print(f"     Option B — Increase entropy pressure:")
        print(f"       gamma  = {GAMMA * 2:.1f}  (was {GAMMA})")
        print(f"       gamma0 = {GAMMA0 * 2:.1f}  (was {GAMMA0})")
        print(f"       zeta   = {ZETA * 0.5:.2f}  (was {ZETA})")
        print(f"")
        print(f"     Option C — Reduce commit weight (let entropy dominate):")
        print(f"       beta = {BETA * 0.2:.3f}  (was {BETA})")
        print(f"       This forces the model to care less about reconstruction quality")
        print(f"       and more about using the codebook.")

    # Bit collapse
    if any("bit" in i.lower() and ("collapse" in i.lower() or "CRITICAL" in i) for i in all_issues):
        rec_num += 1
        print(f"\n  {BOLD}{rec_num}. FIX BIT COLLAPSE{RESET}")
        print(f"     Bits are highly biased (near-0 or near-1). Solutions:")
        print(f"       1. Add bit-balance regularization: penalize |mean(bit) - 0.5|")
        print(f"       2. Increase gamma (entropy weight) → already covered above")
        print(f"       3. Use fewer bits: S1_BITS=8, S2_BITS=8 is easier to train")
        print(f"       4. Check if input features have enough diversity (see Check 1)")

    # Dead S2 stream
    if any("dead_s2" in i or "deterministic_s2" in i for i in all_issues):
        rec_num += 1
        print(f"\n  {BOLD}{rec_num}. FIX DEAD S2 (FINE) STREAM{RESET}")
        print(f"     The fine stream isn't adding value. This means:")
        print(f"       1. The S1 coarse codes already capture most information")
        print(f"       2. The S2 capacity is too large for the residual signal")
        print(f"     Solutions:")
        print(f"       - Reduce S2_BITS to {max(4, S2_BITS - 2)}")
        print(f"       - Decrease S2 post-quant embed dimension")
        print(f"       - Train with higher recon weight on residual (S2 target = x - z_pre)")

    # Reconstruction quality
    if any("recon" in i and "CRITICAL" in i for i in all_issues):
        rec_num += 1
        print(f"\n  {BOLD}{rec_num}. FIX RECONSTRUCTION QUALITY{RESET}")
        print(f"     Options:")
        print(f"       1. Increase capacity: d_model={D_MODEL*2}, ff_dim={FF_DIM*2}")
        print(f"       2. More layers: n_enc={N_ENC+1}, n_dec={N_DEC+1}")
        print(f"       3. Check input normalization (features should be ~N(0,1))")
        print(f"       4. Reduce CODEBOOK_DIM to make quantization easier")
        print(f"       5. Train for more epochs — tokenizer may be under-trained")

    # Encoder issues
    if any("encoder" in i for i in all_issues):
        rec_num += 1
        print(f"\n  {BOLD}{rec_num}. FIX ENCODER OUTPUTS{RESET}")
        print(f"     Solutions:")
        print(f"       1. Check embed layer init (std should be ~1/sqrt(d_model))")
        print(f"       2. Verify encoder layers aren't identity-mapped")
        print(f"       3. Add batch norm or check if encoder is producing diverse outputs")

    # Stuck tokens
    if any("stuck" in i or "long_run" in i for i in all_issues):
        rec_num += 1
        print(f"\n  {BOLD}{rec_num}. FIX STUCK TOKENS{RESET}")
        print(f"       1. Add noise to encoder outputs before quantization during training")
        print(f"       2. Use EMA codebook updates instead of direct gradient")
        print(f"       3. Restart dead codes (replace least-used with random encoder outputs)")

    # Scale issues in input
    if any("scale" in i for i in all_issues):
        rec_num += 1
        print(f"\n  {BOLD}{rec_num}. FIX FEATURE SCALING{RESET}")
        print(f"     Some input features have extreme scales. The tokenizer sees")
        print(f"     unnormalized data. Solutions:")
        print(f"       1. Per-feature z-score normalization before windowing")
        print(f"       2. Robust scaling (median/IQR) instead of mean/std")
        print(f"       3. Clip features to [-5, 5] BEFORE normalization")

    # High correlation
    if any("correlation" in i for i in all_issues):
        rec_num += 1
        print(f"\n  {BOLD}{rec_num}. REDUCE FEATURE REDUNDANCY{RESET}")
        print(f"     Highly correlated features waste codebook capacity:")
        print(f"       1. Remove redundant features from DEFAULT_FEATURE_LIST")
        print(f"       2. Apply PCA to reduce from {len(DEFAULT_FEATURE_LIST)} → fewer dims")
        print(f"       3. This makes the tokenizer's job easier")

    # General training recipe
    print(f"\n  {BOLD}{'─' * 55}{RESET}")
    print(f"  {BOLD}GENERAL TRAINING RECIPE (if retraining from scratch):{RESET}")
    print(f"  {'─' * 55}")
    print(f"     1. Start with smaller codebook: S1_BITS=8, S2_BITS=8")
    print(f"     2. Verify utilization > 70% before increasing bit count")
    print(f"     3. Use these hyperparams as starting point:")
    print(f"        beta=0.25, gamma0=1.0, gamma=1.0, zeta=0.5")
    print(f"     4. Monitor per-epoch utilization and bit balance")
    print(f"     5. Gradually increase bits: 6→8→10 only when previous is stable")
    print(f"     6. Use EMA codebook updates (decay=0.99) for stability")
    print(f"     7. Train for at least 200 epochs with lr=1e-4, cosine decay")
    print(f"     8. Use codebook restart: replace dead codes every epoch")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{BOLD}{'═'*70}")
    print(f"  TOKENIZER DEEP HEALTH AUDIT v2.0")
    print(f"  Hierarchical BSQ: {S1_BITS}-bit coarse + {S2_BITS}-bit fine (total {CODEBOOK_DIM}-bit)")
    print(f"  Codebook size:    {VOCAB_SIZE:,} coarse × {VOCAB_SIZE:,} fine")
    print(f"  Total possible:   2^{CODEBOOK_DIM} = {2**CODEBOOK_DIM:,} combined codes")
    print(f"  SEQ_LEN:          {SEQ_LEN}")
    print(f"  d_model:          {D_MODEL}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*70}{RESET}")

    print("\nLoading data and tokenizer...")
    data  = load_data()
    model = load_model()
    print(f"  Training samples : {len(data):,}")
    print(f"  Features         : {data.shape[1]}")
    print(f"  Feature names    : {DEFAULT_FEATURE_LIST[:5]}...")

    all_issues = []

    # ── Check 1: Data diagnostics ───────────────────────────────────────────
    data_issues = check_data_diagnostics(data.float())
    all_issues += data_issues

    # ── Check 2: Architecture ────────────────────────────────────────────────
    arch_issues = check_architecture(model)
    all_issues += arch_issues

    # Build windows for remaining checks
    windows = build_windows(data, SEQ_LEN, stride=4, max_w=2000)
    results  = encode_full(model, windows)
    results['_model'] = model  # For loss component estimation

    # ── Check 3: Codebook utilization ────────────────────────────────────────
    util_issues, s1, s2 = check_utilization_detailed(results)
    all_issues += util_issues

    # ── Check 4: Bit analysis ────────────────────────────────────────────────
    bit_issues = check_bit_analysis(results)
    all_issues += bit_issues

    # ── Check 5: Reconstruction quality ──────────────────────────────────────
    recon_issues = check_reconstruction_detailed(results, windows)
    all_issues += recon_issues

    # ── Check 6: Token distribution ──────────────────────────────────────────
    dist_issues = check_token_distribution(s1, s2)
    all_issues += dist_issues

    # ── Check 7: Encoder analysis ────────────────────────────────────────────
    enc_issues = check_encoder_analysis(results, model)
    all_issues += enc_issues

    # ── Check 8: Group quantization ──────────────────────────────────────────
    grp_issues = check_group_analysis(results)
    all_issues += grp_issues

    # ── Check 9: Loss component analysis ─────────────────────────────────────
    loss_issues = check_loss_components(results, windows)
    all_issues += loss_issues

    # ── Check 10: Temporal coherence ─────────────────────────────────────────
    temp_issues = check_temporal_coherence(results)
    all_issues += temp_issues

    # ── Final verdict ────────────────────────────────────────────────────────
    verdict(all_issues, data.shape)


if __name__ == "__main__":
    main()
