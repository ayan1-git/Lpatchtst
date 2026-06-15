# ============================================================
# audit_scripts/audit_tokens_only.py
# LPatchTST — TOKENS_ONLY mode diagnostic audit
#
# Run from repo root:
#   python audit_scripts/audit_tokens_only.py
#
# What this covers (tokens_only pipeline only):
#   1. Oracle target construction & distribution
#   2. Prediction bias (short/flat collapse)
#   3. Train / Val loss gap per fold
#   4. Tokenizer vocabulary & entropy
#   5. Fine-tune hyperparameter review
#   6. Data integrity (multi-asset alignment)
#   7. Zero-rate drift (train vs val)
#   8. Best policy sanity
# ============================================================

from __future__ import annotations
import os, sys, re, json, math, warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

# ── bootstrap ────────────────────────────────────────────────
try:
    current_dir = Path(__file__).resolve().parent
except NameError:
    current_dir = Path.cwd()

SEARCH_DIRS = [str(current_dir), str(current_dir.parent), "/kaggle/working/Lpatchtst", os.getcwd()]

def _find_file(fname):
    for d in SEARCH_DIRS:
        if not os.path.exists(d):
            continue
        for root, dirs, files in os.walk(d):
            dirs[:] = [name for name in dirs if name not in (
                'data', 'Data', '__pycache__', '.git', '.cursor', '.kilo', '.ipynb_checkpoints'
            ) and not name.startswith('.')]
            if fname in files:
                return os.path.join(root, fname)
    return None

src_dir = None
for d in SEARCH_DIRS:
    if os.path.exists(os.path.join(d, "config.py")):
        src_dir = d
        break

if src_dir is None:
    config_path = _find_file("config.py")
    if config_path:
        src_dir = os.path.dirname(config_path)

if src_dir is None:
    raise FileNotFoundError("Cannot locate config.py in any search directory.")

print(f"Source directory : {src_dir}")
REPO_ROOT = Path(src_dir)
sys.path.insert(0, str(REPO_ROOT))

import config as CFG
from data_loader  import tokenize_full_series
from oracle       import generate_targets
from tokenizer    import prepare_ohlc_features, KronosTokenizer
from features     import FeatureEngineer, FeatureConfig

AUDIT_OUT = REPO_ROOT / "audit_output"
AUDIT_OUT.mkdir(exist_ok=True)

def _get_data_dir() -> Path:
    data_dir = REPO_ROOT / "Data"
    if not data_dir.exists() or not list(data_dir.glob("*.csv")):
        if (REPO_ROOT.parent / "Data").exists() and list((REPO_ROOT.parent / "Data").glob("*.csv")):
            data_dir = REPO_ROOT.parent / "Data"
        else:
            kaggle_input = Path("/kaggle/input")
            if kaggle_input.exists():
                for subdir in kaggle_input.iterdir():
                    if subdir.is_dir() and list(subdir.glob("*.csv")):
                        return subdir
    return data_dir

def _get_file_path(filename: str) -> Path:
    for base in [REPO_ROOT, REPO_ROOT.parent, Path.cwd()]:
        p = base / filename
        if p.exists():
            return p
    return REPO_ROOT / filename

PASS = "✅ PASS"
WARN = "⚠️  WARN"
FAIL = "❌ FAIL"

results: list[dict] = []

def log(section: str, check: str, status: str, detail: str = ""):
    tag = {"✅ PASS":"PASS","⚠️  WARN":"WARN","❌ FAIL":"FAIL"}.get(status, "INFO")
    msg = f"[{tag}] [{section}] {check}"
    if detail: msg += f"  →  {detail}"
    print(msg)
    results.append({"section": section, "check": check, "status": tag, "detail": detail})

DEFAULT_FEATURE_LIST = [
    'open', 'high', 'low', 'close',
    'ret_norm_1d', 'ret_norm_3d', 'ret_norm_6d', 'ret_norm_13d',
    'ret_norm_26d', 'ret_norm_65d', 'ret_norm_130d', 'ret_norm_260d',
    'macd_8_24', 'macd_26_78', 'macd_52_156',
    'feat_efficiency', 'feat_icp', 'feat_momentum_rsi',
    'feat_vol_asymmetry', 'feat_local_structure',
    'feat_session_sin', 'feat_session_cos', 'feat_vol_squeeze'
]


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _load_fe():
    return FeatureEngineer(FeatureConfig(
        ewma_span             = CFG.FE_VOL_LONG_PERIOD,
        return_horizons       = CFG.FE_RETURN_HORIZONS,
        macd_pairs            = CFG.FE_MACD_PAIRS,
        macd_price_std_window = CFG.FE_MACD_PRICE_STD_WIN,
        macd_signal_std_window= CFG.FE_MACD_SIGNAL_STD_WIN,
        target_clip           = CFG.FE_TARGET_CLIP,
        momentum_period       = CFG.FE_MOMENTUM_PERIOD,
        rsi_period            = CFG.FE_RSI_PERIOD,
        vol_asym_window       = CFG.FE_VOL_ASYM_WINDOW,
        icp_period            = CFG.FE_ICP_PERIOD,
        local_structure_bars  = CFG.FE_LOCAL_STRUCTURE_BARS,
        vol_squeeze_fast      = CFG.FE_VOL_SQUEEZE_FAST,
        vol_squeeze_slow      = CFG.FE_VOL_SQUEEZE_SLOW,
        atr_period            = CFG.ATR_PERIOD,
        session_open          = CFG.FE_SESSION_OPEN,
        session_close         = CFG.FE_SESSION_CLOSE,
        session_tz            = CFG.FE_SESSION_TZ,
        add_session_features  = CFG.FE_ADD_SESSION,
        use_talib             = getattr(CFG, "USE_TALIB", False),
    ))


def _build_one_asset(csv_path: Path, fe):
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.lower().strip() for c in df.columns]
        time_col = next((c for c in df.columns if c in ("date","datetime","time")), None)
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col).sort_index()
        else:
            try: df.index = pd.to_datetime(df.index)
            except: pass

        required = {'open', 'high', 'low', 'close'}
        if not required.issubset(df.columns):
            return None

        df = df.iloc[3276:]
        if len(df) < 100:
            return None

        hl = df["high"] - df["low"]
        hc = (df["high"] - df["close"].shift()).abs()
        lc = (df["low"]  - df["close"].shift()).abs()
        atr = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(CFG.ATR_PERIOD).mean()

        targets = generate_targets(
            df["open"].values, df["high"].values,
            df["low"].values,  df["close"].values,
            atr.values,
            max_hold          = CFG.ORACLE_MAX_HOLD,
            fee_per_side      = CFG.FEE_PER_SIDE,
            slippage          = CFG.SLIPPAGE,
            sl_atr_mult       = CFG.ORACLE_SL_ATR_MULT,
            tp_atr_mult       = CFG.ORACLE_TP_ATR_MULT,
            saturation_factor = CFG.SATURATION_FACTOR,
            mae_penalty       = CFG.MAE_PENALTY,
        )
        thresh  = CFG.SAMPLER_THRESHOLD
        targets = np.where(np.abs(targets) < thresh, 0.0, targets).astype(np.float32)

        prices = df['close']
        ohlc_subset = df[['open', 'high', 'low', 'close']]
        eng_feats = fe.build(prices, ohlc=ohlc_subset, include_target=False, dropna=False)
        df = df.join(eng_feats)

        for col in DEFAULT_FEATURE_LIST:
            if col not in df.columns:
                df[col] = 0.0

        ohlc_ret = df[DEFAULT_FEATURE_LIST].values.astype(np.float32)

        valid_len  = len(targets) - CFG.ORACLE_MAX_HOLD
        final_targets = targets[:valid_len]
        combined = df.iloc[1 : valid_len+1].copy()
        combined["atr"] = atr.iloc[1 : valid_len+1]

        return combined, final_targets, ohlc_ret

    except Exception as e:
        print(f"  [DEBUG] _build_one_asset failed: {str(e)}")
        return None


def _load_tokenizer():
    full_path = _find_file("model.safetensors")
    if full_path is None:
        log("TOKENIZER", "checkpoint found", FAIL, "model.safetensors missing")
        return None
    try:
        tok = KronosTokenizer.from_pretrained(str(full_path), device=str("cpu"))
        tok.eval()
        for p in tok.parameters(): p.requires_grad = False
        log("TOKENIZER", "checkpoint loaded", PASS, str(full_path))
        return tok
    except Exception as e:
        log("TOKENIZER", "checkpoint load", FAIL, str(e))
        return None


# ─────────────────────────────────────────────────────────────
# 1.  ORACLE TARGET AUDIT
# ─────────────────────────────────────────────────────────────

def audit_oracle_targets():
    print("\n" + "="*60)
    print("SECTION 1 — ORACLE TARGET DISTRIBUTION")
    print("="*60)

    data_dir   = _get_data_dir()
    csv_files  = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        log("ORACLE", "CSV files", FAIL, f"None found in {data_dir}")
        return None, None

    fe      = _load_fe()
    result  = _build_one_asset(csv_files[0], fe)
    if result is None:
        log("ORACLE", "Asset load", FAIL, str(csv_files[0]))
        return None, None

    df_combined, targets, ohlc_ret = result

    long_frac  = (targets >  0).mean()
    short_frac = (targets <  0).mean()
    zero_frac  = (targets == 0.0).mean()

    log("ORACLE", "Target balance",
        PASS if 0.20 < long_frac < 0.80 else WARN,
        f"Long={long_frac:.2%}  Short={short_frac:.2%}  Zero/Flat={zero_frac:.2%}")

    nonzero = targets[targets != 0.0]
    below_thresh = (np.abs(nonzero) < CFG.SAMPLER_THRESHOLD).sum()
    log("ORACLE", "Sub-threshold leakage after zeroing",
        PASS if below_thresh == 0 else FAIL,
        f"{below_thresh} nonzero targets with |tgt| < SAMPLER_THRESHOLD={CFG.SAMPLER_THRESHOLD}"
        "  (should be 0 — process_dataset zeroing broken if >0)")

    if len(nonzero) > 10:
        skew = pd.Series(nonzero).skew()
        log("ORACLE", "Nonzero target skewness",
            PASS if abs(skew) < 1.5 else WARN,
            f"skew={skew:.3f}  (|skew|>1.5 → asymmetric signal; model will favor one side)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(targets, bins=80, color="#4C72B0", alpha=0.8)
    axes[0].axvline(0, color="red", lw=1.5, ls="--", label="zero")
    axes[0].set_title(f"All targets  (N={len(targets)})"); axes[0].legend()
    if len(nonzero):
        axes[1].hist(nonzero, bins=60, color="#DD8452", alpha=0.8)
        axes[1].set_title("Nonzero targets only  (actual trade signals)")
    fig.suptitle(f"Oracle Targets — {csv_files[0].name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(AUDIT_OUT / "01_oracle_target_dist.png", dpi=120)
    plt.close(fig)
    log("ORACLE", "Plot saved", PASS, str(AUDIT_OUT / "01_oracle_target_dist.png"))

    return targets, ohlc_ret


# ─────────────────────────────────────────────────────────────
# 2.  PREDICTION BIAS AUDIT
# ─────────────────────────────────────────────────────────────

def audit_prediction_bias():
    print("\n" + "="*60)
    print("SECTION 2 — PREDICTION / SIGNAL BIAS")
    print("="*60)

    metrics_path = _get_file_path("test_metrics.json")
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        da   = m.get("dir_accuracy", m.get("DirAcc", None))
        corr = m.get("correlation",  m.get("Corr",   None))
        if da is not None:
            log("PRED_BIAS", "Test DirAcc",
                PASS if da > 0.52 else (WARN if da > 0.48 else FAIL),
                f"{da:.2%}  (>52% = exploitable signal)")
        if corr is not None:
            log("PRED_BIAS", "Test Correlation",
                PASS if corr > 0.05 else (WARN if corr > 0.0 else FAIL),
                f"{corr:.4f}  (>0.05 = reliable directional signal)")
    else:
        log("PRED_BIAS", "test_metrics.json", WARN, "Not found")

    bt_path = _get_file_path("backtest_results.csv")
    if not bt_path.exists():
        log("PRED_BIAS", "backtest_results.csv", WARN, "Not found — skipping signal distribution check")
        return

    bt = pd.read_csv(bt_path)
    sig_col = next((c for c in ["signal","pred","prediction"] if c in bt.columns), None)
    if sig_col is None:
        log("PRED_BIAS", "Signal column", WARN, f"No signal/pred column. Cols: {list(bt.columns)[:8]}")
        return

    sigs = bt[sig_col].dropna()
    thresh = getattr(CFG, "SAMPLER_THRESHOLD", 0.05)
    long_frac  = (sigs >  thresh).mean()
    short_frac = (sigs < -thresh).mean()
    flat_frac  = ((sigs >= -thresh) & (sigs <= thresh)).mean()

    status = PASS if 0.25 < long_frac < 0.75 else FAIL
    log("PRED_BIAS", "Signal direction balance", status,
        f"Long={long_frac:.2%}  Short={short_frac:.2%}  Flat={flat_frac:.2%}")

    if short_frac > 0.65:
        log("PRED_BIAS", "SHORT BIAS DETECTED", FAIL,
            f"Model outputs Short {short_frac:.1%} of the time.")

    if flat_frac > 0.50:
        log("PRED_BIAS", "FLAT COLLAPSE DETECTED", FAIL,
            f"Model is Flat {flat_frac:.1%} of the time.")

    if "target" in bt.columns:
        tgt_zero = bt["target"] == 0.0
        false_sig = (sigs[tgt_zero].abs() > thresh).mean()
        log("PRED_BIAS", "False signal rate (tgt=0, |pred|>thresh)",
            PASS if false_sig < 0.30 else FAIL,
            f"{false_sig:.1%}  (should be <30%)")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.hist(sigs, bins=80, color="#DD8452", alpha=0.8, edgecolor="none")
    ax.axvline( thresh, color="green",  lw=1.5, ls="--", label=f"+thresh={thresh}")
    ax.axvline(-thresh, color="red",    lw=1.5, ls="--", label=f"-thresh={thresh}")
    ax.axvline(0,       color="black",  lw=1.0, ls=":")
    ax.set_title("Prediction / Signal Distribution"); ax.legend()
    fig.tight_layout()
    fig.savefig(AUDIT_OUT / "02_signal_distribution.png", dpi=120)
    plt.close(fig)
    log("PRED_BIAS", "Plot saved", PASS, str(AUDIT_OUT / "02_signal_distribution.png"))


# ─────────────────────────────────────────────────────────────
# 3.  TRAIN / VAL LOSS GAP
# ─────────────────────────────────────────────────────────────

def audit_loss_gap():
    print("\n" + "="*60)
    print("SECTION 3 — TRAIN/VAL LOSS GAP & GRADIENT NORMS")
    print("="*60)

    candidates = []
    for base in [REPO_ROOT, REPO_ROOT.parent, Path.cwd()]:
        candidates.extend(base.glob("*.txt"))
        candidates.extend(base.glob("*.log"))

    unique_candidates = {p.resolve(): p for p in candidates}
    log_candidates = sorted(
        list(unique_candidates.values()),
        key=lambda p: p.stat().st_size, reverse=True
    )

    train_log = None
    for p in log_candidates:
        txt = p.read_text(errors="ignore")
        if "TrainL=" in txt or "Pre-train" in txt:
            train_log = txt
            log("LOSS_GAP", "Log file", PASS, str(p))
            break
    if train_log is None:
        log("LOSS_GAP", "Log file", WARN, "Not found — place paste.txt in repo root")
        return

    pt_rows = re.findall(
        r"Pre-train.*?Ep\s*(\d+).*?Loss=([\d.]+).*?DirAcc=([\d.]+)%.*?Corr=([-\d.]+)",
        train_log
    )
    if pt_rows:
        df_pt = pd.DataFrame(pt_rows, columns=["ep","loss","dir_acc","corr"]).astype(float)
        final = df_pt.iloc[-1]
        log("LOSS_GAP", "Pre-train final loss",
            PASS if final.loss < 1.5 else WARN, f"{final.loss:.4f}")
        log("LOSS_GAP", "Pre-train final DirAcc",
            PASS if final.dir_acc > 60 else WARN, f"{final.dir_acc:.1f}%")
        log("LOSS_GAP", "Pre-train final Corr",
            PASS if final.corr > 0.4 else WARN, f"{final.corr:.4f}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(df_pt.ep, df_pt.loss, "b-o", ms=3); axes[0].set_title("Pre-train Loss")
        axes[1].plot(df_pt.ep, df_pt.dir_acc, "g-o", ms=3)
        axes[1].axhline(60, color="red", lw=1, ls="--", label="60% baseline")
        axes[1].set_title("Pre-train DirAcc %"); axes[1].legend()
        axes[2].plot(df_pt.ep, df_pt.corr, "r-o", ms=3); axes[2].set_title("Pre-train Corr")
        fig.tight_layout()
        fig.savefig(AUDIT_OUT / "03a_pretrain_curves.png", dpi=120); plt.close(fig)
        log("LOSS_GAP", "Pre-train curve plot", PASS, str(AUDIT_OUT / "03a_pretrain_curves.png"))

    ft_rows = re.findall(
        r"Fold\s+(\d+).*?Ep\s*(\d+)\s*\|.*?TrainL=([\d.]+).*?ValL=([\d.]+).*?DirAcc=([\d.]+)%.*?Corr=([-\d.]+)",
        train_log
    )
    if ft_rows:
        df_ft = pd.DataFrame(ft_rows, columns=["fold","ep","train_l","val_l","dir_acc","corr"]).astype(float)
        n_folds = int(df_ft.fold.nunique())

        for fid, grp in df_ft.groupby("fold"):
            best_val   = grp.val_l.min()
            best_train = grp.train_l.min()
            ratio      = best_val / (best_train + 1e-9)
            best_acc   = grp.dir_acc.max()
            best_corr  = grp.corr.max()

            log("LOSS_GAP", f"Fold {int(fid)} val/train loss ratio",
                PASS if ratio < 1.5 else (WARN if ratio < 2.0 else FAIL),
                f"ValL={best_val:.4f} / TrainL={best_train:.4f} = {ratio:.2f}x")
            log("LOSS_GAP", f"Fold {int(fid)} val DirAcc",
                PASS if best_acc > 52 else FAIL, f"{best_acc:.1f}%")
            log("LOSS_GAP", f"Fold {int(fid)} val Corr",
                PASS if best_corr > 0.05 else FAIL, f"{best_corr:.4f}")

        fig, axes = plt.subplots(n_folds, 2, figsize=(14, 4*n_folds), squeeze=False)
        for ri, (fid, grp) in enumerate(df_ft.groupby("fold")):
            grp = grp.sort_values("ep")
            axes[ri,0].plot(grp.ep, grp.train_l, label="Train", color="steelblue")
            axes[ri,0].plot(grp.ep, grp.val_l,   label="Val",   color="orange")
            axes[ri,0].set_title(f"Fold {int(fid)} Loss"); axes[ri,0].legend()
            axes[ri,1].plot(grp.ep, grp.dir_acc, color="seagreen")
            axes[ri,1].axhline(52, color="red", lw=1, ls="--", label="52%")
            axes[ri,1].set_title(f"Fold {int(fid)} DirAcc %"); axes[ri,1].legend()
        fig.tight_layout()
        fig.savefig(AUDIT_OUT / "03b_finetune_curves.png", dpi=120); plt.close(fig)
        log("LOSS_GAP", "Fine-tune fold curves plot", PASS, str(AUDIT_OUT / "03b_finetune_curves.png"))

    gn_rows = re.findall(r"Ep\s*\d+.*?GN avg=([\d.]+)\s+max=([\d.]+)", train_log)
    if gn_rows:
        df_gn = pd.DataFrame(gn_rows, columns=["avg","max"]).astype(float)
        peak_avg  = df_gn.avg.max()
        peak_max  = df_gn["max"].max()
        late_avg  = df_gn.tail(10).avg.mean()
        log("LOSS_GAP", "Peak GN avg",  PASS if peak_avg < 50  else WARN, f"{peak_avg:.1f}")
        log("LOSS_GAP", "Peak GN max",  PASS if peak_max < 500 else WARN, f"{peak_max:.1f}")
        log("LOSS_GAP", "Late GN avg",  PASS if late_avg < 30  else WARN, f"{late_avg:.2f}")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df_gn.avg,    label="GN avg", color="blue")
        ax.plot(df_gn["max"], label="GN max", color="red", alpha=0.4)
        ax.axhline(50,  color="orange",  lw=1, ls="--", label="avg=50")
        ax.axhline(500, color="darkred", lw=1, ls="--", label="max=500")
        ax.set_title("Gradient Norms"); ax.legend()
        fig.tight_layout()
        fig.savefig(AUDIT_OUT / "03c_gradient_norms.png", dpi=120); plt.close(fig)
        log("LOSS_GAP", "Gradient norm plot", PASS, str(AUDIT_OUT / "03c_gradient_norms.png"))

    drift_rows = re.findall(
        r"zero-rate drift.*?Train=([\d.]+)%.*?Val=([\d.]+)%.*?Δ=([+-][\d.]+)pp",
        train_log
    )
    if drift_rows:
        for tr_z, va_z, delta in drift_rows:
            d = float(delta)
            log("LOSS_GAP", f"Zero-rate drift (Δ={d:+.1f}pp)",
                PASS if abs(d) < 10 else WARN,
                f"Train zero={tr_z}%  Val zero={va_z}%")


# ─────────────────────────────────────────────────────────────
# 4.  TOKENIZER VOCABULARY AUDIT  (FIXED)
# ─────────────────────────────────────────────────────────────

def audit_tokenizer(ohlc_ret=None):
    print("\n" + "="*60)
    print("SECTION 4 — TOKENIZER VOCABULARY & ENTROPY")
    print("="*60)

    tok = _load_tokenizer()
    if tok is None:
        return

    vocab_c = 2 ** getattr(tok, "s1_bits", CFG.TOKENIZER_S1_BITS)
    vocab_f = 2 ** getattr(tok, "s2_bits", CFG.TOKENIZER_S2_BITS)
    log("TOKENIZER", "Vocab sizes", PASS, f"coarse={vocab_c}  fine={vocab_f}")

    if ohlc_ret is None:
        data_dir  = _get_data_dir()
        csv_files = sorted(data_dir.glob("*.csv"))
        if not csv_files:
            log("TOKENIZER", "OHLC data", FAIL, "No CSVs found")
            return
        fe     = _load_fe()
        result = _build_one_asset(csv_files[0], fe)
        if result is None:
            log("TOKENIZER", "Asset build", FAIL, str(csv_files[0]))
            return
        _, _, ohlc_ret = result

    # ── FIX: mirror tokenize_full_series exactly — take [:, -1] per window ──
    try:
        tok.eval()
        device = next(tok.parameters()).device

        S          = getattr(CFG, "TOKENIZER_WINDOW", 90)
        # FIX: Use a slice from the middle/end to avoid initial NaN features
        # and apply nan_to_num to be safe.
        start_idx = max(0, len(ohlc_ret) // 2)
        probe_data = ohlc_ret[start_idx : start_idx + 2000]
        probe_data = np.nan_to_num(probe_data, nan=0.0)
        
        T          = len(probe_data)
        C          = probe_data.shape[1]


        pad    = np.tile(probe_data[0:1], (S - 1, 1))
        padded = np.concatenate([pad, probe_data], axis=0)

        from numpy.lib.stride_tricks import as_strided
        windows = as_strided(
            padded,
            shape=(T, S, C),
            strides=(padded.strides[0], padded.strides[0], padded.strides[1])
        )

        chunk_size = getattr(CFG, "TOKENIZER_CHUNK_SIZE", 1024)
        coarse_parts, fine_parts = [], []

        with torch.no_grad():
            for i in range(0, T, chunk_size):
                batch = torch.from_numpy(
                    np.array(windows[i: i + chunk_size])
                ).to(device).float()
                w_mean = batch.mean(dim=1, keepdim=True)
                w_std  = batch.std(dim=1, keepdim=True) + 1e-5
                batch  = (batch - w_mean) / w_std
                batch  = torch.clamp(batch, -5.0, 5.0)

                idx_c, idx_f = tok.encode(batch, half=True)

                # ✅ CORRECT: [:, -1] mirrors tokenize_full_series exactly
                # Each window's token = the last timestep only
                coarse_parts.append(idx_c[:, -1].cpu())
                fine_parts.append(idx_f[:, -1].cpu())

        coarse = torch.cat(coarse_parts)   # shape (T,) — one token per bar
        fine   = torch.cat(fine_parts)     # shape (T,)

    except Exception as e:
        log("TOKENIZER", "encode probe", FAIL, str(e))
        return

    n_c    = coarse.unique().numel()
    n_f    = fine.unique().numel()
    util_c = n_c / vocab_c
    util_f = n_f / vocab_f
    log("TOKENIZER", "Coarse vocab utilisation",
        PASS if util_c > 0.15 else FAIL,
        f"{n_c}/{vocab_c} = {util_c:.1%}  (<15% = codebook collapse)")
    log("TOKENIZER", "Fine vocab utilisation",
        PASS if util_f > 0.10 else FAIL,
        f"{n_f}/{vocab_f} = {util_f:.1%}  (<10% = codebook collapse)")

    # ── Entropy ────────────────────────────────────────────────────────────
    def token_entropy(toks, vsize):
        cnt   = torch.bincount(toks, minlength=vsize).float()
        probs = cnt / cnt.sum()
        probs = probs[probs > 0]
        return -(probs * probs.log()).sum().item()

    max_H = math.log(vocab_c)
    H_c   = token_entropy(coarse, vocab_c)
    H_f   = token_entropy(fine,   vocab_f)
    log("TOKENIZER", "Coarse entropy ratio",
        PASS if H_c / max_H > 0.30 else WARN,
        f"H={H_c:.3f} / max={max_H:.3f} = {H_c/max_H:.1%}  (>30% = good diversity)")
    log("TOKENIZER", "Fine entropy ratio",
        PASS if H_f / max_H > 0.25 else WARN,
        f"H={H_f:.3f} / max={max_H:.3f} = {H_f/max_H:.1%}")

    # ── Top-10 concentration ───────────────────────────────────────────────
    hist_c     = torch.bincount(coarse, minlength=vocab_c).float()
    top10_conc = (hist_c.topk(10).values.sum() / hist_c.sum()).item()
    log("TOKENIZER", "Top-10 coarse token concentration",
        PASS if top10_conc < 0.40 else WARN,
        f"{top10_conc:.1%}  (>40% = codebook bottleneck)")

    # ── Multi-asset consistency ────────────────────────────────────────────
    data_dir  = _get_data_dir()
    csv_files = sorted(data_dir.glob("*.csv"))[:5]
    fe        = _load_fe()
    entropies = []
    for p in csv_files:
        r = _build_one_asset(p, fe)
        if r is None: continue
        _, _, ohlc_r = r
        try:
            T2  = min(1000, len(ohlc_r))
            pr2 = ohlc_r[:T2]
            C2  = pr2.shape[1]
            pd2 = np.concatenate([np.tile(pr2[0:1], (S-1, 1)), pr2], axis=0)
            w2  = as_strided(pd2, shape=(T2,S,C2),
                             strides=(pd2.strides[0], pd2.strides[0], pd2.strides[1]))
            ac2 = []
            with torch.no_grad():
                for i in range(0, T2, chunk_size):
                    b2 = torch.from_numpy(np.array(w2[i:i+chunk_size])).to(device).float()
                    b2 = (b2 - b2.mean(dim=1,keepdim=True)) / (b2.std(dim=1,keepdim=True)+1e-5)
                    b2 = torch.clamp(b2, -5.0, 5.0)
                    ic2, _ = tok.encode(b2, half=True)
                    ac2.append(ic2.reshape(-1).cpu())
            entropies.append(token_entropy(torch.cat(ac2), vocab_c))
        except:
            pass
    if len(entropies) > 1:
        H_cv = np.std(entropies) / (np.mean(entropies) + 1e-9)
        log("TOKENIZER", "Cross-asset entropy consistency (CV)",
            PASS if H_cv < 0.20 else WARN,
            f"CV={H_cv:.2%}  (>20% = tokenizer encodes some assets better than others)")

    # ── Plot token histograms ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, toks, label, vsize in [
        (axes[0], coarse, "Coarse", vocab_c),
        (axes[1], fine,   "Fine",   vocab_f),
    ]:
        hist = torch.bincount(toks, minlength=vsize).float().numpy()
        nz   = sorted(hist[hist > 0], reverse=True)
        ax.bar(range(len(nz)), nz, color="#55A868", width=1.0)
        ax.set_title(f"{label} Token Frequency (sorted, non-zero only)")
        ax.set_xlabel("rank"); ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(AUDIT_OUT / "04_token_distribution.png", dpi=120); plt.close(fig)
    log("TOKENIZER", "Token histogram plot", PASS, str(AUDIT_OUT / "04_token_distribution.png"))


# ─────────────────────────────────────────────────────────────
# 5.  HYPERPARAMETER AUDIT
# ─────────────────────────────────────────────────────────────

def audit_hyperparams():
    print("\n" + "="*60)
    print("SECTION 5 — FINE-TUNE HYPERPARAMETERS")
    print("="*60)

    checks = [
        ("FINETUNE_HEAD_LR",       (1e-7, 5e-7),  "Head LR",          FAIL),
        ("FINETUNE_FULL_LR",       (5e-8, 2e-7),  "Full fine-tune LR", FAIL),
        ("FINETUNE_FREEZE_EPOCHS", (10,   20),     "Freeze epochs",     WARN),
        ("WFV_PATIENCE",           (15,   40),     "Fine-tune patience",WARN),
        ("GRAD_CLIP",              (0.5,  2.0),    "Gradient clip",     WARN),
        ("WEIGHT_DECAY",           (1e-4, 1e-2),   "Weight decay",      WARN),
        ("DROPOUT",                (0.10, 0.30),   "Dropout rate",      WARN),
        ("PRETRAIN_LR",            (2e-5, 8e-5),   "Pre-train max LR",  WARN),
        ("FALSE_SIGNAL_MARGIN",    (0.0, getattr(CFG,"SAMPLER_THRESHOLD",0.05)),
                                   "Must be <= SAMPLER_THRESHOLD",      FAIL),
    ]

    for attr, (lo, hi), desc, default_severity in checks:
        val = getattr(CFG, attr, None)
        if val is None:
            log("HPARAMS", attr, WARN, f"Not in config.py — {desc}")
            continue
        if lo <= val <= hi:
            log("HPARAMS", attr, PASS, f"{val}  →  {desc}")
        else:
            direction = "TOO HIGH" if val > hi else "TOO LOW"
            log("HPARAMS", attr, default_severity,
                f"{val}  [{direction}]  ideal=[{lo}, {hi}]  →  {desc}")

    fsm = getattr(CFG, "FALSE_SIGNAL_MARGIN", None)
    sth = getattr(CFG, "SAMPLER_THRESHOLD",   None)
    if fsm is not None and sth is not None:
        if fsm > sth:
            log("HPARAMS", "FALSE_SIGNAL_MARGIN > SAMPLER_THRESHOLD", FAIL,
                f"FALSE_SIGNAL_MARGIN={fsm} > SAMPLER_THRESHOLD={sth}. "
                f"Fix: set FALSE_SIGNAL_MARGIN <= SAMPLER_THRESHOLD.")
        else:
            log("HPARAMS", "FALSE_SIGNAL_MARGIN vs SAMPLER_THRESHOLD", PASS, f"{fsm} <= {sth}  ✓")

    hlr = getattr(CFG, "FINETUNE_HEAD_LR", None)
    flr = getattr(CFG, "FINETUNE_FULL_LR", None)
    if hlr and flr:
        ratio = hlr / flr
        log("HPARAMS", "Head/Full LR ratio",
            PASS if 2 <= ratio <= 10 else WARN,
            f"{ratio:.1f}x  (ideal 2-10x)")

    ce = getattr(CFG, "CURRICULUM_RAMP_EPOCHS", None)
    ep = getattr(CFG, "EPOCHS", None)
    if ce is not None and ep is not None:
        log("HPARAMS", "CURRICULUM_RAMP_EPOCHS vs EPOCHS",
            PASS if 0 < ce < ep else WARN,
            f"CURRICULUM_RAMP_EPOCHS={ce}  EPOCHS={ep}")


# ─────────────────────────────────────────────────────────────
# 6.  MULTI-ASSET DATA INTEGRITY
# ─────────────────────────────────────────────────────────────

def audit_data_integrity():
    print("\n" + "="*60)
    print("SECTION 6 — MULTI-ASSET DATA INTEGRITY")
    print("="*60)

    data_dir  = _get_data_dir()
    csv_files = sorted(data_dir.glob("*.csv"))
    log("DATA", "CSV file count", PASS if csv_files else FAIL, f"{len(csv_files)} files")

    lengths, issues = [], []
    for p in csv_files:
        try:
            df = pd.read_csv(p)
            df.columns = [c.lower().strip() for c in df.columns]
            close_col = next((c for c in df.columns if "close" in c), None)
            if close_col is None:
                issues.append(f"{p.name}: no close column"); continue
            n    = len(df)
            n_na = df[close_col].isna().sum()
            lengths.append(n)
            if n_na > 0: issues.append(f"{p.name}: {n_na} NaN close values")
            if (df[close_col] <= 0).any(): issues.append(f"{p.name}: non-positive close prices")
        except Exception as e:
            issues.append(f"{p.name}: {e}")

    for iss in issues:
        log("DATA", "Asset issue", WARN, iss)
    if not issues:
        log("DATA", "All assets readable", PASS)

    if lengths:
        log("DATA", "Bar count range", PASS,
            f"min={min(lengths)}  max={max(lengths)}  mean={np.mean(lengths):.0f}")
        cv = np.std(lengths) / np.mean(lengths)
        log("DATA", "Row count consistency (CV)",
            PASS if cv < 0.10 else WARN, f"CV={cv:.2%}")

    min_bars = min(lengths) if lengths else 0
    n_folds  = getattr(CFG, "N_FOLDS", 5)
    gap      = getattr(CFG, "LOOKBACK_WINDOW", 512)
    val_frac = getattr(CFG, "VAL_FRAC", 0.15)
    needed   = int(min_bars * val_frac / n_folds)
    log("DATA", "Val bars per fold (estimated)",
        PASS if needed > gap * 2 else WARN,
        f"≈{needed} bars  (need >{gap*2} = 2×LOOKBACK_WINDOW)")


# ─────────────────────────────────────────────────────────────
# 7.  BEST POLICY JSON
# ─────────────────────────────────────────────────────────────

def audit_best_policy():
    print("\n" + "="*60)
    print("SECTION 7 — BEST POLICY (best_policy.json)")
    print("="*60)

    p = _get_file_path("best_policy.json")
    if not p.exists():
        log("POLICY", "best_policy.json", WARN, "Not found")
        return
    pol = json.loads(p.read_text())
    print(f"  Keys: {list(pol.keys())}")

    val_loss = pol.get("val_loss", pol.get("best_val_loss", None))
    if val_loss is not None:
        log("POLICY", "Best val_loss",
            PASS if val_loss < 2.0 else FAIL,
            f"{val_loss:.4f}  (>2.0 = checkpoint saved in overfit state)")

    for key in ["head_lr","full_lr","freeze_epochs","dropout","weight_decay"]:
        val = pol.get(key)
        log("POLICY", f"key: {key}",
            PASS if val is not None else WARN,
            str(val) if val is not None else "Missing")


# ─────────────────────────────────────────────────────────────
# 8.  SUMMARY REPORT
# ─────────────────────────────────────────────────────────────

def print_summary():
    print("\n" + "="*60)
    print("AUDIT SUMMARY")
    print("="*60)
    df = pd.DataFrame(results)
    counts = df.status.value_counts()
    print(f"\n  Total : {len(df)}  |  PASS: {counts.get('PASS',0)}  "
          f"WARN: {counts.get('WARN',0)}  FAIL: {counts.get('FAIL',0)}")

    fails = df[df.status == "FAIL"]
    print("\n  ❌ CRITICAL FAILURES:")
    if fails.empty:
        print("    None")
    else:
        for _, r in fails.iterrows():
            print(f"    [{r.section}] {r.check}")
            if r.detail: print(f"      → {r.detail[:150]}")

    print("\n  ⚠️  WARNINGS:")
    for _, r in df[df.status == "WARN"].iterrows():
        print(f"    [{r.section}] {r.check}: {str(r.detail)[:110]}")

    out = AUDIT_OUT / "audit_report.csv"
    df.to_csv(out, index=False)
    print(f"\n  Report → {out}")
    print(f"  Plots  → {AUDIT_OUT}/")


# ─────────────────────────────────────────────────────────────
# ENTRY
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nLPatchTST Audit  (tokens_only mode)")
    print(f"Repo : {REPO_ROOT}")
    print(f"Out  : {AUDIT_OUT}")

    targets, ohlc_ret = audit_oracle_targets()
    audit_prediction_bias()
    audit_loss_gap()
    audit_tokenizer(ohlc_ret=ohlc_ret)
    audit_hyperparams()
    audit_data_integrity()
    audit_best_policy()
    print_summary()