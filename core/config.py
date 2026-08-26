import os
#%%writefile config.py
# #config.py

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
import glob
DATA_DIR = "data"
DATA_FILE = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
# If DATA_FILE is empty, fallback to a single file to avoid crashes
if not DATA_FILE:
    DATA_FILE = ["data/NIFTY 50_30minute.csv"]

LOOKBACK_WINDOW  = 512     # paper's optimal for LPatchTST (was 400)
TOKENIZER_WINDOW   = 90       # normalization window for Kronos tokenizer
ORACLE_MAX_HOLD  = 96
FORECAST_HORIZON = 96
ATR_PERIOD       = 1     # rolling window for ATR (Oracle + backtest)

# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────────────────────────────────────
D_MODEL            = 64
N_HEADS            = 8
N_LAYERS           = 4
PATCH_LEN          = 16
STRIDE             = 12
AGGREGATION_MODE   = "mixing"   # "mixing" | "cls" | "mean"
INFERENCE_SMOOTHING = 3         # rolling window applied to raw predictions

#  # num_patches = (seq_len - patch_len) // stride + 1

# ── Input Mode ───────────────────────────────────────────────────────────────
# MASTER SWITCH: "tokens_only" | "features_only" | "combined"
# The entire pipeline (data_loader, model, train) responds to this flag.
INPUT_MODE      = "features_only" # 
USE_TALIB       = False    # If True, adds ~150 TA-Lib features when in features/combined mode

# ── LPatchTST Architecture ───────────────────────────────────────────────────
USE_LPATCHTST   = True    # False = use vanilla PatchTST, True = LPatchTST
LSTM_LAYERS     = 1      # 1 is sufficient; set 2 for deeper denoising
N_DEC_LAYERS    = 1      # Number of decoder layers
N_QUERIES       = 4      # Number of learnable query tokens

# ─────────────────────────────────────────────────────────────────────────────
# Oracle
# ─────────────────────────────────────────────────────────────────────────────
FEE_PER_SIDE = 0.001
SLIPPAGE = 0.0005

# Entry-ATR based exits
ORACLE_SL_ATR_MULT = 2.0
ORACLE_TP_ATR_MULT = 4.5

# Optional trailing logic
ORACLE_ENABLE_TRAILING = True
ORACLE_TRAIL_ATR_MULT = 3.3

SATURATION_FACTOR = 2.5
MAE_PENALTY = 0.20
MIN_TRADES_TUNE = 30

# ─────────────────────────────────────────────────────────────────────────────
# Quantile Head (Falcon-2.0 / arXiv:2608.13262 inspired)
# ─────────────────────────────────────────────────────────────────────────────
# When True, PredictionHead is replaced by a monotone multi-quantile head.
# Model forward returns (B, Q) with columns ordered ascending by level.
# Buy/sell/hold decisions still use ONLY the median column (q=0.50).
QUANTILE_HEAD    = True
# v10.1: levels retargeted around the ~63%-mass zero-atom of Oracle targets.
# Round-1 analysis: q25/q75 collapsed INTO the atom (coverage frozen at
# 62%/35% for 25 epochs) and carried no signal; the informative quantiles are
# the ones outside the atom. Four per side now, plus the decision median.
QUANTILE_LEVELS  = [0.02, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.98]
PINBALL_WEIGHT   = 1.0   # weight on mean pinball loss over all levels
V9_MEDIAN_WEIGHT = 1.5   # raised from 1.0: false_sig crept 11% -> 22% over pretrain round 1

# Head hidden width multiplier: hidden = (d_model // 2) * QUANTILE_HEAD_HIDDEN_MULT.
# 2 -> d_model wide; round-1's 32-unit bottleneck was tight for 7 correlated outputs.
QUANTILE_HEAD_HIDDEN_MULT = 2

# ─────────────────────────────────────────────────────────────────────────────
# Pretrain control
# ─────────────────────────────────────────────────────────────────────────────
# FORCE_PRETRAIN=True always runs pretrain, even if a compatible checkpoint
# exists. Otherwise: compatible checkpoint -> skip; none/incompatible -> run.
# When running with a legacy (v9) checkpoint present, WARM_START_ENCODER
# transfers stem/encoder/decoder weights (strict=False) and backs the legacy
# file up to models/pretrain_legacy_backup.pth first.
FORCE_PRETRAIN     = False
WARM_START_ENCODER = True

# ─────────────────────────────────────────────────────────────────────────────
# ORBIT — Omni-Range Bootstrap Incremental Training (arXiv:2608.13262)
# ─────────────────────────────────────────────────────────────────────────────
# ORBIT_ENABLE / ORBIT_CTX_MIN govern the legacy "ORBIT-lite" context
# randomization inside FinancialDataset.__getitem__ (train splits only,
# including fine-tune folds). Window END stays fixed; horizon is fixed.
ORBIT_ENABLE = True
ORBIT_CTX_MIN = 128

# Full ORBIT pretraining stream. When ORBIT_STREAM_MODE=True, pretrain()
# replaces ConcatDataset + DistributedWeightedSampler + epoch looping with:
#   1. Bootstrap Multi-Level Sampling  → offline per-asset sample index of
#      five-tuples (record, variable, s, e, p). Record/variable are identity
#      here (one series, one oracle target per asset); levels 3–4 sample the
#      context window [s, e) and prediction horizon p.
#   2. Domain-aware dataset weights    → global training stream assembled by a
#      low-discrepancy greedy blending rule (largest-deficit-first).
#   3. Omni-Range Incremental Training → single pass over a globally shuffled
#      stream; variable-length contexts left-padded per batch with attention
#      masks; horizons drawn from ORBIT_HORIZONS via multi-horizon oracle
#      labels. No epochs, no early stopping on repeated data.
ORBIT_STREAM_MODE   = True
# Candidate prediction horizons in bars (paper Level-4: p ~ Uniform feasible).
# Evaluation/fine-tune stay at FORECAST_HORIZON=96; pretrain sees all.
ORBIT_HORIZONS      = [16, 32, 48, 96]
# Optimizer steps for one pretrain run. Stream length = steps × BATCH_SIZE × world_size.
ORBIT_TOTAL_STEPS   = 20000
# Prescribed dataset (asset) weights, keyed by CSV basename stem.
# None = equal weight per asset (domain-aware balancing default).
ORBIT_ASSET_WEIGHTS = None          # e.g. {"NIFTY 50_30minute": 2.0}
ORBIT_INDEX_SEED    = 20260813      # reproducible offline index construction
ORBIT_INDEX_CACHE   = "models/orbit_index.npz"
ORBIT_EVAL_EVERY    = 500           # steps between logging / checkpoint evals
ORBIT_SMOOTH_N      = 500           # trailing-loss window for best-checkpoint pick

# ─────────────────────────────────────────────────────────────────────────────
# Modern architecture flags (Falcon-2.0 parity experiments)
# ─────────────────────────────────────────────────────────────────────────────
USE_MODERN_NORM = False   # swap LayerNorm -> RMSNorm in encoder/decoder
USE_ROPE = False          # rotary embeddings in self-attention (SDPA path)

# ─────────────────────────────────────────────────────────────────────────────
# Scaler tail handling (robust bucket, post-RobustScaler)
# ─────────────────────────────────────────────────────────────────────────────
# "clip"    : hard clip at ±bound IQR-units (legacy behaviour)
# "arcsinh" : smooth tail compression  x -> tau * arcsinh(x / tau)
# v10.1: enabled — multi-asset global scaler clips 2.84% of OHLC bars at ±3
# IQR (bounds were calibrated on NIFTY 50 alone, 0.44%); arcsinh keeps tail
# ordering instead of discarding it.
SCALER_TAIL_MODE = "arcsinh"
ARCSINH_TAU = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# Training#
BATCH_SIZE      = 128    # per-GPU batch; 256 effective with 2 GPUs
LEARNING_RATE   = 1e-5
PRETRAIN_LR     = 3e-5    # v10.1: was 5e-5 — grad-norm avg climbed to 6.5 with constant spike-clipping by ep45
EPOCHS          = 10
WEIGHT_DECAY    = 1e-3
PRETRAIN_WEIGHT_DECAY = 1e-2
FINETUNE_WEIGHT_DECAY = 5e-3
DROPOUT         = 0.2
PRETRAIN_DROPOUT = 0.3   # was 0.4 — v2: lower to let model learn stronger features
FINETUNE_DROPOUT = 0.2
GRAD_CLIP       = 5.0
PRETRAIN_GRAD_CLIP = 5.0
FINETUNE_GRAD_CLIP_STAGE_A = 5.0
FINETUNE_GRAD_CLIP_STAGE_B = 5.0
NUM_WORKERS     = 4     # parallel data prefetch workers (Kaggle provides 4+ CPU cores)
PREFETCH_FACTOR = 8     # batches prefetched per worker (2 GPUs × 4 workers = 32 queued batches)
USE_AMP         = True

# ─────────────────────────────────────────────────────────────────────────────
# Split Ratios
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ── Robust Clipping ──────────────────────────────────────────────────────────
# Per-column clip bounds in IQR units (NOT std devs).
# Calibrated via clip_audit.py on training data.
#
#   open/high/low/close   : p99.5 ≈ 2.97 IQR-units → bound 3.0 clips ~0.44%.
#
# v2 note: the feature set is fully NO_SCALE — the ROBUST bucket only fires
# for legacy columns (open/high/low/close raw bars and archived v1 builds).
# The feat_vol_squeeze / vs_factor entries below are dead but kept so old
# checkpoint builds remain loadable.
#
# Default bound for any other robust column: 3.0 IQR-units (~0.3% clip rate).
ROBUST_CLIP_BOUNDS: dict[str, float] = {
    "open":               3.0,
    "high":               3.0,
    "low":                3.0,
    "close":              3.0,
    "feat_vol_squeeze":   3.0,
    "vs_factor_span":     2.0,   # prefix match
}
ROBUST_CLIP_BOUND_DEFAULT: float = 3.0

# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering  ←→  features.py / FeatureConfig  (v2 minimal set)
#
# These are the ONLY config keys that feed into FeatureEngineer.
# train.py._make_feature_config() maps every key here to a FeatureConfig field.
# Changing any value here automatically changes what columns are produced,
# what columns data_loader.py routes to each scaler bucket, and what
# input_dim is passed to the model — no code edits required anywhere.
#
# v2 feature families (9 model inputs, close-only):
#   ret_norm_{h}    h ∈ FE_RET_NORM_HORIZONS     r(t,h)/(σ·√h)
#   vol_ratio_{s}   s ∈ FE_VOL_RATIO_SPANS       log(σ_s/σ_norm)
#   log_ewma_vol_{FE_VOL_NORM_SPAN}              log(σ_norm)  (absolute level)
#   mom_norm_{h}    h ∈ FE_MOM_HORIZONS         r(t,h)/(σ·√h)
# ─────────────────────────────────────────────────────────────────────────────

# EWMA volatility spans (bars) emitted as log-ratios against the
# normalization span. Produces vol_ratio_{s} columns. Default [10, 20].
FE_VOL_RATIO_SPANS = [10, 20]

# Span whose σ is the normalization denominator for ret_norm_*, mom_norm_*,
# target_norm_ret, AND the single absolute level feature log_ewma_vol_{span}.
FE_VOL_NORM_SPAN = 60

# Volatility-normalized return horizons (bars). Produces ret_norm_{h}.
FE_RET_NORM_HORIZONS = [1, 5, 20]

# Volatility-normalized momentum horizons (bars). Staggered against
# FE_RET_NORM_HORIZONS so no column is duplicated. Produces mom_norm_{h}.
FE_MOM_HORIZONS = [3, 10, 40]

# Oracle target clip bound. Normalised return targets clipped to ±FE_TARGET_CLIP
# before being used as training labels. Paper default: 20.0.
FE_TARGET_CLIP = 20.0

# Extra bars fetched beyond LOOKBACK_WINDOW at inference time so the feature
# warm-up (≈70 bars for the v2 set) is fully covered after NaN dropping.
FE_WARMUP_BARS = 200

# ── Retired v1 keys (archived pipeline: features_archive_v1.py) ─────────────
# Kept commented for reference; nothing in core/ reads them anymore.
# FE_RETURN_HORIZONS, FE_MACD_PAIRS, FE_MACD_PRICE_STD_WIN,
# FE_MACD_SIGNAL_STD_WIN, FE_MOMENTUM_PERIOD, FE_RSI_PERIOD,
# FE_VOL_ASYM_WINDOW, FE_ICP_PERIOD, FE_LOCAL_STRUCTURE_BARS,
# FE_VOL_SQUEEZE_FAST/SLOW, FE_SESSION_OPEN/CLOSE/TZ, FE_ADD_SESSION,
# USE_TALIB

# ─────────────────────────────────────────────────────────────────────────────
# Sampler
# ─────────────────────────────────────────────────────────────────────────────
BIAS_CORRECTION_POWER = -0.5  # Default: 0.0 (no correction). Negative values boost minority directional class.
# |score| below this threshold → Flat class in WeightedRandomSampler, loss, eval.
SAMPLER_THRESHOLD = 0.08
FLAT_THRESHOLD = SAMPLER_THRESHOLD          # alias for loss / diagnostics
ORACLE_THRESHOLD = SAMPLER_THRESHOLD      # oracle stats use same boundary
# False-signal dead-zone in loss; must stay < FLAT_THRESHOLD.
FALSE_SIGNAL_MARGIN = 0.03

# Epoch count for loss curriculum ramp; val eval uses this for full strictness.
CURRICULUM_RAMP_EPOCHS = 20   # must match loss.py curriculum_ramp_epochs

# When True, tokenize only train OHLC slices (no val/test in tokenizer pass).
TOKENIZE_STRICT_TRAIN_ONLY = True

# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer (Kronos Hierarchical — Pre-trained Specs)
# ─────────────────────────────────────────────────────────────────────────────
TOKENIZER_D_IN       = 24    # Current tokenizer checkpoint: OHLC + features.py matrix, excluding vs_factor
TOKENIZER_EXCLUDE_COLUMNS = ("vs_factor_span",)  # Never feed robust volatility-scaling factor to tokenizer
TOKENIZER_D_MODEL    = 256
TOKENIZER_N_HEADS    = 4
TOKENIZER_FF_DIM     = 512
TOKENIZER_N_ENC      = 4
TOKENIZER_N_DEC      = 4
TOKENIZER_S1_BITS    = 10
TOKENIZER_S2_BITS    = 10
TOKENIZER_GROUP_SIZE = 4
VOCAB_SIZE            = 2 ** (TOKENIZER_S1_BITS + TOKENIZER_S2_BITS)

# Tokenizer Hyperparameters (for training/loss consistency)
TOKENIZER_BETA       = 0.05
TOKENIZER_GAMMA0     = 1.0
TOKENIZER_GAMMA      = 1.5
TOKENIZER_ZETA       = 0.25
TOKENIZER_ATTN_DROPOUT = 0.0
TOKENIZER_FFN_DROPOUT  = 0.0
TOKENIZER_RESID_DROPOUT = 0.0

TOKENIZER_CHUNK_SIZE  = 2048   # Reduced for larger d_model
TOKENIZER_PATH        = "models/model.safetensors"

# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward Validation
WFV_ENABLED    = True
FINETUNE_FULL_LR = 5e-6
FINETUNE_HEAD_LR = 1e-5
WFV_TRAIN_BARS = 21000
WFV_VAL_BARS  = 2500
WFV_STEP_BARS  = 2500
WFV_MIN_FOLDS  = 3
WFV_PATIENCE   = 20
GAP_DENSITY_SAFETY = 0.70  # assume 30% fewer bars/day than median
GAP_MARGIN_DAYS    = 5     # bump margin a bit
MIN_GAP_DAYS    = 7
MIN_TRAIN_WINDOW_YEARS = 3
VAL_WINDOW_MONTHS = 6
N_FOLDS = 1

# ─────────────────────────────────────────────────────────────────────────────
# Runtime — set dynamically, do not edit
# ─────────────────────────────────────────────────────────────────────────────
# Populated by train.py / evaluate.py after feature columns are resolved.
# Value = len(feature_cols) when USE_TOKENIZER=False, else 1.
NUM_FEATURES = None
MODEL_PATH = "models/best_model_final.pth"
