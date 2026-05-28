import os
#%%writefile config.py
# #config.py

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
import glob
DATA_DIR = "Data"
DATA_FILE = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
# If DATA_FILE is empty, fallback to a single file to avoid crashes
if not DATA_FILE:
    DATA_FILE = ["Data/NIFTY 50_30minute.csv"]

LOOKBACK_WINDOW  = 512     # paper's optimal for LPatchTST (was 400)
ORACLE_MAX_HOLD  = 15
FORECAST_HORIZON = 15
ATR_PERIOD       = 14      # rolling window for ATR (Oracle + backtest)

# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────────────────────────────────────
D_MODEL            = 96
N_HEADS            = 4
N_LAYERS           = 5
PATCH_LEN          = 16
STRIDE             = 12
AGGREGATION_MODE   = "mixing"   # "mixing" | "cls" | "mean"
INFERENCE_SMOOTHING = 3         # rolling window applied to raw predictions

#  # num_patches = (seq_len - patch_len) // stride + 1

# ── Input Mode ───────────────────────────────────────────────────────────────
# MASTER SWITCH: "tokens_only" | "features_only" | "combined"
# The entire pipeline (data_loader, model, train) responds to this flag.
INPUT_MODE      = "tokens_only"
USE_TALIB       = False    # If True, adds ~150 TA-Lib features when in features/combined mode

# ── LPatchTST Architecture ───────────────────────────────────────────────────
USE_LPATCHTST   = True    # False = use vanilla PatchTST, True = LPatchTST
LSTM_LAYERS     = 1      # 1 is sufficient; set 2 for deeper denoising

# ─────────────────────────────────────────────────────────────────────────────
# Oracle
# ─────────────────────────────────────────────────────────────────────────────
FEE_PER_SIDE = 0.001
SLIPPAGE = 0.0005

# Entry-ATR based exits
ORACLE_SL_ATR_MULT = 1.6
ORACLE_TP_ATR_MULT = 3.7

# Optional trailing logic
ORACLE_ENABLE_TRAILING = True
ORACLE_TRAIL_ATR_MULT = 3.3

SATURATION_FACTOR = 2.5
MAE_PENALTY = 0.20
MIN_TRADES_TUNE = 30

# ─────────────────────────────────────────────────────────────────────────────
# Training#
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE      = 128
LEARNING_RATE   = 1e-5
EPOCHS          = 100
WEIGHT_DECAY    = 1e-3
DROPOUT         = 0.2
GRAD_CLIP       = 2.0
PRETRAIN_GRAD_CLIP = 5.0
FINETUNE_GRAD_CLIP_STAGE_A = 2.0
FINETUNE_GRAD_CLIP_STAGE_B = 2.0
NUM_WORKERS     = 2     # parallel data prefetch workers
PREFETCH_FACTOR = 4     # batches prefetched per worker
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
#   vs_factor_span260 : p99.5 = 1.493 IQR-units → bound 2.0 clips nothing.
#   feat_vol_squeeze  : p99.5 = 2.821 IQR-units → bound 2.5 clips ~0.9%.
#
ROBUST_CLIP_BOUNDS: dict[str, float] = {
    "vs_factor_span":  2.0,   # prefix match
    "feat_vol_squeeze": 2.5,
}
ROBUST_CLIP_BOUND_DEFAULT: float = 3.0   # fallback for unknown robust columns

# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering  ←→  features.py / FeatureConfig
#
# These are the ONLY config keys that feed into FeatureEngineer.
# train.py._make_feature_config() maps every key here to a FeatureConfig field.
# Changing any value here automatically changes what columns are produced,
# what columns data_loader.py routes to each scaler bucket, and what
# input_dim is passed to the model — no code edits required anywhere.
# ─────────────────────────────────────────────────────────────────────────────

# EWMA volatility span (bars). Controls σ_t used by ret_norm_* and vs_factor.
# Larger span = slower regime adaptation. Maps to FeatureConfig.ewma_span.
FE_VOL_LONG_PERIOD = 260

# Multi-horizon normalised return lookback windows (trading bars).
# Produces columns: ret_norm_1d, ret_norm_5d, ret_norm_21d, …
# All are vol-scaled (≈ z-score) → NO_SCALE bucket in data_loader.py.
FE_RETURN_HORIZONS = [1, 3, 6, 13, 26, 65, 130, 260]

# Multi-scale MACD (short_span, long_span) pairs.
# Produces columns: macd_8_24, macd_16_48, macd_32_96.
# All 3-step normalised (std ≈ 1.05) → NO_SCALE bucket in data_loader.py.
FE_MACD_PAIRS = [(8, 24), (26, 78), (52, 156)]

# MACD Step-2: rolling price std window for per-instrument normalisation.
# Paper default: 63 bars. Maps to FeatureConfig.macd_price_std_window.
FE_MACD_PRICE_STD_WIN = 260

# MACD Step-3: rolling regime std window for cross-sectional normalisation.
# Paper default: 252 bars. Maps to FeatureConfig.macd_signal_std_window.
FE_MACD_SIGNAL_STD_WIN = 3276

# Oracle target clip bound. Normalised return targets clipped to ±FE_TARGET_CLIP
# before being used as training labels. Paper default: 20.0.
FE_TARGET_CLIP = 20.0

# ─────────────────────────────────────────────────────────────────────────────
# OHLC Feature Engineering  (Features 6–13)
# Maps to new FeatureConfig fields in features.py
# ─────────────────────────────────────────────────────────────────────────────

# Kaufman Efficiency Ratio + RSI lookback (bars). Maps to momentum_period.
FE_MOMENTUM_PERIOD     = 26

# Separate RSI period. Set to None to share FE_MOMENTUM_PERIOD.
FE_RSI_PERIOD          = 14

# Rolling window for directional vol asymmetry. Maps to vol_asym_window.
FE_VOL_ASYM_WINDOW     = 65

# Smoothing window for Internal Close Position. Maps to icp_period.
FE_ICP_PERIOD          = 13

# Donchian channel lookback for local structure. ~5 days on 30-min NIFTY.
FE_LOCAL_STRUCTURE_BARS = 65

# Fast/slow ATR windows for vol squeeze ratio. Maps to vol_squeeze_fast/slow.
FE_VOL_SQUEEZE_FAST    = 5
FE_VOL_SQUEEZE_SLOW    = 26
 #
# Session time-of-day encoding (NIFTY 30-min).
FE_SESSION_OPEN        = "09:15"
FE_SESSION_CLOSE       = "15:30"
FE_SESSION_TZ          = "Asia/Kolkata"
FE_ADD_SESSION         = True

# ─────────────────────────────────────────────────────────────────────────────
# Sampler
# ─────────────────────────────────────────────────────────────────────────────
# |score| below this threshold → Flat class in WeightedRandomSampler, loss, eval.
SAMPLER_THRESHOLD = 0.05
FLAT_THRESHOLD = SAMPLER_THRESHOLD          # alias for loss / diagnostics
ORACLE_THRESHOLD = SAMPLER_THRESHOLD      # oracle stats use same boundary
# False-signal dead-zone in loss; must stay < FLAT_THRESHOLD.
FALSE_SIGNAL_MARGIN = 0.05

# Epoch count for loss curriculum ramp; val eval uses this for full strictness.
CURRICULUM_RAMP_EPOCHS = 20   # must match loss.py curriculum_ramp_epochs

# When True, tokenize only train OHLC slices (no val/test in tokenizer pass).
TOKENIZE_STRICT_TRAIN_ONLY = True

# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer (Kronos Hierarchical — Pre-trained Specs)
# ─────────────────────────────────────────────────────────────────────────────
TOKENIZER_D_IN       = 6
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
TOKENIZER_GAMMA      = 1.1
TOKENIZER_ZETA       = 0.05
TOKENIZER_ATTN_DROPOUT = 0.0
TOKENIZER_FFN_DROPOUT  = 0.0
TOKENIZER_RESID_DROPOUT = 0.0

TOKENIZER_CHUNK_SIZE  = 2048   # Reduced for larger d_model
TOKENIZER_PATH        = "model.safetensors"

# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward Validation
# ─────────────────────────────────────────────────────────────────────────────
WFV_ENABLED    = True
WFV_TRAIN_BARS = 21000
WFV_VAL_BARS  = 2500
WFV_STEP_BARS  = 2500
WFV_MIN_FOLDS  = 3
WFV_PATIENCE   = 15
GAP_DENSITY_SAFETY = 0.70  # assume 30% fewer bars/day than median
GAP_MARGIN_DAYS    = 5     # bump margin a bit
MIN_GAP_DAYS    = 7
MIN_TRAIN_WINDOW_YEARS = 3
VAL_WINDOW_MONTHS = 6

# ─────────────────────────────────────────────────────────────────────────────
# Runtime — set dynamically, do not edit
# ─────────────────────────────────────────────────────────────────────────────
# Populated by train.py / evaluate.py after feature columns are resolved.
# Value = len(feature_cols) when USE_TOKENIZER=False, else 1.
NUM_FEATURES = None
