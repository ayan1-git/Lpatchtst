# config.py

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
DATA_FILE        = ["Data /NIFTY 50_30minute.csv"]
LOOKBACK_WINDOW  = 400
ORACLE_MAX_HOLD  = 96
FORECAST_HORIZON = 96
ATR_PERIOD       = 14      # rolling window for ATR (Oracle + backtest)

# ─────────────────────────────────────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────────────────────────────────────
D_MODEL            = 64
N_HEADS            = 4
N_LAYERS           = 2
PATCH_LEN          = 16
STRIDE             = 8
AGGREGATION_MODE   = "mixing"   # "mixing" | "cls" | "mean"
INFERENCE_SMOOTHING = 3         # rolling window applied to raw predictions

# NOTE: The strict geometry check (LOOKBACK_WINDOW - PATCH_LEN) % STRIDE == 0
# has been intentionally removed. torch.unfold() uses floor division internally,
# so any (seq_len, patch_len, stride) triplet where seq_len >= patch_len is valid:
#   num_patches = (seq_len - patch_len) // stride + 1

# ─────────────────────────────────────────────────────────────────────────────
# Oracle
# ─────────────────────────────────────────────────────────────────────────────
FEE_PER_SIDE      = 0.001
SLIPPAGE          = 0.0005
ATR_MULT          = 3.0
SATURATION_FACTOR = 2.5
MAE_PENALTY       = 0.20
MIN_TRADES_TUNE   = 30

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE      = 128
LEARNING_RATE   = 1e-4
EPOCHS          = 50
WEIGHT_DECAY    = 0.05
GRAD_CLIP       = 2.0
NUM_WORKERS     = 4     # parallel data prefetch workers
PREFETCH_FACTOR = 2     # batches prefetched per worker
USE_AMP         = True

# ─────────────────────────────────────────────────────────────────────────────
# Split Ratios
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

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
FE_VOL_LONG_PERIOD = 100

# Multi-horizon normalised return lookback windows (trading bars).
# Produces columns: ret_norm_1d, ret_norm_5d, ret_norm_21d, …
# All are vol-scaled (≈ z-score) → NO_SCALE bucket in data_loader.py.
FE_RETURN_HORIZONS = [1, 5, 21, 63, 126, 252]

# Multi-scale MACD (short_span, long_span) pairs.
# Produces columns: macd_8_24, macd_16_48, macd_32_96.
# All 3-step normalised (std ≈ 1.05) → NO_SCALE bucket in data_loader.py.
FE_MACD_PAIRS = [(8, 24), (16, 48), (32, 96)]

# MACD Step-2: rolling price std window for per-instrument normalisation.
# Paper default: 63 bars. Maps to FeatureConfig.macd_price_std_window.
FE_MACD_PRICE_STD_WIN = 63

# MACD Step-3: rolling regime std window for cross-sectional normalisation.
# Paper default: 252 bars. Maps to FeatureConfig.macd_signal_std_window.
FE_MACD_SIGNAL_STD_WIN = 252

# Oracle target clip bound. Normalised return targets clipped to ±FE_TARGET_CLIP
# before being used as training labels. Paper default: 20.0.
FE_TARGET_CLIP = 20.0

# ─────────────────────────────────────────────────────────────────────────────
# OHLC Feature Engineering  (Features 6–13)
# Maps to new FeatureConfig fields in features.py
# ─────────────────────────────────────────────────────────────────────────────

# Kaufman Efficiency Ratio + RSI lookback (bars). Maps to momentum_period.
FE_MOMENTUM_PERIOD     = 14

# Separate RSI period. Set to None to share FE_MOMENTUM_PERIOD.
FE_RSI_PERIOD          = None

# Rolling window for directional vol asymmetry. Maps to vol_asym_window.
FE_VOL_ASYM_WINDOW     = 20

# Smoothing window for Internal Close Position. Maps to icp_period.
FE_ICP_PERIOD          = 14

# Donchian channel lookback for local structure. ~5 days on 30-min NIFTY.
FE_LOCAL_STRUCTURE_BARS = 65

# Fast/slow ATR windows for vol squeeze ratio. Maps to vol_squeeze_fast/slow.
FE_VOL_SQUEEZE_FAST    = 5
FE_VOL_SQUEEZE_SLOW    = 20

# Session time-of-day encoding (NIFTY 30-min).
FE_SESSION_OPEN        = "09:15"
FE_SESSION_CLOSE       = "15:30"
FE_SESSION_TZ          = "Asia/Kolkata"
FE_ADD_SESSION         = True

# ─────────────────────────────────────────────────────────────────────────────
# Sampler
# ─────────────────────────────────────────────────────────────────────────────
# |score| below this threshold → Flat class in WeightedRandomSampler.
SAMPLER_THRESHOLD = 0.10

# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────
USE_TOKENIZER  = False
TOKENIZER_BITS = 12
VOCAB_SIZE     = 2 ** TOKENIZER_BITS

# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward Validation
# ─────────────────────────────────────────────────────────────────────────────
WFV_ENABLED    = True
WFV_TRAIN_BARS = 8000
WFV_TEST_BARS  = 2000
WFV_STEP_BARS  = 2000
WFV_MIN_FOLDS  = 3
WFV_PATIENCE   = 15

# ─────────────────────────────────────────────────────────────────────────────
# Runtime — set dynamically, do not edit
# ─────────────────────────────────────────────────────────────────────────────
# Populated by train.py / evaluate.py after feature columns are resolved.
# Value = len(feature_cols) when USE_TOKENIZER=False, else 1.
NUM_FEATURES = None