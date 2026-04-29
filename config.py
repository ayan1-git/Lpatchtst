# config.py

# --- Data ---
DATA_FILE        = ["NIFTY 50_30minute.csv"]
LOOKBACK_WINDOW  = 400
ORACLE_MAX_HOLD  = 96
FORECAST_HORIZON = 96
ATR_PERIOD       = 14      # rolling window for ATR (Oracle + backtest)

# --- Model Architecture ---
D_MODEL          = 64
N_HEADS          = 4
N_LAYERS         = 2
PATCH_LEN        = 16
STRIDE           = 8
AGGREGATION_MODE = "mixing"   # "mixing" | "cls" | "mean"
INFERENCE_SMOOTHING = 3       # rolling window for prediction smoothing

# --- Oracle ---
FEE_PER_SIDE     = 0.001
SLIPPAGE         = 0.0005
ATR_MULT         = 3.0
SATURATION_FACTOR = 2.5
MAE_PENALTY      = 0.20
MIN_TRADES_TUNE  = 30

# --- Training ---
BATCH_SIZE       = 128
LEARNING_RATE    = 1e-4
EPOCHS           = 50
WEIGHT_DECAY     = 0.05
GRAD_CLIP        = 2.0
NUM_WORKERS      = 4
PREFETCH_FACTOR  = 2
USE_AMP          = True

# --- Split Ratios ---
TRAIN_RATIO      = 0.70
VAL_RATIO        = 0.15
TEST_RATIO       = 0.15

# --- Feature Engineering (features.py / FeatureConfig) ---
# FE_VOL_LONG_PERIOD → FeatureConfig.ewma_span
# Controls EWMA volatility span for σ_t, ret_norm_*, vs_factor.
# Larger span = slower adaptation to new volatility regimes.
FE_VOL_LONG_PERIOD      = 100

# Multi-horizon normalised return lookback windows (trading days)
# Produces columns: ret_norm_1d, ret_norm_5d, ..., ret_norm_252d
FE_RETURN_HORIZONS      = [1, 5, 21, 63, 126, 252]

# Multi-scale MACD (short_span, long_span) pairs
# Produces columns: macd_8_24, macd_16_48, macd_32_96
FE_MACD_PAIRS           = [(8, 24), (16, 48), (32, 96)]

# MACD Step-2: rolling price std window (paper default: 63 days)
FE_MACD_PRICE_STD_WIN   = 63

# MACD Step-3: rolling regime std window (paper default: 252 days)
FE_MACD_SIGNAL_STD_WIN  = 252

# Oracle target clip bound for normalised return target (paper default: ±20)
FE_TARGET_CLIP          = 20.0

# --- Sampler ---
SAMPLER_THRESHOLD = 0.10   # |score| below this → Flat class in WeightedRandomSampler

# --- Tokenizer ---
USE_TOKENIZER    = False
TOKENIZER_BITS   = 12
VOCAB_SIZE       = 2 ** TOKENIZER_BITS

# --- Walk-Forward Validation ---
WFV_ENABLED      = True
WFV_TRAIN_BARS   = 8000
WFV_TEST_BARS    = 2000
WFV_STEP_BARS    = 2000
WFV_MIN_FOLDS    = 3
WFV_PATIENCE     = 15

# --- Set dynamically at runtime (do not edit) ---
NUM_FEATURES     = None    # set by train.py after feature columns are resolved