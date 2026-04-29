# config.py

# --- Data Parameters ---
DATA_FILE = ["NIFTY 50_30minute.csv"]
LOOKBACK_WINDOW = 400
ORACLE_MAX_HOLD = 96
FORECAST_HORIZON = 96
ATR_PERIOD = 14

# --- Architecture ---
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
PATCH_LEN = 16
STRIDE = 8
AGGREGATION_MODE = "mixing"
INFERENCE_SMOOTHING = 3

# NOTE: The strict geometry check (LOOKBACK_WINDOW - PATCH_LEN) % STRIDE == 0
# has been intentionally removed. torch.unfold() uses floor division internally,
# so any (seq_len, patch_len, stride) triplet where seq_len >= patch_len is valid:
#   num_patches = (seq_len - patch_len) // stride + 1
# The check rejected valid configs (e.g. LOOKBACK_WINDOW=401) unnecessarily.

# --- Oracle Parameters ---
FEE_PER_SIDE = 0.001
SLIPPAGE = 0.0005
ATR_MULT = 3.0
SATURATION_FACTOR = 2.5
MAE_PENALTY = 0.20
MIN_TRADES_TUNE = 30

# --- Training ---
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EPOCHS = 50
WEIGHT_DECAY = 0.05
GRAD_CLIP = 2.0
NUM_WORKERS    = 4   # was 0 — enables parallel data prefetching
PREFETCH_FACTOR = 2  # batches prefetched per worker
USE_AMP = True

# --- Split Ratios ---
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# --- Feature Engineering ---
OB_ATR_MULT = 0.5
OB_INTERNAL_LB = 5
OB_SWING_LB = 20
OB_MAX_OBS = 5
OB_IOU_THRESHOLD = 0.85
FE_MISSING_FILL = 5.0
FE_MOMENTUM_PERIOD = 14
FE_VOL_SHORT_PERIOD = 6
FE_VOL_LONG_PERIOD = 100
FE_SKEW_PERIOD = 28
FE_ZSCORE_PERIOD = 50
FE_ICP_PERIOD = 14
FE_MDS_FAST_WINDOW = 5
FE_MDS_SLOW_WINDOW = 30
FE_VOL_ASYM_WINDOW = 20
FE_STOCH_PERIOD = 14
FE_ADX_PERIOD = 14
FE_BAR_PER_DAY = 13
SAMPLER_THRESHOLD = 0.10
SESSION_OPEN = "09:15"
SESSION_CLOSE = "15:30"
SESSION_TZ = "Asia/Kolkata"

# --- Tokenizer ---
USE_TOKENIZER = False
TOKENIZER_BITS = 12
VOCAB_SIZE = 2 ** TOKENIZER_BITS

# --- Walk-Forward Validation ---
WFV_ENABLED = True
WFV_TRAIN_BARS = 8000
WFV_TEST_BARS = 2000
WFV_STEP_BARS = 2000
WFV_MIN_FOLDS = 3
WFV_PATIENCE = 15