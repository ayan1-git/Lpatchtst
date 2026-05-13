"""
talib_features.py
=================
Automated TA-Lib feature expansion for OHLC data.
Pre-calculates feature buckets at module load time to ensure compatibility with 
the pipeline's static registration system.
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EPS = 1e-8

# ── TA-Lib import guard ───────────────────────────────────────────────────────
try:
    import talib
    _TALIB_AVAILABLE = True
except ImportError:
    _TALIB_AVAILABLE = False
    logger.warning("TA-Lib not installed. Features will be empty.")

# ── Module-level Feature Discovery ───────────────────────────────────────────
TALIB_PASSTHROUGH = []
TALIB_SCALE = []
_FUNC_REGISTRY = [] # List of (func_name, group, is_candle, output_count)

if _TALIB_AVAILABLE:
    _groups = talib.get_function_groups()
    _ignore_groups = ["Volume Indicators", "Math Operators", "Math Transform"]
    
    for _group, _funcs in _groups.items():
        if _group in _ignore_groups:
            continue
            
        for _f_name in _funcs:
            # Candlestick Patterns
            if _group == "Pattern Recognition":
                _name = f"talib_{_f_name.lower()}"
                TALIB_PASSTHROUGH.append(_name)
                _FUNC_REGISTRY.append((_f_name, _group, True, 1))
                continue
            
            # Multi-output function mapping
            _multi = {
                "BBANDS": 3, "MACD": 3, "MACDEXT": 3, "MACDFIX": 3, 
                "STOCH": 2, "STOCHF": 2, "STOCHRSI": 2, "MAMA": 2, "AROON": 2,
                "MINMAX": 2, "HT_PHASOR": 2, "HT_SINE": 2
            }
            _count = _multi.get(_f_name, 1)
            
            # Heuristic for bucket assignment
            _is_osc = any(x in _f_name for x in ["RSI", "MFI", "ADX", "STOCH", "WILLR", "AROON", "ULTOSC", "CCI"])
            
            for _i in range(_count):
                _suffix = f"_{_i}" if _count > 1 else ""
                _name = f"talib_{_f_name.lower()}{_suffix}"
                if _is_osc:
                    TALIB_PASSTHROUGH.append(_name)
                else:
                    TALIB_SCALE.append(_name)
            
            _FUNC_REGISTRY.append((_f_name, _group, False, _count))

def _safe(arr, expected_len: int = None) -> np.ndarray:
    """Cast to float64 and replace inf with nan. Ensures 1D of correct length."""
    if arr is None: return np.array([], dtype=np.float64)
    out = np.array(arr, dtype=np.float64)
    
    # If it's multi-dimensional, try to find a 1D slice that matches expected_len
    if out.ndim > 1:
        if expected_len is not None:
            # Check if any dimension matches
            for axis in range(out.ndim):
                if out.shape[axis] == expected_len:
                    # Take first slice along other dimensions
                    if axis == 0: out = out[0]
                    else: out = out[:, 0] # simplistic
                    break
            else:
                # No match, take first anyway but it might still fail length check
                out = out.reshape(-1)[:expected_len]
        else:
            out = out.flatten()
            
    out[~np.isfinite(out)] = np.nan
    return out

def build_talib_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ALL pre-registered TA-Lib features using OHLC data.
    """
    if not _TALIB_AVAILABLE:
        return pd.DataFrame(index=df_raw.index)

    # Standardize inputs
    c = np.array(df_raw["close"].values, dtype=np.float64)
    o = np.array(df_raw["open"].values, dtype=np.float64) if "open" in df_raw.columns else c
    h = np.array(df_raw["high"].values, dtype=np.float64) if "high" in df_raw.columns else c
    l = np.array(df_raw["low"].values, dtype=np.float64) if "low" in df_raw.columns else c
    v = np.array(df_raw["volume"].values, dtype=np.float64) if "volume" in df_raw.columns else np.zeros_like(c)
    idx = df_raw.index
    n_rows = len(c)

    # Pre-calculate reference scale (approx volatility) for Tanh normalization
    price_std = np.nanstd(c) + _EPS
    
    feats: dict[str, np.ndarray] = {}
    
    for f_name, group, is_candle, count in _FUNC_REGISTRY:
        try:
            func = getattr(talib, f_name)
            
            if is_candle:
                val = _safe(func(o, h, l, c), n_rows)
                if len(val) == n_rows:
                    feats[f"talib_{f_name.lower()}"] = val / 100.0
                continue

            # ── 1. Determine Inputs & Call ──
            # Try most common signatures
            outputs = None
            try:
                outputs = func(o, h, l, c)
            except Exception:
                try:
                    outputs = func(h, l, c)
                except Exception:
                    try:
                        outputs = func(c)
                    except Exception:
                        try:
                            outputs = func(h, l, c, v)
                        except Exception:
                            try:
                                outputs = func(h, l)
                            except Exception:
                                try:
                                    outputs = func(c, v)
                                except Exception:
                                    continue

            # ── 2. Handle Outputs ──
            if count > 1:
                # Tuple of arrays
                if isinstance(outputs, (list, tuple)):
                    for i in range(count):
                        name = f"talib_{f_name.lower()}_{i}"
                        if i < len(outputs):
                            val = _safe(outputs[i], n_rows)
                            if len(val) == n_rows:
                                feats[name] = _apply_smart_normalization(f_name, val, price_std, group, c)
                # Single 2D array
                elif isinstance(outputs, np.ndarray) and outputs.ndim == 2:
                    for i in range(min(count, outputs.shape[0])):
                        name = f"talib_{f_name.lower()}_{i}"
                        val = _safe(outputs[i], n_rows)
                        if len(val) == n_rows:
                            feats[name] = _apply_smart_normalization(f_name, val, price_std, group, c)
            else:
                # Single output
                val = _safe(outputs, n_rows)
                if len(val) == n_rows:
                    feats[f"talib_{f_name.lower()}"] = _apply_smart_normalization(f_name, val, price_std, group, c)
                        
        except Exception:
            continue

    # Final length check before DataFrame creation
    final_feats = {}
    for k, v in feats.items():
        if len(v) == n_rows:
            final_feats[k] = v
        else:
            logger.debug("Dropping %s: length mismatch (%d vs %d)", k, len(v), n_rows)

    df_out = pd.DataFrame(final_feats, index=idx, dtype=np.float64)
    # Ensure all statically registered columns exist (filled with NaN if missing)
    all_expected = TALIB_PASSTHROUGH + TALIB_SCALE
    df_out = df_out.reindex(columns=all_expected)
    return df_out

def _apply_smart_normalization(f_name: str, arr: np.ndarray, price_std: float, group: str, close: np.ndarray) -> np.ndarray:
    # 1. Price Levels (Overlap Studies, Price Transform, and some Statistics)
    if group in ("Overlap Studies", "Price Transform") or any(x in f_name for x in ["LINEARREG", "MIDPOINT", "MIDPRICE", "AVGPRICE", "TSF"]):
        dev = (close - arr) / (arr + _EPS)
        return np.tanh(dev * 100.0)

    # 2. Oscillators & Bounded Indicators
    # a. Center-Zero Oscillators [-100, 100] -> [-1, 1]
    if any(x in f_name for x in ["CMO", "AROONOSC"]):
        return arr / 100.0

    # b. Bounded [0, 100] -> [-1, 1]
    if any(x in f_name for x in ["RSI", "MFI", "ADX", "STOCH", "ULTOSC", "AROON", "DX"]):
        return (arr - 50.0) / 50.0

    # c. Negatively Bounded [-100, 0] -> [-1, 1]
    if "WILLR" in f_name:
        return (arr + 50.0) / 50.0

    # d. Unbounded Oscillators (CCI etc.) -> Soft Clip with Tanh
    if "CCI" in f_name:
        return np.tanh(arr / 100.0)

    # 3. Rate of Change / Momentum / Volatility Normalized
    if "NATR" in f_name or "ROC" in f_name or "MOM" in f_name:
        if "ROC" in f_name or "NATR" in f_name:
            if "ROCR" in f_name:
                base = 100.0 if "100" in f_name else 1.0
                return np.tanh((arr - base) / (base * 0.05 + _EPS))
            return np.tanh(arr / 5.0)
        return np.tanh(arr / (price_std * 0.1 + _EPS))

    # 4. General Tanh scaling
    return np.tanh(arr / (price_std * 0.1 + _EPS))