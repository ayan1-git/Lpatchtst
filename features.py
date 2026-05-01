"""
features.py
===========
Production-grade feature engineering for financial time-series models.

Implements the three feature families from:
  "Deep Learning for Financial Time Series" (VLSTM / PsLSTM benchmark paper).

Original Features (close-only)
-------------------------------
1.  EWMA Volatility  σ_t                         — Eqs. 16–17
2.  Multi-Horizon Normalized Returns r_norm(t,h)  — Eq. 18
3.  Multi-Scale MACD Momentum Signals             — Eqs. 19–21
4.  Volatility Scaling Factor  1/σ_t              — Eq. 22
5.  Normalized Return Target  (training only)     — Eq. 23

Added Features (OHLC-based, all NO_SCALE bucket)
-------------------------------------------------
6.  Kaufman Efficiency Ratio    feat_efficiency       [-1, +1]
7.  Internal Close Position     feat_icp              [-1, +1]
8.  RSI (centered)              feat_momentum_rsi     [-1, +1]
9.  Directional Vol Asymmetry   feat_vol_asymmetry    [-1, +1]
10. Local Structure Position    feat_local_structure  [-1, +1]
11. Session Time Sin            feat_session_sin      [-1, +1]
12. Session Time Cos            feat_session_cos      [-1, +1]
13. Vol Squeeze (ATR ratio)     feat_vol_squeeze      [>0, ROBUST]

Column count: 13 close-only + 8 OHLC-based = 21 total model inputs
(vs_factor is ROBUST; feat_vol_squeeze is ROBUST; all others NO_SCALE)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

_EPS = 1e-10  # guard against zero-division throughout


# ──────────────────────────────────────────────────────────────────────────────
# Inference-mode guard
# ──────────────────────────────────────────────────────────────────────────────

# Set to True before any live inference run. Once True, normalized_return_target()
# and include_target=True in FeatureEngineer.build() will raise RuntimeError.
# Can be controlled via environment variable: FEATURE_INFERENCE_MODE=1
import os as _os
_INFERENCE_MODE: bool = _os.environ.get("FEATURE_INFERENCE_MODE", "0").strip() == "1"


def set_inference_mode(active: bool = True) -> None:
    """
    Globally enable or disable inference mode.

    Call set_inference_mode(True) at the top of any live inference script.
    This makes accidental include_target=True a hard RuntimeError instead of
    a silent look-ahead leak.

    Can also be set via environment variable: FEATURE_INFERENCE_MODE=1
    """
    global _INFERENCE_MODE
    _INFERENCE_MODE = active
    logger.info(
        "Inference mode %s. normalized_return_target() is %s.",
        "ENABLED" if active else "DISABLED",
        "BLOCKED" if active else "PERMITTED",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FeatureConfig:
    """
    Central configuration for all feature engineering hyperparameters.

    Original paper fields
    ---------------------
    ewma_span : int
        Span for the EWMA volatility estimator (σ_t).
        α = 2 / (span + 1). Larger span → slower regime adaptation.

    return_horizons : list[int]
        Lookback windows (bars) for multi-horizon normalised return features.

    macd_pairs : list[tuple[int, int]]
        (short_span, long_span) pairs for multi-scale MACD signals.

    macd_price_std_window : int
        Rolling window for Step-2 price-scale normalisation.

    macd_signal_std_window : int
        Rolling window for Step-3 regime normalisation.

    target_clip : float
        Symmetric clip bound for the normalised return target (±20).

    New OHLC feature fields
    -----------------------
    momentum_period : int
        Lookback for KER (efficiency ratio) and RSI. Default 14.

    rsi_period : int
        Separate RSI period if you want it decoupled from momentum_period.
        If None, falls back to momentum_period.

    vol_asym_window : int
        Rolling window for directional volatility asymmetry. Default 20.

    icp_period : int
        Smoothing window for Internal Close Position. Default 14.

    local_structure_bars : int
        Rolling window for local high/low range. Default 65 bars
        (≈ 5 trading days on 30-min NIFTY data).

    vol_squeeze_fast : int
        Fast ATR window for squeeze ratio. Default 5.

    vol_squeeze_slow : int
        Slow ATR window for squeeze ratio. Default 20.

    atr_period : int
        ATR period used internally. Must match config.ATR_PERIOD. Default 14.

    session_open : str
        Session open time "HH:MM" for cyclic time-of-day encoding.

    session_close : str
        Session close time "HH:MM".

    session_tz : str
        Timezone string (e.g. "Asia/Kolkata").

    add_session_features : bool
        Whether to compute feat_session_sin / feat_session_cos.
        Requires a DatetimeIndex. Default True.
    """

    # ── paper fields ──────────────────────────────────────────────────────────
    ewma_span: int = 260

    return_horizons: list[int] = field(
        default_factory=lambda: [1, 3, 6, 13, 26, 65, 130, 260]
    )

    macd_pairs: list[tuple[int, int]] = field(
        default_factory=lambda: [(8, 24), (26, 78), (52, 156)]
    )

    macd_price_std_window: int = 260
    macd_signal_std_window: int = 3276
    target_clip: float = 20.0

    # ── new OHLC feature fields ───────────────────────────────────────────────
    momentum_period: int = 26
    rsi_period: Optional[int] = 14      # None → use momentum_period
    vol_asym_window: int = 65
    icp_period: int = 13
    local_structure_bars: int = 65        # ~5 days on 30-min NIFTY
    vol_squeeze_fast: int = 5
    vol_squeeze_slow: int = 26
    atr_period: int = 14

    session_open: str = "09:15"
    session_close: str = "15:30"
    session_tz: str = "Asia/Kolkata"
    add_session_features: bool = True

    def __post_init__(self) -> None:
        if self.ewma_span < 1:
            raise ValueError(f"ewma_span must be >= 1, got {self.ewma_span}")
        if not self.return_horizons:
            raise ValueError("return_horizons must not be empty.")
        for s, l in self.macd_pairs:
            if s >= l:
                raise ValueError(
                    f"MACD short_span ({s}) must be < long_span ({l})."
                )
        if self.target_clip <= 0:
            raise ValueError("target_clip must be positive.")
        if self.momentum_period < 2:
            raise ValueError("momentum_period must be >= 2.")
        if self.vol_asym_window < 2:
            raise ValueError("vol_asym_window must be >= 2.")
        if self.icp_period < 1:
            raise ValueError("icp_period must be >= 1.")
        if self.local_structure_bars < 2:
            raise ValueError("local_structure_bars must be >= 2.")
        if self.vol_squeeze_fast < 1 or self.vol_squeeze_slow < 2:
            raise ValueError("vol_squeeze_fast >= 1 and vol_squeeze_slow >= 2.")
        if self.vol_squeeze_fast >= self.vol_squeeze_slow:
            raise ValueError("vol_squeeze_fast must be < vol_squeeze_slow.")

    @property
    def effective_rsi_period(self) -> int:
        return self.rsi_period if self.rsi_period is not None else self.momentum_period


# ──────────────────────────────────────────────────────────────────────────────
# Input validation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _validate_prices(prices: pd.Series) -> None:
    if not isinstance(prices, pd.Series):
        raise TypeError(f"Expected pd.Series, got {type(prices).__name__}.")
    if prices.empty:
        raise ValueError("Price series is empty.")
    non_null = prices.dropna()
    if non_null.empty:
        raise ValueError("Price series contains only NaN values.")
    if (non_null <= 0).any():
        raise ValueError(
            "All prices must be strictly positive; "
            f"found {(non_null <= 0).sum()} non-positive value(s)."
        )
    n_nan = prices.isna().sum()
    if n_nan > 0:
        logger.warning(
            "Price series '%s' has %d NaN value(s). "
            "Features will propagate NaN at those positions.",
            prices.name,
            n_nan,
        )


def _validate_ohlc(ohlc: pd.DataFrame) -> None:
    """Validate an OHLC DataFrame with lowercase column names."""
    required = {"open", "high", "low", "close"}
    missing = required - set(ohlc.columns)
    if missing:
        raise ValueError(f"OHLC DataFrame missing columns: {sorted(missing)}")
    if ohlc.empty:
        raise ValueError("OHLC DataFrame is empty.")

    # ── Bug #7 FIX: integrity checks on OHLC relationships ────────────────

    # 1. All prices must be strictly positive (catches zero-fill and bad adjustments)
    price_cols = ohlc[["open", "high", "low", "close"]]
    non_null_prices = price_cols.dropna()
    if not (non_null_prices > 0).all(axis=None):
        bad_counts = (non_null_prices <= 0).sum()
        raise ValueError(
            f"OHLC contains non-positive prices: {bad_counts[bad_counts > 0].to_dict()}"
        )

    # 2. High must be >= Low on every bar (catches inverted/corrupted bars)
    inverted = ohlc["high"] < ohlc["low"]
    n_inverted = inverted.sum()
    if n_inverted > 0:
        first_bad = ohlc.index[inverted][0]
        raise ValueError(
            f"OHLC has {n_inverted} bar(s) where high < low "
            f"(first occurrence: {first_bad}). "
            "Check for data corruption or bad corporate-action adjustment."
        )

    # 3. Close must be within [low, high] on every bar
    # (use warnings not errors — some vendors report slight violations on adjusted data)
    close_above_high = (ohlc["close"] > ohlc["high"]).sum()
    close_below_low  = (ohlc["close"] < ohlc["low"]).sum()
    if close_above_high > 0:
        logger.warning(
            "_validate_ohlc: %d bar(s) have close > high. "
            "Possible bad price adjustment. feat_icp will be clipped.",
            close_above_high,
        )
    if close_below_low > 0:
        logger.warning(
            "_validate_ohlc: %d bar(s) have close < low. "
            "Possible bad price adjustment. feat_icp will be clipped.",
            close_below_low,
        )

    # 4. Warn (not raise) on NaN counts so partial data is still usable
    n_nan = ohlc[["open", "high", "low", "close"]].isna().sum()
    total_nan = n_nan.sum()
    if total_nan > 0:
        logger.warning(
            "_validate_ohlc: NaN values found — %s. "
            "OHLC features will propagate NaN at those positions.",
            n_nan[n_nan > 0].to_dict(),
        )
    # ────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# Primitive: log returns
# ──────────────────────────────────────────────────────────────────────────────

def log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute log returns: r_t = log(P_t / P_{t-1}).

    Time-additive and approximately symmetric — preferred over simple returns
    for financial time-series modelling.
    """
    # Escape to pure numpy to avoid cudf.pandas np.log() alignment/NaN bugs
    original_index = prices.index
    arr = np.array(prices.values, dtype=np.float64)
    
    log_ret = np.empty_like(arr)
    log_ret[0] = np.nan
    # Use standard numpy division and log
    with np.errstate(divide='ignore', invalid='ignore'):
        log_ret[1:] = np.log(arr[1:] / arr[:-1])
        
    return pd.Series(log_ret, index=original_index, name=prices.name)


def _cumulative_log_return(prices: pd.Series, h: int) -> pd.Series:
    """h-bar cumulative log return via rolling sum (handles NaN gaps)."""
    r = log_returns(prices)
    return r.rolling(window=h, min_periods=h).sum()


# ──────────────────────────────────────────────────────────────────────────────
# Feature 1 — EWMA Volatility  (Eqs. 16–17)
# ──────────────────────────────────────────────────────────────────────────────

def _numpy_ewm_mean(arr: np.ndarray, span: int) -> np.ndarray:
    """Pure numpy EWMA mean — cudf-proof implementation."""
    alpha = 2.0 / (span + 1.0)
    out = np.empty(len(arr), dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        if np.isnan(arr[i]):
            out[i] = np.nan
        else:
            out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _numpy_ewm_mean_skipnan(arr: np.ndarray, span: int) -> np.ndarray:
    """
    EWMA mean that SKIPS NaN inputs (carry-forward) instead of updating.

    At a NaN bar: state is carried forward unchanged — the gap is simply
    not observed. This is the correct behaviour for price-gap robustness.

    Contrast with _numpy_ewm_mean() which PROPAGATES NaN from the first
    NaN onward — that function is still used where NaN-propagation is
    the desired behaviour (e.g. rolling windows that should stay NaN
    during warm-up).
    """
    alpha = 2.0 / (span + 1.0)
    out = np.empty(len(arr), dtype=np.float64)
    out[0] = arr[0]  # NaN at bar 0 is fine — handled by sigma_vals[0]=NaN
    for i in range(1, len(arr)):
        if np.isnan(arr[i]):
            out[i] = out[i - 1]          # ← carry forward, no update
        elif np.isnan(out[i - 1]):
            out[i] = arr[i]              # ← restart from first valid value
        else:
            out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def ewma_volatility(prices: pd.Series, span: int = 63) -> pd.Series:
    """
    Conditional daily volatility σ_t via EWMA (paper Eqs. 16–17).

    α = 2/(span+1).
    σ²_t = α·(r_t − μ_t)² + (1−α)·σ²_{t−1}

    NaN gaps (circuit breakers, corporate actions) are handled by
    carrying forward the last valid EWMA state — never injecting
    phantom zero-returns into the variance estimator.
    """
    _validate_prices(prices)

    # Extract raw numpy array — bypass cudf entirely
    try:
        r_vals = prices.values.get()        # CuPy array → numpy (GPU path)
    except AttributeError:
        r_vals = np.array(prices.values, dtype=np.float64)  # already numpy

    # log returns in numpy
    with np.errstate(divide='ignore', invalid='ignore'):
        log_r = np.log(r_vals[1:] / r_vals[:-1])
    log_r = np.concatenate([[np.nan], log_r])  # restore length

    original_index = prices.index

    # ── FIX: use skip-NaN EWMA — no nan_to_num anywhere ──────────────────────
    ewma_mean   = _numpy_ewm_mean_skipnan(log_r, span)
    # demeaned_sq is NaN wherever log_r is NaN → variance state not updated
    demeaned_sq = np.where(np.isnan(log_r), np.nan, (log_r - ewma_mean) ** 2)
    ewma_var    = _numpy_ewm_mean_skipnan(demeaned_sq, span)
    # ─────────────────────────────────────────────────────────────────────────

    sigma_vals  = np.sqrt(ewma_var)
    sigma_vals[0] = np.nan

    # Rebuild as plain Python list → pandas Series (avoids cudf constructor)
    sigma = pd.Series(
        sigma_vals.tolist(),
        index=original_index,
        name=f"ewma_vol_span{span}",
        dtype="float64",
    )
    return sigma


# ──────────────────────────────────────────────────────────────────────────────
# Feature 2 — Multi-Horizon Normalised Returns  (Eq. 18)
# ──────────────────────────────────────────────────────────────────────────────

def normalized_returns(
    prices: pd.Series,
    horizons: Optional[list[int]] = None,
    span: int = 63,
) -> pd.DataFrame:
    """
    Vol-normalised multi-horizon returns: r_norm(t,h) = r(t,h) / (σ_t · √h).

    Dimensionless, horizon-invariant, concentrated in [-2, 2].
    """
    if horizons is None:
        horizons = FeatureConfig().return_horizons
    _validate_prices(prices)
    sigma = ewma_volatility(prices, span=span)
    out: dict[str, pd.Series] = {}
    for h in horizons:
        cum_ret = _cumulative_log_return(prices, h)
        denom = sigma * np.sqrt(h)
        with np.errstate(invalid="ignore", divide="ignore"):
            norm_ret = cum_ret / denom
        norm_ret = norm_ret.where(denom > _EPS, other=np.nan)
        out[f"ret_norm_{h}d"] = norm_ret
    return pd.DataFrame(out, index=prices.index)


# ──────────────────────────────────────────────────────────────────────────────
# Feature 3 — Multi-Scale MACD Momentum Signal  (Eqs. 19–21)
# ──────────────────────────────────────────────────────────────────────────────

def macd_signal(
    prices: pd.Series,
    short_span: int = 8,
    long_span: int = 24,
    price_std_window: int = 63,
    signal_std_window: int = 252,
) -> pd.Series:
    """
    Three-step normalised MACD: Raw → price-scale → regime normalisation.

    Result is ~unit-variance, concentrated in [-4, 4].
    """
    _validate_prices(prices)

    # ── cudf-safe: extract to numpy, compute EWMA in pure Python loop ────────
    try:
        p_vals = prices.values.get()           # CuPy → numpy (GPU path)
    except AttributeError:
        p_vals = np.array(prices.values, dtype=np.float64)  # CPU path

    ewma_s_arr = _numpy_ewm_mean_skipnan(p_vals, short_span)   # ← fixed
    ewma_l_arr = _numpy_ewm_mean_skipnan(p_vals, long_span)    # ← fixed

    # Reconstruct as pandas Series with original index
    ewma_s = pd.Series(ewma_s_arr.tolist(), index=prices.index, dtype="float64")
    ewma_l = pd.Series(ewma_l_arr.tolist(), index=prices.index, dtype="float64")

    macd_raw = ewma_s - ewma_l

    # ── Step 2 & 3: Normalisation ────────────────────────────────────────────
    mp = price_std_window
    ms = signal_std_window

    price_std = prices.rolling(window=price_std_window, min_periods=mp).std()
    with np.errstate(invalid="ignore", divide="ignore"):
        q = macd_raw / price_std
    q = q.where(price_std > _EPS, other=np.nan)

    q_std = q.rolling(window=signal_std_window, min_periods=ms).std()
    with np.errstate(invalid="ignore", divide="ignore"):
        signal = q / q_std
    signal = signal.where(q_std > _EPS, other=np.nan)
    signal.name = f"macd_{short_span}_{long_span}"
    return signal


def macd_signals_multi(
    prices: pd.Series,
    pairs: Optional[list[tuple[int, int]]] = None,
    price_std_window: int = 63,
    signal_std_window: int = 252,
) -> pd.DataFrame:
    """Compute normalised MACD for multiple (short, long) span pairs."""
    if pairs is None:
        pairs = FeatureConfig().macd_pairs
    signals: dict[str, pd.Series] = {}
    for short_span, long_span in pairs:
        sig = macd_signal(
            prices,
            short_span=short_span,
            long_span=long_span,
            price_std_window=price_std_window,
            signal_std_window=signal_std_window,
        )
        signals[sig.name] = sig
    return pd.DataFrame(signals, index=prices.index)


# ──────────────────────────────────────────────────────────────────────────────
# Feature 4 — Volatility Scaling Factor  (Eq. 22)
# ──────────────────────────────────────────────────────────────────────────────

def volatility_scaling_factor(prices: pd.Series, span: int = 63) -> pd.Series:
    """
    1/σ_t — used for volatility-targeted position sizing (Eq. 22).

    Heavily right-skewed → ROBUST scaling bucket in data_loader.py.
    """
    sigma = ewma_volatility(prices, span=span)
    vs = 1.0 / sigma.where(sigma > _EPS, other=np.nan)
    vs.name = f"vs_factor_span{span}"
    return vs


# ──────────────────────────────────────────────────────────────────────────────
# Feature 5 — Normalised Return Target  (Eq. 23, training only)
# ──────────────────────────────────────────────────────────────────────────────

def normalized_return_target(
    prices: pd.Series,
    span: int = 63,
    clip_value: float = 20.0,
    _allow_in_inference: bool = False,   # escape hatch for deliberate test-time use
) -> pd.Series:
    """
    Clipped vol-normalised next-bar return target (Eq. 23).

    target_t = clip( r_{t+1} / σ_t,  ±clip_value )

    ⚠️  NEVER include during live inference — r_{t+1} is not available.
    Use set_inference_mode(True) in your inference script to enforce this
    as a hard RuntimeError rather than a docstring convention.
    """
    # ── FIX: runtime guard against inference-time misuse ──────────────────
    if _INFERENCE_MODE and not _allow_in_inference:
        raise RuntimeError(
            "normalized_return_target() was called while _INFERENCE_MODE=True. "
            "This function uses r_{t+1} (future data) and must never run during "
            "live inference. If you are in a training context, call "
            "set_inference_mode(False) first, or pass include_target=False to "
            "FeatureEngineer.build()."
        )
    # ──────────────────────────────────────────────────────────────────────

    _validate_prices(prices)
    sigma = ewma_volatility(prices, span=span)
    r_next = log_returns(prices).shift(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        norm_target = r_next / sigma
    norm_target = norm_target.where(sigma > _EPS, other=np.nan)
    norm_target = norm_target.clip(lower=-clip_value, upper=clip_value)
    norm_target.name = "target_norm_ret"
    return norm_target


# ──────────────────────────────────────────────────────────────────────────────
# Feature 6 — Kaufman Efficiency Ratio  (OHLC: uses close)
# ──────────────────────────────────────────────────────────────────────────────

def kaufman_efficiency_ratio(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Kaufman Efficiency Ratio (KER) — directional efficiency of price movement.

    Formula
    -------
        net_move  = |close_t − close_{t−period}|       (directional displacement)
        path_len  = Σ|close_i − close_{i−1}|  over period bars  (total path)
        ker_raw   = net_move / path_len
        ker       = ker_raw · sign(close_t − close_{t−period})

    Interpretation
    --------------
        ±1 → perfectly trending (every bar moves in the same direction)
         0 → perfectly choppy (price returns to start after a random walk)

    Output: [-1, +1] → NO_SCALE bucket.
    Warm-up: period bars.
    """
    net_move = close.diff(period)
    path_len = close.diff().abs().rolling(period, min_periods=period).sum()
    ker_unsigned = net_move.abs() / (path_len + _EPS)
    ker = (ker_unsigned * np.sign(net_move)).clip(-1.0, 1.0)
    ker.name = "feat_efficiency"
    return ker


# ──────────────────────────────────────────────────────────────────────────────
# Feature 7 — Internal Close Position  (OHLC: uses high, low, close)
# ──────────────────────────────────────────────────────────────────────────────

def internal_close_position(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Smoothed Internal Close Position (ICP) — where did close land in the bar range?

    Formula
    -------
        raw_icp  = (close − low) / (high − low + ε)     ∈ [0, 1]
        scaled   = raw_icp * 2 − 1                       ∈ [−1, +1]
        icp      = rolling_mean(scaled, period)

    Interpretation
    --------------
        +1 → close at top of every bar over the window (bullish pressure)
        −1 → close at bottom of every bar (bearish pressure)
         0 → neutral / indecisive

    Output: [-1, +1] → NO_SCALE bucket.
    Warm-up: period bars.
    """
    raw = (close - low) / (high - low + _EPS)
    scaled = (raw * 2.0) - 1.0
    icp = scaled.rolling(period, min_periods=period).mean().clip(-1.0, 1.0)
    icp.name = "feat_icp"
    return icp


# ──────────────────────────────────────────────────────────────────────────────
# Feature 8 — RSI (centered at 0)  (OHLC: uses close)
# ──────────────────────────────────────────────────────────────────────────────

def centered_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder RSI re-centered to [-1, +1].

    Formula
    -------
        delta    = close_t − close_{t−1}
        avg_up   = EWMA(max(delta, 0), α=1/period)
        avg_down = EWMA(max(−delta, 0), α=1/period)
        RS       = avg_up / avg_down
        RSI      = 100 − 100/(1+RS)
        output   = (RSI − 50) / 50             ∈ [−1, +1]

    Interpretation
    --------------
        +1 → RSI=100 (overbought extreme — mean-reversion candidate)
        −1 → RSI=0   (oversold extreme)
         0 → RSI=50  (neutral momentum)

    Output: [-1, +1] → NO_SCALE bucket.
    Warm-up: approx 2*period bars.
    """
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)

    # ── cudf-safe: escape to numpy ────────────────────────────────────────
    try:
        up_vals = up.values.get()
        dn_vals = dn.values.get()
    except AttributeError:
        up_vals = np.array(up.values, dtype=np.float64)
        dn_vals = np.array(dn.values, dtype=np.float64)

    # Wilder uses alpha = 1/period, same as _numpy_ewm_mean with span=(2*period - 1)
    # But alpha=1/period ≠ 2/(span+1) for generic span, so compute directly:
    alpha = 1.0 / float(period)
    wilder_span = int(round(2.0 / alpha - 1))   # = 2*period - 1

    avg_up_arr = _numpy_ewm_mean_skipnan(up_vals, wilder_span)
    avg_dn_arr = _numpy_ewm_mean_skipnan(dn_vals, wilder_span)

    avg_up = pd.Series(avg_up_arr.tolist(), index=close.index, dtype="float64")
    avg_dn = pd.Series(avg_dn_arr.tolist(), index=close.index, dtype="float64")
    # ─────────────────────────────────────────────────────────────────────

    rs = avg_up / (avg_dn + _EPS)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi_centered = ((rsi - 50.0) / 50.0).clip(-1.0, 1.0)
    rsi_centered.name = "feat_momentum_rsi"
    return rsi_centered


# ──────────────────────────────────────────────────────────────────────────────
# Feature 9 — Directional Volatility Asymmetry  (OHLC: uses close)
# ──────────────────────────────────────────────────────────────────────────────

def directional_vol_asymmetry(
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Asymmetry between upside and downside volatility over a rolling window.

    Formula
    -------
        up_vol   = std(max(r_t, 0), window)
        down_vol = std(max(−r_t, 0), window)
        output   = (up_vol − down_vol) / (up_vol + down_vol + ε)

    Interpretation
    --------------
        +1 → upside vol >> downside vol  (unusual; often pre-breakout)
        −1 → downside vol >> upside vol  (crash regime / leverage effect)
         0 → symmetric volatility

    Markets typically show negative asymmetry (leverage effect).
    Output: [-1, +1] → NO_SCALE bucket.
    Warm-up: window bars.
    """
    r = log_returns(close)
    up = r.clip(lower=0.0)
    dn = r.clip(upper=0.0)
    up_vol = up.rolling(window, min_periods=window).std()
    dn_vol = dn.rolling(window, min_periods=window).std()
    asym = (up_vol - dn_vol) / (up_vol + dn_vol + _EPS)
    asym = asym.clip(-1.0, 1.0)
    asym.name = "feat_vol_asymmetry"
    return asym


# ──────────────────────────────────────────────────────────────────────────────
# Feature 10 — Local Structure Position  (OHLC: uses high, low, close)
# ──────────────────────────────────────────────────────────────────────────────

def local_structure_position(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 65,
) -> pd.Series:
    """
    Normalised price position within its recent high-low range (Donchian position).

    Formula
    -------
        roll_high = rolling_max(high, window)
        roll_low  = rolling_min(low, window)
        output    = ((close − roll_low) / (roll_high − roll_low + ε)) * 2 − 1

    Interpretation
    --------------
        +1 → close at top of the N-bar range (strong resistance zone)
        −1 → close at bottom (strong support zone)
         0 → mid-range

    Output: [-1, +1] → NO_SCALE bucket.
    Warm-up: window bars.
    """
    roll_high = high.rolling(window, min_periods=max(5, window // 2)).max()
    roll_low = low.rolling(window, min_periods=max(5, window // 2)).min()
    pos = ((close - roll_low) / (roll_high - roll_low + _EPS)) * 2.0 - 1.0
    pos = pos.clip(-1.0, 1.0)
    pos.name = "feat_local_structure"
    return pos


# ──────────────────────────────────────────────────────────────────────────────
# Feature 11/12 — Session Time-of-Day Encoding  (requires DatetimeIndex)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_hhmm(hhmm: str) -> Tuple[int, int]:
    parts = hhmm.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time string: {hhmm!r} — expected 'HH:MM'.")
    h, m = int(parts[0]), int(parts[1])
    if not (0 <= h <= 23) or not (0 <= m <= 59):
        raise ValueError(f"Invalid time: {hhmm!r}")
    return h, m


def session_cyclic_features(
    index: pd.DatetimeIndex,
    session_open: str = "09:15",
    session_close: str = "15:30",
    tz: str = "Asia/Kolkata",
) -> Tuple[pd.Series, pd.Series]:
    """
    Cyclic sine/cosine encoding of intraday time position.

    Maps each bar's timestamp to its fractional position within the trading
    session, then encodes that fraction as (sin, cos) to avoid discontinuities
    at session boundaries.

    Formula
    -------
        pos   = (minutes_since_open) / session_length_minutes   ∈ [0, 1]
        angle = 2π · pos
        sin_t = sin(angle),   cos_t = cos(angle)

    Why sin/cos instead of raw position
    ------------------------------------
    A raw linear position creates a discontinuity (1→0 jump) the model has to
    learn to ignore. The sin/cos pair encodes it continuously — temporal
    distance between any two points matches their angular distance.

    Interpretation
    --------------
        (sin=0, cos=1)  → session open  (9:15)
        (sin=1, cos=0)  → mid-session   (~12:22)
        (sin=0, cos=−1) → session close (15:30)

    Output: both in [-1, +1] → NO_SCALE bucket.
    Zero warm-up — pure timestamp function.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("session_cyclic_features requires a DatetimeIndex.")

    # ── FIX: handle tz-naive index explicitly ─────────────────────────────
    if index.tz is not None:
        idx = index.tz_convert(tz)
    else:
        logger.warning(
            "session_cyclic_features: DatetimeIndex is tz-naive. "
            "Assuming timestamps are already in %s. "
            "If your data is UTC or another timezone, localize before calling: "
            "index = index.tz_localize('UTC').tz_convert('%s')",
            tz, tz,
        )
        # Treat naive timestamps as already being in the target tz.
        # This is the least-surprising fallback: a tz-naive 09:15 bar
        # is assumed to mean 09:15 local time, not 09:15 UTC.
        idx = index.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    # ─────────────────────────────────────────────────────────────────────

    oh, om = _parse_hhmm(session_open)
    ch, cm = _parse_hhmm(session_close)
    open_min = oh * 60 + om
    close_min = ch * 60 + cm
    session_len = close_min - open_min
    if session_len <= 0:
        raise ValueError("session_close must be after session_open.")

    minutes = pd.Series(
        idx.hour.astype(np.int32) * 60 + idx.minute.astype(np.int32),
        index=index,
    ).clip(lower=open_min, upper=close_min)

    pos = (minutes - open_min) / float(session_len)
    angle = math.pi * pos

    # ── FIX: float64 throughout, no astype(float32) ───────────────────────
    s_sin = pd.Series(np.sin(angle), index=index, name="feat_session_sin", dtype="float64")
    s_cos = pd.Series(np.cos(angle), index=index, name="feat_session_cos", dtype="float64")
    # ─────────────────────────────────────────────────────────────────────
    return s_sin, s_cos


# ──────────────────────────────────────────────────────────────────────────────
# Feature 13 — Vol Squeeze (ATR fast/slow ratio)  (OHLC: uses high, low, close)
# ──────────────────────────────────────────────────────────────────────────────

def vol_squeeze_ratio(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    fast_window: int = 5,
    slow_window: int = 20,
) -> pd.Series:
    """
    Ratio of short-term ATR to medium-term ATR — detects volatility contractions.

    Formula
    -------
        TR_t     = max(H−L, |H−C_{t-1}|, |L−C_{t-1}|)
        ATR_fast = rolling_mean(TR, fast_window)
        ATR_slow = rolling_mean(TR, slow_window)
        output   = ATR_fast / (ATR_slow + ε)

    Interpretation
    --------------
        < 1.0 → current vol below medium-term average (compression / squeeze)
        = 1.0 → neutral
        > 1.0 → expanding volatility (breakout mode)

    Distribution: right-skewed and unbounded above → ROBUST scaling bucket.
    Warm-up: slow_window bars.
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_fast = tr.rolling(fast_window, min_periods=fast_window).mean()
    atr_slow = tr.rolling(slow_window, min_periods=slow_window).mean()
    squeeze = atr_fast / (atr_slow + _EPS)
    squeeze.name = "feat_vol_squeeze"
    return squeeze


# ──────────────────────────────────────────────────────────────────────────────
# Master Feature Builder
# ──────────────────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    End-to-end feature engineering for financial time-series models.

    Orchestrates all 13 feature families into a single aligned DataFrame.

    Usage
    -----
    Close-only (original paper features):
        fe = FeatureEngineer()
        feats = fe.build(close_series, include_target=False, dropna=False)

    Full OHLC features (all 13 families):
        fe = FeatureEngineer()
        feats = fe.build(close_series, ohlc=ohlc_df, include_target=False)

    Notes
    -----
    - ohlc must have lowercase columns: open, high, low, close.
    - ohlc.index must align with prices.index.
    - If ohlc is None, the 8 OHLC-based features are silently skipped
      (fully backward-compatible with existing train.py).
    - feat_session_sin / feat_session_cos require a DatetimeIndex. If the
      index is not a DatetimeIndex, session features are skipped with a warning.
    """

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        self.config = config or FeatureConfig()

    # ──────────────────────────────────────────────────────────────────────────
    # Single-asset build
    # ──────────────────────────────────────────────────────────────────────────

    def build(
        self,
        prices: pd.Series,
        ohlc: Optional[pd.DataFrame] = None,
        include_target: bool = False,
        dropna: bool = False,
    ) -> pd.DataFrame:
        """
        Compute the full feature matrix for one asset.

        Parameters
        ----------
        prices : pd.Series
            Close prices. Index should be a DatetimeIndex for session features.
        ohlc : pd.DataFrame, optional
            DataFrame with columns open/high/low/close (same index as prices).
            If provided, computes the 8 OHLC-based features (6–13).
            If None, only the 4 close-only paper features are computed.
        include_target : bool
            Append the normalised return target column (training only).
        dropna : bool
            Drop NaN warm-up rows. Use True for training, False for inference.

        Returns
        -------
        pd.DataFrame
            Column layout (when ohlc provided):
                Close-only (NO_SCALE):
                    ewma_vol_span{N}
                    ret_norm_1d … ret_norm_252d  (6 cols)
                    macd_8_24, macd_16_48, macd_32_96
                Close-only (ROBUST):
                    vs_factor_span{N}
                OHLC-based (NO_SCALE):
                    feat_efficiency
                    feat_icp
                    feat_momentum_rsi
                    feat_vol_asymmetry
                    feat_local_structure
                    feat_session_sin
                    feat_session_cos
                OHLC-based (ROBUST):
                    feat_vol_squeeze
                Training only:
                    target_norm_ret
        """
        cfg = self.config
        logger.info(
            "Building features for '%s' | %d rows | ohlc=%s.",
            prices.name or "unnamed",
            len(prices),
            ohlc is not None,
        )

        _validate_prices(prices)
        prices = prices.sort_index()

        if ohlc is not None:
            _validate_ohlc(ohlc)
            ohlc = ohlc.reindex(prices.index)

            # ── Bug #8 FIX: check overlap after reindex ──────────────────────────────
            overlap_frac = ohlc.notna().any(axis=1).mean()
            if overlap_frac == 0.0:
                raise ValueError(
                    "OHLC DataFrame has zero overlap with prices.index after reindex. "
                    "This is almost certainly a timezone or frequency mismatch. "
                    f"OHLC index sample: {ohlc.index[:3].tolist()}, "
                    f"Prices index sample: {prices.index[:3].tolist()}. "
                    "Fix: align timezones before calling build() — e.g., "
                    "ohlc.index = ohlc.index.tz_convert('Asia/Kolkata')."
                )
            elif overlap_frac < 0.5:
                logger.warning(
                    "build(): OHLC overlap with prices.index is only %.1f%% after reindex. "
                    "%.1f%% of OHLC bars are NaN — likely a timezone or frequency mismatch. "
                    "OHLC index sample: %s | Prices index sample: %s",
                    overlap_frac * 100,
                    (1 - overlap_frac) * 100,
                    ohlc.index[:3].tolist(),
                    prices.index[:3].tolist(),
                )
            elif overlap_frac < 0.95:
                logger.warning(
                    "build(): OHLC overlap with prices.index is %.1f%% — "
                    "%d NaN bars will propagate into OHLC features.",
                    overlap_frac * 100,
                    int((1 - overlap_frac) * len(ohlc)),
                )
            # ─────────────────────────────────────────────────────────────────────────

        parts: list[pd.DataFrame | pd.Series] = []

        # ── 1. EWMA Volatility ────────────────────────────────────────────────
        vol = ewma_volatility(prices, span=cfg.ewma_span)
        parts.append(vol)

        # ── 2. Multi-Horizon Normalised Returns ───────────────────────────────
        ret_feats = normalized_returns(
            prices,
            horizons=cfg.return_horizons,
            span=cfg.ewma_span,
        )
        parts.append(ret_feats)

        # ── 3. Multi-Scale MACD Momentum ──────────────────────────────────────
        macd_feats = macd_signals_multi(
            prices,
            pairs=cfg.macd_pairs,
            price_std_window=cfg.macd_price_std_window,
            signal_std_window=cfg.macd_signal_std_window,
        )
        parts.append(macd_feats)

        # ── 4. Volatility Scaling Factor (ROBUST) ─────────────────────────────
        vs = volatility_scaling_factor(prices, span=cfg.ewma_span)
        parts.append(vs)

        # ── OHLC-based features (skipped if ohlc=None) ────────────────────────
        if ohlc is not None:
            h = ohlc["high"]
            l = ohlc["low"]
            c = ohlc["close"]

            # 6. Kaufman Efficiency Ratio
            parts.append(kaufman_efficiency_ratio(c, period=cfg.momentum_period))

            # 7. Internal Close Position
            parts.append(internal_close_position(h, l, c, period=cfg.icp_period))

            # 8. Centered RSI
            parts.append(centered_rsi(c, period=cfg.effective_rsi_period))

            # 9. Directional Volatility Asymmetry
            parts.append(directional_vol_asymmetry(c, window=cfg.vol_asym_window))

            # 10. Local Structure Position
            parts.append(local_structure_position(h, l, c, window=cfg.local_structure_bars))

            # 11+12. Session Time-of-Day (sin + cos)
            if cfg.add_session_features:
                if isinstance(prices.index, pd.DatetimeIndex):
                    s_sin, s_cos = session_cyclic_features(
                        index=prices.index,
                        session_open=cfg.session_open,
                        session_close=cfg.session_close,
                        tz=cfg.session_tz,
                    )
                    parts.append(s_sin)
                    parts.append(s_cos)
                else:
                    logger.warning(
                        "Session features skipped — index is not a DatetimeIndex "
                        "(got %s).", type(prices.index).__name__
                    )

            # 13. Vol Squeeze Ratio (ROBUST)
            parts.append(vol_squeeze_ratio(
                h, l, c,
                fast_window=cfg.vol_squeeze_fast,
                slow_window=cfg.vol_squeeze_slow,
            ))

        # ── 5. Target (training only) ─────────────────────────────────────────
        if include_target:
            # ── FIX: block include_target=True during inference ───────────────
            if _INFERENCE_MODE:
                raise RuntimeError(
                    "FeatureEngineer.build() called with include_target=True while "
                    "_INFERENCE_MODE=True. The target column uses r_{t+1} (look-ahead). "
                    "Pass include_target=False for all inference builds, or call "
                    "set_inference_mode(False) if you are intentionally in a training context."
                )
            # ──────────────────────────────────────────────────────────────────
            target = normalized_return_target(
                prices,
                span=cfg.ewma_span,
                clip_value=cfg.target_clip,
            )
            parts.append(target)

        result = pd.concat(parts, axis=1)

        # Sanity-check uniform dtypes after concat
        assert (result.dtypes == "float64").all(), \
            f"Mixed dtypes after concat: {result.dtypes[result.dtypes != 'float64']}"

        if dropna:
            n_before = len(result)
            result = result.dropna()
            n_dropped = n_before - len(result)
            logger.info(
                "Warm-up rows dropped: %d / %d  (%.1f%%)",
                n_dropped,
                n_before,
                100 * n_dropped / n_before,
            )

        logger.info(
            "Feature matrix built: shape=%s | NaN count=%d",
            result.shape,
            result.isna().sum().sum(),
        )
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Multi-asset
    # ──────────────────────────────────────────────────────────────────────────

    def build_multi_asset(
        self,
        price_df: pd.DataFrame,
        ohlc_dict: Optional[dict[str, pd.DataFrame]] = None,
        include_target: bool = False,
        dropna: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Build features for every asset (column) in a price panel.

        Parameters
        ----------
        price_df : pd.DataFrame
            Columns = ticker symbols, values = close prices.
        ohlc_dict : dict[str, pd.DataFrame], optional
            {ticker: ohlc_df} for OHLC-based features per asset.
        """
        result: dict[str, pd.DataFrame] = {}
        for ticker in price_df.columns:
            series = price_df[ticker].dropna()
            series.name = ticker
            ohlc = ohlc_dict.get(ticker) if ohlc_dict else None
            try:
                result[ticker] = self.build(
                    series,
                    ohlc=ohlc,
                    include_target=include_target,
                    dropna=dropna,
                )
            except Exception as exc:
                logger.warning("Feature build FAILED for '%s': %s", ticker, exc)
        return result

    def stack_for_model(
        self,
        feature_dict: dict[str, pd.DataFrame],
        lookback: int = 63,
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
        """
        Stack per-asset feature DataFrames into aligned 3-D tensors.

        Returns
        -------
        X : np.ndarray  shape (T, K, d)
        y : np.ndarray  shape (T, K)  or empty if no target column
        dates : list[str]
        tickers : list[str]
        """
        tickers = list(feature_dict.keys())
        has_target = "target_norm_ret" in next(iter(feature_dict.values())).columns

        # ── Step 1: find common date index ───────────────────────────────────────
        common_idx = feature_dict[tickers[0]].index
        for df in feature_dict.values():
            common_idx = common_idx.intersection(df.index)
        common_idx = common_idx[lookback:]

        # ── FIX: compute intersection of ALL tickers' columns ────────────────────
        # Guarantees (a) no KeyError, (b) no silent column drop from one side,
        # (c) enforces a canonical sorted order so stacking is always aligned.

        all_col_sets = [set(df.columns) for df in feature_dict.values()]
        common_col_set = set.intersection(*all_col_sets)

        # Warn about any ticker-specific columns that were dropped
        for ticker, df in feature_dict.items():
            dropped = set(df.columns) - common_col_set
            if dropped:
                logger.warning(
                    "stack_for_model: ticker '%s' has extra columns not present "
                    "in all tickers — dropping from tensor: %s",
                    ticker,
                    sorted(dropped),
                )

        # Exclude target, then sort for canonical cross-ticker order
        feature_cols = sorted(
            col for col in common_col_set
            if col != "target_norm_ret"
        )

        if not feature_cols:
            raise ValueError(
                "stack_for_model: no feature columns common to all tickers. "
                "Check that all assets were built with the same FeatureConfig."
            )

        logger.info(
            "stack_for_model: %d common feature columns across %d tickers.",
            len(feature_cols), len(tickers),
        )
        # ─────────────────────────────────────────────────────────────────────────

        X_list, y_list = [], []
        for ticker in tickers:
            df = feature_dict[ticker].loc[common_idx]
            X_list.append(df[feature_cols].values)
            if has_target:
                y_list.append(df["target_norm_ret"].values)

        X = np.stack(X_list, axis=1).astype(np.float32)
        y = np.stack(y_list, axis=1).astype(np.float32) if has_target else np.array([])
        return X, y, [str(d) for d in common_idx], tickers


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test / demo  (python features.py)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    np.random.seed(42)
    N = 10000

    def _gbm(n: int, mu: float = 0.0002, sigma: float = 0.015, s0: float = 100.0) -> pd.DataFrame:
        start = pd.Timestamp("2020-01-02 09:15:00", tz="Asia/Kolkata")
        idx = pd.date_range(start, periods=n * 5, freq="30min", tz="Asia/Kolkata")
        idx = idx[(idx.hour * 60 + idx.minute >= 9 * 60 + 15) &
                  (idx.hour * 60 + idx.minute <= 15 * 60 + 30)][:n]
        log_r = np.random.normal(mu, sigma, len(idx))
        closes = s0 * np.exp(np.cumsum(log_r))
        highs = closes * (1 + np.abs(np.random.normal(0, 0.003, len(idx))))
        lows  = closes * (1 - np.abs(np.random.normal(0, 0.003, len(idx))))
        opens = np.roll(closes, 1); opens[0] = s0
        return pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes},
            index=idx,
        )

    raw   = _gbm(N)
    close = raw["close"].rename("NIFTY")

    cfg = FeatureConfig()

    fe    = FeatureEngineer(config=cfg)
    feats = fe.build(close, ohlc=raw, include_target=True, dropna=True)

    print("\n" + "=" * 65)
    print("  FEATURE MATRIX (close-only + OHLC, dropna=True)")
    print("=" * 65)
    print(f"  Shape : {feats.shape}")
    print(f"  Columns ({len(feats.columns)}):")
    for col in feats.columns:
        s = feats[col]
        print(f"    {col:<30}  mean={s.mean():+.4f}  std={s.std():.4f}  "
              f"min={s.min():+.4f}  max={s.max():+.4f}")
    print("\n  NaN remaining:", feats.isna().sum().sum())