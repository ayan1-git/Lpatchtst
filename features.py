"""
features.py
===========
Production-grade feature engineering for financial time-series models.

Implements the three feature families from:
  "Deep Learning for Financial Time Series" (VLSTM / PsLSTM benchmark paper).

Features
--------
1.  EWMA Volatility  σ_t           — Eqs. 16–17
2.  Multi-Horizon Normalized Returns r_norm(t,h) — Eq. 18
3.  Multi-Scale MACD Momentum Signals            — Eqs. 19–21  (3-step normalization)
4.  Volatility Scaling Factor  1/σ_t             — Eq. 22
5.  Normalized Return Target  (training only)    — Eq. 23

All derived from daily close prices only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

_EPS = 1e-10  # guard against zero-division throughout


# ──────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FeatureConfig:
    """
    Central configuration for all feature engineering hyperparameters.

    Attributes
    ----------
    ewma_span : int
        Span for the EWMA volatility estimator.
        λ = 2 / (span + 1).  span=63 ≈ 3 calendar months.
        Larger span → slower adaptation to new volatility regimes.

    return_horizons : list[int]
        Lookback windows (trading days) for multi-horizon return features.
        Default maps to: 1D, 1W, 1M, 3M, 6M, 1Y.

    macd_pairs : list[tuple[int, int]]
        (short_span, long_span) pairs for multi-scale MACD signals.
        Inspired by CTA trend-following convention (8/24, 16/48, 32/96).

    macd_price_std_window : int
        Rolling window for Step-2 price-scale normalisation (paper: 63 days).

    macd_signal_std_window : int
        Rolling window for Step-3 regime normalisation (paper: 252 days).

    target_clip : float
        Symmetric clip bound for the normalised return target (paper: ±20).
        Suppresses Black-Swan events that would dominate gradient updates.
    """

    ewma_span: int = 63

    return_horizons: list[int] = field(
        default_factory=lambda: [1, 5, 21, 63, 126, 252]
    )

    macd_pairs: list[tuple[int, int]] = field(
        default_factory=lambda: [(8, 24), (16, 48), (32, 96)]
    )

    macd_price_std_window: int = 63
    macd_signal_std_window: int = 252

    target_clip: float = 20.0

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


# ──────────────────────────────────────────────────────────────────────────────
# Input validation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _validate_prices(prices: pd.Series) -> None:
    """
    Strict price validation.

    Checks
    ------
    - Must be a pd.Series.
    - Must not be empty.
    - All non-NaN values must be strictly positive.
    - Warns if the series contains any NaN values.
    """
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


# ──────────────────────────────────────────────────────────────────────────────
# Primitive: log returns
# ──────────────────────────────────────────────────────────────────────────────

def log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute daily log returns: r_t = log(P_t / P_{t-1}).

    Preferred over simple returns because:
    - Time-additive: r(t,h) = Σ r_i over h days.
    - Symmetric: log-return for +10% ≈ −(−10%) log-return.
    - Better statistical properties (closer to normality for short horizons).

    Returns
    -------
    pd.Series
        Daily log returns. First element is NaN (no prior price).
    """
    return np.log(prices / prices.shift(1))


def _cumulative_log_return(prices: pd.Series, h: int) -> pd.Series:
    """
    h-day cumulative log return: log(P_t / P_{t-h}).

    Uses rolling sum of daily log returns rather than direct price ratio
    to correctly handle gaps / NaNs within the window.
    """
    r = log_returns(prices)
    return r.rolling(window=h, min_periods=max(1, h // 2)).sum()


# ──────────────────────────────────────────────────────────────────────────────
# Feature 1 — EWMA Volatility
# ──────────────────────────────────────────────────────────────────────────────

def ewma_volatility(
    prices: pd.Series,
    span: int = 63,
) -> pd.Series:
    """
    Estimate daily conditional volatility σ_t via EWMA (paper Eqs. 16–17).

    Algorithm
    ---------
    Let α = 2 / (span + 1)  (decay factor, called λ in the paper).

    Step 1 — EWMA mean (tracks the slowly-drifting return mean):
        μ_t = α·r_t + (1-α)·μ_{t-1}                     [Eq. 16]

    Step 2 — EWMA variance (tracks time-varying volatility):
        σ²_t = α·(r_t - μ_t)² + (1-α)·σ²_{t-1}          [Eq. 17]

    Step 3 — Standard deviation:
        σ_t = √(σ²_t)

    Implementation notes
    --------------------
    - `adjust=False` enforces the exact recursive formula (not the
      "expanding window" form that Pandas uses by default).
    - This gives different (smaller) results than
      `prices.pct_change().ewm(span).std()`, which applies an
      unbiased correction factor. The paper uses the biased version.
    - σ_t is in **daily units**. Multiply by √252 to annualise.

    Parameters
    ----------
    prices : pd.Series
        Strictly positive daily close prices.
    span : int
        EWMA span. α = 2/(span+1). Larger → slower adaptation.

    Returns
    -------
    pd.Series  (name=``ewma_vol_span{span}``)
        Daily conditional volatility, daily units.
        NaN at index 0 (no prior return available).
    """
    _validate_prices(prices)
    r = log_returns(prices)

    ewma_mean = r.ewm(span=span, adjust=False).mean()
    demeaned_sq = (r - ewma_mean) ** 2
    ewma_var = demeaned_sq.ewm(span=span, adjust=False).mean()

    sigma = np.sqrt(ewma_var)
    sigma.name = f"ewma_vol_span{span}"
    return sigma


# ──────────────────────────────────────────────────────────────────────────────
# Feature 2 — Multi-Horizon Normalised Returns
# ──────────────────────────────────────────────────────────────────────────────

def normalized_returns(
    prices: pd.Series,
    horizons: Optional[list[int]] = None,
    span: int = 63,
) -> pd.DataFrame:
    """
    Multi-horizon volatility-normalised returns (paper Eq. 18).

    Formula
    -------
        r_norm(t, h) = r(t, h) / (σ_t · √h)

    where:
        r(t, h) = log(P_t / P_{t-h})   — h-day cumulative log return
        σ_t     = EWMA daily volatility at time t   (known at prediction time)
        √h      = time-scaling correction

    Rationale for each term
    -----------------------
    σ_t in the denominator:
        Removes the current volatility regime.
        A 2% move in a quiet market is very different from a 2% move during
        a crash — dividing by σ_t makes them comparable ("how many σ's").

    √h in the denominator:
        Under a random walk, variance grows linearly with h, so σ grows as √h.
        Dividing by √h makes the expected magnitude of the signal horizon-invariant.
        Without it, a 1-year normalised return would be ~√252 ≈ 16× smaller
        than a 1-day normalised return.

    Result: a dimensionless signal concentrated around [-2, 2] regardless
    of the asset, the date, or the horizon. This is ideal for neural network
    input (bounded, roughly symmetric, no re-scaling required).

    Parameters
    ----------
    prices : pd.Series
        Daily close prices.
    horizons : list[int], optional
        Lookback windows in trading days.
        Default [1, 5, 21, 63, 126, 252] = 1D, 1W, 1M, 3M, 6M, 1Y.
    span : int
        EWMA span for σ_t. Must match the span used elsewhere in the pipeline.

    Returns
    -------
    pd.DataFrame
        Columns: ``ret_norm_{h}d`` for each horizon h.
        NaN rows: first `h` rows per horizon + first `span` rows
        while volatility warms up.
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
# Feature 3 — Multi-Scale MACD Momentum Signal
# ──────────────────────────────────────────────────────────────────────────────

def macd_signal(
    prices: pd.Series,
    short_span: int = 8,
    long_span: int = 24,
    price_std_window: int = 63,
    signal_std_window: int = 252,
) -> pd.Series:
    """
    Three-step normalised MACD momentum signal (paper Eqs. 19–21).

    MACD is the classic trend-following signal: "is the recent price above
    or below its longer-term trend?"  The three normalisation steps make it
    regime-stable and cross-asset comparable.

    ┌─────────────────────────────────────────────────────────────────────┐
    │ Step 1 — Raw MACD  (Eq. 19)                                         │
    │                                                                     │
    │   MACD_t = EWMA(P, short_span)_t − EWMA(P, long_span)_t           │
    │                                                                     │
    │   • Positive  → short EMA > long EMA → price trending upward.     │
    │   • Problem: value is in price units (dollars, bps, …).            │
    │   • Incomparable across assets (gold price ≠ T-bill price).        │
    └─────────────────────────────────────────────────────────────────────┘
    ┌─────────────────────────────────────────────────────────────────────┐
    │ Step 2 — Price-Scale Normalisation  (Eq. 20)                        │
    │                                                                     │
    │   q_t = MACD_t / RollingStd(P, price_std_window)_t                │
    │                                                                     │
    │   • Divides by the rolling standard deviation of the *price level* │
    │     (not the return).  Window = 63 days (≈ 1 quarter).            │
    │   • Now dimensionless. A q=1 means the same "magnitude of trend"  │
    │     for gold, bonds, and equity futures alike.                     │
    │   • Problem: q_t distribution still shifts across market regimes.  │
    └─────────────────────────────────────────────────────────────────────┘
    ┌─────────────────────────────────────────────────────────────────────┐
    │ Step 3 — Regime Normalisation  (Eq. 21)                             │
    │                                                                     │
    │   Signal_t = q_t / RollingStd(q, signal_std_window)_t             │
    │                                                                     │
    │   • Divides by the rolling standard deviation of q over ~1 year.   │
    │   • Stabilises the signal distribution across low- and high-vol    │
    │     regimes.  Result is ~unit-variance, concentrated in [-2, 2].  │
    │   • The paper reports nearly all values fall within [-4, 4].       │
    └─────────────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    prices : pd.Series
        Daily close prices (strictly positive).
    short_span, long_span : int
        EWMA spans for the two exponential moving averages.
    price_std_window : int
        Rolling window for Step-2 price std normalisation (paper: 63 days).
    signal_std_window : int
        Rolling window for Step-3 regime normalisation (paper: 252 days).

    Returns
    -------
    pd.Series  (name=``macd_{short_span}_{long_span}``)
        Normalised MACD signal. Approximately symmetric around 0.
    """
    _validate_prices(prices)

    mp = max(1, price_std_window // 2)
    ms = max(1, signal_std_window // 2)

    # Step 1: Raw MACD (price units)
    ewma_s = prices.ewm(span=short_span, adjust=False).mean()
    ewma_l = prices.ewm(span=long_span,  adjust=False).mean()
    macd_raw = ewma_s - ewma_l

    # Step 2: Remove price-level scale
    price_std = prices.rolling(window=price_std_window, min_periods=mp).std()
    with np.errstate(invalid="ignore", divide="ignore"):
        q = macd_raw / price_std
    q = q.where(price_std > _EPS, other=np.nan)

    # Step 3: Remove regime-level distribution drift
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
    """
    Compute normalised MACD for multiple (short, long) span pairs.

    Using multiple scale pairs captures trend signals at different
    time horizons simultaneously — short spans react quickly (noise-prone),
    long spans are stable but slow. The model learns to weight them.

    Parameters
    ----------
    prices : pd.Series
        Daily close prices.
    pairs : list of (short_span, long_span), optional
        Span pairs. Defaults to [(8,24), (16,48), (32,96)].
    price_std_window, signal_std_window : int
        Shared normalisation windows across all pairs.

    Returns
    -------
    pd.DataFrame
        Columns: ``macd_{short}_{long}`` for each pair.
    """
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
# Feature 4 — Volatility Scaling Factor
# ──────────────────────────────────────────────────────────────────────────────

def volatility_scaling_factor(
    prices: pd.Series,
    span: int = 63,
) -> pd.Series:
    """
    Compute the volatility scaling factor used in portfolio construction (Eq. 22).

    Formula
    -------
        vs_factor_t = 1 / σ_t

    Portfolio weight:
        w_t = ŷ_t · (σ_tgt / σ_t) = ŷ_t · σ_tgt · vs_factor_t

    where ŷ_t ∈ [-1, 1] is the neural network's directional signal and
    σ_tgt is the target portfolio volatility (paper: 10% annualised).

    Why this matters
    ----------------
    Different assets have vastly different baseline volatilities
    (e.g., S&P 500 futures vs. 2-year Treasury futures).  Without scaling,
    a fixed position in each asset would lead to equity futures dominating
    total portfolio risk.  Multiplying by 1/σ_t dynamically equalises
    risk contributions across the cross-sectional universe.

    Distribution nuance
    -------------------
    vs_factor_t is heavily right-skewed.  During calm regimes, σ_t → 0
    causes vs_factor_t → ∞.  This is the key reason linear models
    fail here — the relationship between leverage and volatility is
    threshold-like and asymmetric, not linear.

    Parameters
    ----------
    prices : pd.Series
        Daily close prices.
    span : int
        EWMA span for σ_t.

    Returns
    -------
    pd.Series  (name=``vs_factor_span{span}``)
        1/σ_t. NaN where σ_t is undefined or zero.
    """
    sigma = ewma_volatility(prices, span=span)
    vs = 1.0 / sigma.where(sigma > _EPS, other=np.nan)
    vs.name = f"vs_factor_span{span}"
    return vs


# ──────────────────────────────────────────────────────────────────────────────
# Feature 5 — Normalised Return Target  (training only)
# ──────────────────────────────────────────────────────────────────────────────

def normalized_return_target(
    prices: pd.Series,
    span: int = 63,
    clip_value: float = 20.0,
) -> pd.Series:
    """
    Compute the clipped volatility-normalised next-day return target (Eq. 23).

    Formula
    -------
        target_t = clip( r_{t+1} / σ_t,  -clip_value,  +clip_value )

    where:
        r_{t+1}    = log(P_{t+1} / P_t)  — next-day log return
        σ_t        = EWMA volatility known at time t  (no lookahead)
        clip_value = 20  (paper default)

    Alignment note
    --------------
    target_t is labelled at index t (the *feature* time), not t+1
    (the *return* time).  This means row t contains both the input
    features and the label for the same date, enabling simple
    supervised learning without offset bookkeeping.

    ⚠️  NEVER include this column during live inference —
    next-day returns are not available at prediction time.

    Why clip at ±20?
    ----------------
    Financial returns are heavy-tailed.  A single crash day (e.g.,
    Black Monday 1987: −22% S&P 500) would produce a normalised return
    of ~ −22% / (daily σ of ~1%) ≈ −22, dominating the loss for that
    entire mini-batch.  Clipping at ±20 removes the most extreme ~0.1%
    of observations while retaining the bulk of directional signal.

    Parameters
    ----------
    prices : pd.Series
        Daily close prices.
    span : int
        EWMA span. Must match the span used for input features.
    clip_value : float
        Symmetric clip bound.

    Returns
    -------
    pd.Series  (name=``target_norm_ret``)
        Clipped normalised next-day return, indexed at time t.
        Last row is NaN (no t+1 available).
    """
    _validate_prices(prices)
    sigma = ewma_volatility(prices, span=span)
    r_next = log_returns(prices).shift(-1)   # r_{t+1}, forward-shifted

    with np.errstate(invalid="ignore", divide="ignore"):
        norm_target = r_next / sigma
    norm_target = norm_target.where(sigma > _EPS, other=np.nan)
    norm_target = norm_target.clip(lower=-clip_value, upper=clip_value)
    norm_target.name = "target_norm_ret"
    return norm_target


# ──────────────────────────────────────────────────────────────────────────────
# Master Feature Builder
# ──────────────────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    End-to-end feature engineering for daily close price data.

    Orchestrates all five feature families into a single aligned DataFrame.
    Supports single-asset and multi-asset workflows.

    Parameters
    ----------
    config : FeatureConfig, optional
        Hyperparameter configuration. Uses paper defaults if not supplied.

    Examples
    --------
    >>> fe = FeatureEngineer()
    >>> features = fe.build(close_price_series, include_target=True, dropna=True)

    >>> price_panel = pd.DataFrame({"ES": ..., "ZN": ..., "GC": ...})
    >>> all_features = fe.build_multi_asset(price_panel, include_target=True)
    """

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        self.config = config or FeatureConfig()

    # ------------------------------------------------------------------
    # Single-asset
    # ------------------------------------------------------------------

    def build(
        self,
        prices: pd.Series,
        include_target: bool = False,
        dropna: bool = False,
    ) -> pd.DataFrame:
        """
        Compute the full feature matrix for one asset.

        Parameters
        ----------
        prices : pd.Series
            Daily close prices. Index should be a DatetimeIndex.
        include_target : bool
            Append the normalised return target column.
            Set to False during live inference.
        dropna : bool
            Drop rows containing any NaN (removes the warm-up period).
            Recommended for training; not recommended for inference
            (would silently drop the most recent row on the live edge).

        Returns
        -------
        pd.DataFrame
            Column layout (left to right):
                ewma_vol_span{N}          — EWMA daily volatility
                ret_norm_1d  … ret_norm_252d  — 6 horizon-normalised returns
                macd_8_24 / macd_16_48 / macd_32_96  — MACD momentum signals
                vs_factor_span{N}         — volatility scaling factor
                target_norm_ret           — normalised return target (if requested)
        """
        cfg = self.config
        logger.info(
            "Building features for '%s' | %d rows.",
            prices.name or "unnamed",
            len(prices),
        )

        _validate_prices(prices)
        prices = prices.sort_index()

        parts: list[pd.DataFrame | pd.Series] = []

        # ── 1. EWMA Volatility ────────────────────────────────────────
        vol = ewma_volatility(prices, span=cfg.ewma_span)
        parts.append(vol)

        # ── 2. Multi-Horizon Normalised Returns ───────────────────────
        ret_feats = normalized_returns(
            prices,
            horizons=cfg.return_horizons,
            span=cfg.ewma_span,
        )
        parts.append(ret_feats)

        # ── 3. Multi-Scale MACD Momentum ──────────────────────────────
        macd_feats = macd_signals_multi(
            prices,
            pairs=cfg.macd_pairs,
            price_std_window=cfg.macd_price_std_window,
            signal_std_window=cfg.macd_signal_std_window,
        )
        parts.append(macd_feats)

        # ── 4. Volatility Scaling Factor ──────────────────────────────
        vs = volatility_scaling_factor(prices, span=cfg.ewma_span)
        parts.append(vs)

        # ── 5. Target (training only) ─────────────────────────────────
        if include_target:
            target = normalized_return_target(
                prices,
                span=cfg.ewma_span,
                clip_value=cfg.target_clip,
            )
            parts.append(target)

        result = pd.concat(parts, axis=1)

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

    # ------------------------------------------------------------------
    # Multi-asset
    # ------------------------------------------------------------------

    def build_multi_asset(
        self,
        price_df: pd.DataFrame,
        include_target: bool = False,
        dropna: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Build features for every asset (column) in a price panel.

        Parameters
        ----------
        price_df : pd.DataFrame
            Columns = ticker symbols, rows = dates, values = close prices.
        include_target : bool
            Include normalised return target per asset.
        dropna : bool
            Drop NaN rows per asset (independent per column).

        Returns
        -------
        dict[str, pd.DataFrame]
            {ticker: feature_DataFrame}
            Failed tickers are logged and omitted from the output.
        """
        result: dict[str, pd.DataFrame] = {}
        for ticker in price_df.columns:
            series = price_df[ticker].dropna()
            series.name = ticker
            try:
                result[ticker] = self.build(
                    series,
                    include_target=include_target,
                    dropna=dropna,
                )
            except Exception as exc:
                logger.warning(
                    "Feature build FAILED for '%s': %s",
                    ticker,
                    exc,
                )
        return result

    def stack_for_model(
        self,
        feature_dict: dict[str, pd.DataFrame],
        lookback: int = 63,
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
        """
        Stack per-asset feature DataFrames into aligned 3-D tensors for model input.

        Finds the common date range across all assets, then creates:
            X  —  shape (T, K, d)  — T timesteps, K assets, d features
            y  —  shape (T, K)     — T timesteps, K assets   (if target present)

        Parameters
        ----------
        feature_dict : dict[str, pd.DataFrame]
            Output of ``build_multi_asset``.
        lookback : int
            Minimum warm-up rows to skip (ensures rolling features are stable).

        Returns
        -------
        X : np.ndarray   shape (T, K, d)
        y : np.ndarray   shape (T, K)  or empty array if no target column
        dates : list[str]
            Date strings for the T time-steps.
        tickers : list[str]
            Asset names for the K positions.
        """
        tickers = list(feature_dict.keys())
        has_target = "target_norm_ret" in next(iter(feature_dict.values())).columns

        # Identify common date index
        common_idx = feature_dict[tickers[0]].index
        for df in feature_dict.values():
            common_idx = common_idx.intersection(df.index)
        common_idx = common_idx[lookback:]

        feature_cols = [
            c for c in next(iter(feature_dict.values())).columns
            if c != "target_norm_ret"
        ]

        X_list, y_list = [], []
        for ticker in tickers:
            df = feature_dict[ticker].loc[common_idx]
            X_list.append(df[feature_cols].values)
            if has_target:
                y_list.append(df["target_norm_ret"].values)

        X = np.stack(X_list, axis=1).astype(np.float32)  # (T, K, d)
        y = np.stack(y_list, axis=1).astype(np.float32) if has_target else np.array([])

        return X, y, [str(d) for d in common_idx], tickers


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke-test / demo  (run: python features.py)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    np.random.seed(42)
    N = 800   # ~3 years of trading days

    # Simulate two synthetic price series with GBM
    def _gbm(n: int, mu: float = 0.0002, sigma: float = 0.015, s0: float = 100.0) -> pd.Series:
        dates = pd.date_range("2019-01-02", periods=n, freq="B")
        log_r  = np.random.normal(mu, sigma, n)
        prices = s0 * np.exp(np.cumsum(log_r))
        return pd.Series(prices, index=dates, name="synthetic")

    prices_a = _gbm(N); prices_a.name = "AssetA"
    prices_b = _gbm(N, sigma=0.008); prices_b.name = "AssetB"

    cfg = FeatureConfig()
    fe  = FeatureEngineer(config=cfg)

    # Single asset
    features = fe.build(prices_a, include_target=True, dropna=True)
    print("\n" + "="*60)
    print("  SINGLE-ASSET FEATURE MATRIX")
    print("="*60)
    print(f"  Shape : {features.shape}")
    print(f"  Columns ({len(features.columns)}):")
    for col in features.columns:
        print(f"    {col:<30}  mean={features[col].mean():+.4f}  std={features[col].std():.4f}")

    # Multi-asset
    panel  = pd.DataFrame({"AssetA": prices_a, "AssetB": prices_b})
    all_f  = fe.build_multi_asset(panel, include_target=True, dropna=True)
    X, y, dates, tickers = fe.stack_for_model(all_f, lookback=cfg.ewma_span)
    print("\n" + "="*60)
    print("  MULTI-ASSET STACKED TENSORS")
    print("="*60)
    print(f"  X shape : {X.shape}   (timesteps × assets × features)")
    print(f"  y shape : {y.shape}   (timesteps × assets)")
    print(f"  Tickers : {tickers}")
    print(f"  Date range: {dates[0]}  →  {dates[-1]}")
    print("\n  NaN in X:", np.isnan(X).sum())
    print("  NaN in y:", np.isnan(y).sum())
    print()