"""
features.py
===========
Minimal vol-normalized feature engineering for financial time-series models.

Feature Families (close-only, 9 model inputs)
-----------------------------------------------
1.  Vol-Normalized Returns    ret_norm_{1,5,20}            r(t,h) / (σ60·√h)
2.  Vol Regime Ratios         vol_ratio_{10,20}            log(σ_s / σ60), s ∈ {10, 20}
3.  Absolute Vol Level        log_ewma_vol_60              log(σ60)
4.  Vol-Normalized Momentum   mom_norm_{3,10,40}           r(t,h) / (σ60·√h), staggered horizons

Training-only Target
--------------------
5.  Normalized Return Target  target_norm_ret              clip(r_{t+1}/σ60, ±clip)

Normalization σ: span-60 EWMA volatility (`vol_norm_span`) — the most stable
span — used consistently as the denominator in ret_norm, mom_norm, and the
target.

Scale audit note: a raw 1-bar log return (std ≈ 0.003 on NIFTY 30-min) is
~350x smaller than the unit-variance normalized features and was dropped —
its information content equals ret_norm_1 = r/σ. The fast-vol features are
expressed as log-ratios against σ60 so they hover near 0 instead of sitting
at log(σ) ≈ −6.5; only log_ewma_vol_60 keeps the absolute level. All 9
features live in a comparable magnitude band → NO_SCALE bucket in
data_loader.py.

Column count: 9 model inputs (+ 1 training-only target).
Previous 20-feature pipeline (MACD, OHLC indicators, session encodings,
TA-Lib) is archived at core/archive/features_archive_v1.py.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EPS = 1e-10  # guard against zero-division throughout


# ──────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FeatureConfig:
    """
    Central configuration for the v2 feature set.

    Attributes
    ----------
    vol_ratio_spans : list[int]
        Spans whose volatility is expressed as a log-ratio against the
        normalization span, i.e. log(σ_s / σ_norm). Default [10, 20].

    vol_norm_span : int
        Span whose σ is used both as the normalization denominator for
        ret_norm / mom_norm / target AND as the absolute level feature
        `log_ewma_vol_{vol_norm_span}`. Default 60.

    ret_norm_horizons : list[int]
        Horizons (bars) for volatility-normalized returns. Default [1, 5, 20].

    mom_horizons : list[int]
        Horizons (bars) for volatility-normalized momentum. Staggered against
        ret_norm_horizons to avoid redundant columns. Default [3, 10, 40].

    target_clip : float
        Symmetric clip bound for the normalized return target. Default ±20.
    """

    vol_ratio_spans: list[int] = field(default_factory=lambda: [10, 20])
    vol_norm_span: int = 60
    ret_norm_horizons: list[int] = field(default_factory=lambda: [1, 5, 20])
    mom_horizons: list[int] = field(default_factory=lambda: [3, 10, 40])
    target_clip: float = 20.0

    def __post_init__(self) -> None:
        if any(s < 1 for s in self.vol_ratio_spans):
            raise ValueError("All vol_ratio_spans must be >= 1.")
        if self.vol_norm_span < 1:
            raise ValueError(
                f"vol_norm_span must be >= 1, got {self.vol_norm_span}."
            )
        if not self.ret_norm_horizons:
            raise ValueError("ret_norm_horizons must not be empty.")
        if not self.mom_horizons:
            raise ValueError("mom_horizons must not be empty.")
        for h in [*self.ret_norm_horizons, *self.mom_horizons]:
            if h < 1:
                raise ValueError(f"All horizons must be >= 1, got {h}.")
        if self.target_clip <= 0:
            raise ValueError("target_clip must be positive.")

    @property
    def feature_columns(self) -> list[str]:
        """Canonical ordered list of the 9 model-input column names."""
        cols: list[str] = [f"ret_norm_{h}" for h in self.ret_norm_horizons]
        cols += [f"vol_ratio_{s}" for s in self.vol_ratio_spans]
        cols += [f"log_ewma_vol_{self.vol_norm_span}"]
        cols += [f"mom_norm_{h}" for h in self.mom_horizons]
        return cols


# ──────────────────────────────────────────────────────────────────────────────
# Input validation helpers (carried over verbatim from archived v1)
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


# ──────────────────────────────────────────────────────────────────────────────
# Primitive: log returns (cudf-safe numpy escape carried over from v1)
# ──────────────────────────────────────────────────────────────────────────────

def _prices_to_numpy(prices: pd.Series) -> np.ndarray:
    """Extract close prices as float64 numpy array (CuPy → numpy on GPU path)."""
    try:
        return prices.values.get()  # type: ignore[union-attr]  # CuPy array → numpy
    except AttributeError:
        return np.array(prices.values, dtype=np.float64)


def log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute log returns: r_t = log(P_t / P_{t-1}).

    Time-additive and approximately symmetric.
    """
    original_index = prices.index
    arr = _prices_to_numpy(prices)

    log_ret = np.empty_like(arr)
    log_ret[0] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret[1:] = np.log(arr[1:] / arr[:-1])

    return pd.Series(log_ret, index=original_index, name=prices.name)


def _cumulative_log_return(prices: pd.Series, h: int) -> pd.Series:
    """h-bar cumulative log return via rolling sum (handles NaN gaps)."""
    r = log_returns(prices)
    return r.rolling(window=h, min_periods=h).sum()


# ──────────────────────────────────────────────────────────────────────────────
# EWMA volatility (Wilder-seeded, cudf-safe — carried over from v1)
# ──────────────────────────────────────────────────────────────────────────────

def _ewm_wilder_seeded(arr: np.ndarray, alpha: float, seed_period: int) -> np.ndarray:
    """
    EWM with correct warm-up: seed = simple mean of first `seed_period` valid bars.
    Carry-forward across NaN gaps.
    """
    n = len(arr)
    out = np.full(n, np.nan)
    valid = np.where(~np.isnan(arr))[0]

    if len(valid) < seed_period:
        return out   # not enough data — all NaN

    i = 0
    while i < len(valid):
        seg_end = i + seed_period
        if seg_end > len(valid):
            break
        seed_idx = valid[i:seg_end]
        seed_val = np.mean(arr[seed_idx])
        out[seed_idx[-1]] = seed_val          # first valid output

        prev = seed_idx[-1]
        for j in range(seed_idx[-1] + 1, n):
            if np.isnan(arr[j]):
                out[j] = out[prev]            # carry-forward
            else:
                out[j] = alpha * arr[j] + (1.0 - alpha) * out[prev]
                prev = j
        break   # single contiguous series — done

    return out


def ewma_volatility(prices: pd.Series, span: int = 60) -> pd.Series:
    """
    Conditional volatility σ_t via EWMA of squared demeaned log returns.

    α = 2/(span+1).
    σ²_t = α·(r_t − μ_t)² + (1−α)·σ²_{t−1}

    NaN gaps are handled by carrying forward the last valid EWMA state —
    never injecting phantom zero-returns into the variance estimator.
    """
    _validate_prices(prices)

    p_vals = _prices_to_numpy(prices)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_r = np.log(p_vals[1:] / p_vals[:-1])
    log_r = np.concatenate([[np.nan], log_r])  # restore length

    alpha = 2.0 / (span + 1.0)
    seed_period = min(span, 30)
    ewma_mean = _ewm_wilder_seeded(log_r, alpha, seed_period)
    demeaned_sq = np.where(np.isnan(log_r), np.nan, (log_r - ewma_mean) ** 2)
    ewma_var = _ewm_wilder_seeded(demeaned_sq, alpha, seed_period)

    sigma_vals = np.sqrt(ewma_var)
    sigma_vals[0] = np.nan

    sigma = pd.Series(sigma_vals.tolist(), index=prices.index, dtype="float64")
    return sigma


def vol_regime_features(
    prices: pd.Series,
    ratio_spans: list[int],
    norm_span: int,
) -> pd.DataFrame:
    """
    Vol-regime features.

    For each s in `ratio_spans` (the fast spans) produce
    ``vol_ratio_{s} = log(σ_s / σ_norm)`` — a zero-centered measure of how
    much faster / slower current short-term vol is vs. the stable baseline.

    Plus a single absolute level feature ``log_ewma_vol_{norm_span} = log(σ_norm)``.

    Why the ratio split
    -------------------
    Emitting log(σ) directly leaves the features off-center around log(σ) ≈ -6
    (since NIFTY 30-min σ ≈ 0.003 → log ≈ -5.8). The |mean|/std ratio of ~12
    pollutes every dot-product downstream. A log-ratio is by construction
    near zero (σ_s and σ_norm track the same underlying regime), so the
    regime features live in a tight band, while the absolute level is
    preserved in the one log_ewma_vol column.

    NaN propagation: if either σ_s or σ_norm is invalid at bar t, the
    ratio is NaN at t.

    Output columns: vol_ratio_{s} for s in ratio_spans,
                     log_ewma_vol_{norm_span}.
    NO_SCALE bucket.
    Warm-up: ~min(span, 30) bars for each underlying σ.
    """
    sigma_norm = ewma_volatility(prices, span=norm_span)
    with np.errstate(invalid="ignore", divide="ignore"):
        log_norm = np.log(sigma_norm.where(sigma_norm > _EPS, other=np.nan))
    log_norm.name = f"log_ewma_vol_{norm_span}"

    # Emit in the canonical order: ratios first, then absolute level last.
    out: dict[str, pd.Series] = {}
    for s in ratio_spans:
        sigma_s = ewma_volatility(prices, span=s)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.log(
                sigma_s.where(sigma_s > _EPS, other=np.nan)
                / sigma_norm.where(sigma_norm > _EPS, other=np.nan)
            )
        ratio.name = f"vol_ratio_{s}"
        out[f"vol_ratio_{s}"] = ratio
    out[f"log_ewma_vol_{norm_span}"] = log_norm
    return pd.DataFrame(out, index=prices.index)


# ──────────────────────────────────────────────────────────────────────────────
# Vol-normalized returns & momentum (shared formula, staggered horizons)
# ──────────────────────────────────────────────────────────────────────────────

def _normalized_return_series(
    prices: pd.Series,
    horizons: list[int],
    sigma: pd.Series,
    prefix: str,
) -> pd.DataFrame:
    """
    Shared helper for ret_norm_* and mom_norm_*:

        x(t,h) = r(t,h) / (σ_t · √h)

    where r(t,h) = h-bar cumulative log return. Dimensionless, horizon-
    invariant, concentrated in roughly [-3, +3].
    """
    out: dict[str, pd.Series] = {}
    for h in horizons:
        cum_ret = _cumulative_log_return(prices, h)
        denom = sigma * np.sqrt(h)
        with np.errstate(invalid="ignore", divide="ignore"):
            norm = cum_ret / denom
        norm = norm.where(denom > _EPS, other=np.nan)
        norm.name = f"{prefix}_{h}"
        out[f"{prefix}_{h}"] = norm
    return pd.DataFrame(out, index=prices.index)


# ──────────────────────────────────────────────────────────────────────────────
# Training-only Target
# ──────────────────────────────────────────────────────────────────────────────

def normalized_return_target(
    prices: pd.Series,
    sigma: pd.Series,
    clip_value: float = 20.0,
    inference_mode: bool = False,
    _allow_in_inference: bool = False,   # escape hatch for deliberate test-time use
) -> pd.Series:
    """
    Clipped vol-normalized next-bar return target (training label only).

        target_t = clip( r_{t+1} / σ_t ,  ±clip_value )

    ⚠️  NEVER include during live inference — r_{t+1} is not available.
    The inference_mode guard raises RuntimeError rather than relying on
    docstring convention.
    """
    if inference_mode and not _allow_in_inference:
        raise RuntimeError(
            "normalized_return_target() was called while inference_mode=True. "
            "This function uses r_{t+1} (future data) and must never run during "
            "live inference. Pass inference_mode=False to the FeatureEngineer "
            "constructor for training builds."
        )

    _validate_prices(prices)
    r_next = log_returns(prices).shift(-1)
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
    End-to-end feature engineering — v2 minimal vol-normalized set.

    Usage
    -----
    Training:
        fe = FeatureEngineer()
        feats = fe.build(close_series, include_target=True, dropna=True)

    Inference:
        fe = FeatureEngineer(inference_mode=True)
        feats = fe.build(close_series, include_target=False, dropna=False)

    Notes
    -----
    - `ohlc` is accepted for backward compatibility with existing callers
      (train.py still passes it) but is IGNORED — the v2 feature set is
      close-only.
    """

    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        inference_mode: bool = False,
    ) -> None:
        self.config = config or FeatureConfig()
        self.inference_mode = inference_mode  # instance-level, survives pickling

    def build(
        self,
        prices: pd.Series,
        ohlc: Optional[pd.DataFrame] = None,
        include_target: bool = False,
        dropna: bool = False,
    ) -> pd.DataFrame:
        """
        Compute the v2 feature matrix for one asset.

        Parameters
        ----------
        prices : pd.Series
            Close prices.
        ohlc : ignored (v2 features are close-only; kept for API compat).
        include_target : bool
            Append the normalized return target column (training only).
        dropna : bool
            Drop NaN warm-up rows. Use True for training, False for inference.

        Returns
        -------
        pd.DataFrame
            Columns in canonical order (see FeatureConfig.feature_columns),
            plus 'target_norm_ret' when include_target=True.
        """
        cfg = self.config
        logger.info(
            "Building v2 features for '%s' | %d rows | ohlc=%s (ignored — "
            "v2 feature set is close-only).",
            prices.name or "unnamed",
            len(prices),
            ohlc is not None,
        )

        _validate_prices(prices)
        prices = prices.sort_index()

        # σ used as normalization denominator (most stable configured span)
        sigma_norm = ewma_volatility(prices, span=cfg.vol_norm_span)

        parts: list[pd.DataFrame | pd.Series] = []

        # ── 1. Vol-normalized multi-horizon returns ───────────────────────
        parts.append(_normalized_return_series(
            prices, cfg.ret_norm_horizons, sigma_norm, prefix="ret_norm"
        ))

        # ── 2. Vol regime (ratios) + absolute level ───────────────────────
        parts.append(vol_regime_features(
            prices,
            ratio_spans=cfg.vol_ratio_spans,
            norm_span=cfg.vol_norm_span,
        ))

        # ── 3. Vol-normalized multi-horizon momentum ──────────────────────
        parts.append(_normalized_return_series(
            prices, cfg.mom_horizons, sigma_norm, prefix="mom_norm"
        ))

        # ── 4. Target (training only) ─────────────────────────────────────
        if include_target:
            if self.inference_mode:
                raise RuntimeError(
                    "FeatureEngineer.build() called with include_target=True while "
                    "inference_mode=True. The target column uses r_{t+1} (look-ahead). "
                    "Pass inference_mode=False to the FeatureEngineer constructor "
                    "for training builds."
                )
            parts.append(normalized_return_target(
                prices,
                sigma=sigma_norm,
                clip_value=cfg.target_clip,
                inference_mode=self.inference_mode,
            ))

        result = pd.concat(parts, axis=1)

        # Sanity-check uniform dtypes after concat
        if not (result.dtypes == "float64").all():
            raise ValueError(
                f"Mixed dtypes after concat: "
                f"{result.dtypes[result.dtypes != 'float64'].to_dict()}"
            )

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

    def build_multi_asset(
        self,
        price_df: pd.DataFrame,
        ohlc_dict: Optional[dict[str, pd.DataFrame]] = None,
        include_target: bool = False,
        dropna: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Build features for every asset (column) in a price panel.

        `ohlc_dict` accepted for API compat; unused by the v2 close-only set.
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
        if not feature_dict:
            raise ValueError(
                "stack_for_model: feature_dict is empty — all ticker builds failed. "
                "Check the warnings logged by build_multi_asset() for per-ticker errors."
            )

        tickers = list(feature_dict.keys())
        has_target = "target_norm_ret" in next(iter(feature_dict.values())).columns

        min_len_ticker = min(feature_dict, key=lambda t: len(feature_dict[t]))
        min_len = len(feature_dict[min_len_ticker])
        if min_len <= lookback:
            logger.warning(
                "stack_for_model(): ticker '%s' has only %d bars — shorter than "
                "lookback=%d. This will likely produce an empty common_idx after slicing.",
                min_len_ticker,
                min_len,
                lookback,
            )

        common_idx = feature_dict[tickers[0]].index
        for df in feature_dict.values():
            common_idx = common_idx.intersection(df.index)

        n_common = len(common_idx)
        common_idx = common_idx[lookback:]

        if len(common_idx) == 0:
            raise ValueError(
                f"stack_for_model(): no dates remain after applying lookback={lookback}. "
                f"The common index across {len(tickers)} ticker(s) contained only "
                f"{n_common} bar(s) — which is <= lookback ({lookback}). "
                "Fix: reduce lookback, supply more history, or check that all tickers "
                "cover the same date range."
            )
        logger.info(
            "stack_for_model(): common_idx after lookback trim: %d bars "
            "(%d dropped as warm-up, spanning %s → %s).",
            len(common_idx),
            lookback,
            common_idx[0] if len(common_idx) > 0 else "None",
            common_idx[-1] if len(common_idx) > 0 else "None",
        )

        all_col_sets = [set(df.columns) for df in feature_dict.values()]
        common_col_set = set.intersection(*all_col_sets)

        for ticker, df in feature_dict.items():
            dropped = set(df.columns) - common_col_set
            if dropped:
                logger.warning(
                    "stack_for_model: ticker '%s' has extra columns not present "
                    "in all tickers — dropping from tensor: %s",
                    ticker,
                    sorted(dropped),
                )

        feature_cols = [
            col for col in cfg_feature_order(self.config)
            if col in common_col_set
        ]
        # The training target must never enter the feature tensor.
        extras = sorted(common_col_set - set(feature_cols) - {"target_norm_ret"})
        if extras:
            logger.warning(
                "stack_for_model: unexpected common columns outside canonical "
                "order — appending: %s", extras,
            )
            feature_cols += extras

        if not feature_cols:
            raise ValueError(
                "stack_for_model: no feature columns common to all tickers. "
                "Check that all assets were built with the same FeatureConfig."
            )

        logger.info(
            "stack_for_model: %d common feature columns across %d tickers.",
            len(feature_cols), len(tickers),
        )

        X_list, y_list = [], []
        for ticker in tickers:
            df = feature_dict[ticker].loc[common_idx]
            X_list.append(df[feature_cols].values)
            if has_target:
                y_list.append(df["target_norm_ret"].values)

        X = np.stack(X_list, axis=1).astype(np.float32)
        y = np.stack(y_list, axis=1).astype(np.float32) if has_target else np.array([])
        return X, y, [str(d) for d in common_idx], tickers


def cfg_feature_order(cfg: FeatureConfig) -> list[str]:
    """Canonical model-input column order (excludes the training target)."""
    return cfg.feature_columns


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test / demo  (python features.py)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    np.random.seed(42)
    N = 10000

    idx = pd.date_range(
        pd.Timestamp("2020-01-02 09:15:00", tz="Asia/Kolkata"),
        periods=N,
        freq="30min",
        tz="Asia/Kolkata",
    )
    log_r = np.random.normal(0.0002, 0.015, N)
    closes = 100.0 * np.exp(np.cumsum(log_r))
    close = pd.Series(closes, index=idx, name="NIFTY")

    cfg = FeatureConfig()

    fe = FeatureEngineer(config=cfg)
    feats_train = fe.build(close, include_target=True, dropna=False)
    feats_inf = FeatureEngineer(config=cfg, inference_mode=True).build(
        close, include_target=False, dropna=False
    )

    print("\n" + "=" * 65)
    print("  V2 FEATURE MATRIX")
    print("=" * 65)
    print(f"  Train shape (with target): {feats_train.shape}")
    print(f"  Infer shape (no target):   {feats_inf.shape}")
    print(f"\n  Canonical order: {cfg.feature_columns}")
    assert list(feats_inf.columns) == cfg.feature_columns
    assert list(feats_train.columns) == cfg.feature_columns + ["target_norm_ret"]

    print("\n  Per-column stats:")
    for col in feats_train.columns:
        s = feats_train[col].dropna()
        print(f"    {col:<18}  mean={s.mean():+.4f}  std={s.std():.4f}  "
              f"min={s.min():+.4f}  max={s.max():+.4f}")

    first_valid = feats_train.notna().all(axis=1).idxmax()
    print(f"\n  First fully-valid row: {first_valid} "
          f"(warm-up = {feats_train.index.get_loc(first_valid)} bars)")
    print("  NaN remaining after warm-up:",
          int(feats_train.loc[first_valid:].isna().sum().sum()))
