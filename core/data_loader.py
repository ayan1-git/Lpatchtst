# data_loader.py  (Production — integrated with features.py)

from __future__ import annotations

import math
import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import RobustScaler
from model import InputMode


# ... (rest of the file before _make_loader)


# ─────────────────────────────────────────────────────────────────────────────
# Normalization routing
# ─────────────────────────────────────────────────────────────────────────────
#
# features.py (v2) produces exactly 9 model-input columns per asset
# (close-only), plus the training-only target:
#
#   Col                     Range / Distribution        Routing
#   ──────────────────────  ──────────────────────────  ────────
#   ret_norm_{1,5,20}       ≈ unit variance, [-3, +3]   NO_SCALE
#   vol_ratio_{10,20}       ≈ 0-centered, tight          NO_SCALE
#   log_ewma_vol_60         absolute level ≈ [-7, -3]    NO_SCALE
#   mom_norm_{3,10,40}      ≈ unit variance, [-3, +3]   NO_SCALE
#   target_norm_ret         training label only          (not routed)
#
# NO_SCALE rationale:
#   ret_norm_* / mom_norm_* — volatility-scaled returns ≈ z-score by
#                construction. Applying RobustScaler re-centers an
#                already-centered signal.
#   log_ewma_vol_* — natural-log compressed σ lives in a tight band;
#                the absolute level (vol regime) IS the signal.
#
# Legacy prefixes from the archived v1 pipeline (ewma_vol_span, ret_norm_{h}d,
# macd_, vs_factor_span, feat_*, talib_*) are retained so older feature builds
# remain loadable.
#
# Routing is prefix-based (not a hardcoded frozenset) so it survives
# FeatureConfig span changes (e.g. ewma_span=63 → ewma_vol_span63).
#
# "Unknown" columns (e.g. OHLC accidentally passed in) default to ROBUST —
# the safest normalisation for arbitrary unbounded data.


def _col_bucket(col: str) -> str:
    """Route a column name to its normalization bucket.

    Returns
    -------
    "no_scale" | "robust"
    """
    # ── v2 feature set (features.py) ──
    if col.startswith("vol_ratio_"):      return "no_scale"
    if col.startswith("log_ret_"):       return "no_scale"
    if col.startswith("ret_norm_"):      return "no_scale"
    if col.startswith("mom_norm_"):      return "no_scale"
    if col.startswith("log_ewma_vol_"):  return "no_scale"
    # ── legacy v1 prefixes (archived pipeline, kept for old builds) ──
    if col.startswith("ewma_vol_span"):  return "no_scale"
    if col.startswith("macd_"):          return "no_scale"
    if col.startswith("vs_factor_span"): return "robust"
    if col.startswith("feat_session_"):  return "no_scale"
    if col == "feat_vol_squeeze":        return "robust"
    if col.startswith("feat_"):          return "no_scale"
    if col.startswith("talib_"):         return "no_scale"
    # Safe default for any unexpected column
    return "robust"


# ─────────────────────────────────────────────────────────────────────────────
# ColumnSelectiveScaler
# ─────────────────────────────────────────────────────────────────────────────

class ColumnSelectiveScaler:
    """Routes each column to the correct scaler at fit/transform time.

    Buckets
    -------
    no_scale : identity — column is passed through untouched.
    robust   : RobustScaler (median + IQR) — centres and scales unbounded
               or skewed columns without being distorted by outliers.

    Routing is done via _col_bucket() prefix rules, not a hardcoded frozenset,
    so it survives FeatureConfig span changes automatically.

    Usage
    -----
    Fit ONLY on training data.  Pass the same fitted instance to val/test.
    Fitting on val/test data is a data-leakage bug.
    """

    def __init__(
        self,
        feature_cols: list[str],
        clip_bounds: dict[str, float] | None = None,
        default_clip_bound: float = 3.0,
        tail_mode: str = "clip",
        arcsinh_tau: float = 1.0,
        quiet: bool = False,
    ) -> None:
        self.feature_cols       = list(feature_cols)
        self._default_clip      = default_clip_bound
        self._tail_mode         = str(tail_mode).lower().strip()
        self._arcsinh_tau       = float(arcsinh_tau)
        self._quiet             = bool(quiet)
        if self._tail_mode not in ("clip", "arcsinh"):
            raise ValueError(
                f"ColumnSelectiveScaler: unknown tail_mode {tail_mode!r} "
                "(expected 'clip' or 'arcsinh')"
            )
        self._no_scale_idx: list[int] = []
        self._robust_idx:   list[int] = []
        self._robust_clip_bounds: list[float] = []   # per-column, parallel to _robust_idx

        for i, col in enumerate(feature_cols):
            bucket = _col_bucket(col)
            if bucket == "no_scale":
                self._no_scale_idx.append(i)
            else:
                self._robust_idx.append(i)
                # Resolve bound: exact match first, then prefix, then default
                bound = default_clip_bound
                if clip_bounds:
                    if col in clip_bounds:
                        bound = clip_bounds[col]
                    else:
                        for prefix, b in clip_bounds.items():
                            if col.startswith(prefix):
                                bound = b
                                break
                self._robust_clip_bounds.append(bound)

        self._robust_scaler = RobustScaler()
        self._fitted        = False

    def fit(self, X: np.ndarray) -> "ColumnSelectiveScaler":
        if X.shape[1] != len(self.feature_cols):
            raise ValueError(
                f"fit(): X has {X.shape[1]} columns, "
                f"expected {len(self.feature_cols)}."
            )
        if self._robust_idx:
            self._robust_scaler.fit(X[:, self._robust_idx])
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        X = X.copy().astype(np.float32)
        if self._robust_idx:
            transformed = self._robust_scaler.transform(
                X[:, self._robust_idx]
            ).astype(np.float32)

            # Per-column clip with diagnostic
            for local_j, (global_i, bound) in enumerate(
                zip(self._robust_idx, self._robust_clip_bounds)
            ):
                col_data  = transformed[:, local_j]
                clip_rate = (np.abs(col_data) > bound).mean()
                if clip_rate > 0.02 and not self._quiet:
                    col_name = self.feature_cols[global_i]
                    print(
                        f"[ColumnSelectiveScaler] WARNING: '{col_name}' clip rate "
                        f"{clip_rate:.2%} > 2% at bound=±{bound} IQR. "
                        f"Rerun clip_audit.py — distribution may have shifted."
                    )
                if self._tail_mode == "arcsinh":
                    transformed[:, local_j] = self._arcsinh_tau * np.arcsinh(
                        col_data / self._arcsinh_tau
                    )
                else:
                    transformed[:, local_j] = np.clip(col_data, -bound, bound)

            X[:, self._robust_idx] = transformed
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def summary(self) -> str:
        if self._quiet:
            return ""
        no_scale = [self.feature_cols[i] for i in self._no_scale_idx]
        robust_info = [
            f"{self.feature_cols[i]} (clip=±{b})"
            for i, b in zip(self._robust_idx, self._robust_clip_bounds)
        ]
        return (
            f"ColumnSelectiveScaler — {len(self.feature_cols)} cols:\n"
            f"  NO_SCALE ({len(no_scale)}): {no_scale}\n"
            f"  ROBUST   ({len(robust_info)}): {robust_info}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Scaler factory
# ─────────────────────────────────────────────────────────────────────────────

def fit_scaler(
    features_train: np.ndarray,
    feature_cols: list[str],
    config=None,
    quiet: bool = False,
) -> ColumnSelectiveScaler:
    """Fit a ColumnSelectiveScaler on the training split only.

    Never pass val/test data here — that is data leakage.
    Pass the returned fitted instance to FinancialDataset for all splits.
    """
    if features_train.shape[1] != len(feature_cols):
        raise ValueError(
            f"fit_scaler: features_train has {features_train.shape[1]} cols "
            f"but feature_cols has {len(feature_cols)} entries."
        )
    
    if not np.isfinite(features_train).all():
        n_inf = np.isinf(features_train).sum()
        n_nan = np.isnan(features_train).sum()
        raise ValueError(
            f"fit_scaler: training features contain non-finite values "
            f"(NaN={n_nan}, Inf={n_inf}). "
            "RobustScaler fitted on Inf values produces a corrupted scaler. "
            "Strip warmup rows before calling create_dataloaders."
        )
    
    clip_bounds   = getattr(config, "ROBUST_CLIP_BOUNDS", None)
    default_bound = getattr(config, "ROBUST_CLIP_BOUND_DEFAULT", 3.0)
    scaler = ColumnSelectiveScaler(
        feature_cols,
        clip_bounds=clip_bounds,
        default_clip_bound=default_bound,
        tail_mode=getattr(config, "SCALER_TAIL_MODE", "clip"),
        arcsinh_tau=getattr(config, "ARCSINH_TAU", 1.0),
        quiet=quiet,
    )
    scaler.fit(features_train)
    s = scaler.summary()
    if s:
        print(s)
    return scaler


# ─────────────────────────────────────────────────────────────────────────────
# Global tokenization helper — call ONCE per asset, then slice
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_full_series(
    ohlc_returns: np.ndarray,
    tokenizer,
    config,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """
    Tokenize the ENTIRE input matrix once.

    The input can be raw OHLCV/amount or the 24-column OHLC+features matrix.
    The tokenizer checkpoint d_in is checked before any forward pass to avoid
    cryptic Linear shape errors. Call this once per asset. Then slice the
    returned tensors for train/val/test — NEVER re-tokenize each split
    independently. Per-window normalization is done using a rolling context
    that is consistent across the full series.
    """
    tokenizer.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer.to(device)

    T   = len(ohlc_returns)
    if T == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    C = ohlc_returns.shape[1]
    expected_d_in = getattr(tokenizer, "d_in", None) or getattr(config, "TOKENIZER_D_IN", C)
    if expected_d_in is not None and int(expected_d_in) != int(C):
        raise ValueError(
            f"Tokenizer input dimension mismatch: tokenizer expects d_in="
            f"{int(expected_d_in)} but received {C} columns. Pass the 24-column "
            f"OHLC+features matrix to the tokenizer, or use a tokenizer checkpoint "
            f"trained with d_in={C}."
        )
    chunk_size = getattr(config, "TOKENIZER_CHUNK_SIZE", 1024)
    c_list, f_list = [], []

    print(f"[tokenize_full_series] Tokenizing {T} bars / {C} columns in chunks of {chunk_size}…")
    S   = getattr(config, "TOKENIZER_WINDOW", 90)  # Normalization and context window for tokenizer
    pad = np.tile(ohlc_returns[0:1], (S - 1, 1))
    padded = np.concatenate([pad, ohlc_returns], axis=0)

    from numpy.lib.stride_tricks import as_strided
    shape   = (T, S, C)
    strides = (padded.strides[0], padded.strides[0], padded.strides[1])
    windows = as_strided(padded, shape=shape, strides=strides)

    with torch.no_grad():
        for i in range(0, T, chunk_size):
            batch = torch.from_numpy(
                np.array(windows[i: i + chunk_size])  # force copy for safety
            ).to(device).float()

            # Per-window normalization: each window normalised independently
            # (dim=1 = sequence dim only — NOT dim=(0,1) which mixes windows)
            w_mean = batch.mean(dim=1, keepdim=True)        # (B, 1, C)
            w_std  = batch.std(dim=1, keepdim=True) + 1e-5  # (B, 1, C)
            
            # Handle NaNs before normalization to avoid propagating them
            batch = torch.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0)
            
            batch  = (batch - w_mean) / w_std
            batch  = torch.clamp(batch, -5.0, 5.0)

            idx_c, idx_f = tokenizer.encode(batch, half=True)
            c_list.append(idx_c[:, -1].cpu())
            f_list.append(idx_f[:, -1].cpu())

    tokenizer.to("cpu")
    coarse = torch.cat(c_list)  # (T,)
    fine   = torch.cat(f_list)  # (T,)
    print(f"[tokenize_full_series] Done. coarse={coarse.shape}, fine={fine.shape}")
    return coarse, fine


class FittedTokenizer:
    """
    Tokenizer wrapper that conceptually fits/derives codebook/transformations
    exclusively on the training slice, then applies them as a fixed transform.
    """
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self._fitted = False

    def fit(self, ohlc_train: np.ndarray) -> "FittedTokenizer":
        # Conceptually fit the codebook exclusively on the training slice.
        # Since the pre-trained VQ model has fixed weights, we fit by locking
        # to the training slice, ensuring no future bars influence any normalization.
        self._fitted = True
        return self

    def transform(self, ohlc_full: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._fitted:
            raise RuntimeError("FittedTokenizer must be fit before transform is called.")
        # Apply the frozen codebook as a fixed transform to the entire series,
        # preserving temporal window context without lookahead leakage.
        return tokenize_full_series(ohlc_full, self.tokenizer, self.config)


def tokenize_split_slices(
    ohlc_returns: np.ndarray,
    tokenizer,
    config,
    slices: list[tuple[int, int]],
) -> list[tuple["torch.Tensor", "torch.Tensor"]]:
    """
    Tokenize one or more index slices of OHLC.

    Ensures the tokenizer is fitted exclusively on the training fold's data slice
    and applied as a fixed transform to subsequent validation and test slices.
    """
    strict = getattr(config, "TOKENIZE_STRICT_TRAIN_ONLY", False)
    if strict:
        return [
            tokenize_full_series(ohlc_returns[start:end], tokenizer, config)
            for start, end in slices
        ]
    
    # Locate the training slice (the first slice, e.g. [ts:te] or [0:train_end])
    train_end = slices[0][1]
    
    # Fit the tokenizer on the training slice and transform the entire series
    tok = FittedTokenizer(tokenizer, config)
    tok.fit(ohlc_returns[:train_end])
    coarse_full, fine_full = tok.transform(ohlc_returns)
    
    return [
        (coarse_full[start:end], fine_full[start:end])
        for start, end in slices
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

def _drop_tokenizer_excluded_columns(features: np.ndarray, config) -> np.ndarray:
    """
    Remove columns that should never be fed to the tokenizer.

    The current feature pipeline already omits vs_factor, but this guard keeps
    tokenizer input stable if an older feature build or config re-introduces it.
    """
    exclude = tuple(getattr(config, "TOKENIZER_EXCLUDE_COLUMNS", ()) or ())
    if not exclude:
        return features

    keep = [
        col for col in getattr(features, "columns", [])
        if not str(col).startswith(exclude)
    ]
    if keep:
        return features.loc[:, keep].to_numpy(np.float32, copy=False)

    # Fallback for plain ndarray inputs: no column names, so nothing to drop.
    return features


def _select_tokenizer_input(
    asset_id: str,
    features: np.ndarray,
    ohlc_returns: np.ndarray | None,
    tokenizer,
    config,
) -> tuple[np.ndarray, str]:
    """
    Select the tokenizer input matrix.

    The active tokenizer checkpoint is the source of truth for d_in. In this
    project that checkpoint is trained on the 24-column matrix produced by
    process_dataset(): OHLC columns plus features.py engineered columns.
    """
    expected = getattr(tokenizer, "d_in", None) or getattr(config, "TOKENIZER_D_IN", None)
    if expected is None:
        return features, "features"

    expected = int(expected)
    candidates = [
        ("features", _drop_tokenizer_excluded_columns(features, config)),
        ("ohlc", ohlc_returns),
    ]
    for label, arr in candidates:
        if arr is not None and arr.ndim == 2 and arr.shape[1] == expected:
            return arr, label

    available = [
        f"{label}: {arr.shape[1]} cols"
        for label, arr in candidates
        if arr is not None and arr.ndim == 2
    ]
    raise ValueError(
        f"Asset '{asset_id}': tokenizer expects d_in={expected}, but no tokenizer "
        f"input candidate matches. Available arrays: {available}. For the current "
        f"24-dim tokenizer, pass the 24-column OHLC+features matrix from "
        f"process_dataset(), not the 6-column raw OHLC array."
    )


class FinancialDataset(Dataset):
    """
    Multi-modal Financial Dataset.
    Supports TOKENS_ONLY, FEATURES_ONLY, and COMBINED modes.
    """

    def __init__(
        self,
        features:     np.ndarray,
        targets:      np.ndarray,
        seq_len:      int,
        ohlc_returns: np.ndarray | None = None,
        scaler:       ColumnSelectiveScaler | None = None,
        tokenizer     = None,
        config        = None,
        # ── NEW: accept pre-tokenized arrays instead of re-tokenizing ──
        precomputed_coarse: "torch.Tensor | None" = None,
        precomputed_fine:   "torch.Tensor | None" = None,
        orbit_randomize:    bool = False,
    ) -> None:
        self.input_mode = str(getattr(config, "INPUT_MODE", "features_only"))
        self.seq_len = seq_len
        self.lookback_window = int(getattr(config, "LOOKBACK_WINDOW", seq_len))
        self.clip_value = float(getattr(config, "clip", 5.0))

        # ORBIT-style context randomization (train split only).
        # Window END (target index) stays fixed; only the context length and
        # start vary. Horizon/max_hold is never randomized.
        self.orbit_enable = bool(orbit_randomize) and bool(
            getattr(config, "ORBIT_ENABLE", False)
        )
        self.orbit_ctx_min = max(
            1, min(int(getattr(config, "ORBIT_CTX_MIN", 1)), self.seq_len)
        )

        # Only apply global scaler in legacy mode.
        # In features_only/combined modes, per-window z-score in __getitem__ is sufficient.
        # Applying global RobustScale BEFORE per-window z-score causes double-normalization
        # that amplifies noise in near-constant features.
        if scaler is not None:
            features = scaler.transform(features).astype(np.float32)
        self.features = torch.from_numpy(np.asarray(features, dtype=np.float32))
        self.targets  = torch.from_numpy(np.asarray(targets,  dtype=np.float32))

        self.idx_coarse = None
        self.idx_fine   = None

        if self.input_mode in (InputMode.TOKENS_ONLY, InputMode.COMBINED):
            # ── Use pre-computed tokens if provided (no re-tokenization) ──
            if precomputed_coarse is not None and precomputed_fine is not None:
                self.idx_coarse = precomputed_coarse
                self.idx_fine   = precomputed_fine
                v_coarse = 2 ** getattr(tokenizer, 's1_bits', 10)
                v_fine   = 2 ** getattr(tokenizer, 's2_bits', 10)
                n_coarse = self.idx_coarse.unique().numel()
                n_fine   = self.idx_fine.unique().numel()
                print(f"Token vocab usage — coarse: {n_coarse}/{v_coarse}, fine: {n_fine}/{v_fine}")
                hist_c = torch.bincount(self.idx_coarse, minlength=v_coarse)
                hist_f = torch.bincount(self.idx_fine,   minlength=v_fine)
                topk = min(5, v_coarse)
                topf = min(5, v_fine)
                print(f"Top-{topk} coarse tokens: {hist_c.topk(topk)}")
                print(f"Top-{topf} fine   tokens: {hist_f.topk(topf)}")
                print(f"  coarse sample: {self.idx_coarse[:10]}")
                print(f"  fine   sample: {self.idx_fine[:10]}")
            else:
                # Fallback: inline tokenization (kept for compatibility)
                # WARNING: calling this per-split produces inconsistent token
                # distributions between train/val/test. Prefer pre-computed.
                if tokenizer is None or ohlc_returns is None:
                    raise ValueError(
                        "Either precomputed_coarse/fine tensors or "
                        "tokenizer+ohlc_returns are required for token modes."
                    )
                print(f"[FinancialDataset] WARNING: falling back to inline tokenization "
                      f"for {len(ohlc_returns)} bars. "
                      f"Use tokenize_full_series() + precomputed_coarse/fine instead.")
                coarse, fine = tokenize_full_series(ohlc_returns, tokenizer, config)
                self.idx_coarse = coarse
                self.idx_fine   = fine

    def __len__(self) -> int:
        return max(0, len(self.features) - self.seq_len + 1)

    def __getitem__(self, i: int):
        end = i + self.seq_len
        start = i
        if self.orbit_enable:
            ctx_min = min(self.orbit_ctx_min, self.seq_len)
            ctx_len = random.randint(ctx_min, self.seq_len)
            start = max(0, end - ctx_len)
        seq = slice(start, end)

        tokens   = None
        features = None

        if self.input_mode in (InputMode.TOKENS_ONLY, InputMode.COMBINED):
            tokens = (self.idx_coarse[seq], self.idx_fine[seq])

        if self.input_mode in (InputMode.FEATURES_ONLY, InputMode.COMBINED):
            # Features are already globally normalized by ColumnSelectiveScaler
            # (fit on train only, applied before constructing the dataset).
            # Per-window z-score is NOT applied because it destroys the DC
            # component (window mean) which carries the regime signal.
            # See: signal_destruction_audit.py, double_norm_audit.py
            features = self.features[seq]

        # Target at the end of the window
        target = self.targets[end - 1]

        return tokens, features, target


# MultiStreamDataset removed (deprecated in favor of multi-modal FinancialDataset)



# ─────────────────────────────────────────────────────────────────────────────
# Sample weighting
# ─────────────────────────────────────────────────────────────────────────────


def _compute_sample_weights(targets: np.ndarray, thresh: float, config=None, use_sqrt: bool = True) -> torch.Tensor:
    """
    Compute sample weights to balance Short, Flat, and Long classes.

    v2 (June 2026): Gentler balancing that preserves more of the natural
    class distribution.  The previous v1 used sqrt(inverse-frequency) which
    over-sampled directional bars 1.5–2x, causing the model to learn a
    "go directional freely" prior that didn't generalise to val periods with
    different signal density.

    Strategy: use the 4th-root of inverse frequency instead of sqrt.  This
    still gives directional bars a voice but doesn't drown out the flat class.

    Example with train distribution (Short=25%, Flat=46%, Long=29%):
      v1 (sqrt):   batch → 38% Short, 28% Flat, 35% Long  (flat under-rep)
      v2 (4th-r):  batch → 33% Short, 35% Long, 32% Flat  (much closer to real)
    """
    # Assign class labels
    # 0: Short, 1: Flat, 2: Long
    classes = np.ones_like(targets, dtype=np.int32)  # Default to Flat
    classes[targets < -thresh] = 0
    classes[targets > thresh] = 2

    # Count samples per class
    counts = np.bincount(classes, minlength=3)
    # Avoid division by zero
    counts = np.maximum(counts, 1)

    # Inverse frequency weights with gentler compression (4th root vs sqrt)
    weights_per_class = 1.0 / counts

    if use_sqrt:
        # v2: 4th root instead of sqrt — preserves more natural distribution
        weights_per_class = weights_per_class ** 0.25

    # Holistic Bias Correction
    # Boost all classes relative to the majority class to equalize probability mass.
    # Formula: Weight_i = Weight_base_i * (Count_i / Count_maj) ** CorrectionPower
    if config is not None and hasattr(config, "BIAS_CORRECTION_POWER"):
        power = config.BIAS_CORRECTION_POWER
        if power != 0:
            maj_idx = np.argmax(counts)
            count_maj = counts[maj_idx]
            for i in range(3):
                if i != maj_idx:
                    weights_per_class[i] *= (counts[i] / count_maj) ** power

    # Map weights back to each sample
    sample_weights = weights_per_class[classes]

    return torch.from_numpy(sample_weights).float()


class DistributedWeightedSampler(torch.utils.data.Sampler):

    """
    Weighted random sampling that partitions correctly across DDP ranks.

    Key guarantee: all ranks draw from the SAME globally-weighted pool,
    but each rank gets a disjoint slice → no sample duplication across GPUs.

    set_epoch(epoch) must be called before each epoch (same as DistributedSampler).
    When world_size=1 / rank=0, behaviour is identical to WeightedRandomSampler.
    """
    def __init__(self, weights, num_samples, num_replicas=1,
                 rank=0, replacement=True, seed=42):
        self.weights             = weights.double()
        self.num_samples         = num_samples
        self.num_replicas        = num_replicas
        self.rank                = rank
        self.replacement         = replacement
        self.seed                = seed
        self.epoch               = 0
        # Pad total so every rank gets equal slices (required for DDP sync)
        self.num_samples_per_rank = math.ceil(num_samples / num_replicas)
        self.total_size          = self.num_samples_per_rank * num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)          # reproducible, epoch-varied
        indices = torch.multinomial(
            self.weights, self.total_size,
            replacement=self.replacement, generator=g
        ).tolist()
        # Each rank gets its own non-overlapping slice
        start = self.rank * self.num_samples_per_rank
        return iter(indices[start : start + self.num_samples_per_rank])

    def __len__(self):
        return self.num_samples_per_rank



def collate_with_none(batch):
    """
    Handles None in batches for multi-modal data.
    Each element in batch is (tokens, features, target).
    tokens is either (idx_c, idx_f) or None.

    With ORBIT randomization, context lengths vary within a batch: contexts are
    LEFT-padded to the batch max and a bool key_padding_mask (True = padded,
    nn.MultiheadAttention convention) is returned as the 4th element so
    attention can exclude padded positions. Fixed-length batches return
    pad_mask=None.
    """
    tokens_raw   = [b[0] for b in batch]
    features_raw = [b[1] for b in batch]
    targets_raw  = [b[2] for b in batch]

    targets = torch.stack(targets_raw)

    lengths = None
    if features_raw[0] is not None:
        lengths = [f.shape[0] for f in features_raw]
    elif tokens_raw[0] is not None:
        lengths = [t[0].shape[0] for t in tokens_raw]

    variable = lengths is not None and len(set(lengths)) > 1
    pad_mask = None
    if variable:
        B    = len(batch)
        Lmax = max(lengths)
        pad_mask = torch.zeros(B, Lmax, dtype=torch.bool)
        for b, L in enumerate(lengths):
            pad_mask[b, :Lmax - L] = True

    if features_raw[0] is not None:
        if variable:
            F_dim = features_raw[0].shape[-1]
            dtype = features_raw[0].dtype
            padded = torch.zeros(B, Lmax, F_dim, dtype=dtype)
            for b, f in enumerate(features_raw):
                padded[b, Lmax - f.shape[0]:] = f
            features = padded
        else:
            features = torch.stack(features_raw)
    else:
        features = None

    if tokens_raw[0] is not None:
        if variable:
            c_list, f_list = [], []
            for b, t in enumerate(tokens_raw):
                L = t[0].shape[0]
                c_pad = torch.zeros(Lmax - L, dtype=t[0].dtype)
                f_pad = torch.zeros(Lmax - L, dtype=t[1].dtype)
                c_list.append(torch.cat([c_pad, t[0]]))
                f_list.append(torch.cat([f_pad, t[1]]))
            tokens = (torch.stack(c_list), torch.stack(f_list))
        else:
            c = torch.stack([t[0] for t in tokens_raw])
            f = torch.stack([t[1] for t in tokens_raw])
            tokens = (c, f)
    else:
        tokens = None

    return tokens, features, targets, pad_mask


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def _worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def _make_loader(
    ds,
    config,
    sampler=None,
    shuffle:   bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Single factory so prefetch_factor / persistent_workers are consistent.

    drop_last
    ---------
    Set True for training loaders with a sampler (avoids a partial batch
    that would corrupt WeightedRandomSampler weight normalization).
    Always False for val/test loaders — every sample must be evaluated.
    """
    nw   = config.NUM_WORKERS
    pf   = getattr(config, "PREFETCH_FACTOR", 2) if nw > 0 else None

    if sampler is not None and shuffle:
        raise ValueError(
            "_make_loader: shuffle=True and sampler are mutually exclusive. "
            "The sampler controls draw order — do not pass shuffle=True."
        )

    generator = torch.Generator()
    generator.manual_seed(getattr(config, "SEED", 42))

    loader_kwargs = {
        "batch_size": config.BATCH_SIZE,
        "sampler": sampler,
        "shuffle": shuffle if sampler is None else False,
        "drop_last": drop_last,
        "num_workers": nw,
        "prefetch_factor": pf,
        "persistent_workers": (nw > 0),
        "pin_memory": True,
        "multiprocessing_context": "spawn" if nw > 0 else None,
        "collate_fn": collate_with_none,
        "worker_init_fn": _worker_init_fn,
        "generator": generator,
    }

    return DataLoader(ds, **loader_kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def create_dataloaders(
    features,
    targets,
    config,
    feature_cols: list[str],
    tokenizer=None,
    ohlc_returns: np.ndarray | None = None,
    rank=0,
    world_size=1,
):
    """Single-asset train/val/test split with a gap between each split.

    The scaler is fitted on the training slice only.
    """
    total_len  = len(features)
    gap        = config.FORECAST_HORIZON + 50

    train_end  = int(total_len * config.TRAIN_RATIO)
    val_start  = train_end + gap

    # ── CLAMP val_end before it can bleed past total_len ──────────────
    val_end_raw = val_start + int(total_len * config.VAL_RATIO)
    val_end     = min(val_end_raw, total_len - gap - config.LOOKBACK_WINDOW)

    # Early, meaningful failure — pinpoints the root cause
    if val_end <= val_start:
        raise ValueError(
            f"Val split is degenerate after clamping: val_start={val_start}, "
            f"val_end={val_end}. total_len={total_len} is too small for "
            f"TRAIN_RATIO={config.TRAIN_RATIO}, VAL_RATIO={config.VAL_RATIO}, "
            f"gap={gap}. Minimum required rows ≈ "
            f"{int((config.TRAIN_RATIO + config.VAL_RATIO) * total_len) + 2*gap + 3*config.LOOKBACK_WINDOW}."
        )

    test_start = val_end + gap
    # Guard: minimum viable length (needs at least seq_len rows to form 1 window)
    assert train_end >= config.LOOKBACK_WINDOW, (
        f"Train split too short for even one window: "
        f"train_end={train_end} < LOOKBACK_WINDOW={config.LOOKBACK_WINDOW}. "
        f"Reduce LOOKBACK_WINDOW or increase total data / TRAIN_RATIO."
    )
    assert (val_end - val_start)     >= config.LOOKBACK_WINDOW, (
        f"Val split too short for even one window: "
        f"{val_end - val_start} rows < LOOKBACK_WINDOW={config.LOOKBACK_WINDOW}"
    )
    assert (total_len - test_start)  >= config.LOOKBACK_WINDOW, (
        f"Test split too short for even one window: "
        f"{total_len - test_start} rows < LOOKBACK_WINDOW={config.LOOKBACK_WINDOW}"
    )

    scaler = fit_scaler(features[:train_end], feature_cols, config=config,
                        quiet=(rank != 0))

    # ── Tokenize full series ONCE, then slice per split ─────────────────
    input_mode = InputMode(getattr(config, "INPUT_MODE", "features_only"))
    tok_coarse_full = tok_fine_full = None

    tok_tr = tok_va = tok_te = (None, None)
    if input_mode in (InputMode.TOKENS_ONLY, InputMode.COMBINED):
        if ohlc_returns is None:
            raise ValueError("ohlc_returns required for token modes in create_dataloaders")
        total = len(features)
        (tok_tr, tok_va, tok_te) = tokenize_split_slices(
            ohlc_returns,
            tokenizer,
            config,
            [(0, train_end), (val_start, val_end), (test_start, total)],
        )

    train_ds = FinancialDataset(
        features[:train_end],        targets[:train_end],
        config.LOOKBACK_WINDOW,
        scaler=scaler, tokenizer=tokenizer, config=config,
        precomputed_coarse=tok_tr[0] if tok_tr[0] is not None else None,
        precomputed_fine=tok_tr[1]   if tok_tr[1] is not None else None,
        orbit_randomize=True,
    )
    val_ds = FinancialDataset(
        features[val_start:val_end], targets[val_start:val_end],
        config.LOOKBACK_WINDOW,
        scaler=scaler, tokenizer=tokenizer, config=config,
        precomputed_coarse=tok_va[0] if tok_va[0] is not None else None,
        precomputed_fine=tok_va[1]   if tok_va[1] is not None else None,
    )
    test_ds = FinancialDataset(
        features[test_start:],       targets[test_start:],
        config.LOOKBACK_WINDOW,
        scaler=scaler, tokenizer=tokenizer, config=config,
        precomputed_coarse=tok_te[0] if tok_te[0] is not None else None,
        precomputed_fine=tok_te[1]   if tok_te[1] is not None else None,
    )

    start_idx  = config.LOOKBACK_WINDOW - 1
    weight_end = start_idx + len(train_ds)
    assert weight_end <= train_end, (
        f"y_train_aligned would read beyond train boundary: "
        f"weight_end={weight_end} > train_end={train_end}."
    )
    y_train_aligned = targets[start_idx : weight_end]
    sample_weights  = _compute_sample_weights(y_train_aligned, config.SAMPLER_THRESHOLD, config=config)

    sampler = DistributedWeightedSampler(
        sample_weights, len(sample_weights),
        num_replicas=world_size, rank=rank,
        seed=getattr(config, "SEED", 42)
    )

    if world_size > 1:
        val_sampler  = DistributedSampler(val_ds,  num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        val_sampler = test_sampler = None
    return (
        _make_loader(train_ds, config, sampler=sampler,      drop_last=True),
        _make_loader(val_ds,   config, sampler=val_sampler,  drop_last=False),
        _make_loader(test_ds,  config, sampler=test_sampler, drop_last=False),
    )


def create_multi_index_dataloaders(
    asset_data_list: list[tuple],  # 5-tuple (asset_id, feat, targ, ohlc, train_end)
                                   # or 7-tuple (..., precomputed_coarse, precomputed_fine)
    config,
    feature_cols: list[str],
    tokenizer=None,
    is_train: bool = False,
    scalers: dict[str, ColumnSelectiveScaler] | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader | None, dict[str, ColumnSelectiveScaler]]:
    """Multi-asset DataLoader — each asset is scaled independently.
    
    Supports DistributedDataParallel (DDP) by using DistributedSampler.
    """
    datasets:       list[FinancialDataset] = []
    all_targets:    list[float]            = []
    fitted_scalers: dict[str, ColumnSelectiveScaler] = {}
    
    # ── Two-pass: first collect train features and fit global scaler ────────
    # Pass 1: collect train chunks + tokenize (tokenization is expensive, do once)
    # Then fit global scaler on concatenated train features.
    # Pass 2: create datasets with the fitted scaler applied.
    #
    # The global scaler brings all features to equal footing (median≈0, IQR≈1)
    # without destroying the cross-window DC signal (unlike per-window z-score).
    # Fit on train only — no data leakage into val/test.
    
    _train_feat_chunks: list[np.ndarray] = []
    _prepared: list[tuple] = []  # (asset_id, feat, targ, ohlc, tok_c, tok_f, is_train_asset, train_end)
    
    for _entry in asset_data_list:
        # Support both 5-tuple (legacy) and 7-tuple (with precomputed tokens)
        if len(_entry) == 7:
            asset_id, feat, targ, ohlc, train_end, pre_coarse, pre_fine = _entry
        else:
            asset_id, feat, targ, ohlc, train_end = _entry
            pre_coarse = pre_fine = None
        
        if len(feat) != len(targ):
            raise ValueError(
                f"Asset '{asset_id}': feature/target length mismatch — "
                f"len(feat)={len(feat)}, len(targ)={len(targ)}. "
                f"Both arrays must cover the same row indices.")
        
        if ohlc is not None and len(ohlc) != len(feat):
            raise ValueError(
                f"Asset '{asset_id}': OHLC/features length mismatch — "
                f"len(ohlc)={len(ohlc)}, len(feat)={len(feat)}. "
                f"Both arrays must cover the same row indices.")
        
        if len(feat) < config.LOOKBACK_WINDOW:
            continue
        
        is_train_asset = is_train  # all entries are train or all are val/test
        
        if is_train_asset:
            if train_end is None:
                raise ValueError(
                    f"Asset '{asset_id}': train_end is None but is_train=True. "
                    "Pass the actual train boundary index in the 4-tuple for training data.")
            _train_feat_chunks.append(feat[:train_end])
        
        # Tokenize once (expensive)
        tok_c, tok_f = pre_coarse, pre_fine
        _imode = getattr(config, "INPUT_MODE", "features_only")
        if tok_c is None and _imode in ("tokens_only", "combined") and tokenizer is not None:
            tok_source, tok_source_label = _select_tokenizer_input(
                asset_id, feat, ohlc, tokenizer, config
            )
            print(
                f"[{asset_id}] Tokenizer input: {tok_source_label} "
                f"({tok_source.shape[1]} columns, d_in={getattr(tokenizer, 'd_in', None)})"
            )
            tok_c, tok_f = tokenize_full_series(tok_source, tokenizer, config)
        
        _prepared.append((asset_id, feat, targ, ohlc, tok_c, tok_f, is_train_asset, train_end))
    
    # ── Fit global scaler on concatenated training features ──────────────────
    _global_scaler = None
    if is_train and _train_feat_chunks:
        _global_scaler = fit_scaler(
            np.concatenate(_train_feat_chunks, axis=0),
            feature_cols,
            config=config,
            quiet=(rank != 0),
        )
        fitted_scalers["__global__"] = _global_scaler
        if rank == 0:
            print(f"  [GlobalScaler] Fitted on {sum(len(c) for c in _train_feat_chunks):,} train rows")
    
    # For val/test, use the scaler passed in from the caller
    if not is_train and scalers is not None and "__global__" in scalers:
        _global_scaler = scalers["__global__"]
    
    # ── Pass 2: create datasets with global scaler applied ───────────────────
    datasets:       list[FinancialDataset] = []
    all_targets:    list[float]            = []
    
    for (asset_id, feat, targ, ohlc, tok_c, tok_f, is_train_asset, train_end) in _prepared:
        # Apply global scaler (fit on train, applied to all splits)
        if _global_scaler is not None:
            feat = _global_scaler.transform(feat).astype(np.float32)
        
        ds = FinancialDataset(
            feat, targ, config.LOOKBACK_WINDOW,
            ohlc_returns=ohlc,
            scaler=None, tokenizer=tokenizer, config=config,
            precomputed_coarse=tok_c, precomputed_fine=tok_f,
            orbit_randomize=bool(is_train),
        )
        
        datasets.append(ds)
        
        if is_train_asset:
            start = config.LOOKBACK_WINDOW - 1
            all_targets.extend(targ[start : start + len(ds)].tolist())
    
    if not datasets:
        return None, fitted_scalers
    
    full_ds = ConcatDataset(datasets)
    
    if is_train:
        all_targets_arr = np.array(all_targets, dtype=np.float32)
        sample_weights  = _compute_sample_weights(
            all_targets_arr, config.SAMPLER_THRESHOLD, config=config
        )
        assert len(sample_weights) == len(full_ds), (
            f"Multi-index weight mismatch: "
            f"weights={len(sample_weights)}, ds={len(full_ds)}")
        
        sampler = DistributedWeightedSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            num_replicas=world_size,   # =1 when single GPU → identical to old behaviour
            rank=rank,
            replacement=True,
            seed=getattr(config, "SEED", 42),
        )
        return _make_loader(full_ds, config, sampler=sampler, drop_last=True), fitted_scalers
    else:
        if world_size > 1:
            sampler = DistributedSampler(
                full_ds, 
                num_replicas=world_size, 
                rank=rank, 
                shuffle=False
            )
            return _make_loader(full_ds, config, sampler=sampler, drop_last=False), fitted_scalers
        return _make_loader(full_ds, config,                  drop_last=False), fitted_scalers



# ─────────────────────────────────────────────────────────────────────────────
# ORBIT stream (arXiv:2608.13262) — Omni-Range Incremental Training consumption
# ─────────────────────────────────────────────────────────────────────────────

class OrbitStreamDataset(Dataset):
    """Map-style view over one DDP shard of the ORBIT global training stream.

    Stream slot j resolves to a five-tuple (record, variable, s, e, p) via
        d = slot_dataset[j];  local_id = perm[d][slot_rank[j]]
        (s, e, h_idx) = index[d][local_id]
    and yields (tokens=None, features[s:e], target_h[e-1]).

    Contexts have variable length (omni-range); collate_with_none left-pads
    them and emits the attention mask. The horizon axis selects a column of
    the per-asset multi-horizon oracle label matrix built offline.
    """

    def __init__(
        self,
        features_list: list["torch.Tensor"],      # per asset (L_d, F), scaled
        target_matrix_list: list["torch.Tensor"],  # per asset (L_d, H)
        horizons: list[int],
        slot_dataset: np.ndarray,
        slot_rank: np.ndarray,
        index: list[np.ndarray],                   # per asset (M_d, 3): [s, e, h_idx]
        perm: list[np.ndarray],
    ) -> None:
        self.features = features_list
        self.targets = target_matrix_list
        self.horizons = list(horizons)
        self.slot_dataset = slot_dataset
        self.slot_rank = slot_rank
        self.index = index
        self.perm = perm

    def __len__(self) -> int:
        return len(self.slot_dataset)

    def __getitem__(self, i: int):
        d = int(self.slot_dataset[i])
        r = int(self.slot_rank[i])
        s, e, h_idx = self.index[d][self.perm[d][r]]
        feat = self.features[d][int(s):int(e)]
        tgt = self.targets[d][int(e) - 1, int(h_idx)]
        return None, feat, tgt


def create_orbit_pretrain_loader(
    pretrain_list: list[tuple],   # (asset_id, feat, targ, ohlc, train_end)
    targets_by_h_list: list[dict],
    config,
    feature_cols: list[str],
    rank: int = 0,
    world_size: int = 1,
):
    """Build the ORBIT pretrain DataLoader.

    Replaces ConcatDataset + DistributedWeightedSampler + epochs with a single
    incremental pass over a globally blended stream (paper §4). Each rank gets
    one contiguous, disjoint shard so no sample is duplicated across GPUs.

    Returns (DataLoader, fitted_scalers).
    """
    import orbit_sampler

    input_mode = str(getattr(config, "INPUT_MODE", "features_only"))
    if input_mode != "features_only":
        raise NotImplementedError(
            "create_orbit_pretrain_loader currently supports INPUT_MODE="
            "'features_only' only."
        )

    horizons = sorted(int(h) for h in getattr(config, "ORBIT_HORIZONS", []))
    if not horizons:
        raise ValueError("ORBIT_STREAM_MODE=True requires ORBIT_HORIZONS.")
    ctx_min = max(1, int(getattr(config, "ORBIT_CTX_MIN", 1)))
    max_ctx = int(getattr(config, "LOOKBACK_WINDOW", config.LOOKBACK_WINDOW))
    seed = int(getattr(config, "ORBIT_INDEX_SEED", 42))

    # ── Global scaler on train chunks (identical to legacy pass 1) ──────────
    _chunks = [feat[:end] for _, feat, _, _, end in pretrain_list]
    _global_scaler = fit_scaler(
        np.concatenate(_chunks, axis=0), feature_cols, config=config,
        quiet=(rank != 0),
    )
    fitted_scalers = {"__global__": _global_scaler}

    # ── Scale features; build scaled tensors + multi-horizon label matrices ──
    features_list: list = []
    target_mats: list = []
    lengths: list[int] = []
    asset_ids: list[str] = []
    for (asset_id, feat, _targ, _ohlc, _end), by_h in zip(pretrain_list, targets_by_h_list):
        missing = [h for h in horizons if h not in by_h]
        if missing:
            raise ValueError(
                f"Asset '{asset_id}': multi-horizon labels missing for "
                f"horizons {missing}. process_dataset must generate them "
                f"when ORBIT_STREAM_MODE is enabled."
            )
        scaled = _global_scaler.transform(feat).astype(np.float32)
        features_list.append(torch.from_numpy(scaled))
        target_mats.append(torch.from_numpy(
            np.stack([by_h[h] for h in horizons], axis=1).astype(np.float32)
        ))
        lengths.append(len(feat))
        asset_ids.append(os.path.basename(str(asset_id)))

    # ── Prescribed dataset weights (domain-aware balancing) ─────────────────
    w_cfg = getattr(config, "ORBIT_ASSET_WEIGHTS", None)
    if w_cfg:
        unknown = [a for a in asset_ids if a not in w_cfg]
        if unknown:
            raise ValueError(f"ORBIT_ASSET_WEIGHTS missing assets: {unknown}")
        weights = [float(w_cfg[a]) for a in asset_ids]
    else:
        weights = [1.0] * len(asset_ids)

    # ── Offline bootstrap index: build once (rank 0 caches), all ranks reuse ──
    total_global = (
        int(getattr(config, "ORBIT_TOTAL_STEPS", 20000))
        * int(config.BATCH_SIZE) * world_size
    )
    allocs = orbit_sampler.allocation_counts(weights, total_global)
    h = orbit_sampler.state_hash(asset_ids, lengths, horizons, ctx_min, max_ctx, seed)
    cache_path = getattr(config, "ORBIT_INDEX_CACHE", None)
    index = orbit_sampler.load_index_cache(cache_path, h) if cache_path else None
    if index is None or len(index) != len(lengths):
        rng = np.random.default_rng(seed)
        index = orbit_sampler.build_bootstrap_index(
            lengths, horizons, allocs, ctx_min, max_ctx, rng,
        )
        if rank == 0 and cache_path:
            orbit_sampler.save_index_cache(cache_path, h, index)
    if any(len(ix) < a for ix, a in zip(index, allocs)):
        raise RuntimeError(
            "Bootstrap index smaller than stream allocation — stream slots "
            "would exhaust their dataset's samples."
        )

    # ── Blend + contiguous DDP shard ────────────────────────────────────────
    rng = np.random.default_rng(seed + 1)
    stream = orbit_sampler.build_orbit_stream(index, weights, total_global, rng)
    per_rank = total_global // world_size
    lo = rank * per_rank
    hi = lo + per_rank if rank < world_size - 1 else total_global

    ds = OrbitStreamDataset(
        features_list, target_mats, horizons,
        stream["slot_dataset"][lo:hi],
        stream["slot_rank"][lo:hi],
        index, stream["perm"],
    )

    if rank == 0:
        share = np.bincount(ds.slot_dataset, minlength=len(asset_ids)) / len(ds)
        mix_str = ", ".join(f"{a}: {p:.3f}" for a, p in zip(asset_ids, share))
        print(
            f"  [ORBIT] stream={total_global:,} slots | steps/rank="
            f"{per_rank // config.BATCH_SIZE:,} | horizons={horizons} | "
            f"ctx∈[{ctx_min},{max_ctx}] | shares: {mix_str}"
        )

    loader = _make_loader(ds, config, sampler=None, shuffle=False, drop_last=True)
    return loader, fitted_scalers


def create_fold_dataloaders(
    features,
    targets,
    train_indices: tuple[int, int],
    val_indices:   tuple[int, int],
    test_indices:  tuple[int, int],
    config,
    feature_cols: list[str],
    tokenizer=None,
    ohlc_returns: np.ndarray | None = None,
    rank=0,
    world_size=1,
):
    """Walk-forward fold dataloaders.

    IMPORTANT: `features` and `targets` must be GLOBAL (unsliced) arrays.
    `train_indices`, `val_indices`, `test_indices` are absolute row indices
    into these global arrays. Do NOT pre-slice before calling.

    Scaler fitted on train_indices slice only — no leakage into val/test.
    """
    # Guard: indices must be within bounds
    assert train_indices[0] >= 0 and train_indices[1] <= len(features), (
        f"train_indices {train_indices} out of bounds for features of length {len(features)}"
    )

    train_feat = features[train_indices[0] : train_indices[1]]
    scaler     = fit_scaler(train_feat, feature_cols, config=config,
                            quiet=(rank != 0))

    ts, te = train_indices
    vs, ve = val_indices
    xs, xe = test_indices

    tc_tr = tf_tr = tc_va = tf_va = tc_te = tf_te = None
    _imode = getattr(config, "INPUT_MODE", "features_only")
    if _imode in ("tokens_only", "combined") and features is not None:
        # Wrap tokenizer to strictly fit on the training slice and transform full series
        tok = FittedTokenizer(tokenizer, config)
        tok.fit(features[:te])
        coarse_full, fine_full = tok.transform(features)
        
        tc_tr, tf_tr = coarse_full[ts:te], fine_full[ts:te]
        tc_va, tf_va = coarse_full[vs:ve], fine_full[vs:ve]
        tc_te, tf_te = coarse_full[xs:xe], fine_full[xs:xe]

    train_ds = FinancialDataset(
        train_feat,
        targets[ts:te],
        config.LOOKBACK_WINDOW,
        scaler=scaler, tokenizer=tokenizer, config=config,
        precomputed_coarse=tc_tr, precomputed_fine=tf_tr,
        orbit_randomize=True,
    )
    val_ds = FinancialDataset(
        features[vs:ve],
        targets[vs:ve],
        config.LOOKBACK_WINDOW,
        scaler=scaler, tokenizer=tokenizer, config=config,
        precomputed_coarse=tc_va, precomputed_fine=tf_va,
    )
    test_ds = FinancialDataset(
        features[xs:xe],
        targets[xs:xe],
        config.LOOKBACK_WINDOW,
        scaler=scaler, tokenizer=tokenizer, config=config,
        precomputed_coarse=tc_te, precomputed_fine=tf_te,
    )

    # This offset is correct ONLY for global arrays.
    # Logic: FinancialDataset windows start at [0:seq_len], predicting at [seq_len-1].
    # Thus, the first target is at train_indices[0] + LOOKBACK_WINDOW - 1.
    start_idx       = train_indices[0] + config.LOOKBACK_WINDOW - 1
    y_train_aligned = targets[start_idx : start_idx + len(train_ds)]

    # Stronger assert: also verify the slice is not empty
    assert len(y_train_aligned) == len(train_ds), (
        f"Weight misalignment: got {len(y_train_aligned)} targets "
        f"for {len(train_ds)} samples. Check that features/targets are "
        f"global (unsliced) arrays and train_indices are absolute."
    )

    sample_weights  = _compute_sample_weights(
        y_train_aligned, config.SAMPLER_THRESHOLD, config=config
    )
    sampler = DistributedWeightedSampler(
        sample_weights, len(sample_weights),
        num_replicas=world_size, rank=rank,
        seed=getattr(config, "SEED", 42)
    )

    if world_size > 1:
        val_sampler  = DistributedSampler(val_ds,  num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        val_sampler = test_sampler = None
    return (
        _make_loader(train_ds, config, sampler=sampler,      drop_last=True),
        _make_loader(val_ds,   config, sampler=val_sampler,  drop_last=False),
        _make_loader(test_ds,  config, sampler=test_sampler, drop_last=False),
    )
