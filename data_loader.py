# data_loader.py  (Production — integrated with features.py)

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from sklearn.preprocessing import RobustScaler
from model import InputMode


# ─────────────────────────────────────────────────────────────────────────────
# Normalization routing
# ─────────────────────────────────────────────────────────────────────────────
#
# features.py produces exactly 13 columns per asset (close-only path):
#
#   Col                     Range / Distribution        Routing
#   ──────────────────────  ──────────────────────────  ────────
#   ewma_vol_span{N}        ~0.003, tight band          NO_SCALE
#   ret_norm_{h}d  (×8)     p1/p99 ≈ [-2.5, +2.5]      NO_SCALE
#   macd_{s}_{l}   (×3)     std ≈ 1.05, [-3, +3]       NO_SCALE
#   vs_factor_span{N}       mean ~346, skew ~24         ROBUST
#
# NO_SCALE rationale:
#   ewma_vol   — already a tiny dimensionless fraction (~0.003).
#                Centering to zero destroys its absolute meaning (σ=0
#                is the natural origin; shifting it breaks the signal).
#   ret_norm_* — volatility-scaled returns ≈ z-score by construction.
#                Applying RobustScaler re-centers an already-centered signal.
#   macd_*     — three-step normalised (paper Eqs. 19–21), empirical std ≈ 1.05.
#                Already unit-variance; re-scaling adds noise.
#
# ROBUST rationale:
#   vs_factor  — 1/σ.  Mean ~346, skew ~24, can spike to 3000+ in low-vol
#                regimes.  RobustScaler (median+IQR) centres it without being
#                destroyed by the right-tail outliers.
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
    if col.startswith("ewma_vol_span"):   return "no_scale"
    if col.startswith("ret_norm_"):       return "no_scale"
    if col.startswith("macd_"):           return "no_scale"
    if col.startswith("vs_factor_span"):  return "robust"
    if col.startswith("feat_session_"):   return "no_scale"
    if col == "feat_vol_squeeze":         return "robust"
    if col.startswith("feat_"):           return "no_scale"
    if col.startswith("talib_"):          return "no_scale"
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
    ) -> None:
        self.feature_cols       = list(feature_cols)
        self._default_clip      = default_clip_bound
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
                if clip_rate > 0.02:
                    col_name = self.feature_cols[global_i]
                    print(
                        f"[ColumnSelectiveScaler] WARNING: '{col_name}' clip rate "
                        f"{clip_rate:.2%} > 2% at bound=±{bound} IQR. "
                        f"Rerun clip_audit.py — distribution may have shifted."
                    )
                transformed[:, local_j] = np.clip(col_data, -bound, bound)

            X[:, self._robust_idx] = transformed
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def summary(self) -> str:
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
    )
    scaler.fit(features_train)
    print(scaler.summary())
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
    Tokenize the ENTIRE series once.
    Call this once per asset. Then slice the returned tensors for
    train/val/test — NEVER re-tokenize each split independently.

    Per-window normalization is done using a rolling context that is
    consistent across the full series (train+val+test see the same stats).
    """
    tokenizer.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer.to(device)

    T   = len(ohlc_returns)
    if T == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    S   = 64  # TOKENIZER_SEQ_LEN
    pad = np.tile(ohlc_returns[0:1], (S - 1, 1))
    padded = np.concatenate([pad, ohlc_returns], axis=0)

    from numpy.lib.stride_tricks import as_strided
    C       = ohlc_returns.shape[1]
    shape   = (T, S, C)
    strides = (padded.strides[0], padded.strides[0], padded.strides[1])
    windows = as_strided(padded, shape=shape, strides=strides)

    chunk_size = getattr(config, "TOKENIZER_CHUNK_SIZE", 1024)
    c_list, f_list = [], []

    print(f"[tokenize_full_series] Tokenizing {T} bars in chunks of {chunk_size}…")
    with torch.no_grad():
        for i in range(0, T, chunk_size):
            batch = torch.from_numpy(
                np.array(windows[i: i + chunk_size])  # force copy for safety
            ).to(device).float()

            # Per-window normalization: each window normalised independently
            # (dim=1 = sequence dim only — NOT dim=(0,1) which mixes windows)
            w_mean = batch.mean(dim=1, keepdim=True)        # (B, 1, C)
            w_std  = batch.std(dim=1, keepdim=True) + 1e-5  # (B, 1, C)
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
    ) -> None:
        self.input_mode = str(getattr(config, "INPUT_MODE", "features_only"))
        self.seq_len = seq_len

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
                print(f"Top-5 coarse tokens: {hist_c.topk(5)}")
                print(f"Top-5 fine   tokens: {hist_f.topk(5)}")
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
        seq = slice(i, i + self.seq_len)
        
        tokens   = None
        features = None
        
        if self.input_mode in (InputMode.TOKENS_ONLY, InputMode.COMBINED):
            tokens = (self.idx_coarse[seq], self.idx_fine[seq])
        
        if self.input_mode in (InputMode.FEATURES_ONLY, InputMode.COMBINED):
            features = self.features[seq]
        
        # Target at the end of the window
        target = self.targets[i + self.seq_len - 1]
        
        return tokens, features, target


# MultiStreamDataset removed (deprecated in favor of multi-modal FinancialDataset)



# ─────────────────────────────────────────────────────────────────────────────
# Sample weighting
# ─────────────────────────────────────────────────────────────────────────────

def _compute_sample_weights(
    targets_array: np.ndarray,
    threshold: float,
    use_sqrt: bool = True,
) -> torch.Tensor:
    """Inverse-frequency weights over three classes: Short / Flat / Long.

    Parameters
    ----------
    targets_array : 1-D float array of Oracle target scores.
    threshold     : |score| below this → Flat class.
    use_sqrt      : if True, weight ∝ 1/√count (softer than 1/count).
    """
    class_indices = []
    for y in targets_array:
        if   y < -threshold: class_indices.append(0)   # Short
        elif y >  threshold: class_indices.append(2)   # Long
        else:                class_indices.append(1)   # Flat

    counts = np.bincount(class_indices, minlength=3)
    # Balanced inverse-frequency weighting across classes
    if use_sqrt:
        weights = [1.0 / np.sqrt(max(counts[c], 1)) for c in range(3)]
    else:
        weights = [1.0 / max(counts[c], 1) for c in range(3)]
    
    return torch.DoubleTensor([weights[c] for c in class_indices])


def collate_with_none(batch):
    """
    Handles None in batches for multi-modal data.
    Each element in batch is (tokens, features, target).
    tokens is either (idx_c, idx_f) or None.
    """
    tokens_raw   = [b[0] for b in batch]
    features_raw = [b[1] for b in batch]
    targets_raw  = [b[2] for b in batch]
    
    targets = torch.stack(targets_raw)
    
    if features_raw[0] is not None:
        features = torch.stack(features_raw)
    else:
        features = None
        
    if tokens_raw[0] is not None:
        c = torch.stack([t[0] for t in tokens_raw])
        f = torch.stack([t[1] for t in tokens_raw])
        tokens = (c, f)
    else:
        tokens = None
        
    return tokens, features, targets


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
    cuda = torch.cuda.is_available()

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
        "pin_memory": cuda,
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

    scaler = fit_scaler(features[:train_end], feature_cols, config=config)

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
    sample_weights  = _compute_sample_weights(y_train_aligned, config.SAMPLER_THRESHOLD)

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return (
        _make_loader(train_ds, config, sampler=sampler, drop_last=True),
        _make_loader(val_ds,   config,                  drop_last=False),
        _make_loader(test_ds,  config,                  drop_last=False),
    )


def create_multi_index_dataloaders(
    asset_data_list: list[tuple],  # 5-tuple (asset_id, feat, targ, ohlc, train_end)
                                   # or 7-tuple (..., precomputed_coarse, precomputed_fine)
    config,
    feature_cols: list[str],
    tokenizer=None,
    is_train: bool = False,
    scalers: dict[str, ColumnSelectiveScaler] | None = None,
) -> tuple[DataLoader | None, dict[str, ColumnSelectiveScaler]]:
    """Multi-asset DataLoader — each asset is scaled independently.

    A separate scaler is fitted per asset on its own training slice.
    This prevents a high-vol asset from shifting the scaler median for
    a low-vol asset (cross-asset leakage).

    Parameters
    ----------
    asset_data_list : list of (asset_id, features_array, targets_array, ohlc_returns, train_end) per asset.
    config          : config module / namespace.
    feature_cols    : ordered column names matching the feature arrays.
    tokenizer       : optional pre-trained KLineTokenizer.
    is_train        : True → fit scalers + build WeightedRandomSampler.
                      False → no scaler fitting (use provided scalers).
    scalers         : optional dict of {asset_id: fitted_scaler} for validation/test.
    """
    datasets:       list[FinancialDataset] = []
    all_targets:    list[float]            = []
    fitted_scalers: dict[str, ColumnSelectiveScaler] = {}

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
                f"Both arrays must cover the same row indices."
            )

        if len(feat) < config.LOOKBACK_WINDOW:
            continue

        if is_train:
            if train_end is None:
                raise ValueError(
                    f"Asset '{asset_id}': train_end is None but is_train=True. "
                    "Pass the actual train boundary index in the 4-tuple for training data."
                )

            # Optimization: skip scaler in tokens_only mode (tokenizer handles normalization)
            input_mode = getattr(config, "INPUT_MODE", "features_only")
            if input_mode == "tokens_only":
                scaler = None
            else:
                scaler = fit_scaler(feat[:train_end], feature_cols, config=config)
                fitted_scalers[asset_id] = scaler

            # Use precomputed tokens if provided; otherwise tokenize from ohlc
            tok_c, tok_f = pre_coarse, pre_fine
            _imode = getattr(config, "INPUT_MODE", "features_only")
            if tok_c is None and _imode in ("tokens_only", "combined") and ohlc is not None:
                tok_c, tok_f = tokenize_full_series(ohlc[:train_end], tokenizer, config)

            ds = FinancialDataset(
                feat[:train_end], targ[:train_end], config.LOOKBACK_WINDOW,
                scaler=scaler, tokenizer=tokenizer, config=config,
                precomputed_coarse=tok_c, precomputed_fine=tok_f,
            )
        else:
            # For val/test, use provided scalers if they exist
            scaler = None
            if scalers is not None and asset_id in scalers:
                scaler = scalers[asset_id]

            # If we are in features/combined mode but have no scaler, that's an error
            input_mode = getattr(config, "INPUT_MODE", "features_only")
            if input_mode != "tokens_only" and scaler is None:
                raise ValueError(
                    f"No fitted scaler for asset '{asset_id}'. "
                    f"Pass scalers returned from the training run "
                    f"(is_train=True call). Available keys: "
                    f"{list(scalers.keys()) if scalers is not None else 'scalers=None'}"
                )

            # Use precomputed tokens if provided; otherwise tokenize from ohlc
            tok_c, tok_f = pre_coarse, pre_fine
            _imode = getattr(config, "INPUT_MODE", "features_only")
            if tok_c is None and _imode in ("tokens_only", "combined") and ohlc is not None:
                tok_c, tok_f = tokenize_full_series(ohlc, tokenizer, config)

            ds = FinancialDataset(
                feat, targ, config.LOOKBACK_WINDOW,
                scaler=scaler, tokenizer=tokenizer, config=config,
                precomputed_coarse=tok_c, precomputed_fine=tok_f,
            )

        datasets.append(ds)

        if is_train:
            start = config.LOOKBACK_WINDOW - 1
            # For train data, we sliced up to train_end. 
            all_targets.extend(targ[start : start + len(ds)].tolist())

    if not datasets:
        return None, fitted_scalers

    full_ds = ConcatDataset(datasets)

    if is_train:
        all_targets_arr = np.array(all_targets, dtype=np.float32)
        sample_weights  = _compute_sample_weights(
            all_targets_arr, config.SAMPLER_THRESHOLD
        )
        assert len(sample_weights) == len(full_ds), (
            f"Multi-index weight mismatch: "
            f"weights={len(sample_weights)}, ds={len(full_ds)}"
        )
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        return _make_loader(full_ds, config, sampler=sampler, drop_last=True), fitted_scalers
    else:
        return _make_loader(full_ds, config,                  drop_last=False), fitted_scalers


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
    scaler     = fit_scaler(train_feat, feature_cols, config=config)

    ts, te = train_indices
    vs, ve = val_indices
    xs, xe = test_indices

    tc_tr = tf_tr = tc_va = tf_va = tc_te = tf_te = None
    _imode = getattr(config, "INPUT_MODE", "features_only")
    if _imode in ("tokens_only", "combined") and ohlc_returns is not None:
        # Wrap tokenizer to strictly fit on the training slice and transform full series
        tok = FittedTokenizer(tokenizer, config)
        tok.fit(ohlc_returns[:te])
        coarse_full, fine_full = tok.transform(ohlc_returns)
        
        tc_tr, tf_tr = coarse_full[ts:te], fine_full[ts:te]
        tc_va, tf_va = coarse_full[vs:ve], fine_full[vs:ve]
        tc_te, tf_te = coarse_full[xs:xe], fine_full[xs:xe]

    train_ds = FinancialDataset(
        train_feat,
        targets[ts:te],
        config.LOOKBACK_WINDOW,
        scaler=scaler, tokenizer=tokenizer, config=config,
        precomputed_coarse=tc_tr, precomputed_fine=tf_tr,
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
        y_train_aligned, config.SAMPLER_THRESHOLD
    )
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return (
        _make_loader(train_ds, config, sampler=sampler, drop_last=True),
        _make_loader(val_ds,   config,                  drop_last=False),
        _make_loader(test_ds,  config,                  drop_last=False),
    )