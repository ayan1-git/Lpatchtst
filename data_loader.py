# data_loader.py  (Production — integrated with features.py)

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import RobustScaler


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
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FinancialDataset(Dataset):
    """Windowed time-series dataset.

    The scaler is pre-fitted on the training split and passed in —
    no per-window statistics are computed here.

    If a tokenizer is provided, the full feature sequence is tokenised
    once at construction time and stored; __getitem__ returns token
    windows rather than raw feature windows.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets:  np.ndarray,
        seq_len:  int,
        scaler:   ColumnSelectiveScaler | None = None,
        tokenizer = None,
    ) -> None:
        # Guard: alignment
        if len(features) != len(targets):
            raise ValueError(
                f"FinancialDataset: len(features)={len(features)} != "
                f"len(targets)={len(targets)}. Arrays must be row-aligned."
            )

        # Guard: NaN detection (Early failure is better than NaN loss)
        nan_count = np.isnan(features).sum()
        if nan_count > 0:
            raise ValueError(
                f"FinancialDataset: features contain {nan_count} NaN values. "
                "The training pipeline does not support NaN inputs. "
                "Ensure FeatureEngineer.build(..., dropna=True) was used."
            )

        # Guard: must be able to form at least one window
        n_windows = len(features) - seq_len + 1
        if n_windows <= 0:
            raise ValueError(
                f"FinancialDataset: features has {len(features)} rows but "
                f"seq_len={seq_len} requires at least {seq_len} rows to form "
                f"one window. Got n_windows={n_windows}. "
                f"Check split indices or reduce LOOKBACK_WINDOW."
            )

        if scaler is not None:
            print("Applying normalization to features…")
            features = scaler.transform(features).astype(np.float32)
        else:
            print("Warning: No scaler provided — features will be used in their raw state.")

        # ── Diagnostic: Feature Matrix Statistics ─────────────────────────────
        print("\n" + "="*85)
        print(f"{'FEATURE MATRIX STATISTICS (After Normalization)':^85}")
        print("-" * 85)
        print(f"{'Column':<30} | {'Mean':>10} | {'Std':>10} | {'Min':>10} | {'Max':>10}")
        print("-" * 85)

        col_names = getattr(scaler, "feature_cols", [])
        for i in range(features.shape[1]):
            col_data = features[:, i]
            c_name = col_names[i] if i < len(col_names) else f"Col_{i}"
            print(f"{c_name[:30]:<30} | {np.mean(col_data):10.4f} | {np.std(col_data):10.4f} | {np.min(col_data):10.4f} | {np.max(col_data):10.4f}")
        print("="*85 + "\n")

        # Store as CPU tensor — workers can only access CPU memory.
        # .to(device) happens in the training loop (train_fold).
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets  = torch.tensor(targets,  dtype=torch.float32)
        self.seq_len  = seq_len
        self.tokens   = None

        if tokenizer is not None:
            print(f"Pre-tokenising dataset ({len(features)} rows)…")
            was_training = tokenizer.training        # capture state before any mutation
            tokenizer.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer.to(device)

            try:
                with torch.no_grad():
                    feat_tensor = torch.tensor(self.features, dtype=torch.float32)
                    chunk_size  = getattr(config, "TOKENIZER_CHUNK_SIZE", 4096)
                    token_chunks: list[torch.Tensor] = []

                    for start in range(0, len(feat_tensor), chunk_size):
                        chunk = feat_tensor[start : start + chunk_size].to(device)
                        # (1, chunk_size, F) → encode → (1, chunk_size) → (chunk_size,)
                        # This preserves temporal structure across the chunk.
                        # Temporal continuity is broken only at chunk boundaries.
                        toks  = tokenizer.encode(chunk.unsqueeze(0)).squeeze(0)
                        token_chunks.append(toks.cpu())

                    self.tokens = torch.cat(token_chunks, dim=0)   # (N,)
            finally:
                tokenizer.to("cpu")
                if was_training:
                    tokenizer.train()                # always restore, even if encode() raised
            print("Tokenisation complete.")

    def __len__(self) -> int:
        # Safe: constructor already guaranteed len(features) >= seq_len
        return len(self.features) - self.seq_len + 1

    def __getitem__(self, idx: int):
        if self.tokens is not None:
            x = self.tokens[idx : idx + self.seq_len]
        else:
            x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len - 1]
        return x, y


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

    counts  = np.bincount(class_indices, minlength=3)
    weights = (
        1.0 / (np.sqrt(counts) + 1e-6)
        if use_sqrt
        else 1.0 / (counts + 1e-6)
    )
    return torch.DoubleTensor([weights[c] for c in class_indices])


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    }

    # pin_memory_device available since PyTorch 2.1
    if cuda:
        try:
            v = torch.__version__.split(".")
            major, minor = int(v[0]), int(v[1])
            if (major > 2) or (major == 2 and minor >= 1):
                loader_kwargs["pin_memory_device"] = "cuda"
        except (ValueError, IndexError):
            pass

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

    train_ds = FinancialDataset(
        features[:train_end],        targets[:train_end],
        config.LOOKBACK_WINDOW, scaler=scaler, tokenizer=tokenizer,
    )
    val_ds = FinancialDataset(
        features[val_start:val_end], targets[val_start:val_end],
        config.LOOKBACK_WINDOW, scaler=scaler, tokenizer=tokenizer,
    )
    test_ds = FinancialDataset(
        features[test_start:],       targets[test_start:],
        config.LOOKBACK_WINDOW, scaler=scaler, tokenizer=tokenizer,
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
        num_samples=len(sample_weights) // 2,
        replacement=True,
    )

    return (
        _make_loader(train_ds, config, sampler=sampler, drop_last=True),
        _make_loader(val_ds,   config,                  drop_last=False),
        _make_loader(test_ds,  config,                  drop_last=False),
    )


def create_multi_index_dataloaders(
    asset_data_list: list[tuple[str, np.ndarray, np.ndarray, int]],
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
    asset_data_list : list of (asset_id, features_array, targets_array, train_end) per asset.
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

    for asset_id, feat, targ, train_end in asset_data_list:
        if len(feat) != len(targ):
            raise ValueError(
                f"Asset '{asset_id}': feature/target length mismatch — "
                f"len(feat)={len(feat)}, len(targ)={len(targ)}. "
                f"Both arrays must cover the same row indices."
            )

        if len(feat) < config.LOOKBACK_WINDOW:
            continue

        if is_train:
            scaler = fit_scaler(feat[:train_end], feature_cols, config=config)
            fitted_scalers[asset_id] = scaler
            ds = FinancialDataset(
                feat[:train_end], targ[:train_end], config.LOOKBACK_WINDOW,
                scaler=scaler, tokenizer=tokenizer,
            )
        else:
            if scalers is None or asset_id not in scalers:
                raise ValueError(
                    f"No fitted scaler for asset '{asset_id}'. "
                    f"Pass scalers returned from the training run "
                    f"(is_train=True call). Available keys: "
                    f"{list(scalers.keys()) if scalers is not None else 'scalers=None'}"
                )
            scaler = scalers[asset_id]
            ds = FinancialDataset(
                feat, targ, config.LOOKBACK_WINDOW,
                scaler=scaler, tokenizer=tokenizer,
            )

        datasets.append(ds)

        if is_train:
            start = config.LOOKBACK_WINDOW - 1
            # For train data, we sliced up to train_end. 
            all_targets.extend(targ[start : start + len(ds)].tolist())

    if not datasets:
        return None, fitted_scalers

    full_ds = torch.utils.data.ConcatDataset(datasets)

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
            num_samples=len(sample_weights) // 2,
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

    train_ds = FinancialDataset(
        train_feat,
        targets[train_indices[0] : train_indices[1]],
        config.LOOKBACK_WINDOW, scaler=scaler, tokenizer=tokenizer,
    )
    val_ds = FinancialDataset(
        features[val_indices[0]  : val_indices[1]],
        targets[val_indices[0]   : val_indices[1]],
        config.LOOKBACK_WINDOW, scaler=scaler, tokenizer=tokenizer,
    )
    test_ds = FinancialDataset(
        features[test_indices[0] : test_indices[1]],
        targets[test_indices[0]  : test_indices[1]],
        config.LOOKBACK_WINDOW, scaler=scaler, tokenizer=tokenizer,
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
        num_samples=len(sample_weights) // 2,
        replacement=True,
    )

    return (
        _make_loader(train_ds, config, sampler=sampler, drop_last=True),
        _make_loader(val_ds,   config,                  drop_last=False),
        _make_loader(test_ds,  config,                  drop_last=False),
    )