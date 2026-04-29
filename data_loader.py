import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import RobustScaler, MinMaxScaler  # FIX: MinMaxScaler was missing


class FinancialDataset(Dataset):
    """
    Windowed time-series dataset.
    Scaler is pre-fitted on the training split and passed in — no per-window stats.
    """

    def __init__(self, features, targets, seq_len, scaler=None, tokenizer=None):
        if scaler is not None:
            features = scaler.transform(features).astype(np.float32)

        self.features = torch.FloatTensor(features)
        self.targets  = torch.FloatTensor(targets)
        self.seq_len  = seq_len
        self.tokens   = None

        if tokenizer is not None:
            print(f"Pre-calculating tokens for dataset (size: {len(features)})...")
            tokenizer.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer.to(device)
            with torch.no_grad():
                feat_tensor = self.features.to(device)
                all_tokens  = tokenizer.encode(feat_tensor.unsqueeze(1))
                self.tokens = all_tokens.squeeze(1).cpu()
            tokenizer.to("cpu")
            print("Tokenization complete.")

    def __len__(self):
        return len(self.features) - self.seq_len + 1

    def __getitem__(self, idx):
        if self.tokens is not None:
            x = self.tokens[idx: idx + self.seq_len]
        else:
            x = self.features[idx: idx + self.seq_len]
        y = self.targets[idx + self.seq_len - 1]
        return x, y


# ── Normalization contracts ────────────────────────────────────────────────────
#
# NO_SCALE  — already in [-1,1] or [0,1] by construction in features.py
#             (clip_scale, tanh, wick ratio, binary flags, sin/cos, vol_asymmetry)
#             Applying any scaler would shift a correctly-calibrated signal.
#
# MINMAX    — feat_ob_dist_supp / feat_ob_dist_res have hard known bounds [0, 5.0]
#             where 5.0 is an explicit sentinel (no active zone present).
#             RobustScaler treats 5.0 as a tail outlier and crushes the real
#             proximity signal [0, ~0.1] down to near-zero.
#             MinMaxScaler with [0,5] clip maps cleanly: 0=at zone, 1=sentinel.
#
# ROBUST    — unbounded (log-ratios, z-scores, rolling skewness, MDS ratio)
#             and raw OHLC price levels (~10k-25k). RobustScaler (median+IQR)
#             is outlier-resistant against news spikes and intraday regime shifts.

_NO_SCALE_COLS = frozenset([
    "feat_ob_supp_active", "feat_ob_res_active",
    "feat_session_sin",    "feat_session_cos",
    "feat_icp",            "feat_efficiency",
    "feat_ob_supp_touches","feat_ob_res_touches",
    "feat_momentum_rsi",   "feat_rejection_upper",
    "feat_rejection_lower","feat_local_structure",
    "feat_vol_asymmetry",
])

_MINMAX_COLS = frozenset(["feat_ob_dist_supp", "feat_ob_dist_res"])

# Everything else (remaining SCALE_FEATURES + OHLC) falls into RobustScaler
# by residual — derived at fit-time from feature_cols so new columns
# automatically get a safe default normalization.


class ColumnSelectiveScaler:
    """
    Routes each column to the correct scaler:
      NO_SCALE cols  → identity (untouched)
      MINMAX cols    → MinMaxScaler([0,1]) with [0,5] clip
      all others     → RobustScaler (median + IQR)

    Fit on training data only; pass the same fitted instance to val/test.
    """

    def __init__(self, feature_cols: list[str]):
        self.feature_cols   = list(feature_cols)
        self._no_scale_idx  = [i for i, c in enumerate(feature_cols) if c in _NO_SCALE_COLS]
        self._minmax_idx    = [i for i, c in enumerate(feature_cols) if c in _MINMAX_COLS]
        self._robust_idx    = [
            i for i, c in enumerate(feature_cols)
            if c not in _NO_SCALE_COLS and c not in _MINMAX_COLS
        ]
        self._robust_scaler = RobustScaler()
        self._minmax_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        self._fitted        = False

    def fit(self, X: np.ndarray) -> "ColumnSelectiveScaler":
        if self._robust_idx:
            self._robust_scaler.fit(X[:, self._robust_idx])
        if self._minmax_idx:
            self._minmax_scaler.fit(np.clip(X[:, self._minmax_idx], 0.0, 5.0))
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")
        X = X.copy().astype(np.float32)
        if self._robust_idx:
            X[:, self._robust_idx] = self._robust_scaler.transform(
                X[:, self._robust_idx]
            ).astype(np.float32)
        if self._minmax_idx:
            X[:, self._minmax_idx] = self._minmax_scaler.transform(
                np.clip(X[:, self._minmax_idx], 0.0, 5.0)
            ).astype(np.float32)
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def summary(self) -> str:
        no_scale = [self.feature_cols[i] for i in self._no_scale_idx]
        minmax   = [self.feature_cols[i] for i in self._minmax_idx]
        robust   = [self.feature_cols[i] for i in self._robust_idx]
        return (
            f"ColumnSelectiveScaler over {len(self.feature_cols)} cols:\n"
            f"  NO_SCALE ({len(no_scale)}): {no_scale}\n"
            f"  MINMAX   ({len(minmax)}):   {minmax}\n"
            f"  ROBUST   ({len(robust)}):   {robust}"
        )


def fit_scaler(
    features_train: np.ndarray,
    feature_cols: list[str],
) -> ColumnSelectiveScaler:
    """
    Fit a ColumnSelectiveScaler on training data only.
    Never pass val/test data here — that is data leakage.
    Pass the returned scaler to FinancialDataset for all splits.
    """
    if features_train.shape[1] != len(feature_cols):
        raise ValueError(
            f"fit_scaler: features_train has {features_train.shape[1]} columns "
            f"but feature_cols has {len(feature_cols)} entries."
        )
    scaler = ColumnSelectiveScaler(feature_cols)
    scaler.fit(features_train)
    print(scaler.summary())
    return scaler


def _compute_sample_weights(
    targets_array: np.ndarray,
    threshold: float,
    use_sqrt: bool = True,
) -> torch.Tensor:
    class_indices = []
    for y in targets_array:
        if y < -threshold:
            class_indices.append(0)   # Short
        elif y > threshold:
            class_indices.append(2)   # Long
        else:
            class_indices.append(1)   # Flat
    counts  = np.bincount(class_indices)
    weights = 1.0 / (np.sqrt(counts) + 1e-6) if use_sqrt else 1.0 / (counts + 1e-6)
    return torch.DoubleTensor([weights[c] for c in class_indices])


def _make_loader(ds, config, sampler=None, shuffle=False) -> DataLoader:
    """Single helper so prefetch_factor / persistent_workers are set consistently
    for every DataLoader in this file."""
    is_cuda  = torch.cuda.is_available()
    nw       = config.NUM_WORKERS
    pf       = getattr(config, "PREFETCH_FACTOR", 2) if nw > 0 else None
    pw       = nw > 0
    return DataLoader(
        ds,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        drop_last=(sampler is not None),   # drop_last only when using sampler
        num_workers=nw,
        prefetch_factor=pf,
        persistent_workers=pw,
        pin_memory=is_cuda,
    )


def create_dataloaders(features, targets, config, feature_cols, tokenizer=None):
    total_len  = len(features)
    gap        = config.FORECAST_HORIZON + 50

    train_end  = int(total_len * config.TRAIN_RATIO)
    val_start  = train_end + gap
    val_end    = val_start + int(total_len * config.VAL_RATIO)
    test_start = val_end + gap

    # FIX: pass feature_cols so the scaler routes columns correctly
    scaler = fit_scaler(features[:train_end], feature_cols)

    train_ds = FinancialDataset(features[:train_end],        targets[:train_end],        config.LOOKBACK_WINDOW, scaler=scaler, tokenizer=tokenizer)
    val_ds   = FinancialDataset(features[val_start:val_end], targets[val_start:val_end], config.LOOKBACK_WINDOW, scaler=scaler, tokenizer=tokenizer)
    test_ds  = FinancialDataset(features[test_start:],       targets[test_start:],       config.LOOKBACK_WINDOW, scaler=scaler, tokenizer=tokenizer)

    print("Computing sample weights...")
    start_idx       = config.LOOKBACK_WINDOW - 1
    y_train_aligned = targets[start_idx: start_idx + len(train_ds)]
    sample_weights  = _compute_sample_weights(y_train_aligned, config.SAMPLER_THRESHOLD)

    assert len(sample_weights) == len(train_ds), \
        f"Weight mismatch: weights={len(sample_weights)}, ds={len(train_ds)}"

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights) // 2, replacement=True)

    train_loader = _make_loader(train_ds, config, sampler=sampler)
    val_loader   = _make_loader(val_ds,   config)
    test_loader  = _make_loader(test_ds,  config)

    return train_loader, val_loader, test_loader


def create_multi_index_dataloaders(asset_data_list, config, feature_cols, tokenizer=None, is_train=False):
    """
    FIX: Added feature_cols parameter — required by fit_scaler().
    The same feature_cols list from process_dataset() must be passed in.
    """
    datasets    = []
    all_targets = []

    for feat, targ in asset_data_list:
        if len(feat) < config.LOOKBACK_WINDOW:
            continue
        # FIX: pass feature_cols to fit_scaler — was the crash site
        scaler = fit_scaler(feat, feature_cols) if is_train else None
        ds     = FinancialDataset(feat, targ, config.LOOKBACK_WINDOW, scaler=scaler, tokenizer=tokenizer)
        datasets.append(ds)
        if is_train:
            start = config.LOOKBACK_WINDOW - 1
            all_targets.extend(targ[start: start + len(ds)].tolist())

    if not datasets:
        return None

    full_ds = torch.utils.data.ConcatDataset(datasets)

    if is_train:
        all_targets_arr = np.array(all_targets, dtype=np.float32)
        sample_weights  = _compute_sample_weights(all_targets_arr, config.SAMPLER_THRESHOLD)

        assert len(sample_weights) == len(full_ds), \
            f"Multi-index weight mismatch: weights={len(sample_weights)}, ds={len(full_ds)}"

        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights) // 2, replacement=True)
        return _make_loader(full_ds, config, sampler=sampler)
    else:
        return _make_loader(full_ds, config)


def create_fold_dataloaders(features, targets, train_indices, val_indices, test_indices, config, feature_cols, tokenizer=None):
    """
    FIX: Added feature_cols parameter — required by fit_scaler().
    Scaler is fitted on train_indices slice only — no leakage into val/test.
    """
    train_feat = features[train_indices[0]:train_indices[1]]
    # FIX: pass feature_cols to fit_scaler
    scaler     = fit_scaler(train_feat, feature_cols)

    train_ds = FinancialDataset(train_feat,                                targets[train_indices[0]:train_indices[1]], config.LOOKBACK_WINDOW, scaler=scaler, tokenizer=tokenizer)
    val_ds   = FinancialDataset(features[val_indices[0]:val_indices[1]],   targets[val_indices[0]:val_indices[1]],   config.LOOKBACK_WINDOW, scaler=scaler, tokenizer=tokenizer)
    test_ds  = FinancialDataset(features[test_indices[0]:test_indices[1]], targets[test_indices[0]:test_indices[1]], config.LOOKBACK_WINDOW, scaler=scaler, tokenizer=tokenizer)

    start_idx       = train_indices[0] + config.LOOKBACK_WINDOW - 1
    y_train_aligned = targets[start_idx: start_idx + len(train_ds)]
    sample_weights  = _compute_sample_weights(y_train_aligned, config.SAMPLER_THRESHOLD)
    sampler         = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights) // 2, replacement=True)

    train_loader = _make_loader(train_ds, config, sampler=sampler)
    val_loader   = _make_loader(val_ds,   config)
    test_loader  = _make_loader(test_ds,  config)

    return train_loader, val_loader, test_loader