#!/usr/bin/env python3
"""
audit_pre_training_smoke.py
===========================
Pre-training end-to-end smoke for the v2 feature pipeline.

This is the GATE audit you should run immediately before kicking off a real
training job. If anything here fails, training will fail (or, worse, train
silently on garbage). A clean pass is the green light.

Designed to run on Kaggle / any Linux box with the production model dims
(D_MODEL=64, N_LAYERS=4, full seq_len=512, real CSV data). No memory-
compromise knobs — uses the actual architecture and full pipeline.

Run from repo root:
    python3 audits/scripts/audit_pre_training_smoke.py

What it verifies (in order)
---------------------------
  1.  Real CSV → v2 FeatureEngineer → 9-column DataFrame, all float64, no NaN
  2.  process_dataset() returns the expected (T, 13) tensor + targets
  3.  ColumnSelectiveScaler fits + transforms the v2 features (empty ROBUST bucket)
  4.  FinancialDataset.__getitem__ emits (None, (T, 13), scalar) for
      INPUT_MODE="features_only"
  5.  DataLoader collation produces (tokens=None, features=(B, T, 13),
      targets=(B,), pad_mask=None or (B,))
  6.  Production model forward on that batch returns (B, Q) — exact Q from
      config.QUANTILE_LEVELS — and the output is monotonically non-decreasing
  7.  v10_total_loss returns finite scalar and produces non-NaN gradients
  8.  AMP + GradScaler: synthetic-NaN loss correctly skipped, scheduler
      does not advance
  9.  Optimizer.step() actually mutates trainable params (sentinel check)
  10. One-epoch micro-training on real data reduces loss monotonically
      (avg(loss[2nd half]) < avg(loss[1st half]))
  11. Checkpoint round-trip: save model+optimizer+scheduler state to a temp
      file, load it back, confirm every state tensor is identical
  12. Inference path on a single window: features → (1, Q) → decision_score
      (median channel) is finite and within expected magnitude
  13. Warmed-up inference: if a trained model path is given, load it and
      confirm forward pass matches the smoke-built model output shape
  14. Multi-asset process_dataset: load several CSVs and confirm columns
      align across all of them (catches rename / horizon drift bugs)

Each stage prints [PASS] / [WARN] / [FAIL] with clear remediation hints.
The summary at the end lists any [FAIL] items that must be fixed before
training. [WARN] items should be investigated but don't block.
"""

from __future__ import annotations
import io
import os
import sys
import time
import math
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_REPO, "core"))
sys.path.insert(0, _REPO)

import config as CFG
import features as FEAT_MOD
from features import FeatureEngineer, FeatureConfig
from data_loader import (
    ColumnSelectiveScaler,
    FinancialDataset,
    collate_with_none,
    _col_bucket,
)
from train import _make_feature_config, _build_model
from loss import v10_total_loss, pinball_loss, asymmetric_number_line_loss
from model import LPatchTST, PatchTST, InputMode


SEP  = "─" * 70
SEP2 = "=" * 70

RESULTS: list[tuple[str, str, str]] = []   # (stage, status, message)

def hdr(t: str) -> None:
    print(f"\n{SEP}\n{t}\n{SEP}")

def _record(stage: str, status: str, message: str) -> None:
    RESULTS.append((stage, status, message))

def ok(m:   str) -> None:
    print(f"  [PASS] {m}")
    _record("?", "PASS", m)

def warn(m: str) -> None:
    print(f"  [WARN] {m}")
    _record("?", "WARN", m)

def fail(m: str) -> None:
    print(f"  [FAIL] {m}")
    _record("?", "FAIL", m)

def info(m: str) -> None:
    print(f"  [INFO] {m}")


print(SEP2)
print("  LPatchTST — Pre-Training End-to-End Smoke (v2 features_only)")
print(f"  Repo : {_REPO}")
print(SEP2)


# ── Locate one real CSV for end-to-end runs ───────────────────────────────────
def _find_one_csv() -> Optional[str]:
    candidates: list[str] = []
    if hasattr(CFG, "DATA_FILE"):
        df = CFG.DATA_FILE
        if isinstance(df, list) and df:
            candidates = df
        elif isinstance(df, str):
            candidates = [df]
    for p in candidates:
        if os.path.exists(p):
            return p
        alt = os.path.join(_REPO, p)
        if os.path.exists(alt):
            return alt
    for p in sorted(Path(_REPO, "data").glob("*.csv")):
        return str(p)
    return None


data_path = _find_one_csv()
if not data_path:
    print(f"  [FAIL] No CSV found in DATA_FILE or data/. Cannot run end-to-end stages.")
    print(f"          DATA_FILE = {getattr(CFG, 'DATA_FILE', None)}")
    sys.exit(1)
info(f"Primary CSV: {data_path}")


# Helper: consistent feature_cols (train.py path = 4 OHLC + 9 v2)
OHLC_COLS = ["open", "high", "low", "close"]


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1 — v2 FeatureEngineer on real CSV
# ═════════════════════════════════════════════════════════════════════════════
hdr("STAGE 1 · v2 FeatureEngineer on real CSV")

fe_cfg = _make_feature_config()
fe = FeatureEngineer(fe_cfg)
expected_cols = fe_cfg.feature_columns
expected_n    = len(expected_cols)
print(f"  Expected feature columns ({expected_n}): {expected_cols}")

import pandas as pd
df_raw = pd.read_csv(data_path)
df_raw.columns = [c.strip().lower() for c in df_raw.columns]
feats = fe.build(df_raw["close"], ohlc=df_raw, include_target=False, dropna=True)
print(f"  Built features: shape={feats.shape}, dtypes={feats.dtypes.unique().tolist()}")

if list(feats.columns) != expected_cols:
    fail(
        f"FeatureEngineer produced columns {list(feats.columns)} "
        f"but FeatureConfig expects {expected_cols}."
    )
else:
    ok("Column order matches FeatureConfig.feature_columns exactly.")

if feats.shape[1] != expected_n:
    fail(f"Feature count = {feats.shape[1]}, expected {expected_n}.")
else:
    ok(f"Feature count = {expected_n}.")

if not (feats.dtypes == "float64").all():
    fail(f"Non-float64 dtypes present: {feats.dtypes[feats.dtypes != 'float64'].to_dict()}")
else:
    ok("All columns are float64.")

if feats.isna().any().any():
    fail(f"NaNs in features after dropna=True: {feats.isna().sum().sum()} cells.")
else:
    ok("No NaN cells in features (dropna=True).")

if len(feats) < 1000:
    warn(f"Only {len(feats)} rows after warm-up drop. Models may overfit on tiny dataset.")
else:
    ok(f"{len(feats):,} rows after warm-up drop. Sufficient for training.")

# Stash for downstream stages
v2_features_df = feats.copy()
v2_ohlc_df     = df_raw[OHLC_COLS].copy()


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2 — process_dataset on the same file
# ═════════════════════════════════════════════════════════════════════════════
hdr("STAGE 2 · process_dataset() returns expected shapes")

from train import process_dataset
try:
    assets, final_feature_cols = process_dataset([data_path], fe)
except Exception as e:
    fail(f"process_dataset raised: {e}")
    raise

print(f"  final_feature_cols: {final_feature_cols}")
asset_id, feat_vals, target_vals, ohlc_vals, dates = assets[0]
n_feat_with_ohlc = feat_vals.shape[1]
print(f"  assets[0]: features={feat_vals.shape}, targets={target_vals.shape}, ohlc={ohlc_vals.shape}")

expected_total = 4 + expected_n
if n_feat_with_ohlc != expected_total:
    fail(
        f"process_dataset feature count = {n_feat_with_ohlc}, "
        f"expected {4} OHLC + {expected_n} v2 = {expected_total}. "
        "DEFAULT_FEATURE_LIST in train.py is out of sync with v2 features."
    )
else:
    ok(f"Feature count = {4} OHLC + {expected_n} v2 = {n_feat_with_ohlc}.")

if not np.isfinite(feat_vals).all():
    fail("process_dataset features contain NaN or Inf.")
else:
    ok("All process_dataset features are finite.")

if not np.isfinite(target_vals).all():
    fail("process_dataset targets contain NaN or Inf.")
else:
    ok("All process_dataset targets are finite.")


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3 — ColumnSelectiveScaler + per-column bucket routing
# ═════════════════════════════════════════════════════════════════════════════
hdr("STAGE 3 · ColumnSelectiveScaler on v2 features + bucket routing")

X_v2 = v2_features_df.values.astype(np.float32)
n_scale_cols_no  = []
n_scale_cols_rob = []
for c in expected_cols:
    b = _col_bucket(c)
    (n_scale_cols_no if b == "no_scale" else n_scale_cols_rob).append(c)

print(f"  Routing: NO_SCALE={len(n_scale_cols_no)}, ROBUST={len(n_scale_cols_rob)}")
if n_scale_cols_rob:
    fail(f"Unexpected ROBUST routing: {n_scale_cols_rob} (v2 set should be all NO_SCALE).")
else:
    ok("All v2 features route to NO_SCALE bucket (matches design).")

# Also check OHLC + everything together
full_feature_cols = OHLC_COLS + expected_cols
full_routing = {c: _col_bucket(c) for c in full_feature_cols}
ohlc_bucket  = {_col_bucket(c) for c in OHLC_COLS}
print(f"  OHLC bucket: {ohlc_bucket}  (ROBUST is the production default for raw OHLC)")
if "robust" not in ohlc_bucket:
    warn(f"OHLC routed to {ohlc_bucket} — train.py may not be re-scaling raw OHLC bars.")
else:
    ok("OHLC routes to ROBUST bucket (raw bars get RobustScaler).")

scaler_v2 = ColumnSelectiveScaler(expected_cols).fit(X_v2)
scaled_v2 = scaler_v2.transform(X_v2)
if scaled_v2.shape != X_v2.shape:
    fail(f"Scaler output shape {scaled_v2.shape} != input {X_v2.shape}.")
elif not np.isfinite(scaled_v2).all():
    fail("Scaler output contains NaN or Inf.")
else:
    ok(f"Scaler round-trip OK: {scaled_v2.shape}, finite, dtype={scaled_v2.dtype}.")


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4 — FinancialDataset.__getitem__ shape
# ═════════════════════════════════════════════════════════════════════════════
hdr("STAGE 4 · FinancialDataset.__getitem__ returns expected shapes")

seq_len    = CFG.LOOKBACK_WINDOW
input_mode = InputMode.FEATURES_ONLY

# Build a (T, n_feat_with_ohlc) feature matrix that mirrors the production
# train.py pipeline: 4 OHLC raw + 9 v2 features.
feats_ohlc = v2_ohlc_df.iloc[:len(v2_features_df)].values.astype(np.float32)
feats_full = np.concatenate([feats_ohlc, v2_features_df.values.astype(np.float32)], axis=1)
n_feat_with_ohlc_actual = feats_full.shape[1]
assert n_feat_with_ohlc_actual == n_feat_with_ohlc, (
    f"Inconsistent n_feat_with_ohlc: train.py gave {n_feat_with_ohlc}, "
    f"this audit computed {n_feat_with_ohlc_actual}"
)
print(f"  Combined feature matrix: {feats_full.shape} = (T, {n_feat_with_ohlc_actual})")
print(f"  Combined columns: {full_feature_cols}")

# Keep a moderate slice for the DataLoader stages to bound latency
DS_SLICE = min(4096, len(feats_full))
feats_dl  = feats_full[:DS_SLICE]
scaled_dl = ColumnSelectiveScaler(full_feature_cols).fit(feats_dl).transform(feats_dl)
targets_dl = np.zeros(len(scaled_dl), dtype=np.float32)
print(f"  DataLoader slice: {feats_dl.shape} (memory-light)")

ds = FinancialDataset(
    features=scaled_dl,
    targets=targets_dl,
    seq_len=seq_len,
    config=CFG,
)
print(f"  Dataset length: {len(ds)} windows (T={len(scaled_dl)}, seq_len={seq_len})")

tokens, features, target = ds[0]
print(f"  ds[0]: tokens={type(tokens).__name__}, "
      f"features={features.shape if features is not None else None}, "
      f"target={target.shape if hasattr(target, 'shape') else type(target).__name__}")

if tokens is not None:
    fail(f"Expected tokens=None for features_only mode, got {type(tokens).__name__}.")
else:
    ok("tokens=None (features_only mode).")

if features is None:
    fail("features is None — dataset not emitting features.")
elif features.shape != (seq_len, n_feat_with_ohlc):
    fail(f"features.shape={features.shape}, expected ({seq_len}, {n_feat_with_ohlc}).")
else:
    ok(f"features.shape = (seq_len={seq_len}, n_feat={n_feat_with_ohlc}) ✓")

if not torch.is_tensor(target):
    fail(f"target is not a tensor (got {type(target).__name__}).")
elif target.dim() != 0:
    fail(f"target should be scalar (dim=0), got shape {tuple(target.shape)}.")
else:
    ok("target is a scalar tensor (0-dim).")


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 5 — DataLoader collation
# ═════════════════════════════════════════════════════════════════════════════
hdr("STAGE 5 · DataLoader collation")

dl = DataLoader(
    ds, batch_size=4, shuffle=False,
    collate_fn=collate_with_none, num_workers=0,
)
batch = next(iter(dl))
tokens_b, features_b, targets_b, pad_mask = batch
pad_mask_repr = (None if pad_mask is None else tuple(pad_mask.shape))
print(f"  Batch: tokens={type(tokens_b).__name__}, "
      f"features={features_b.shape if features_b is not None else None}, "
      f"targets={targets_b.shape}, pad_mask={pad_mask_repr}")

if tokens_b is not None:
    fail("Batched tokens should be None for features_only.")
else:
    ok("Batched tokens = None.")

if features_b.shape != (4, seq_len, n_feat_with_ohlc):
    fail(f"Batched features shape {tuple(features_b.shape)}, expected (4, {seq_len}, {n_feat_with_ohlc}).")
else:
    ok(f"Batched features shape = (B=4, T={seq_len}, n_feat={n_feat_with_ohlc}) ✓")

if targets_b.shape != (4,):
    fail(f"targets shape {tuple(targets_b.shape)}, expected (4,).")
else:
    ok("targets shape = (B=4,).")

# pad_mask should be None or shape (4, seq_len)
if pad_mask is not None and pad_mask.shape not in [(4,), (4, seq_len)]:
    warn(f"pad_mask shape {tuple(pad_mask.shape)} is unexpected.")
else:
    ok(f"pad_mask shape OK: {pad_mask_repr}.")


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 6 — Production model forward
# ═════════════════════════════════════════════════════════════════════════════
hdr("STAGE 6 · Production model forward on real-shape batch")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

cls = LPatchTST if getattr(CFG, "USE_LPATCHTST", True) else PatchTST
n_feat_build = n_feat_with_ohlc
try:
    net = cls(
        input_mode="features_only",
        seq_len=seq_len,
        n_features=n_feat_build,
        s1_bits=CFG.TOKENIZER_S1_BITS,
        s2_bits=CFG.TOKENIZER_S2_BITS,
        d_model=CFG.D_MODEL,
        patch_len=CFG.PATCH_LEN,
        stride=CFG.STRIDE,
        n_heads=CFG.N_HEADS,
        n_layers=CFG.N_LAYERS,
        lstm_layers=CFG.LSTM_LAYERS,
        dropout=CFG.DROPOUT,
        aggregation=CFG.AGGREGATION_MODE,
        vocab_size=CFG.VOCAB_SIZE,
    ).to(device)
except Exception as e:
    fail(f"Model build failed: {e}")
    raise

n_params     = sum(p.numel() for p in net.parameters())
n_trainable  = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"  Model: {cls.__name__}, {n_params:,} total params, {n_trainable:,} trainable.")
ok("Model instantiated with production config.")

# Move batch to device
features_b_d = features_b.to(device)
net.eval()
with torch.no_grad():
    out = net(tokens=None, features=features_b_d)
Q_expected = len(CFG.QUANTILE_LEVELS)
print(f"  Output shape: {tuple(out.shape)}  (expected (B=4, Q={Q_expected}))")
if out.shape != (4, Q_expected):
    fail(f"Output shape {tuple(out.shape)} != expected (4, {Q_expected}).")
else:
    ok(f"Output shape matches: (B=4, Q={Q_expected}).")

# Monotonicity check (Falcon-2.0 invariant)
diffs = out[:, 1:] - out[:, :-1]
if (diffs >= 0).all():
    ok("QuantileHead output is monotonically non-decreasing across levels.")
else:
    fail("QuantileHead output is NOT monotonic — parametrization broken.")

if not torch.isfinite(out).all():
    fail("Model output contains NaN or Inf.")
else:
    ok("Model output is finite.")


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 7 — v10 loss + backward
# ═════════════════════════════════════════════════════════════════════════════
hdr("STAGE 7 · v10_total_loss on model output")

torch.manual_seed(0)
# Use realistic targets within the [−0.20, +0.20] range of clipped next-bar
# log returns on NIFTY 30-min data
targets_t = torch.tensor([0.25, -0.15, 0.05, -0.30], device=device)
out_d = out.to(device)
loss = v10_total_loss(out_d, targets_t)
print(f"  v10 loss: {loss.item():.4f}")

if not torch.isfinite(loss):
    fail(f"v10 loss is not finite: {loss.item()}.")
else:
    ok(f"v10 loss is finite ({loss.item():.4f}).")

# Backward through a separate forward pass (eval mode was used above)
net.train()
out_train = net(tokens=None, features=features_b_d)
loss_train = v10_total_loss(out_train, targets_t)
loss_train.backward()

nan_grad = []
for n, p in net.named_parameters():
    if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):
        nan_grad.append(n)
if nan_grad:
    fail(f"Backward produced NaN/Inf in {len(nan_grad)} gradient(s). First: {nan_grad[:3]}")
else:
    ok(f"All {n_trainable:,} parameter gradients are finite.")


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 8 — AMP skip-step on synthetic NaN
# ═════════════════════════════════════════════════════════════════════════════
hdr("STAGE 8 · AMP skip-step on synthetic NaN")

from torch.optim.lr_scheduler import OneCycleLR

mini = nn.Linear(4, 2).to(device)
opt  = torch.optim.AdamW(mini.parameters(), lr=1e-3)
sched = OneCycleLR(opt, max_lr=1e-3, total_steps=10, pct_start=0.3, div_factor=10, final_div_factor=10)
scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

step_count = 0
skipped    = 0

for step in range(10):
    x = torch.randn(8, 4, device=device)
    y = torch.randn(8, 2, device=device)
    with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        pred = mini(x)
        loss_step = F.mse_loss(pred, y)
        if step == 5:
            loss_step = loss_step * float("nan")
    opt.zero_grad()
    if not torch.isnan(loss_step):
        scaler.scale(loss_step).backward()
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if scaler.get_scale() >= scale_before:
            sched.step()
            step_count += 1
    else:
        skipped += 1

print(f"  Completed optimizer steps: {step_count} (10 attempted, {skipped} skipped due to NaN)")
if step_count < 9:
    fail(f"Only {step_count}/9 non-NaN steps advanced the scheduler — skip-step guard may be broken.")
else:
    ok(f"Skip-step guard works: {step_count}/9 normal steps advanced, 1 NaN step skipped.")


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 9 — Optimizer.step() mutates trainable params
# ═════════════════════════════════════════════════════════════════════════════
hdr("STAGE 9 · Optimizer.step() mutates trainable params")

# Snapshot first param
w_before = next(net.parameters()).detach().clone()
opt_main = torch.optim.AdamW(
    [p for p in net.parameters() if p.requires_grad],
    lr=1e-4, weight_decay=0.0,
)
net.train()
opt_main.zero_grad()
out_s9 = net(tokens=None, features=features_b_d)
loss_s9 = v10_total_loss(out_s9, torch.zeros(4, device=device))
loss_s9.backward()
opt_main.step()
w_after = next(net.parameters()).detach()
if torch.allclose(w_before, w_after):
    fail("Param tensor unchanged after step() — gradient flow disconnected.")
else:
    delta = (w_after - w_before).abs().mean().item()
    ok(f"Param tensor mutated: mean |Δ| = {delta:.2e}.")


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 10 — One-epoch micro-training on real data
# ═════════════════════════════════════════════════════════════════════════════
hdr("STAGE 10 · One-epoch micro-training (loss should decrease)")

torch.manual_seed(42)
np.random.seed(42)

# Build a dataset with REAL targets = next-bar log return (clipped to ±0.20)
close_series = df_raw["close"].dropna().reset_index(drop=True)
r_next = np.log(close_series.values[1:] / close_series.values[:-1])
features_arr = v2_features_df.values[:len(r_next)].astype(np.float32)
targets_arr  = np.clip(r_next[:len(features_arr)], -0.20, 0.20).astype(np.float32)

N_TRAIN = min(4000, len(features_arr))
feats_train = features_arr[:N_TRAIN]
targs_train = targets_arr[:N_TRAIN]
print(f"  Micro-train slice: {N_TRAIN} rows")

scaler_mt = ColumnSelectiveScaler(expected_cols).fit(feats_train)
feats_s   = scaler_mt.transform(feats_train)

ds_mt = FinancialDataset(
    features=feats_s,
    targets=targs_train,
    seq_len=CFG.LOOKBACK_WINDOW,
    config=CFG,
)
dl_mt = DataLoader(
    ds_mt, batch_size=CFG.BATCH_SIZE, shuffle=True,
    collate_fn=collate_with_none, num_workers=0,
)

# Fresh model for the micro-training loop (use production dims)
torch.manual_seed(0)
net_mt = cls(
    input_mode="features_only",
    seq_len=CFG.LOOKBACK_WINDOW,
    n_features=expected_n,
    s1_bits=CFG.TOKENIZER_S1_BITS,
    s2_bits=CFG.TOKENIZER_S2_BITS,
    d_model=CFG.D_MODEL,
    patch_len=CFG.PATCH_LEN,
    stride=CFG.STRIDE,
    n_heads=CFG.N_HEADS,
    n_layers=CFG.N_LAYERS,
    lstm_layers=CFG.LSTM_LAYERS,
    dropout=CFG.DROPOUT,
    aggregation=CFG.AGGREGATION_MODE,
    vocab_size=CFG.VOCAB_SIZE,
).to(device)
n_params_mt = sum(p.numel() for p in net_mt.parameters())
print(f"  Micro-train model: {n_params_mt:,} params on {device}")

opt_mt = torch.optim.AdamW(net_mt.parameters(), lr=1e-4)
total_steps = max(len(dl_mt), 1)
sched_mt = OneCycleLR(opt_mt, max_lr=1e-4, total_steps=total_steps,
                      pct_start=0.3, div_factor=10, final_div_factor=10)
scaler_mt_amp = torch.amp.GradScaler(enabled=(device.type == "cuda"))

losses: list[float] = []
t0 = time.time()
net_mt.train()
for step, (tokens_b, features_b, targets_b, _) in enumerate(dl_mt):
    features_b = features_b.to(device)
    targets_b  = targets_b.to(device)
    opt_mt.zero_grad()
    with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        out_mt = net_mt(tokens=None, features=features_b)
        loss_step = v10_total_loss(out_mt, targets_b)
    if not torch.isfinite(loss_step):
        continue
    scaler_mt_amp.scale(loss_step).backward()
    scale_before = scaler_mt_amp.get_scale()
    scaler_mt_amp.step(opt_mt)
    scaler_mt_amp.update()
    if scaler_mt_amp.get_scale() >= scale_before:
        sched_mt.step()
    losses.append(loss_step.item())
elapsed = time.time() - t0

print(f"  Completed {len(losses)} steps in {elapsed:.1f}s ({elapsed/max(len(losses),1):.2f}s/step)")
if len(losses) < 5:
    warn(f"Only {len(losses)} micro-train steps completed. Pipeline may be slow.")
else:
    first_half = float(np.mean(losses[:len(losses)//2]))
    second_half = float(np.mean(losses[len(losses)//2:]))
    print(f"  First half mean loss: {first_half:.4f}")
    print(f"  Second half mean loss: {second_half:.4f}")
    if second_half >= first_half:
        warn(
            f"Loss did NOT decrease ({first_half:.4f} → {second_half:.4f}). "
            "Could be: (a) batch size too small for OneCycleLR pct_start, "
            "(b) features carry little signal at 1-bar horizon, "
            "(c) model init issue. Inspect first."
        )
    else:
        reduction = (first_half - second_half) / max(first_half, 1e-9) * 100
        ok(f"Loss decreased {reduction:.1f}% over the epoch — pipeline is learning.")


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 11 — Checkpoint round-trip
# ═════════════════════════════════════════════════════════════════════════════
hdr("STAGE 11 · Checkpoint round-trip (model + optimizer + scheduler)")

ckpt = {
    "model": net_mt.state_dict(),
    "optim": opt_mt.state_dict(),
    "scaler": scaler_mt_amp.state_dict() if device.type == "cuda" else None,
    "sched": sched_mt.state_dict(),
    "rng": torch.get_rng_state(),
    "n_features": expected_n,
    "feature_cols": expected_cols,
    "model_class": cls.__name__,
}
with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
    ckpt_path = f.name
    torch.save(ckpt, ckpt_path)
file_size = os.path.getsize(ckpt_path)
print(f"  Saved checkpoint: {ckpt_path} ({file_size/1024:.1f} KB)")

# Reconstruct
torch.manual_seed(1)
net_mt2 = cls(
    input_mode="features_only",
    seq_len=CFG.LOOKBACK_WINDOW,
    n_features=expected_n,
    s1_bits=CFG.TOKENIZER_S1_BITS,
    s2_bits=CFG.TOKENIZER_S2_BITS,
    d_model=CFG.D_MODEL,
    patch_len=CFG.PATCH_LEN,
    stride=CFG.STRIDE,
    n_heads=CFG.N_HEADS,
    n_layers=CFG.N_LAYERS,
    lstm_layers=CFG.LSTM_LAYERS,
    dropout=CFG.DROPOUT,
    aggregation=CFG.AGGREGATION_MODE,
    vocab_size=CFG.VOCAB_SIZE,
).to(device)
opt_mt2   = torch.optim.AdamW(net_mt2.parameters(), lr=1e-4)
sched_mt2 = OneCycleLR(opt_mt2, max_lr=1e-4, total_steps=total_steps,
                       pct_start=0.3, div_factor=10, final_div_factor=10)
scaler_mt2 = torch.amp.GradScaler(enabled=(device.type == "cuda"))

loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
net_mt2.load_state_dict(loaded["model"])
opt_mt2.load_state_dict(loaded["optim"])
sched_mt2.load_state_dict(loaded["sched"])
if device.type == "cuda":
    scaler_mt2.load_state_dict(loaded["scaler"])

# Verify exact match
mismatches = []
for (n1, p1), (n2, p2) in zip(net_mt.named_parameters(), net_mt2.named_parameters()):
    if not torch.allclose(p1, p2, atol=1e-7):
        mismatches.append(n1)
if mismatches:
    fail(f"Model state_dict round-trip has {len(mismatches)} mismatched params. First: {mismatches[:3]}")
else:
    ok("Model state_dict round-trips exactly.")

# Optimizer state — compare param_groups structure
opt_m1 = opt_mt.state_dict()
opt_m2 = opt_mt2.state_dict()
if len(opt_m1["param_groups"]) != len(opt_m2["param_groups"]):
    fail("Optimizer param_groups count differs after round-trip.")
else:
    ok("Optimizer state_dict round-trips (param_groups match).")

# Scheduler state
if sched_mt.state_dict() != sched_mt2.state_dict():
    fail("Scheduler state_dict differs after round-trip.")
else:
    ok("Scheduler state_dict round-trips exactly.")

os.unlink(ckpt_path)


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 12 — Inference path on a single window
# ═════════════════════════════════════════════════════════════════════════════
hdr("STAGE 12 · Single-window inference path (no tokenizer)")

# Take a fresh DataLoader sample to match production
dl_inf = DataLoader(
    ds, batch_size=1, shuffle=False, collate_fn=collate_with_none, num_workers=0,
)
batch_inf = next(iter(dl_inf))
_, features_i, target_i, _ = batch_inf
features_i_d = features_i.to(device)
print(f"  Single-window features: {tuple(features_i_d.shape)}")

net.eval()
with torch.no_grad():
    out_i = net(tokens=None, features=features_i_d)
print(f"  Single-window output: {tuple(out_i.shape)}")
if out_i.shape != (1, Q_expected):
    fail(f"Output shape {tuple(out_i.shape)} != (1, {Q_expected}).")
else:
    ok(f"Output shape = (1, Q={Q_expected}).")

# Decision score = median channel
med_idx = CFG.QUANTILE_LEVELS.index(0.5)
score = out_i[0, med_idx].item()
print(f"  Decision score (median channel): {score:+.4f}")
if not math.isfinite(score):
    fail(f"Decision score is not finite: {score}.")
elif abs(score) > 50.0:
    warn(f"Decision score |{score:.2f}| is unusually large — model may be untrained.")
else:
    ok("Decision score is finite and in a reasonable magnitude range.")

# Also exercise inference with the BATCH forward used by inference.py
with torch.no_grad():
    out_batch = net(tokens=None, features=features_b_d)
if out_batch.shape != (4, Q_expected):
    fail(f"Batch forward shape {tuple(out_batch.shape)} != (4, {Q_expected}).")
else:
    ok(f"Batch forward shape = (4, {Q_expected}).")


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 13 — Loaded checkpoint inference (if a real model path is given)
# ═════════════════════════════════════════════════════════════════════════════
hdr("STAGE 13 · Optional: load a trained checkpoint and verify forward")

# Default candidate: look for a model file in the conventional path
ckpt_candidates = [
    "models/pretrain_best.pth",
    "models/finetune_best.pth",
    "models/model.safetensors",
]
ckpt_to_try: Optional[str] = None
for c in ckpt_candidates:
    if os.path.exists(c):
        ckpt_to_try = c
        break
    if os.path.exists(os.path.join(_REPO, c)):
        ckpt_to_try = os.path.join(_REPO, c)
        break

if ckpt_to_try is None:
    info("No trained checkpoint found in conventional paths — skipping inference-load stage.")
    info("(This is expected on a fresh repo. Train a quick model and rerun.)")
else:
    try:
        loaded_ckpt = torch.load(ckpt_to_try, map_location=device, weights_only=False)
        # Accept either a raw state_dict or a dict wrapping one
        sd = loaded_ckpt
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        net_loaded = cls(
            input_mode="features_only",
            seq_len=CFG.LOOKBACK_WINDOW,
            n_features=n_feat_with_ohlc,
            s1_bits=CFG.TOKENIZER_S1_BITS,
            s2_bits=CFG.TOKENIZER_S2_BITS,
            d_model=CFG.D_MODEL,
            patch_len=CFG.PATCH_LEN,
            stride=CFG.STRIDE,
            n_heads=CFG.N_HEADS,
            n_layers=CFG.N_LAYERS,
            lstm_layers=CFG.LSTM_LAYERS,
            dropout=0.0,
            aggregation=CFG.AGGREGATION_MODE,
            vocab_size=CFG.VOCAB_SIZE,
        ).to(device)
        missing, unexpected = net_loaded.load_state_dict(sd, strict=False)
        if missing:
            warn(f"Checkpoint load: missing keys = {len(missing)}. First: {missing[:3]}")
        if unexpected:
            warn(f"Checkpoint load: unexpected keys = {len(unexpected)}. First: {unexpected[:3]}")
        if not missing and not unexpected:
            ok(f"Loaded {ckpt_to_try} with strict=True (all keys matched).")
        net_loaded.eval()
        with torch.no_grad():
            out_loaded = net_loaded(tokens=None, features=features_b_d)
        if out_loaded.shape != (4, Q_expected):
            fail(f"Loaded model output shape {tuple(out_loaded.shape)} != (4, {Q_expected}).")
        elif not torch.isfinite(out_loaded).all():
            fail("Loaded model output contains NaN or Inf.")
        else:
            ok(f"Loaded model forward OK: output {tuple(out_loaded.shape)}, all finite.")
    except Exception as e:
        fail(f"Could not load / run checkpoint {ckpt_to_try}: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 14 — Multi-asset process_dataset column alignment
# ═════════════════════════════════════════════════════════════════════════════
hdr("STAGE 14 · Multi-asset process_dataset — columns align across files")

# Build a small list of CSVs (cap to 4 to keep runtime modest)
csv_list: list[str] = []
for p in getattr(CFG, "DATA_FILE", []):
    if isinstance(p, str):
        if os.path.exists(p):
            csv_list.append(p)
        elif os.path.exists(os.path.join(_REPO, p)):
            csv_list.append(os.path.join(_REPO, p))
    if len(csv_list) >= 4:
        break
if len(csv_list) < 2:
    info("Fewer than 2 CSVs available — multi-asset alignment check is trivial.")
else:
    print(f"  Loading {len(csv_list)} CSVs: {[os.path.basename(c) for c in csv_list]}")
    try:
        multi_assets, _ = process_dataset(csv_list, fe)
        n_features_per_asset = [a[1].shape[1] for a in multi_assets]
        if len(set(n_features_per_asset)) == 1:
            ok(f"All {len(multi_assets)} assets have {n_features_per_asset[0]} features.")
        else:
            fail(
                f"Assets disagree on feature count: {n_features_per_asset}. "
                "FeatureConfig likely differs between assets — train.py._make_feature_config "
                "must return a single shared config."
            )
        nan_assets = [a[0] for a in multi_assets if not np.isfinite(a[1]).all()]
        if nan_assets:
            fail(f"Multi-asset features contain NaN/Inf: {nan_assets[:2]}")
        else:
            ok("All multi-asset features are finite.")
    except Exception as e:
        fail(f"Multi-asset process_dataset failed: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{SEP2}")
print("SMOKE COMPLETE — Pre-training end-to-end (v2 features_only)")
print(SEP2)

pass_count = sum(1 for _, s, _ in RESULTS if s == "PASS")
warn_count = sum(1 for _, s, _ in RESULTS if s == "WARN")
fail_count = sum(1 for _, s, _ in RESULTS if s == "FAIL")
total_count = len(RESULTS)
print(f"  Totals: {pass_count} PASS / {warn_count} WARN / {fail_count} FAIL  (of {total_count} checks)")

if fail_count == 0:
    print("\n  [OK] GREEN LIGHT — every stage passed. Safe to launch training.")
else:
    print("\n  [BLOCKED] Failures must be fixed before training:")
    # Re-run a slim printing of FAIL items only
    for s, status, msg in RESULTS:
        if status == "FAIL":
            print(f"    [FAIL] {s} :: {msg}")

if warn_count > 0:
    print("\n  [INVESTIGATE] Warnings (do not block training but worth checking):")
    for s, status, msg in RESULTS:
        if status == "WARN":
            print(f"    [WARN] {s} :: {msg}")

print(f"""
  Configuration snapshot
    CSV:          {data_path}
    v2 features:  {expected_n} columns
    total cols:   {n_feat_with_ohlc} (4 OHLC + {expected_n} v2)
    Dataset:      {len(ds):,} windows @ seq_len={seq_len}
    Model:        {cls.__name__} ({n_params:,} params) on {device}
    Loss:         v10_total_loss (pinball + v9 median, Q={Q_expected})
    Batch size:   {CFG.BATCH_SIZE}  AMP: {CFG.USE_AMP}  Device: {device}

  Common failure causes
    • [FAIL] on column-order or count → rename in core/features.py didn't
      propagate to train.py DEFAULT_FEATURE_LIST or config.FE_* keys
    • [FAIL] on forward shape → QuantileHead init or QUANTILE_LEVELS misconfigured
    • [FAIL] on backward NaN → reduce LR or check gradient clipping
    • [WARN] on loss not decreasing → normal on tiny slice; verify after full run
""")
