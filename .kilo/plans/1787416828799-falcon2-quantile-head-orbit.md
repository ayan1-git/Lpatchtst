# Falcon-2.0-Inspired Upgrade: Quantile Head, Coverage Metrics, ORBIT Sampling

## Context

Verified against arXiv:2608.13262 (Falcon-2.0 / ORBIT) and the actual codebase (`core/` is source of truth; `notebooks/lpatch.ipynb` just launches `torchrun ... core/train.py`).

Current state (all verified):
- Scalar head: `SATURATION * tanh(x)` = 2.5·tanh (`core/model.py:158`)
- Loss v9: α=2.0 undershoot / β=1.5 overshoot quadratic + L1 flat-push (w=2.0) + smooth-L1 edge nudge (w=1.5), mag-weight `|t|^0.5` clamped [0.5, 2.0] (`core/loss.py:74-83`)
- Oracle 5.0: SL 2.0×ATR, TP 4.5×ATR, trail 3.3×ATR, max-hold 96 → target = ±tanh(r_net/2.5), then **zeroed below 0.08** (`core/train.py:244`) → **bimodal targets with point mass at exactly 0** (~46%+ of bars)
- Sequential windows `slice(i, i+seq_len)` + `DistributedWeightedSampler` (4th-root inverse-frequency class weights) (`core/data_loader.py:496-516, 528-578`)
- RobustScaler routing + hard IQR clip at 3.0 (`core/data_loader.py:138-163`)
- Learned positional embedding with linear interpolation for variable lengths (`core/model.py:309-319`)
- Inference: rolling-mean smoothing (3), policy threshold+bias from `models/best_policy.json`, decision = smoothed median score > ±threshold (`core/inference.py:344-385`)

Corrections to the original proposal that this plan incorporates:
1. The sketched one-sided head (`median + Σ softplus`) would force q05 ≥ q50. Use a symmetric two-sided parametrization.
2. Because targets have a point mass at 0, per-quantile coverage diagnostics are meaningful primarily on edge bars; on flat bars they saturate by construction.

## Decisions (user-confirmed)

| Decision | Choice |
|---|---|
| Buy/Sell/Hold rule | Median channel only (identical semantics to today). Spread logged + tuned offline first; live spread-gating deferred |
| Retraining | Warm-start: load `models/pretrain_best.pth` with `strict=False` (stem/encoder/decoder/pos-emb transfer; old scalar head dropped), then normal pretrain→finetune |
| RMSNorm/RoPE | Behind config flags, default OFF; A/B only after Phases 1–3 validate |

## Out of Scope

- Missingness channels (bars are contiguous)
- Horizon/max_hold randomization (labels are path-dependent SL/TP/trail outcomes)
- Rank-Guided Cross-Depth Alignment (paper §3.4)
- Live spread-gating until offline backtest validates it
- Any Kronos tokenizer changes

---

## Phase 1 — Quantile Head + Combined Loss

**`core/config.py`** — add:
```python
QUANTILE_HEAD      = True
QUANTILE_LEVELS    = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]   # 7 symmetric; includes q10/q90 spread anchors + q05/q95 tails
PINBALL_WEIGHT     = 1.0
V9_MEDIAN_WEIGHT   = 1.0    # λ on v9 loss applied to the median channel
WARM_START_ENCODER = True
```
Keep `SATURATION_FACTOR` (Oracle-only now).

**`core/model.py`** — add `QuantileHead`:
```python
class QuantileHead(nn.Module):
    # pool = "cls"|"mean" (reuse PredictionHead pooling semantics; "mixing"→"cls")
    def __init__(self, d_model, levels, dropout=0.0, pool="mean"):
        self.net = Linear(d_model, d_model//2), GELU, Dropout, Linear(d_model//2, len(levels))
        self.m = levels.index(0.5)
    def forward(self, dec_out):                      # (B, K, d) -> (B, Q) ascending
        x = dec_out[:, 0] if cls else dec_out.mean(1)
        raw = self.net(x).float()
        med  = raw[:, self.m:self.m+1]
        lo_inc = F.softplus(raw[:, :self.m]).flip(-1).cumsum(-1)   # distances below median
        hi_inc = F.softplus(raw[:, self.m+1:]).cumsum(-1)          # distances above
        lo = med - lo_inc.flip(-1)                       # q[m-1] >= ... wait: q05 = med - sum of all lower incs
        q = cat([med - lo_inc.flip(-1).cumsum... ])
        return ordered ascending concat                   # NO tanh anywhere; unbounded tails
```
Exact construction (must be two-sided): `lower_j = med − reverse(cumsum(softplus(raw[:m])))`, `upper_j = med + cumsum(softplus(raw[m+1:]))`; output `[lower_Q-1 … lower_0, med, upper_0 … upper_Q-1]`. Monotone by construction. No tanh saturation.

In `PatchTST`: when `config.QUANTILE_HEAD`, instantiate `QuantileHead` in place of `PredictionHead`/linear head; `forward` returns `(B, Q)`. Add method `decision_score(out)` returning the median column so all downstream consumers stay scalar. Update head-specific weight init loop for the new module.

**`core/loss.py`** — add:
```python
def pinball_loss(q_pred, target):        # (B,Q),(B,) in float32; mean over Q and B
    # max(q*(y−ŷ), (q−1)*(y−ŷ)) per level
def v10_total_loss(q_pred, target, **kw):
    return config.PINBALL_WEIGHT * pinball_loss(q_pred, target) \
         + config.V9_MEDIAN_WEIGHT * asymmetric_number_line_loss(q_pred[:, med_idx], target)
```
Keep `continuous_weighted_direction_loss` alias intact (audits import it). All math in float32 (AMP-safe, matches v9).

**`core/train.py`** — `_run_epoch` and `_full_eval_diagnostics`: detect `(B, Q)` output; compute loss via `v10_total_loss`; derive per-batch diagnostics (dir_acc, corr) from the median column. Warm-start in `pretrain()`: if `WARM_START_ENCODER` and ckpt exists, `load_state_dict(clean(ckpt), strict=False)` and print missing/unexpected keys (expect only head keys).

**Checkpoint note:** `models/*.pth` remain v9-lineage only; new runs save new-shaped heads.

## Phase 2 — Coverage & Spread Metrics

Extend `core/train.py::_full_eval_diagnostics` / `_print_diagnostics` and `core/evaluate.py` summary:
- **Per-quantile coverage on edge bars** (`target ≠ 0`): fraction `target ≥ ŷ_q` vs nominal q; flag drift > ±5pp. This replaces `mag_calibration` as the calibration diagnostic (keep `mag_calibration` on the median for run-over-run comparability with v9).
- Flat-bar coverage reported separately (informational; saturates due to zero point mass).
- **Spread**: mean `(q90−q10)` on edge vs flat bars; Spearman-ish corr(spread, |median|); precision at ±threshold stratified by spread terciles — the principled successor to `diag_false_sig.py` hunting.
- Per-quantile mean pinball loss.
- `evaluate.py`: policy search stays on median threshold/bias; write observed spread percentiles into `best_policy.json` output (inert until Phase-gated later).

## Phase 3 — ORBIT-Style Context Randomization (train split only)

**`core/config.py`**: `ORBIT_ENABLE=True`, `ORBIT_CTX_MIN=128`.

**`core/data_loader.py::FinancialDataset`**: add `orbit_randomize=False` ctor flag (True only for the train dataset in `create_multi_index_dataloaders` / `create_dataloaders` / `create_fold_dataloaders`). In `__getitem__(i)`:
- Window END fixed: `t = i + seq_len − 1` → target and sampler-class alignment unchanged (`DistributedWeightedSampler` untouched).
- Draw `L ~ uniform_int[ORBIT_CTX_MIN, seq_len]`; `start = max(0, t+1−L)`; slice features/tokens `[start : t+1]`.
- Use the worker-seeded `random` module (already seeded by `_worker_init_fn`); never global torch RNG.
- Val/test datasets always deterministic sequential.
- Variable lengths flow through existing pos-emb interpolation (`model.py:309-319`) and the length-agnostic LSTM/decoder. Do NOT pad/batch-trim — collate stacks equal-length-per-batch only if L happens to match; **add a collate fallback** that left-pads shorter contexts to batch-max with an attention/loss mask, OR simpler: bucket by drawing L once per `set_epoch` per index… Preferred simple route: draw L per item, and since batches are formed by the sampler, add padding+mask support to `collate_with_none` and a mask argument through `PatchTST.forward` (MultiheadAttention `key_padding_mask`). Keep masks excluding padded positions from attention; loss unaffected (targets are per-window-end scalars).

**Add audit** `audits/verify_orbit.py`: assert `y == targets[t]` for sampled items, L distribution spans [min,max], val/test loaders still yield fixed 512-length windows.

## Phase 4 — RMSNorm + RoPE (flags, default OFF)

**`core/config.py`**: `USE_MODERN_NORM=False`, `USE_ROPE=False`.

**`core/model.py`**:
- `RMSNorm(d_model)` (eps 1e-6, weight init ones); swap `nn.LayerNorm` in encoder/decoder layers when `USE_MODERN_NORM`.
- Minimal SDPA-based attention replacement (q/k/v/out projections + dropout) used ONLY when `USE_ROPE`, applying precomputed cos/sin rotary tables (max len = `num_patches`) to Q/K in **self-attention** paths. Decoder cross-attention: no RoPE (learned query tokens). Default path (`nn.MultiheadAttention`) untouched.
- Validate with a forward/backward smoke test both flag states; DDP `find_unused_parameters=True` already covers unused branches.

A/B only after Phases 1–3 results are recorded.

## Phase 5 — arcsinh vs IQR-Clip A/B

**`core/config.py`**: `SCALER_TAIL_MODE="clip"` (default) | `"arcsinh"`, `ARCSINH_TAU=1.0` (IQR units).

**`core/data_loader.py::ColumnSelectiveScaler.transform`**: robust bucket post-RobustScaler transform becomes `τ·asinh(x/τ)` instead of `np.clip` when mode=`arcsinh`. Keep the >2% tail-mass warning (computed vs old bound for comparability). Compare distributions with existing `audits/clip_audit.py`; A/B on one finetune fold: val coverage table + false_sig_rate + backtest PF.

---

## Validation Plan

1. **Unit smoke (Phase 1):** construct model, random `(B,K,d)` → head output ascending strictly monotone; pinball grad bounded ≤ max(q,1−q); AMP step NaN-free; warm-start prints expected missing-keys (head only).
2. **Short pretrain (~5 epochs):** loss ↓; val diagnostics: edge-bar coverage table near-nominal, false_sig_rate ≤ 28% (v8/v9 baseline), mag_calibration(median) ≥ 0.36× baseline.
3. **Full run:** pretrain → finetune folds → `evaluate.py` → backtest. Record PF / Net Ret / Win Rate / Test Std as a new entry in `docs/research_ledger.md` (KEEP/REVERT verdict per house convention).
4. Each subsequent phase lands only if its gate passes and ledger entry says KEEP.

## Risks

- Low quantiles collapse toward 0 on many samples (bimodal labels) — expected; interpret spread as P(edge)·magnitude uncertainty, not pure magnitude variance.
- Pinball + v9 double-penalize the median channel — tune `V9_MEDIAN_WEIGHT` down (0.5) if flat-bar behavior regresses.
- ORBIT padding/mask path touches collate + forward — highest regression risk; covered by `verify_orbit.py` + fixed-window val/test invariant.
