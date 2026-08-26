# ORBIT Training Paradigm — Faithful Implementation Plan

## Source & Goal

Implement **ORBIT (Omni-Range Bootstrap Incremental Training)** from arXiv 2608.13262 ("Into the ORBIT for Time Series: Training Regimes for Foundation Models", Falcon-2.0 tech report) properly in this repo's pretrain pipeline, replacing the current "ORBIT-lite" (uniform context randomization only).

**Paper spec summary (verified against full text):**
- **Bootstrap Multi-Level Sampling**: per dataset, an offline sample index of five-tuples `(record r, variable v, context start s, context end e, horizon p)` built via four stochastic draws (Algorithm 1). Validity: `L_r ≥ 2P`, `0 ≤ s < e < L_r`, `e−s ≥ P`, `P ≤ p ≤ T_max`, `e+p ≤ L_r`. Contexts/horizons sampled ONCE offline; never resampled during optimization.
- **Dataset weighting + blending**: prescribed domain-aware weights → global stream via low-discrepancy greedy rule: for each slot j pick `argmax_d(w_d·j − c_d)`, where `c_d` is the cumulative count assigned to dataset d so far.
- **Omni-Range Incremental Training**: globally-shuffled stream consumed incrementally step-based (no epochs); batch assembly left-pads contexts / right-pads targets to mini-batch maxima with attention + loss masks excluding padding; memory-mapped/on-demand loading of tuples.
- Ablations confirm: joint (context × horizon) bootstrap sampling beats sliding-window enumeration and fixed-length variants on all metrics.

## User Decisions (resolved)

1. **Horizon axis**: Multi-horizon Oracle grid — precompute oracle labels for holds `{16, 32, 48, 96}` bars; Level-4 samples `p ~ Uniform` over feasible grid entries. Evaluation stays at 96 (= `FORECAST_HORIZON`/`ORACLE_MAX_HOLD`). Fine-tune unchanged (96 only).
2. **Sampler interaction**: Pretrain uses the ORBIT stream exclusively (replaces `ConcatDataset` + `DistributedWeightedSampler`). Class-weighted sampler remains for fine-tune folds only.
3. **Consumption**: Step-based single-pass stream in pretrain (no epochs, no early stopping on repeated data); LR schedule steps per optimizer step.
4. **Degenerate levels**: Each asset CSV = one "dataset"; each asset has one series and one supervised target (oracle score), so paper Levels 1–2 (record, variable) degenerate to identity within a dataset. Real work happens at corpus weighting/blending and Levels 3–4.

## Out of Scope

- Rank-Guided Cross-Depth Alignment, triple-channel tokenization, Falcon-2.0 backbone changes, multi-stage autoregressive inference (model stays LPatchTST as-is).
- Fine-tune pipeline (`finetune_fold`) — unchanged except it consumes the new checkpoint format transparently.
- Known-future covariates / group attention.

---

## Implementation Tasks

### Task 1 — Config additions (`core/config.py`)

```python
# ORBIT (arXiv:2608.13262)
ORBIT_STREAM_MODE   = True            # master switch; False = legacy epoch path
ORBIT_HORIZONS      = [16, 32, 48, 96]  # multi-horizon oracle grid (bars)
ORBIT_TOTAL_STEPS   = 20000           # stream length = TOTAL_STEPS * BATCH_SIZE * world_size
ORBIT_ASSET_WEIGHTS = None            # None = equal weight per asset; else {"<csv-stem>": w}
ORBIT_INDEX_SEED    = 20260813        # reproducible index construction
ORBIT_INDEX_CACHE   = "models/orbit_index.npz"
```

Keep existing `ORBIT_ENABLE`/`ORBIT_CTX_MIN` governing fine-tune context randomization untouched.

### Task 2 — Multi-horizon oracle labels (`core/train.py::process_dataset`, `core/oracle.py`)

- For each asset, run `generate_targets` once per hold `h ∈ ORBIT_HORIZONS` (numba-cached, ~19 assets × ~30k bars × 4 = cheap). Store `{h: target_array}` alongside existing features.
- Apply the same post-threshold zeroing (`|tgt| < SAMPLER_THRESHOLD → 0`) per horizon array.
- **Verify tail semantics first**: confirm how `generate_targets` handles bars within `h−1` of array end (forced-exit path incomplete). If labels are degenerate there, restrict valid split points to `e ≤ L − max(ORBIT_HORIZONS)` when building the index (Task 3) rather than trusting tail values.

### Task 3 — Bootstrap sample index builder (new `core/orbit_sampler.py`)

Faithful port of paper Algorithm 1 adapted to this repo:

- Per asset `d`: valid split range `e ∈ {ctx_min … L_d − P_max}` where `ctx_min = ORBIT_CTX_MIN` (≥ P), `P_max = max(ORBIT_HORIZONS)`. Draw:
  - `e ~ Uniform{ctx_min, L_d − P_max}`
  - `s ~ Uniform{max(0, e − LOOKBACK_WINDOW), e − ORBIT_CTX_MIN}` (context `[s, e−1]`; mirrors Eq. 31–32)
  - `p_idx ~ Uniform` over feasible horizons in `ORBIT_HORIZONS` with `e + p ≤ L_d` (Eq. 33 analog)
  - record/variable = identity (documented degenerate case)
- Index entry: `(asset_id, s, e, h)` stored as int arrays; `M_D` per asset proportional to its weight share of the total stream.
- Construct ONCE with seeded `np.random.Generator(PCG64(ORBIT_INDEX_SEED))`; save/load via `ORBIT_INDEX_CACHE` keyed by a config hash (assets, lengths, horizons, ctx bounds, seed) so ranks reuse instead of rebuilding (paper §5.3 determinism).

### Task 4 — Low-discrepancy greedy blending (`core/orbit_sampler.py`)

- Weights: normalize `ORBIT_ASSET_WEIGHTS` or uniform across assets.
- Stream length `M_stream = ORBIT_TOTAL_STEPS × BATCH_SIZE` (per-rank streams are disjoint slices).
- Greedy rule: slot j → `argmax_d(w_d·j − c_d)`; increment `c_d`; pop next local id from that asset's **independently shuffled** index deque (two-level shuffle per paper §4.3). Then shuffle the assembled stream once globally.
- Expose `build_orbit_stream(index, weights, M_stream) -> np.ndarray` pure function (unit-testable).

### Task 5 — Stream dataset + DDP sharding (`core/data_loader.py`)

- New `OrbitStreamDataset(IterableDataset)` yielding tuples from a contiguous shard of the stream: rank r gets slice `[r·M/B … (r+1)·M/B)` of the global stream (mirrors disjoint sharding in `DistributedWeightedSampler`).
- `__iter__` fetches per tuple: `features[s:e]` from the asset tensor, `target = targets[h][e−1]`. No copies of full windows cached (on-demand indexing of in-memory tensors ≈ memmap role here).
- Collate reuses `collate_with_none` left-padding + `key_padding_mask` (variable contexts coexist in one batch = Omni-Range assembly; single-step targets make loss masking trivial).

### Task 6 — Step-based pretrain loop (`core/train.py::pretrain`)

When `ORBIT_STREAM_MODE`:
- Build stream → `OrbitStreamDataset` loader; iterate exactly `ORBIT_TOTAL_STEPS` optimizer steps, one pass.
- Replace OneCycleLR with AdamW + linear warmup (`0.001` fraction) → cosine decay to `min_lr = lr/10` stepped per iteration (paper Table 4 analog).
- Drop early stopping; keep periodic val eval every N steps (val loader unchanged: fixed 512-context, horizon-96 labels, deterministic) for checkpoint selection (`pretrain_best.pth`) and logging.
- Legacy epoch path preserved under `ORBIT_STREAM_MODE=False`.
- Warm-start encoder logic and AMP/manual grad-norm clipping flow unchanged.

### Task 7 — Verification (`audits/verify_orbit.py` extension)

Add checks:
1. **Blending fidelity**: cumulative per-asset share within tolerance (±0.5%) of prescribed weights at every 1000-slot prefix of the stream.
2. **Index validity**: all tuples satisfy bounds (`s ≥ 0`, `e−s ≥ ORBIT_CTX_MIN`, `e+s ≤ L`, `p ∈ ORBIT_HORIZONS`, feasibility).
3. **Reproducibility**: same seed → byte-identical index + stream; different seed → different.
4. **DDP sharding**: shards disjoint, union == full stream.
5. **Target alignment**: fetched `(context, target, h)` equals direct recomputation from raw CSV through `process_dataset`.
6. **Tail safety**: no tuple's label bar lies in the oracle-incomplete tail region.
7. **Mask correctness**: variable-length batch → padding positions masked in attention and contribute zero loss.
8. **Smoke test**: tiny `ORBIT_TOTAL_STEPS` end-to-end CPU run (config-overridable).

### Task 8 — Docs

Update config comments (replace the current "ORBIT-style" block at `core/config.py:94–101` with pointers to the real implementation) — no separate README unless requested.

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Oracle tail labels degenerate near array end | Verify `generate_targets` tail behavior before index construction; clamp `e ≤ L − P_max` |
| Class imbalance in pretrain without weighted sampler (Flat-heavy) | Acceptable per decision #2 (fine-tune re-balances); log per-class batch composition every N steps |
| Horizon-grid labels shift score distribution vs 96-only training | Val metric stays 96-horizon; if pretrain val regresses, shrink grid via config (no code change) |
| Stream too short/long vs old epochs×dataset size | `ORBIT_TOTAL_STEPS` configurable; default sized ≈ prior pretrain pass count × steps-per-epoch |
| DDP world-size change invalidates cached stream sharding | Cache stores the INDEX (per-asset); stream is rebuilt cheaply per run/world-size |

## Validation Plan

1. Run extended `audits/verify_orbit.py` — all 8 checks pass.
2. Smoke pretrain: `ORBIT_TOTAL_STEPS=50` CPU run completes, loss decreases, checkpoint saved.
3. Short GPU pretrain (~2k steps): monitor blending shares in logs match weights; val loss (96-horizon) not worse than legacy path at equal wall-time.
4. Confirm fine-tune fold runs unchanged on the ORBIT-pretrained checkpoint and backtest engine still loads the model.
