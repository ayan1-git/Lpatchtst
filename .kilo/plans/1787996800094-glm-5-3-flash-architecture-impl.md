# Small-Scale GLM-5.3-Flash Architecture for LPatchTST — Implementation Plan

## 1. Goal

Replace the current `EncoderLayer` / `DecoderLayer` stack in
`core/model.py` with a faithful small-scale replica of the GLM-5.3-Flash
language-model architecture, sized to fit the existing LPatchTST pipeline
(`D_MODEL=64`, `N_LAYERS=4`, `N_HEADS=8`, `LOOKBACK_WINDOW=512`,
`N_QUERIES=4`, `BATCH_SIZE=128`).

**Keep from current code:** `InputStem` (3 input modes), `PredictionHead`,
`QuantileHead`, loss, train loop, data loader, ORBIT sampler, tokenizer.
**Replace:** the encoder–decoder body. **Add:** mHC residual, KDA linear
attention, sparse MLA + lightning indexer, noaux_tc MoE FFN.

Out of scope: vision tower, MTP head, IndexPool, FP8, Chinese-chip serving,
DSA `topk=2048` (we use `topk=64` to match our 512-token context).

The plan targets *numerical correctness of the four pieces*, not
reproduction of the released weights (we have none of those — there is no
public GLM-5.3-Flash at 64-dim). Acceptance is a forward pass that runs,
trains, and matches the four component unit tests.

## 2. Existing project context (what we are building on)

From `core/config.py` and `core/model.py`:

| Setting | Value |
| --- | --- |
| `D_MODEL` | 64 |
| `N_HEADS` | 8 |
| `N_LAYERS` (encoder) | 4 |
| `N_DEC_LAYERS` (decoder) | 1 |
| `N_QUERIES` | 4 learnable query tokens |
| `LOOKBACK_WINDOW` | 512 |
| `PATCH_LEN` / `STRIDE` | 16 / 12 → ~42 patches (decoder-side) |
| `VOCAB_SIZE` | 2^(10+10) = 1 048 576 (Kronos hierarchical) |
| `FE_*` features | 9 model inputs (ret_norm, vol_ratio, log_ewma_vol, mom_norm) |
| `QUANTILE_LEVELS` | 9 |
| `SATURATION_FACTOR` | 2.5 |
| `HIDDEN_ACT` | silu (SwiGLU) |
| `USE_ROPE` | False (currently) |
| `USE_MODERN_NORM` | False (currently LayerNorm; the new code will use RMSNorm) |

Existing pieces we **re-use verbatim** (no edits):

- `InputStem`, `RMSNorm` (already in `model.py`), `FeedForward`
  (the SwiGLU we will mutate into MoE), `PredictionHead`, `QuantileHead`,
  the `dec_self_mask` causal decoder mask, the `_init_weights` rules.

## 3. Scaled-down architecture (the four GLM pieces, mini)

| GLM-5.3-Flash | LPatchTST scale | Rationale |
| --- | --- | --- |
| hidden 4096, 45 layers, 64 heads, 4 dense stem + 41 MoE | **hidden 64, 4 layers, 8 heads, 0 dense stem + 4 MoE** | Fits in <1M params. No "dense stem" at 4 layers — first layer is already MoE. |
| KDA: 64 heads × head_dim 128, short_conv k=4, 34 layers | **KDA: 8 heads × head_dim 64, short_conv k=4, 3 of 4 layers** | Mirrors the 3:1 KDA:Sparse ratio; head_dim=64 equals `D_MODEL//N_HEADS`. |
| Sparse MLA + Lightning Indexer: `q_lora_rank=1536, kv_lora_rank=512, qk_nope=256, v=256, indexer 32h×128d, topk=2048` | **Sparse MLA: `q_lora=64, kv_lora=32, qk_nope=64, v=64, indexer 4h×32d, topk=64`** | All ranks scaled by `D_MODEL/4096 = 1/64`. Topk=64 = 1/8 of context. |
| mHC: n=4 streams, 20 Sinkhorn iters, mapping `nH→n²+2n` per sublayer | **mHC: n=2 streams, 8 Sinkhorn iters, mapping `2H→6`** | `n=2` halves the parameter overhead. 8 iters is plenty at this scale. |
| MoE: 288 routed / 1 shared / top-8 / int=2048, sigmoid gate, noaux_tc, scale 2.5 | **MoE: 8 routed / 1 shared / top-2 / int=128, sigmoid gate, noaux_tc, scale 2.5** | Top-2 keeps active params at 2× expert + 1× shared. 8 experts is enough to make routing learnable; int=128 is 2× hidden. |
| SwiGLU `clamp(±10)` | **SwiGLU `clamp(±10)`** (unchanged) | Same as GLM. |
| NoPE on sparse layers | **NoPE on sparse layers, RoPE on KDA** (we will add RoPE because KDA benefits from positional info and the input is monotonic time) | Compromise: KDA keeps RoPE, sparse MLA does not — matches the asymmetry in GLM (where sparse has NoPE and KDA is a learned recurrence). |

Resulting encoder body (~50 K params, fits in 1 GB HBM at BF16 + 32-bit
optimizer):

```
Layer 0  : KDA  (linear attention, recurrence)  +  MoE(8x128, top-2)
Layer 1  : KDA                                  +  MoE
Layer 2  : KDA                                  +  MoE
Layer 3  : Sparse-MLA (NoPE) + Lightning Indexer + MoE
```

This is exactly the GLM pattern `[KDA, KDA, KDA, Sparse]` × N + tail. With
N=1 it is `[KDA×3, Sparse×1]`. The decoder remains a small causal-MHA +
cross-MHA stack over the 4 learnable queries (unchanged from current), so
we do not break the existing oracle/quantile interface.

## 4. Component specifications (small-scale)

### 4.1 mHC residual (`src/glm53/nn/mhc.py`)

Per sublayer, wrap a `(B, T, H)` stream into `(B, T, n*H)`, run the
sublayer on the aggregated 1-stream, mix back. Following Xie et al.
(arXiv 2512.24880) with n=2 and 8 Sinkhorn iterations.

```
x_n : [B, T, 2H]   (init = [x, x] for layer 0; identity mix for deeper layers)
x_flat = x_n.reshape(B, T, -1)                # [B, T, 2H]
x_n = RMSNorm(x_n)                            # per-stream
proj = Linear(2H, n²+2n)                      # 2H -> 6
alpha = [α_pre, α_post, α_res]  (3 learnable scalars, init 0)
bias  = [6]
h_pre  = sigmoid(proj[..., 0:2]  * α_pre  + bias[0:2])              # [B,T,2]
h_post = 2*sigmoid(proj[..., 2:4] * α_post + bias[2:4])             # [B,T,2]
h_res0 = exp(proj[..., 4:8].view(B,T,2,2) * α_res + bias[4:8].view(2,2))
H_res  = sinkhorn_knopp(h_res0, iters=8, eps=1e-6)                  # [B,T,2,2] doubly stochastic
# Aggregate n -> 1
z = h_pre[..., 0:1] * x_n[..., 0, :] + h_pre[..., 1:2] * x_n[..., 1, :]
# Sublayer
y = Sublayer(z)
# Expand + residual mix
y_n = stack(h_post[..., 0:1] * y, h_post[..., 1:2] * y, dim=-2)     # [B,T,2,H]
x_out = einsum("btsc, btsr -> btrc", y_n, H_res) + x_n              # [B,T,2,H]
```

Sinkhorn runs in fp32 with row-max subtraction for stability (same as
the mHC reference implementation, but with 8 iters instead of 20 — fine
at n=2, since the 2×2 doubly-stochastic set is a 1-D segment after
normalizing one row/col).

Tests: 100-layer mHC chain on `x~N(0,1)` keeps `‖x‖₂/√T ∈ [0.9, 1.1]`
(no explosion/vanish, the GLM paper's stability claim).

### 4.2 KDA linear attention (`src/glm53/nn/kda.py`)

Kimi Delta Attention as in Yang et al. 2025, with the bounded-sigmoid
decay from Kimi-K3 (`g_min = -5`, fixed). 8 heads, head_dim 64.

```
q, k = L2Norm(Swish(Conv1d(k=4, depthwise, causal)(W_qk x)))   # [B,T,8,64]
v    = Swish(Conv1d(k=4, depthwise, causal)(W_v x))           # [B,T,8,64]
beta = sigmoid(W_beta x)                                       # [B,T,8] per-head
g_logit = e^A * (W_alpha_up @ W_alpha_down x) + b_alpha        # [B,T,8,64] per-channel
g     = g_min * sigmoid(g_logit)                               # ∈ (g_min, 0)
alpha = exp(g)                                                 # ∈ (e^-5, 1)

RoPE on q, k only (per-head, GLM does not add RoPE to KDA, but we
do — see §3 rationale). Use the existing `RotarySelfAttention._apply_rope`
helper.

S_t   = (I - β k kᵀ) Diag(α) S_{t-1} + β k vᵀ
o     = S_t^T q
o     = RMSNorm(o) (head-wise)
o     = W_o (sigmoid(W_o_up @ W_o_down x) ⊙ o)                # input-dep output gate
```

For the forward we use the **recurrent** path (`for t in 0..T-1`),
not the chunkwise kernel — the dataset is short (≤512) so chunkwise
overhead is not worth the implementation cost. State shape per
sequence per head: `[64, 64]`. Total state at L=3, H=8, D=64 is
`3 × 8 × 64 × 64 = 98 304` floats per sample — trivial.

Tests: at fixed `alpha=1, beta=1`, the recurrence equals
`S_T = Σ_{t≤T} k_t v_tᵀ` and `o_T = q_Tᵀ S_T = Σ_{t≤T} (q_Tᵀ k_t) v_t`,
which is the unnormalized linear-attention formula (sanity check).

### 4.3 Sparse MLA + Lightning Indexer (`src/glm53/nn/sparse_mla.py`)

NoPE variant of DeepSeek's MLA with a DeepSeek-style indexer (we omit
`IndexPool` — at 512 tokens it is not needed and the project does not
run 1M contexts).

```
c_t   = W_dkv x_t                              # [B,T, kv_lora=32]
c_t   = RMSNorm(c_t)                           # kv_a_layernorm
k_C   = W_uk c_t                               # [B,T,8, qk_nope=64]
v     = W_uv c_t                               # [B,T,8, v=64]
q_c   = W_dq x_t                               # [B,T, q_lora=64]
q_c   = RMSNorm(q_c)                           # q_a_layernorm
q     = W_uq q_c                               # [B,T,8, qk_nope=64]   (NoPE)
                                      # qk_rope_head_dim = 0  -> no RoPE

# Lightning indexer (4 heads × head_dim 32, top-k=64)
i_q   = Wq_b q_c                               # [B,T,4*32] = [B,T,128]
i_k   = wk @ RMSNorm(x_t)                      # [B,T,128]
idx_logits = (i_q @ i_k.transpose) / sqrt(32)  # [B,4,T,T]
top_idx    = idx_logits.topk(64, dim=-1).indices  # [B,4,T,64]

# Sparse attention: gather then SDPA
k_gathered = k_C.gather(2, top_idx_2d)         # [B,8,T,64] (broadcast top_idx over H)
v_gathered = v.gather(2, top_idx_2d)
attn_out   = F.scaled_dot_product_attention(q, k_gathered, v_gathered, is_causal=True)
o          = W_o attn_out.flatten(-2)          # [B,T,H]
```

`top_idx_2d` is the same per-query index set broadcast across the 8
attention heads (the indexer is only 4 heads; we take the union of
the topk-64 sets and use that — this is the standard DSA "shared topk
across heads" simplification).

Tests: when `topk = T`, output matches plain SDPA with `is_causal=True`
to 1e-4 (no-RoPE parity check against `nn.MultiheadAttention` on the
same Q,K,V).

### 4.4 noaux_tc MoE (`src/glm53/nn/moe.py`)

```
gate = sigmoid(W_gate x) + e_score_correction_bias   # [B,T,8]
# noaux_tc: with n_group=1 and topk_group=1 this degenerates to
#   pick top-2 globally (no group scoring).
top2_w, top2_idx = gate.topk(2, dim=-1)
top2_w = top2_w / top2_w.sum(-1, keepdim=True) * routed_scaling_factor   # 2.5
shared_out = silu(W_up_shared x) * silu(W_gate_shared x) clamped ±10
              -> W_down_shared -> H
routed_out = sum_k top2_w[k] * Expert_k(x)
out = routed_out + shared_out
```

Each `Expert` is `down( silu(gate(x)).clamp(±10) * up(x).clamp(±10) )`
with `gate, up: H→int=128`, `down: 128→H`. Expert weights stored as
two 3-D params: `[8, H, 2*int]` and `[8, int, H]`, indexed with
`weight[top2_idx]` for efficiency. `e_score_correction_bias` is a
length-8 `nn.Parameter` initialized to 0.

Tests: with `gate_weight = 0`, `e_score_correction_bias = 0`, all
experts produce the same output; with `e_score_correction_bias[0] = 10`,
top-2 always includes expert 0.

## 5. New `EncoderLayer` and unchanged `DecoderLayer`

`EncoderLayer` becomes the mHC-wrapped GLM block:

```python
class GlmBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_experts, topk, expert_dim,
                 use_sparse_mla=False, use_mhc=True):
        self.use_mhc  = use_mhc
        self.use_sparse = use_sparse_mla
        self.hc_attn  = MHCWrapper(d_model, n_streams=2, iters=8) if use_mhc else None
        self.attn     = SparseMLA(...) if use_sparse_mla else KDA(...)
        self.hc_ffn   = MHCWrapper(...) if use_mhc else None
        self.moe      = NoAuxMoE(d_model, n_experts=8, topk=2,
                                 expert_dim=128, n_shared=1, scale=2.5)
        self.norm_attn = RMSNorm(d_model)
        self.norm_moe  = RMSNorm(d_model)

    def forward(self, x):
        # attn sublayer
        if self.use_mhc:
            x = self.hc_attn(self.attn, self.norm_attn, x)
        else:
            x = x + self.attn(self.norm_attn(x))
        # moe sublayer
        if self.use_mhc:
            x = self.hc_ffn(self.moe, self.norm_moe, x)
        else:
            x = x + self.moe(self.norm_moe(x))
        return x
```

`DecoderLayer` and the decoder stack remain the existing
`nn.MultiheadAttention`-based cross-attention over the 4 learnable
queries — this is the cleanest way to keep the quantile head's
input shape unchanged. The encoder body is what we are replacing;
the decoder stays small.

`PatchTST.__init__` therefore drops the `lstm` (KDA subsumes the
recurrent smoothing role) and replaces
`self.encoder_layers = nn.ModuleList([EncoderLayer(...) * N_LAYERS])`
with `self.encoder_layers = nn.ModuleList([GlmBlock(... use_sparse=(i==N_LAYERS-1))])`
plus an `mhc_expand` op that turns the 1-stream `input_stem` output
into a 2-stream input (just `cat([x, x], dim=-1)`) and an `mhc_collapse`
op that sums the 2 streams back to 1 before the final `encoder_norm`.

## 6. Code plan (atomic, ordered)

Each step ends in a runnable check. Files live in `core/` so we keep
the existing layout; subcomponents go in `core/glm53/nn/` for clarity.

1. **`core/glm53/nn/sinkhorn.py`** — `sinkhorn_knopp(M, iters, eps)`,
   fp32, row-max subtract. Smoke: random `[B,T,2,2]` input, all
   returned rows/cols sum to 1 ± 1e-5.

2. **`core/glm53/nn/mhc.py`** — `MHCSublayer(sublayer, d_model,
   n_streams=2, iters=8)`. Implements §4.1. Unit test: 100-layer chain
   on `x~N(0,1)` stays bounded; identity init (`alpha=0, bias=0`) gives
   `H_res = uniform = [[0.5,0.5],[0.5,0.5]]` and `H_pre = H_post = 0.5`,
   so output ≈ `0.5 * Sublayer(x) + x` at init (same gradient as a
   0.5-weighted residual).

3. **`core/glm53/nn/kda.py`** — `KDA(d_model, n_heads, head_dim,
   short_conv_k=4)`. Implements §4.2. Forward returns
   `(out, [B,T,n_heads,head_dim,head_dim])` so future work can read
   the recurrent state if needed; current code ignores it.

4. **`core/glm53/nn/sparse_mla.py`** — `SparseMLA(d_model, n_heads,
   q_lora_rank, kv_lora_rank, qk_nope_dim, v_dim, topk,
   indexer_n_heads, indexer_head_dim)`. Implements §4.3.

5. **`core/glm53/nn/moe.py`** — `NoAuxMoE(d_model, n_experts, topk,
   expert_dim, n_shared=1, scale=2.5, swiglu_limit=10.0)`.
   Implements §4.4. `forward` returns `(B, T, d_model)`. Router weight
   is fp32 per `moe_router_dtype` convention; everything else BF16.

6. **`core/glm53/block.py`** — `GlmBlock` (§5).

7. **Wire into `core/model.py`.** In `PatchTST.__init__`:
   - Replace `self.encoder_layers` (4 × `EncoderLayer`) with 4 ×
     `GlmBlock` where block 3 sets `use_sparse_mla=True`.
   - Add `self.mhc_expand = lambda x: torch.cat([x, x], dim=-1)` and
     `self.mhc_collapse = lambda x: x.sum(dim=-2)` (operates on the
     2-stream axis we will insert into the residual).
   - Keep `self.lstm = None` (set `LSTM_LAYERS = 0` in `config.py`).
   - In `forward`, between `input_stem + pos` and the encoder loop,
     call `mhc_expand`; before `encoder_norm`, call `mhc_collapse`.
   - Update `__init__` kwarg signature to accept the new MoE/MLA
     parameters, but keep every existing kwarg working (so old
     checkpoints / `train.py` calls don't break).

8. **`core/config.py`** — add the small-scale constants:
   ```python
   N_EXPERTS     = 8
   TOPK          = 2
   EXPERT_DIM    = 128
   Q_LORA_RANK   = 64
   KV_LORA_RANK  = 32
   QK_NOPE_DIM   = 64
   V_HEAD_DIM    = 64
   INDEX_N_HEADS = 4
   INDEX_HEAD_DIM= 32
   SPARSE_TOPK   = 64
   MHC_N_STREAMS = 2
   SINKHORN_ITERS= 8
   USE_MHC       = True
   USE_SPARSE_LAYER = True   # adds the 4th-layer sparse MLA
   ```
   All default to safe values. `LSTM_LAYERS` set to 0 implicitly via
   this change (we'll flip the existing `LSTM_LAYERS=1` to `0` in
   `config.py` and add a one-line comment that KDA replaces it).

9. **`core/train.py`** — no change to the training loop. `model.py`
   still returns `(B, Q)` or `(B, 1)`, the `QuantileHead` is intact,
   the ORBIT stream is intact. We just need to make sure the
   `forward` output shape is the same: a single tensor of shape
   `(B, len(QUANTILE_LEVELS))` for `QUANTILE_HEAD=True` or `(B,1)`
   otherwise. The decoder pipeline is unchanged.

10. **`core/inference.py`, `core/evaluate.py`, `core/backtest_engine.py`**
    — no changes expected; they call `model(tokens=..., features=...)`
    and read `model.decision_score(pred)`. Verified by inspection.

11. **Smoke test (`scripts/smoke_glm_block.py`)** — instantiates
    `GlmBlock(d=64)` four times (last one `use_sparse=True`), runs
    a `(B=4, T=128, H=64)` input through, asserts output is finite
    and same shape; also runs `(B=2, T=512)` to confirm KDA is
    numerically stable at the max context.

12. **End-to-end smoke (`scripts/smoke_full.py`)** — build
    `LPatchTST(input_mode="features_only", d_model=64, n_layers=4,
    n_heads=8)`, run a batch of `(B=2, T=512, F=9)` features through,
    check the output is `(B, 9)` (quantile) and finite. Then a
    `QUANTILE_HEAD=False` run for `(B, 1)`.

13. **Regression check (`tests/test_architecture_swap.py`)** — the
    four component unit tests, runnable via `pytest -k glm53`.
    Catches future regressions if someone edits the new modules.

## 7. Implementation order (single PR, in this order)

| # | File | Done when |
| -- | ---- | --------- |
| 1 | `core/glm53/nn/sinkhorn.py` | `pytest -k sinkhorn` passes |
| 2 | `core/glm53/nn/mhc.py` | `pytest -k mhc` passes; 100-layer chain stable |
| 3 | `core/glm53/nn/kda.py` | `pytest -k kda` passes; parity vs naive recurrence |
| 4 | `core/glm53/nn/sparse_mla.py` | `pytest -k sparse_mla` passes; `topk=T` ≡ plain MHA |
| 5 | `core/glm53/nn/moe.py` | `pytest -k moe` passes; bias selects expert 0 |
| 6 | `core/glm53/block.py` | Imports clean; `GlmBlock(use_sparse_mla=True/False)` constructs |
| 7 | `core/model.py` edits | `LPatchTST` builds and forward runs |
| 8 | `core/config.py` edits | `LSTM_LAYERS=0`, new constants exposed |
| 9 | `scripts/smoke_glm_block.py` | script runs, prints finite-output OK |
| 10 | `scripts/smoke_full.py` | full `LPatchTST` forward returns `(B, 9)` |
| 11 | `tests/test_architecture_swap.py` | `pytest -k glm53` green |
| 12 | end-to-end: `python core/train.py --pretrain-only --steps 100` | one tiny pretrain step runs without OOM and loss is finite |

## 8. Validation criteria (acceptance)

1. `pytest -k glm53` passes 100% (4 component tests).
2. `python core/train.py --pretrain-only --steps 100` (we will wire
   this flag if it doesn't exist; if not, run a 100-step subset of
   `pretrain()` directly) finishes a single mixed-precision step with
   finite loss on `LOOKBACK_WINDOW=512, BATCH_SIZE=128`.
3. Forward + backward peak HBM on the 4-layer, d=64, B=128, T=512
   setup < 4 GB (this is a sanity check; the original was < 2 GB).
4. `param_count(model) < 2_000_000` (target ~500K – 1.5M).
5. `model.decision_score(...)` still returns `(B,)` so the existing
   backtest engine and inference scripts work unchanged.

## 9. Risks and mitigations

- **KDA numerical stability at 512 tokens.** Channel-wise `alpha` with
  `g_min = -5` keeps per-step retention `> e^-5 ≈ 0.0067`, so 512-step
  cumulative decay is `> e^-5*512 = e^-2560 ≈ 0` in float32 — fine.
  Worst case we may need to renormalize the state matrix every 64
  steps; the unit test in step 3 will catch this.
- **Lightning Indexer topk=64 is small enough to gather cheaply, but
  the indexer itself is a 4×32 head full-attention over 512 tokens.**
  That is `B*4*512*512*2` FLOPs per sparse layer — at B=128 it is
  ~0.27 GFLOPs, fine on a single GPU. If too slow, lower indexer
  to 2 heads × 16 dim (still gives 32-dim indexer key space).
- **MoE routing with 8 experts / top-2 may collapse (always pick the
  same 2).** This is a known failure mode of small-expert MoE.
  Mitigation: add a small `router_z_loss` (logsumexp penalty) of
  weight `1e-3` in the loss, gated by `config.MOE_AUX_LOSS_COEF`
  defaulting to `1e-3` (set to 0 to disable). We will not modify
  `loss.py` — instead, the model returns the raw load-balancing
  statistics via `model.last_router_logits` so the training loop can
  add the aux term. We add this in `core/train.py` as a single
  additive line, gated on `getattr(model, 'last_router_logits',
  None) is not None`.
- **mHC at 2 streams doubles the residual bandwidth.** At our scale
  this is irrelevant (<0.1 ms/layer). We keep n=2 for simplicity.
- **NoPE on the sparse layer + RoPE on KDA asymmetry.** The model
  may under-train the position-invariant features. Mitigation: add
  a single `pos_embed` linear that injects a learned time-sinusoid
  into the sparse layer's input — same trick GLM uses for the
  indexer. We skip this in the first cut; the smoke test will tell
  us if loss explodes.

## 10. Out of scope (defer)

- Vision tower (no images in this project).
- MTP / speculative decoding (training a single next-step target
  is what `QuantileHead` already does).
- IndexPool (no 1M context here).
- FP8 weights / serving.
- Fused Triton kernels. We use plain PyTorch + `F.scaled_dot_product_attention`.
  At d=64, L=512, B=128, 4 layers, this runs comfortably on a single
  consumer GPU.

## 11. Open questions for the implementer

- **`LSTM_LAYERS=1` in current `config.py`** — the LSTM will be set
  to 0 in the swap. Confirm: OK to drop LSTM entirely, or do you
  want a tiny `nn.LSTM(d_model, d_model, num_layers=1)` in front of
  the encoder body for residual smoothing? Plan default: **drop**.
- **Router aux loss**: should the new MoE add a load-balancing aux
  loss into `v10_total_loss` automatically (we can monkey-patch
  inside `model.py.forward` to add a small term to a returned
  auxiliary tensor that the training loop picks up), or should we
  keep `loss.py` untouched and only return router stats? Plan
  default: **return stats only, leave `loss.py` untouched**, then
  add the aux term in `train.py` as a one-liner.

---

### References

- GLM-5.3-Flash: `huggingface.co/zai-org/GLM-5.3-Flash`,
  Z.ai blog 2026-08-26, Sebastian Raschka notes 2026-08-26.
- mHC: Xie et al., arXiv 2512.24880.
- KDA: Yang et al., "Kimi Linear" arXiv 2510.14976; Kimi K3 bounded
  decay (arXiv 2607.24653).
- DeepSeek MLA + Sparse (lightning indexer): DeepSeek-V3.2-Exp.
- Project source: `core/model.py`, `core/config.py`, `core/loss.py`,
  `core/train.py`, `docs/research_ledger.md`.
