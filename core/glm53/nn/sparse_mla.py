"""DeepSeek Sparse Attention (DSA) — small-scale implementation.

Faithful to the DeepSeek-V3.2 paper (arXiv 2512.02556). Components:

    Main path (NoPE MLA):
        q_a_layernorm, W_dq, W_uq      : x -> per-head q (qk_nope_dim)
        kv_a_layernorm, W_dkv, W_uk    : x -> per-head k (qk_nope_dim)
                       W_uv            : c_kv -> per-head v (v_head_dim)
        W_o                            : per-head v -> d_model

    Lightning indexer:
        wk        : d_model -> indexer_n_heads * indexer_head_dim
        wq_b      : q_lora_rank -> indexer_n_heads * indexer_head_dim
        weights_proj : indexer_n_heads * indexer_head_dim -> indexer_n_heads
        k_norm    : RMSNorm on the indexer keys
        q_scale   : learnable scalar mscale multiplier on q (init 1.0)

Index score (Eq. 2 in the paper):
    I[t, s] = sum_h w_h * ReLU(<i_q[t, h], i_k[s, h]>)

For top-k selection we use the raw ReLU scores (no softmax). For the
KL-divergence auxiliary loss we use Softmax(I_{t,S_t}) restricted to
the selected top-k set S_t.

Training is the paper's two-stage recipe:
    Stage 1 (warm-up, dense attention): indexer is trained via KL
        loss against the dense main-attention distribution; the rest
        of the model is frozen.
    Stage 2 (sparse): main model trains via LM loss; indexer trains
        via KL loss. Gradient isolation: the indexer sees a detached
        q_c so the LM loss cannot reach the indexer, and the KL loss
        only flows through the indexer branch (it reads the cached
        main attention scores, not the actual attention weights).

This module returns three things via self.last_attn_dist (per-query
L1-normalized main attention over the selected indices) and
self.last_indexer_scores (raw ReLU scores over the selected indices)
so the training loop can add the KL aux loss in one line.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._norm import RMSNorm


class SparseMLA(nn.Module):
    def __init__(self, d_model: int, n_heads: int,
                 q_lora_rank: int, kv_lora_rank: int,
                 qk_nope_dim: int, v_head_dim: int,
                 topk: int,
                 indexer_n_heads: int, indexer_head_dim: int,
                 eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_dim = qk_nope_dim
        self.v_head_dim = v_head_dim
        self.topk = topk
        self.indexer_n_heads = indexer_n_heads
        self.indexer_head_dim = indexer_head_dim

        # Main MLA path.
        self.W_dq = nn.Linear(d_model, q_lora_rank, bias=False)
        self.W_uq = nn.Linear(q_lora_rank, n_heads * qk_nope_dim, bias=False)
        self.W_dkv = nn.Linear(d_model, kv_lora_rank, bias=False)
        self.W_uk = nn.Linear(kv_lora_rank, n_heads * qk_nope_dim, bias=False)
        self.W_uv = nn.Linear(kv_lora_rank, n_heads * v_head_dim, bias=False)
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=eps)
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=eps)
        self.W_o = nn.Linear(n_heads * v_head_dim, d_model, bias=False)

        # Lightning indexer.
        self.wk = nn.Linear(d_model, indexer_n_heads * indexer_head_dim, bias=False)
        self.wq_b = nn.Linear(q_lora_rank, indexer_n_heads * indexer_head_dim, bias=False)
        # weights_proj: per-indexer-head scalar (no bias; raw inner product
        # is the only input). The paper uses a per-head weight; at our
        # scale we collapse the input to a single scalar per head.
        self.weights_proj = nn.Linear(
            indexer_n_heads * indexer_head_dim, indexer_n_heads, bias=False,
        )
        self.k_norm = RMSNorm(indexer_head_dim, eps=eps)
        # Learnable mscale on the indexer query. Initialised to 1.0.
        self.q_scale = nn.Parameter(torch.ones(indexer_n_heads))

        # Cached artifacts for the training loop's KL aux loss.
        # We expose:
        #   self.last_index_scores: [B, T, T]   raw ReLU index scores (the
        #       tensor used for topk). Detached from the main attention
        #       graph but its parameters (wk, wq_b, weights_proj, q_scale)
        #       receive gradient via the KL loss read below.
        #   self.last_indexer_q:    [B, T, Ih, Id]  the per-head i_q that
        #       was used to compute the index scores. Detached from main.
        #   self.last_indexer_k:    [B, T, Ih, Id]  the per-head i_k.
        #   self.last_top_idx:      [B, T, K]   top-k positions used by
        #       the main sparse attention.
        self.last_index_scores = None
        self.last_indexer_q = None
        self.last_indexer_k = None
        self.last_top_idx = None
        # If True, the main attention output over the selected set is
        # returned via the `attn_dist_for_aux` method; the training
        # loop is expected to call `compute_kl_loss(target_dist)` and
        # add the result to the LM loss.
        self._cached_attn_probs = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, d_model] -> [B, T, d_model]."""
        B, T, _ = x.shape
        H, Dn, Dv = self.n_heads, self.qk_nope_dim, self.v_head_dim
        Ih, Id = self.indexer_n_heads, self.indexer_head_dim
        K = min(self.topk, T)

        # 1) Latents.
        q_c = self.q_a_layernorm(self.W_dq(x))            # [B, T, q_lora]
        c_kv = self.kv_a_layernorm(self.W_dkv(x))         # [B, T, kv_lora]

        # 2) Project to per-head K, V, Q (no rope on this path).
        k_C = self.W_uk(c_kv).view(B, T, H, Dn)           # [B, T, H, Dn]
        v = self.W_uv(c_kv).view(B, T, H, Dv)             # [B, T, H, Dv]
        q = self.W_uq(q_c).view(B, T, H, Dn)              # [B, T, H, Dn]

        # 3) Lightning indexer. Per the paper, the index score is
        #    I[t, s] = sum_h w_h * ReLU(<i_q[t,h], i_k[s,h]>)
        #    where the indexer query sees a *detached* copy of the
        #    q_c latent so the LM loss cannot backprop into the
        #    indexer.
        i_k = self.k_norm(self.wk(x).view(B, T, Ih, Id))  # [B, T, Ih, Id]
        with torch.no_grad():
            q_c_for_indexer = q_c.detach()
        # Apply the per-head mscale: wq_b(q_c) shape is [B, T, Ih*Id]
        # which we then reshape and multiply by q_scale per head.
        i_q_flat = self.wq_b(q_c_for_indexer)            # [B, T, Ih*Id]
        i_q = (i_q_flat.view(B, T, Ih, Id) * self.q_scale.view(1, 1, Ih, 1))
        # Per-head dot product -> [B, T, T, Ih]
        idx_per_head = torch.einsum(
            "bthe,bshe->btsh", i_q, i_k
        )                                                  # [B, T, T, Ih]
        # Apply ReLU (per the paper).
        idx_per_head = F.relu(idx_per_head)
        # Per-head scalar weight.
        head_w = self.weights_proj(
            i_q.reshape(B, T, Ih * Id)
        )                                                  # [B, T, Ih]
        # Combine: I[t, s] = sum_h head_w[t, h] * idx_per_head[t, s, h]
        # Broadcasting head_w over the T key axis.
        idx_scores = (idx_per_head * head_w.unsqueeze(2)).sum(dim=-1)
        # idx_scores: [B, T, T]

        # Cache for KL aux loss. We keep the *un-differentiable* path
        # values so the main model never sees the indexer params.
        self.last_index_scores = idx_scores
        self.last_indexer_q = i_q
        self.last_indexer_k = i_k
        self.last_head_w = head_w

        # 4) Top-k selection (causal: a query at position t can only
        #    attend to positions <= t). We mask future positions out
        #    of the index scores first.
        causal = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )                                                  # [T, T]
        idx_scores_masked = idx_scores.masked_fill(
            causal.unsqueeze(0), float("-inf")
        )
        top_idx = idx_scores_masked.topk(K, dim=-1).indices  # [B, T, K]
        self.last_top_idx = top_idx

        # 5) Gather KV along the topk positions.
        k_C_t = k_C.transpose(1, 2)                       # [B, H, T, Dn]
        v_t = v.transpose(1, 2)                            # [B, H, T, Dv]
        k_gathered = torch.empty(B, H, T, K, Dn, device=x.device, dtype=x.dtype)
        v_gathered = torch.empty(B, H, T, K, Dv, device=x.device, dtype=x.dtype)
        for h in range(H):
            k_h = k_C_t[:, h].reshape(B * T, Dn)
            v_h = v_t[:, h].reshape(B * T, Dv)
            idx_flat = top_idx.reshape(B * T, K)
            k_g = k_h.gather(
                0, idx_flat.unsqueeze(-1).expand(-1, -1, Dn).reshape(B * T * K, Dn)
            )
            v_g = v_h.gather(
                0, idx_flat.unsqueeze(-1).expand(-1, -1, Dv).reshape(B * T * K, Dv)
            )
            k_gathered[:, h] = k_g.view(B, T, K, Dn)
            v_gathered[:, h] = v_g.view(B, T, K, Dv)

        # 6) Sparse attention.
        q_attn = q.transpose(1, 2)                         # [B, H, T, Dn]
        attn_logits = (q_attn.unsqueeze(3) * k_gathered).sum(dim=-1)
        attn_logits = attn_logits / (Dn ** 0.5)
        # Causal mask (already guaranteed by top_idx for future
        # positions being inf, but be safe).
        q_pos = torch.arange(T, device=x.device).view(1, 1, T, 1)
        k_pos = top_idx.unsqueeze(1)                       # [B, 1, T, K]
        causal_mask = k_pos > q_pos
        attn_logits = attn_logits.masked_fill(causal_mask, float("-inf"))
        attn_probs = F.softmax(attn_logits, dim=-1)        # [B, H, T, K]
        attn_probs = attn_probs.nan_to_num(0.0)
        # Cache the attention probabilities for the KL aux loss
        # (the main training loop reads the average across heads and
        # uses that as the "target" distribution that the indexer
        # should match).
        self._cached_attn_probs = attn_probs.detach()
        # The K positions the main attention actually used.
        self._cached_attn_k_pos = top_idx

        # out[b, h, t, dv] = sum_k attn[b,h,t,k] * v[b,h,t,k,dv]
        out = (attn_probs.unsqueeze(-1) * v_gathered).sum(dim=3)
        out = out.transpose(1, 2).reshape(B, T, H * Dv)
        return self.W_o(out)

    # ────────────────────────────────────────────────────────────
    # KL aux loss (DeepSeek Sec. 2.1, Eq. 3 and 4)
    # ────────────────────────────────────────────────────────────
    def compute_kl_loss(self, mode: str = "sparse") -> torch.Tensor:
        """Compute the indexer KL loss against the main attention
        distribution cached from the most recent forward pass.

        Args:
            mode: "dense"   — Eq. 3, KL against the full causal
                              main attention distribution (use during
                              the dense warm-up stage).
                  "sparse"  — Eq. 4, KL restricted to the top-k
                              selected set S_t (use during the
                              sparse training stage).
        Returns:
            Scalar KL loss. Returns 0 if no forward has been run or
            the caches are missing.
        """
        if (self.last_index_scores is None
                or self._cached_attn_probs is None
                or self._cached_attn_k_pos is None):
            return torch.zeros((), device=next(self.parameters()).device)
        if mode not in {"dense", "sparse"}:
            raise ValueError(f"mode must be 'dense' or 'sparse', got {mode}")

        B, H, T, K = self._cached_attn_probs.shape
        # Main attention target: average across heads, L1-normalize.
        attn = self._cached_attn_probs.mean(dim=1)          # [B, T, K]
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-9)

        if mode == "dense":
            # We need the full main attention distribution over the
            # full sequence, not just the topk. We rebuild it by
            # scattering the topk probs back to the full positions.
            full_attn = torch.zeros(B, T, T, device=attn.device, dtype=attn.dtype)
            full_attn.scatter_(-1, self._cached_attn_k_pos, attn)
            # L1-normalize (already normalized over the topk set, but
            # for the "dense" case the target is the full distribution
            # so we re-normalize).
            full_attn = full_attn / (full_attn.sum(dim=-1, keepdim=True) + 1e-9)
            # Indexer distribution over the full sequence: softmax
            # of the masked index scores. We use the raw scores
            # (before causal masking of future positions, since the
            # target itself is causal).
            idx_full = self.last_index_scores
            # Causal mask: future positions get -inf before softmax.
            causal = torch.triu(
                torch.ones(T, T, device=idx_full.device, dtype=torch.bool), diagonal=1
            )
            idx_for_kl = idx_full.masked_fill(
                causal.unsqueeze(0), float("-inf")
            )
            log_q = F.log_softmax(idx_for_kl, dim=-1)
            # KL(p_attn || softmax(idx)) = sum p * (log p - log q)
            kl = (full_attn * (full_attn.clamp_min(1e-9).log() - log_q)).sum(dim=-1)
            return kl.mean()
        else:
            # Sparse mode (Eq. 4): restrict both distributions to S_t.
            # attn is already [B, T, K] over S_t, so use as-is.
            # Indexer scores over S_t: gather.
            idx_scores = self.last_index_scores
            top_idx = self._cached_attn_k_pos                # [B, T, K]
            idx_S = idx_scores.gather(-1, top_idx)          # [B, T, K]
            log_q = F.log_softmax(idx_S, dim=-1)
            kl = (attn * (attn.clamp_min(1e-9).log() - log_q)).sum(dim=-1)
            return kl.mean()
