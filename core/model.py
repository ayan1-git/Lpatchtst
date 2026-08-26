from __future__ import annotations
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
import config

SATURATION = getattr(config, "SATURATION_FACTOR", 2.5)

class InputMode(str, Enum):
    TOKENS_ONLY   = "tokens_only"    # discrete tokens from KronosTokenizer only
    FEATURES_ONLY = "features_only"  # 21 engineered features only (original mode)
    COMBINED      = "combined"       # tokens + features concatenated then projected


class InputStem(nn.Module):
    """
    Converts raw inputs into a unified (B, L, d_model) tensor
    regardless of input_mode. The rest of the model never needs
    to know which mode is active.
    """
    def __init__(self, input_mode: InputMode, d_model: int,
                 n_tokens: int,         # vocab size for token embedding (hierarchical sum)
                 n_features: int,       # number of engineered features (e.g. 21)
                 s1_bits: int = 6,
                 s2_bits: int = 6,
                 dropout: float = 0.2):
        super().__init__()
        self.mode = InputMode(input_mode)
        self.d_model = d_model
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits

        if self.mode in (InputMode.TOKENS_ONLY, InputMode.COMBINED):
            expected_vocab = 2 ** (s1_bits + s2_bits)
            if n_tokens is not None and int(n_tokens) != expected_vocab:
                raise ValueError(
                    f"InputStem hierarchical vocab_size mismatch: "
                    f"n_tokens={n_tokens}, expected {expected_vocab} from "
                    f"s1_bits={s1_bits}, s2_bits={s2_bits}"
                )
            self.embed_coarse = nn.Embedding(2 ** s1_bits, d_model)
            self.embed_fine   = nn.Embedding(2 ** s2_bits, d_model)
            self.tok_dropout = nn.Dropout(dropout)

        if self.mode in (InputMode.FEATURES_ONLY, InputMode.COMBINED):
            self.feature_proj = nn.Linear(n_features, d_model)

        if self.mode == InputMode.COMBINED:
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model * 2),
                nn.SiLU(),
                nn.Linear(d_model * 2, d_model)
            )

    def forward(self, tokens=None, features=None):
        if self.mode == InputMode.TOKENS_ONLY:
            assert tokens is not None, "tokens required for tokens_only mode"
            idx_c, idx_f = tokens
            emb = self.embed_coarse(idx_c) + self.embed_fine(idx_f)
            return self.tok_dropout(emb)

        elif self.mode == InputMode.FEATURES_ONLY:
            assert features is not None, "features required for features_only mode"
            return self.feature_proj(features)

        elif self.mode == InputMode.COMBINED:
            assert tokens is not None and features is not None
            idx_c, idx_f = tokens
            tok_emb  = self.embed_coarse(idx_c) + self.embed_fine(idx_f)
            feat_emb = self.feature_proj(features)
            fused = self.gate(torch.cat([tok_emb, feat_emb], dim=-1))
            return fused


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


_USE_MODERN_NORM = lambda: getattr(config, "USE_MODERN_NORM", False)


def _make_norm(d_model: int) -> nn.Module:
    return RMSNorm(d_model) if _USE_MODERN_NORM() else nn.LayerNorm(d_model)


class RotarySelfAttention(nn.Module):
    """Minimal SDPA-based multi-head self-attention with optional RoPE.

    Drop-in replacement for nn.MultiheadAttention on self-attention paths.
    Callable signature mirrors MHA: forward(q, k, v, ...) -> (out, None).
    Masks are converted to additive float masks internally (inputs follow the
    nn.MultiheadAttention convention: True = blocked).
    """

    def __init__(self, d_model: int, n_heads: int, max_len: int,
                 dropout: float = 0.0, use_rope: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout
        self.use_rope = use_rope

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model)

        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("rope_cos", freqs.cos(), persistent=False)
        self.register_buffer("rope_sin", freqs.sin(), persistent=False)

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, H, D); rotate consecutive pairs per head dim
        L, D = x.shape[1], x.shape[-1]
        cos = self.rope_cos[:L].to(dtype=x.dtype).view(1, L, 1, D // 2)
        sin = self.rope_sin[:L].to(dtype=x.dtype).view(1, L, 1, D // 2)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        return out

    def forward(self, query, key, value,
                attn_mask: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None):
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        q = self.q_proj(query).view(B, Lq, self.n_heads, self.head_dim)
        k = self.k_proj(key).view(B, Lk, self.n_heads, self.head_dim)
        v = self.v_proj(value).view(B, Lk, self.n_heads, self.head_dim)

        if self.use_rope:
            q = self._apply_rope(q)
            k = self._apply_rope(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        add_mask = None
        if attn_mask is not None or key_padding_mask is not None:
            add_mask = torch.zeros(B, 1, Lq, Lk,
                                   device=query.device, dtype=query.dtype)
            if attn_mask is not None:
                add_mask = add_mask.masked_fill(
                    attn_mask.bool().view(1, 1, Lq, Lk), float("-inf"))
            if key_padding_mask is not None:
                add_mask = add_mask.masked_fill(
                    key_padding_mask.bool().view(B, 1, 1, Lk), float("-inf"))

        o = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=add_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        o = o.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.out_proj(o), None


class FeedForward(nn.Module):
    """SwiGLU feed-forward network: w2(SiLU(w1(x)) * w3(x))"""
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.w3 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.w2 = nn.Linear(dim_feedforward, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class EncoderLayer(nn.Module):
    """Bidirectional self-attention encoder layer with pre-LN (or pre-RMSNorm)."""
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int,
                 dropout: float = 0.0, max_len: int = 4096):
        super().__init__()
        self.norm1 = _make_norm(d_model)
        if getattr(config, "USE_ROPE", False):
            self.self_attn: nn.Module = RotarySelfAttention(
                d_model, n_heads, max_len=max_len, dropout=dropout,
                use_rope=True,
            )
        else:
            self.self_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            )
        self.norm2 = _make_norm(d_model)
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        attn_out = self.self_attn(h, h, h, key_padding_mask=key_padding_mask)[0]
        x = x + self.dropout(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x


class DecoderLayer(nn.Module):
    """Causal self-attention + cross-attention decoder layer with pre-LN."""
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int,
                 dropout: float = 0.0, max_len: int = 4096):
        super().__init__()
        self.norm1 = _make_norm(d_model)
        if getattr(config, "USE_ROPE", False):
            self.self_attn: nn.Module = RotarySelfAttention(
                d_model, n_heads, max_len=max_len, dropout=dropout,
                use_rope=True,
            )
        else:
            self.self_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            )
        self.norm2 = _make_norm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm3 = _make_norm(d_model)
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor,
                self_attn_mask: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=self_attn_mask
        )[0])
        x = x + self.dropout(self.cross_attn(
            self.norm2(x), enc_output, enc_output,
            key_padding_mask=key_padding_mask
        )[0])
        x = x + self.ffn(self.norm3(x))
        return x


class PredictionHead(nn.Module):
    """Pools decoder query outputs and projects to a scalar prediction."""
    def __init__(self, d_model: int, dropout: float = 0.0, pool: str = "mean"):
        super().__init__()
        self.pool = pool.lower().strip()
        if self.pool == "mixing":
            self.pool = "cls"
        if self.pool not in {"mean", "cls"}:
            raise ValueError(f"PredictionHead pool must be 'mean', 'cls', or 'mixing', got {pool!r}")
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, decoder_output: torch.Tensor) -> torch.Tensor:
        # decoder_output: (B, K, d_model) → pool over queries → (B, d_model)
        if self.pool == "cls":
            x = decoder_output[:, 0]
        else:
            x = decoder_output.mean(dim=1)
        return SATURATION * torch.tanh(self.net(x))


class QuantileHead(nn.Module):
    """Monotone multi-quantile head (Falcon-2.0 style).

    Pools decoder query outputs and emits len(levels) quantiles in ascending
    order via a symmetric two-sided parametrization around the median:

        q_{m+j}   = median + cumsum(softplus(raw_upper))_j
        q_{m-1-j} = median - cumsum(softplus(raw_lower))_j

    Monotone by construction; unbounded tails (no tanh saturation).
    """
    def __init__(self, d_model: int, levels, dropout: float = 0.0, pool: str = "mean",
                 hidden_mult: int = 1):
        super().__init__()
        p = pool.lower().strip()
        if p == "mixing":
            p = "cls"
        if p not in {"mean", "cls"}:
            raise ValueError(f"QuantileHead pool must be 'mean', 'cls', or 'mixing', got {pool!r}")
        self.pool = p
        self.levels = [float(q) for q in levels]
        if 0.5 not in self.levels:
            raise ValueError("QUANTILE_LEVELS must contain 0.5 (median channel)")
        self.m = self.levels.index(0.5)
        n_upper = len(self.levels) - self.m - 1
        if self.m == 0 or n_upper == 0:
            raise ValueError("QUANTILE_LEVELS need at least one level below and above 0.5")
        hidden = max(8, (d_model // 2) * int(hidden_mult))
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, len(self.levels)),
        )

    @property
    def output_dim(self) -> int:
        return len(self.levels)

    def forward(self, decoder_output: torch.Tensor) -> torch.Tensor:
        if self.pool == "cls":
            x = decoder_output[:, 0]
        else:
            x = decoder_output.mean(dim=1)
        raw = self.net(x).float()
        med = raw[:, self.m:self.m + 1]

        lower_inc = F.softplus(raw[:, :self.m])
        lower = med - torch.flip(
            torch.cumsum(torch.flip(lower_inc, dims=[-1]), dim=-1), dims=[-1]
        )
        upper_inc = F.softplus(raw[:, self.m + 1:])
        upper = med + torch.cumsum(upper_inc, dim=-1)

        return torch.cat([lower, med, upper], dim=-1)


class PatchTST(nn.Module):
    """
    Encoder-Decoder PatchTST with multi-modal input support.

    Architecture:
    1. InputStem: (tokens, features) -> (B, L, d_model)
    2. Positional Embedding (with interpolation for variable lengths)
    3. LSTM (optional): Temporal smoothing/context
    4. Encoder: Bidirectional self-attention over full sequence
    5. Decoder: Learnable query tokens with causal self-attention + cross-attention to encoder
    6. PredictionHead: Pool queries -> projection -> tanh
    """

    def __init__(
        self,
        seq_len: int = 400,
        num_features: int = 21,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        n_dec_layers: int = 1,
        n_queries: int = 4,
        lstm_layers: int = 0,
        dropout: float = 0.2,
        aggregation: str = "mixing",
        input_mode: str = "features_only",
        vocab_size: int = 4096,
        s1_bits: int = 6,
        s2_bits: int = 6,
        n_features: int | None = None,
        **legacy_kwargs,
    ):
        super().__init__()

        if n_features is not None:
            num_features = n_features

        self.seq_len      = int(seq_len)
        self.num_features = int(num_features)
        self.patch_len    = int(patch_len)
        self.stride       = int(stride)
        self.d_model      = int(d_model)
        self.n_heads      = int(n_heads)
        self.n_layers     = int(n_layers)
        self.n_dec_layers = int(n_dec_layers)
        self.n_queries    = int(n_queries)
        self.lstm_layers  = int(lstm_layers)
        self.dropout_rate = float(dropout)
        self.aggregation  = aggregation.lower().strip()
        self.input_mode   = InputMode(input_mode)

        # ── 1. Input Stem ────────────────────────────────────────────────────
        self.input_stem = InputStem(
            input_mode=self.input_mode,
            d_model=self.d_model,
            n_tokens=vocab_size,
            n_features=self.num_features,
            s1_bits=s1_bits,
            s2_bits=s2_bits,
            dropout=dropout
        )

        # ── 2. Positional Embedding ──────────────────────────────────────────
        self.num_patches = self.seq_len
        self.register_buffer(
            "pos_embedding_base",
            torch.randn(1, self.num_patches, self.d_model) * 0.02
        )
        self.dropout = nn.Dropout(dropout)

        # ── 3. LSTM (Temporal Context) ───────────────────────────────────────
        if self.lstm_layers > 0:
            self.lstm = nn.LSTM(
                input_size=self.d_model,
                hidden_size=self.d_model,
                num_layers=self.lstm_layers,
                batch_first=True,
                dropout=dropout if self.lstm_layers > 1 else 0,
            )
        else:
            self.lstm = None

        # ── 4. Encoder (Bidirectional) ───────────────────────────────────────
        ff_dim = self.d_model * 2
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(self.d_model, self.n_heads, ff_dim, dropout,
                         max_len=self.num_patches)
            for _ in range(self.n_layers)
        ])
        self.encoder_norm = _make_norm(self.d_model)

        # ── 5. Decoder (Causal Self-Attn + Cross-Attn) ───────────────────────
        self.query_embed = nn.Embedding(self.n_queries, self.d_model)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(self.d_model, self.n_heads, ff_dim, dropout,
                         max_len=self.n_queries)
            for _ in range(self.n_dec_layers)
        ])
        self.decoder_norm = _make_norm(self.d_model)

        # Pre-compute causal mask for decoder self-attention (K x K)
        dec_mask = torch.triu(
            torch.ones(self.n_queries, self.n_queries), diagonal=1
        ).bool()
        self.register_buffer("dec_self_mask", dec_mask)

        # ── 6. Prediction Head ───────────────────────────────────────────────
        self.use_quantile_head = bool(getattr(config, "QUANTILE_HEAD", False))
        if self.use_quantile_head:
            self.head = None
            self.feature_head: nn.Module = QuantileHead(
                self.d_model, config.QUANTILE_LEVELS, dropout,
                pool=self.aggregation,
                hidden_mult=getattr(config, "QUANTILE_HEAD_HIDDEN_MULT", 1),
            )
        elif self.aggregation == "mean":
            self.head = nn.Linear(self.d_model, 1)
            self.feature_head = None
        else:
            self.head = None
            self.feature_head = PredictionHead(self.d_model, dropout, pool=self.aggregation)

        self.apply(self._init_weights)

        for proj in filter(None, [getattr(self, "head", None), getattr(self, "feature_head", None)]):
            if isinstance(proj, nn.Linear):
                nn.init.trunc_normal_(proj.weight, std=0.02)
                nn.init.zeros_(proj.bias)
            elif isinstance(proj, (PredictionHead, QuantileHead)):
                for layer in proj.net:
                    if isinstance(layer, nn.Linear):
                        nn.init.trunc_normal_(layer.weight, std=0.02)
                        nn.init.zeros_(layer.bias)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)

    @staticmethod
    def median_index(levels=None) -> int:
        lvls = list(levels if levels is not None else getattr(config, "QUANTILE_LEVELS", []))
        if not lvls:
            raise ValueError("No QUANTILE_LEVELS configured")
        return lvls.index(0.5)

    def decision_score(self, pred: torch.Tensor) -> torch.Tensor:
        """Collapse a (B, Q) quantile output to the scalar decision score (median)."""
        if self.use_quantile_head and pred.dim() > 1 and pred.size(-1) > 1:
            return pred[..., self.median_index()]
        return pred

    def forward(self, tokens=None, features=None,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B = tokens[0].shape[0] if tokens is not None else features.shape[0]

        # Step 1: Unified embedding via stem
        x = self.input_stem(tokens=tokens, features=features)   # (B, L, d_model)

        # Step 2: Positional Embedding — interpolate if seq len changed
        num_tokens_actual = x.shape[1]
        if num_tokens_actual == self.num_patches:
            pos = self.pos_embedding_base
        else:
            pos = F.interpolate(
                self.pos_embedding_base.transpose(1, 2),
                size=num_tokens_actual,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        x = x + pos
        x = self.dropout(x)

        # Step 3: LSTM (if present)
        if self.lstm is not None:
            x, _ = self.lstm(x)

        # Step 4: Encoder (bidirectional, no causal mask)
        for layer in self.encoder_layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        enc_output = self.encoder_norm(x)  # (B, L, d_model)

        # Step 5: Decoder (shared, for direction query tokens)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, K, d_model)
        for layer in self.decoder_layers:
            queries = layer(queries, enc_output,
                            self_attn_mask=self.dec_self_mask,
                            key_padding_mask=key_padding_mask)
        dec_output = self.decoder_norm(queries)  # (B, K, d_model)

        # Step 6: Prediction Head
        if self.use_quantile_head:
            return self.feature_head(dec_output)   # (B, Q)
        if self.aggregation == "mean":
            x = self.head(dec_output.mean(dim=1))
            return SATURATION * torch.tanh(x)
        return self.feature_head(dec_output)


class LPatchTST(PatchTST):
    """
    Refined LPatchTST with encoder-decoder architecture.
    """
    def __init__(self,
                 input_mode: str = "combined",
                 n_features: int = 21,
                 vocab_size: int = 4096,
                 s1_bits: int = 6,
                 s2_bits: int = 6,
                 d_model: int = 128,
                 patch_len: int = 8,
                 stride: int = 4,
                 n_dec_layers: int = 1,
                 n_queries: int = 4,
                 **kwargs):
        super().__init__(
            input_mode=input_mode,
            num_features=n_features,
            vocab_size=vocab_size,
            s1_bits=s1_bits,
            s2_bits=s2_bits,
            d_model=d_model,
            patch_len=patch_len,
            stride=stride,
            n_dec_layers=n_dec_layers,
            n_queries=n_queries,
            **kwargs
        )
