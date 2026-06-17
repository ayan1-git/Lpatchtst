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
    """Bidirectional self-attention encoder layer with pre-LN."""
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.ffn(self.norm2(x))
        return x


class DecoderLayer(nn.Module):
    """Causal self-attention + cross-attention decoder layer with pre-LN."""
    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor,
                self_attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=self_attn_mask
        )[0])
        x = x + self.dropout(self.cross_attn(
            self.norm2(x), enc_output, enc_output
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
            EncoderLayer(self.d_model, self.n_heads, ff_dim, dropout)
            for _ in range(self.n_layers)
        ])
        self.encoder_norm = nn.LayerNorm(self.d_model)

        # ── 5. Decoder (Causal Self-Attn + Cross-Attn) ───────────────────────
        self.query_embed = nn.Embedding(self.n_queries, self.d_model)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(self.d_model, self.n_heads, ff_dim, dropout)
            for _ in range(self.n_dec_layers)
        ])
        self.decoder_norm = nn.LayerNorm(self.d_model)

        # Pre-compute causal mask for decoder self-attention (K x K)
        dec_mask = torch.triu(
            torch.ones(self.n_queries, self.n_queries), diagonal=1
        ).bool()
        self.register_buffer("dec_self_mask", dec_mask)

        # ── 6. Prediction Head ───────────────────────────────────────────────
        if self.aggregation == "mean":
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
            elif isinstance(proj, PredictionHead):
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

    def forward(self, tokens=None, features=None) -> torch.Tensor:
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
            x = layer(x)
        enc_output = self.encoder_norm(x)  # (B, L, d_model)

        # Step 5: Decoder (shared, for direction query tokens)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, K, d_model)
        for layer in self.decoder_layers:
            queries = layer(queries, enc_output, self_attn_mask=self.dec_self_mask)
        dec_output = self.decoder_norm(queries)  # (B, K, d_model)

        # Step 6: Prediction Head
        if self.aggregation == "mean":
            x = self.head(dec_output.mean(dim=1))
            return SATURATION * torch.tanh(x)
        else:
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
