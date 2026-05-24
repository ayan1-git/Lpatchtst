from __future__ import annotations
import contextlib
import torch
import torch.nn as nn
from enum import Enum

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
                 s2_bits: int = 6):
        super().__init__()
        self.mode = InputMode(input_mode)
        self.d_model = d_model
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits

        if self.mode in (InputMode.TOKENS_ONLY, InputMode.COMBINED):
            self.embed_coarse = nn.Embedding(2 ** s1_bits, d_model)
            self.embed_fine   = nn.Embedding(2 ** s2_bits, d_model)
            self.tok_dropout = nn.Dropout(0.15)  # NEW: Prevent token memorization

        if self.mode in (InputMode.FEATURES_ONLY, InputMode.COMBINED):
            self.feature_proj = nn.Linear(n_features, d_model)

        if self.mode == InputMode.COMBINED:
            # After summing token_emb + feature_proj, project back to d_model
            # Use a gating mechanism so model learns how much to trust each source
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model * 2),
                nn.SiLU(),
                nn.Linear(d_model * 2, d_model)
            )

    def forward(self, tokens=None, features=None):
        """
        tokens   : tuple (idx_coarse, idx_fine) each (B, L) — from tokenizer.encode(half=True)
                   OR None if mode is features_only
        features : (B, L, n_features) float tensor
                   OR None if mode is tokens_only
        Returns  : (B, L, d_model)
        """
        if self.mode == InputMode.TOKENS_ONLY:
            assert tokens is not None, "tokens required for tokens_only mode"
            idx_c, idx_f = tokens
            emb = self.embed_coarse(idx_c) + self.embed_fine(idx_f)   # (B, L, d_model)
            return self.emb_dropout(emb)   # Apply dropout here

        elif self.mode == InputMode.FEATURES_ONLY:
            assert features is not None, "features required for features_only mode"
            return self.feature_proj(features)                          # (B, L, d_model)

        elif self.mode == InputMode.COMBINED:
            assert tokens is not None and features is not None
            idx_c, idx_f = tokens
            tok_emb  = self.embed_coarse(idx_c) + self.embed_fine(idx_f)  # (B, L, d_model)
            feat_emb = self.feature_proj(features)                         # (B, L, d_model)
            # Gated fusion — model learns weighting between discrete and continuous
            fused = self.gate(torch.cat([tok_emb, feat_emb], dim=-1))     # (B, L, d_model)
            return fused


class PatchTST(nn.Module):
    """
    Channel-independent PatchTST with support for multi-modal input stems.
    
    Architecture:
    1. InputStem: (tokens, features) -> (B, L, d_model)
    2. Patching: Unfold -> (B, num_patches, patch_len * d_model) -> Linear(d_model)
    3. Positional Embedding
    4. LSTM (Optional): Temporal smoothing/context
    5. Transformer Encoder: Patch-to-Patch attention
    6. Aggregation Head: Global pooling -> Projection
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
        lstm_layers: int = 0,
        dropout: float = 0.2,
        aggregation: str = "mixing",
        input_mode: str = "features_only",
        vocab_size: int = 4096,
        s1_bits: int = 6,
        s2_bits: int = 6,
        **legacy_kwargs,
    ):
        super().__init__()

        self.seq_len      = int(seq_len)
        self.num_features = int(num_features)
        self.patch_len    = int(patch_len)
        self.stride       = int(stride)
        self.d_model      = int(d_model)
        self.n_heads      = int(n_heads)
        self.n_layers     = int(n_layers)
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
            s2_bits=s2_bits
        )

        # ── 2. Patching ──────────────────────────────────────────────────────
        self.patch_embed = nn.Linear(self.patch_len * self.d_model, self.d_model)
        self.num_patches = (self.seq_len - self.patch_len) // self.stride + 1
        self.register_buffer(
            "pos_embedding_base",
            torch.randn(1, self.num_patches, self.d_model) * 0.02
        )
        self.dropout = nn.Dropout(dropout)

        # ── 4. LSTM (Temporal Context) ───────────────────────────────────────
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

        # ── 5. Transformer Encoder ───────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.enc_dropout = nn.Dropout(dropout)

        # ── 6. Head ──────────────────────────────────────────────────────────
        if self.aggregation == "mean":
            self.head = nn.Linear(self.num_patches * self.d_model, 1)
        else: # "mixing"
            self.feature_head = nn.Linear(self.d_model, 1)

        self.apply(self._init_weights)

        for proj in filter(None, [getattr(self, "head", None), getattr(self, "feature_head", None)]):
            nn.init.trunc_normal_(proj.weight, std=0.02)
            nn.init.zeros_(proj.bias)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            # Suppress OOV noise: unseen tokens will default to near-zero rather than heavy noise
            nn.init.normal_(m.weight, std=0.005)

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)

    def forward(self, tokens=None, features=None) -> torch.Tensor:
        """
        Args:
            tokens   : tuple (s1, s2) each (B, L) or None
            features : (B, L, F) float or None
        """
        # Step 1: Unified embedding via stem
        x = self.input_stem(tokens=tokens, features=features)   # (B, L, d_model)

        # Step 2: Patching
        # unfold: (B, L, d_model) -> (B, num_patches, patch_len, d_model)
        x = x.unfold(1, self.patch_len, self.stride)
        # flatten: (B, num_patches, patch_len * d_model)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # project: (B, num_patches, d_model)
        x = self.patch_embed(x)

        # Step 3: Positional Embedding — interpolate if seq len changed
        num_patches_actual = x.shape[1]
        if num_patches_actual == self.num_patches:
            pos = self.pos_embedding_base
        else:
            # Linear interpolation to handle variable-length sequences at val/test
            pos = torch.nn.functional.interpolate(
                self.pos_embedding_base.transpose(1, 2),   # (1, d_model, num_patches)
                size=num_patches_actual,
                # Linear interpolation for 1D sequence data
                mode='linear',
                align_corners=False
            ).transpose(1, 2)                              # (1, num_patches_actual, d_model)
        x = x + pos
        x = self.dropout(x)

        # Step 4: LSTM (if present)
        if self.lstm is not None:
            x, _ = self.lstm(x)

        # Step 5: Transformer Encoder
        x = self.encoder(x)
        x = self.enc_dropout(x)

        # Step 6: Aggregation
        if self.aggregation == "mean":
            x_flat = x.reshape(x.shape[0], -1)
            return torch.tanh(self.head(x_flat))
        else:
            # Global average pooling over patches
            pooled = torch.mean(x, dim=1)
            return torch.tanh(self.feature_head(pooled))


class LPatchTST(PatchTST):
    """
    Refined LPatchTST that uses InputStem and follows the 
    Patch -> LSTM -> Transformer -> Head pipeline.
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
            **kwargs
        )