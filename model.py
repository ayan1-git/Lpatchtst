from __future__ import annotations
import contextlib
import torch
import torch.nn as nn


class PatchTST(nn.Module):
    """
    Channel-independent PatchTST with two aggregation modes:
    - aggregation="mean"   : Legacy mode. Flattens patches -> Linear -> Mean.
    - aggregation="mixing" : New mode. Pools patches -> Linear -> Learnable Mixing.

    Backward compatible with older keyword arguments:
    seqlen, numfeatures, patchlen, dmodel, nheads, nlayers

    Fixes applied (vs. previous version):
    - num_patches now uses floor division (//) instead of float division.
    - Geometry check no longer requires (seq_len - patch_len) % stride == 0;
      torch.unfold handles non-divisible strides via floor division natively.
    - Added ValueError when use_tokenizer=True and num_features != 1 to
      prevent silent incorrect forward passes.
    - score_dropout moved from the (B, F) scalar scores to post-encoder
      representations, preventing catastrophic feature zeroing with small F.
    """

    def __init__(
        self,
        seq_len: int = 400,
        num_features: int = 21,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
        aggregation: str = "mixing",
        use_tokenizer: bool = False,
        vocab_size: int = 4096,
        **legacy_kwargs,
    ):
        # ---- Backward-compatible arg aliases ----
        if "seqlen"       in legacy_kwargs: seq_len      = legacy_kwargs.pop("seqlen")
        if "numfeatures"  in legacy_kwargs: num_features = legacy_kwargs.pop("numfeatures")
        if "patchlen"     in legacy_kwargs: patch_len    = legacy_kwargs.pop("patchlen")
        if "dmodel"       in legacy_kwargs: d_model      = legacy_kwargs.pop("dmodel")
        if "nheads"       in legacy_kwargs: n_heads      = legacy_kwargs.pop("nheads")
        if "nlayers"      in legacy_kwargs: n_layers     = legacy_kwargs.pop("nlayers")

        if legacy_kwargs:
            raise TypeError(f"Unexpected kwargs: {sorted(legacy_kwargs.keys())}")

        super().__init__()

        # ---- Validation ----
        if seq_len <= 0 or num_features <= 0:
            raise ValueError("seq_len and num_features must be positive.")
        if patch_len <= 0 or stride <= 0:
            raise ValueError("patch_len and stride must be positive.")
        if seq_len < patch_len:
            raise ValueError("seq_len must be >= patch_len.")
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads}). "
                f"head_dim would be {d_model / n_heads:.2f} (non-integer)."
            )
        if not (0 < d_model and 0 < n_heads and 0 < n_layers):
            raise ValueError("d_model, n_heads, and n_layers must all be positive integers.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}.")

        # BUG FIX 3: use_tokenizer=True hard-codes F=1 in forward(); passing
        # num_features > 1 would silently ignore all but the first channel.
        if use_tokenizer and num_features != 1:
            raise ValueError(
                "use_tokenizer=True requires num_features=1 "
                "(the sequence is treated as a single discrete token stream). "
                f"Got num_features={num_features}."
            )

        aggregation = aggregation.lower().strip()
        if aggregation not in ("mean", "mixing"):
            raise ValueError("aggregation must be one of: 'mean', 'mixing'.")

        self.seq_len       = int(seq_len)
        self.num_features  = int(num_features)
        self.patch_len     = int(patch_len)
        self.stride        = int(stride)
        self.d_model       = int(d_model)
        self.aggregation   = aggregation
        self.use_tokenizer = use_tokenizer
        self.vocab_size    = vocab_size

        # BUG FIX 1: Use floor division (//) instead of float division (/).
        # Previously: int((seq_len - patch_len) / stride) + 1
        # BUG FIX 2: Removed the overly strict % stride == 0 check.
        # torch.unfold() uses floor division internally, so any stride that
        # keeps num_patches >= 1 is valid. The old check rejected configs like
        # (seq_len=401, patch_len=16, stride=8) that unfold handles perfectly.
        self.num_patches = (self.seq_len - self.patch_len) // self.stride + 1

        if self.num_patches < 1:
            raise ValueError(
                f"Configuration produces num_patches={self.num_patches} < 1. "
                f"Ensure seq_len ({seq_len}) >= patch_len ({patch_len})."
            )

        # ---- Patching & Encoding ----
        if self.use_tokenizer:
            # Token Mode: embed discrete tokens, then project patches.
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            # Patch aggregation: (patch_len * d_model) -> d_model
            self.patch_embedding = nn.Linear(self.patch_len * d_model, d_model)
        else:
            self.token_embedding = None
            self.patch_embedding = nn.Linear(self.patch_len, self.d_model)

        # pos_embedding scaled to 0.02 (standard BERT/GPT convention) so the
        # positional signal adds to patch content rather than overwhelming it.
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, self.d_model) * 0.02
        )
        self.dropout = nn.Dropout(dropout)

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

        # ---- Heads ----
        if self.aggregation == "mean":
            self.head         = nn.Linear(self.num_patches * self.d_model, 1)
            self.feature_head = None
            self.mixing_layer = None
        else:
            self.head         = None
            self.feature_head = nn.Linear(self.d_model, 1)

            mixing_in_dim = 1 if self.use_tokenizer else self.num_features
            if mixing_in_dim > 1:
                # BUG FIX 4: enc_dropout applied post-encoder before pooling.
                # Old score_dropout was on (B, F) scalars — with F=6, p=0.2
                # it randomly zeroed entire feature contributions ~20% of the
                # time, which is catastrophic. Dropout on the high-dimensional
                # (B*F, num_patches, d_model) tensor is safe and effective.
                self.mixing_layer = nn.Linear(mixing_in_dim, 1)
            else:
                # Tokenizer mode (F=1): no mixing needed, no dropout on scalars.
                self.mixing_layer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L)    longs  — if use_tokenizer=True
               (B, L, F) floats — if use_tokenizer=False
        Returns:
            (B, 1) float in [-1, 1] after tanh.
        """
        if self.use_tokenizer:
            if x.dim() != 2:
                raise ValueError(
                    f"Expected 2D input (B, L) for tokenizer mode, got {tuple(x.shape)}"
                )
            B, L = x.shape
            if L != self.seq_len:
                raise ValueError(f"Sequence length mismatch: got {L}, expected {self.seq_len}.")

            # 1. Embed tokens: (B, L) -> (B, L, d_model)
            x = self.token_embedding(x)

            # 2. Patching over sequence dim:
            #    unfold -> (B, num_patches, d_model, patch_len)
            #    permute -> (B, num_patches, patch_len, d_model)
            x_patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
            x_patches = x_patches.permute(0, 1, 3, 2).contiguous()

            # 3. Project: (B, num_patches, patch_len * d_model) -> (B, num_patches, d_model)
            enc_in = self.patch_embedding(x_patches.reshape(B, self.num_patches, -1))
            F = 1  # single feature stream in tokenizer mode

        else:
            if x.dim() != 3:
                raise ValueError(
                    f"Expected 3D input (B, L, F), got {tuple(x.shape)}"
                )
            B, L, F = x.shape
            if L != self.seq_len or F != self.num_features:
                raise ValueError(
                    f"Shape mismatch: got (L={L}, F={F}), "
                    f"expected (L={self.seq_len}, F={self.num_features})."
                )

            # 1. Channel-independent patching: (B, L, F) -> (B, F, L)
            x = x.permute(0, 2, 1).contiguous()
            # Unfold: (B, F, num_patches, patch_len)
            x_patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
            # Merge batch & feature dims: (B*F, num_patches, patch_len)
            x_patches = x_patches.reshape(B * F, self.num_patches, self.patch_len)
            # Project: (B*F, num_patches, d_model)
            enc_in = self.patch_embedding(x_patches)

        # ---- Positional embedding + Encoder ----
        # pos_embedding (1, num_patches, d_model) broadcasts over B or B*F.
        enc_in  = enc_in + self.pos_embedding
        enc_in  = self.dropout(enc_in)
        enc_out = self.encoder(enc_in)  # (B or B*F, num_patches, d_model)
        enc_out = self.enc_dropout(enc_out)

        # ---- Aggregation & Output ----
        if self.aggregation == "mean":
            enc_flat    = enc_out.reshape(enc_out.shape[0], -1)   # (B*F, num_patches * d_model)
            per_feature = self.head(enc_flat).squeeze(-1).view(B, F)          # (B, F)
            out         = per_feature.mean(dim=1, keepdim=True)   # (B, 1)
            return torch.tanh(out)

        else:  # "mixing"
            # Pool over patches: (B or B*F, d_model)
            pooled = torch.mean(enc_out, dim=1)
            # Per-feature score: (B*F, 1) -> (B, F)
            feature_scores = self.feature_head(pooled).squeeze(-1).view(B, F)

            if F > 1:
                out = self.mixing_layer(feature_scores)  # (B, 1)
            else:
                out = feature_scores                     # (B, 1) directly — tokenizer mode

            return torch.tanh(out)


class LPatchTST(nn.Module):
    """
    Two-stage hybrid: LSTM channel-wise denoiser → PatchTST encoder.
    Stage 1: Shared LSTM(input_size=1, hidden_size=d_model) per channel.
    Stage 2: PatchTST Transformer on denoised hidden state patches.
    """
    def __init__(self, seq_len, num_features, d_model, patch_len, stride,
                 n_heads, n_layers, lstm_layers=1, dropout=0.2, aggregation="mixing"):
        super().__init__()
        # BUG FIX 3: validation
        if aggregation != "mixing":
            raise ValueError("LPatchTST only supports aggregation='mixing'.")

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads}). "
                f"head_dim would be {d_model / n_heads:.2f} (non-integer)."
            )
        if not (0 < d_model and 0 < n_heads and 0 < n_layers):
            raise ValueError("d_model, n_heads, and n_layers must all be positive integers.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}.")

        self.seq_len = seq_len
        self.num_features = num_features
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.lstm_layers = lstm_layers

        # Guard for num_patches
        self.num_patches = (seq_len - patch_len) // stride + 1
        if self.num_patches < 1:
            raise ValueError(
                f"Configuration produces num_patches={self.num_patches} < 1. "
                f"Ensure seq_len ({seq_len}) >= patch_len ({patch_len})."
            )

        # Stage 1: shared LSTM, channel-independent
        # Using hidden_size = d_model to ensure alignment with Stage 2.
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )


        # Stage 2: PatchTST encoder (reuse your existing building blocks)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head (mixing mode)
        self.enc_dropout = nn.Dropout(dropout)
        self.feature_head = nn.Linear(d_model, 1)
        self.mixing_layer = nn.Linear(num_features, 1)

    def forward(self, x):  # x: (B, L, F)
        B, L, F = x.shape
        # BUG FIX 2: validation
        if not x.is_floating_point():
            raise ValueError(
                f"LPatchTST expects float input, got dtype={x.dtype}. "
                "Did you forget to cast integer indices before passing to the float branch?"
            )
        if L != self.seq_len:
            raise ValueError(
                f"seq_len mismatch: got {L}, expected {self.seq_len}"
            )
        if F != self.num_features:
            raise ValueError(
                f"num_features mismatch: got {F}, expected {self.num_features}"
            )

        orig_dtype  = x.dtype
        x_ci        = x.permute(0, 2, 1).reshape(B * F, L, 1).to(torch.float32)
        device_type = x.device.type

        # Disable autocast for LSTM only on backends that support the context manager.
        # MPS does not support torch.amp.autocast (PyTorch < 2.3 raises, ≥ 2.3 is a no-op
        # since MPS LSTM already runs in float32). Use nullcontext() as a safe passthrough.
        _autocast_ctx = (
            torch.amp.autocast(device_type=device_type, enabled=False)
            if device_type in ("cpu", "cuda")
            else contextlib.nullcontext()
        )
        with _autocast_ctx:
            h, _ = self.lstm(x_ci)          # always float32 inside

        # BUG FIX 4: Do NOT restore to orig_dtype here.
        # h stays float32 through ALL of Stage 2 (unfold, mean, encoder).
        # Restoring to float16 mid-forward causes precision loss and NaN risk
        # in patches.mean() and TransformerEncoder attention/layernorm.


        # Patch the hidden states
        # BUG FIX 1: use last hidden state per patch window
        patches = h.unfold(1, self.patch_len, self.stride)  # (B*F, N, d_model, P)
        enc_in = patches.mean(dim=-1)  # summarise full LSTM window

        # Stage 2: PatchTST encoder
        enc_in = enc_in + self.pos_embedding
        enc_in = self.dropout(enc_in)
        enc_out = self.encoder(enc_in)              # (B*F, N, d_model)
        enc_out = self.enc_dropout(enc_out)

        pooled = enc_out.mean(dim=1)               # (B*F, d_model)
        scores = self.feature_head(pooled).squeeze(-1).view(B, F)  # (B, F)
        out = self.mixing_layer(scores)            # (B, 1)

        # Restore dtype at the OUTPUT boundary only, not mid-forward.
        # This ensures the returned tensor matches the caller's expected dtype
        # (e.g., float16 under AMP, bfloat16 on TPU) without compromising
        # internal numerical stability.
        return torch.tanh(out).to(orig_dtype)