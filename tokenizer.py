# tokenizer.py — full replacement

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class BSQ(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        z = F.normalize(z, dim=-1)
        z_q = StraightThroughEstimator.apply(z)
        bits = (z_q + 1) / 2
        n_bits = bits.shape[-1]
        powers = 2 ** torch.arange(n_bits - 1, -1, -1, device=z.device)
        indices = torch.sum(bits * powers, dim=-1).long()
        return z_q, indices


class KLineTokenizer(nn.Module):
    """
    Kronos-faithful tokenizer with causal Transformer encoder.
    - input_dim : 21  (your feature set)
    - n_bits    : 12  → coarse 6 + fine 6
    - seq_len   : 512 (matches LOOKBACK_WINDOW — used for positional embedding)
    
    Encoder: 21 → d_enc=64 projection → 2-layer causal TransformerEncoder
             → Linear(64, 12) → BSQ
    """
    def __init__(
        self,
        input_dim: int = 21,
        n_bits:    int = 12,
        d_enc:     int = 64,       # encoder d_model — small, T4 has headroom
        n_heads:   int = 4,        # head_dim = 64/4 = 16
        n_enc_layers: int = 2,     # 2 layers is enough for tokenizer
        seq_len:   int = 512,      # for learnable positional embedding
        dropout:   float = 0.1,
    ):
        super().__init__()
        assert n_bits % 2 == 0
        self.input_dim  = input_dim
        self.n_bits     = n_bits
        self.half_bits  = n_bits // 2   # 6
        self.d_enc      = d_enc
        self.seq_len    = seq_len

        # ── Feature projection ──────────────────────────────────────────────
        # Projects 21 raw features into d_enc=64 space before self-attention.
        self.feat_proj = nn.Sequential(
            nn.Linear(input_dim, d_enc),
            nn.LayerNorm(d_enc),
        )

        # ── Learnable positional embedding ───────────────────────────────────
        # Unlike sinusoidal, learnable pos embeddings adapt to your 30-min
        # NIFTY bar rhythm. Sized to LOOKBACK_WINDOW=512.
        self.pos_emb = nn.Embedding(seq_len, d_enc)

        # ── Causal Transformer encoder ───────────────────────────────────────
        # norm_first=True (Pre-LN) → more stable training, less LR sensitivity.
        # is_causal=True in forward() → bar t only attends to bars 0..t
        # (no future leakage into the tokenizer latent).
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_enc,
            nhead=n_heads,
            dim_feedforward=d_enc * 4,   # 256
            dropout=dropout,
            batch_first=True,
            norm_first=True,             # Pre-LN
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_enc_layers)

        # ── Latent projection ────────────────────────────────────────────────
        self.to_bits = nn.Linear(d_enc, n_bits)

        self.bsq = BSQ()

        # ── Decoders (unchanged from your original) ──────────────────────────
        hidden_dim = 256
        self.decoder_coarse = nn.Sequential(
            nn.Linear(self.half_bits, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
        )
        self.decoder_fine = nn.Sequential(
            nn.Linear(n_bits, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
        )

        self._init_weights()

    def _init_weights(self):
        # GPT-2 style: scale residual projections by 1/sqrt(n_layers)
        for name, p in self.named_parameters():
            if "out_proj.weight" in name or "fc2.weight" in name:
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * self.transformer.num_layers))
            elif p.dim() >= 2:
                nn.init.xavier_uniform_(p)

    def _encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, 21)  — works for any L ≤ seq_len
        returns: (B, L, n_bits) continuous latent BEFORE BSQ
        """
        B, L, _ = x.shape

        # 1. Project features
        h = self.feat_proj(x)                           # (B, L, 64)

        # 2. Add positional embeddings
        positions = torch.arange(L, device=x.device)
        h = h + self.pos_emb(positions).unsqueeze(0)   # (B, L, 64)

        # 3. Causal self-attention: bar t attends only to bars 0..t
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            L, device=x.device
        )
        h = self.transformer(h, mask=causal_mask, is_causal=True)  # (B, L, 64)

        # 4. Project to bit space
        z = self.to_bits(h)                             # (B, L, 12)
        return z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, 21) — sequence required (not single bar)
        returns: (B, L) packed token indices [0, 4096)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)   # (1, L, 21) — single sequence fallback
        z = self._encode_latent(x)
        zc, zf = z[..., :self.half_bits], z[..., self.half_bits:]
        _, idx_c = self.bsq(zc)
        _, idx_f = self.bsq(zf)
        full = (idx_c << self.half_bits) | idx_f
        return full.squeeze(0) if full.shape[0] == 1 else full

    def encode_hierarchical(self, x: torch.Tensor):
        """Returns (idx_coarse, idx_fine) separately — for downstream hierarchical prediction."""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        z = self._encode_latent(x)
        zc, zf = z[..., :self.half_bits], z[..., self.half_bits:]
        _, idx_c = self.bsq(zc)
        _, idx_f = self.bsq(zf)
        if idx_c.shape[0] == 1:
            return idx_c.squeeze(0), idx_f.squeeze(0)
        return idx_c, idx_f

    def forward(self, x: torch.Tensor):
        """
        x: (B, L, 21)
        Returns: x_recon_coarse, x_recon_fine, full_index, idx_coarse, idx_fine
        """
        z = self._encode_latent(x)
        zc_cont = z[..., :self.half_bits]
        zf_cont = z[..., self.half_bits:]

        zq_c, idx_c = self.bsq(zc_cont)
        zq_f, idx_f = self.bsq(zf_cont)

        x_recon_coarse = self.decoder_coarse(zq_c)
        x_recon_fine   = self.decoder_fine(torch.cat([zq_c, zq_f], dim=-1))
        full_index     = (idx_c << self.half_bits) | idx_f

        return x_recon_coarse, x_recon_fine, full_index, idx_c, idx_f