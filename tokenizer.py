# tokenizer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from einops import rearrange, reduce


# ─────────────────────────────────────────────────────────────
# 1. Kronos Official: DifferentiableEntropyFunction
# ─────────────────────────────────────────────────────────────
class DifferentiableEntropyFunction(Function):
    @staticmethod
    def forward(ctx, zq, basis, K, eps):
        zb = (zq + 1) / 2
        zi = ((zb * basis).sum(-1)).to(torch.int64)
        cnt = torch.scatter_reduce(
            torch.zeros(2 ** K, device=zq.device, dtype=zq.dtype),
            0, zi.flatten(),
            torch.ones_like(zi.flatten()).to(zq.dtype), 'sum'
        )
        prob = (cnt + eps) / (cnt + eps).sum()
        H = -(prob * torch.log(prob)).sum()
        ctx.save_for_backward(zq, zi, prob)
        ctx.K = K
        return H

    @staticmethod
    def backward(ctx, grad_output):
        zq, zi, prob = ctx.saved_tensors
        grad_array = -grad_output * (torch.log(prob) + 1) / zi.numel() / ctx.K
        reord_grad = grad_array[zi.flatten()].reshape(zi.shape)
        grad_input = reord_grad.unsqueeze(-1) * zq
        return grad_input, None, None, None


def codebook_entropy(zq, basis, K, eps=1e-4):
    return DifferentiableEntropyFunction.apply(zq, basis, K, eps)


# ─────────────────────────────────────────────────────────────
# 2. Kronos Official: BinarySphericalQuantizer
# ─────────────────────────────────────────────────────────────
class BinarySphericalQuantizer(nn.Module):
    def __init__(self, embed_dim, beta, gamma0, gamma, zeta,
                 soft_entropy=True, group_size=6,
                 persample_entropy_compute='analytical',
                 cb_entropy_compute='group',
                 l2_norm=True, inv_temperature=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.beta = beta
        self.gamma0 = gamma0
        self.gamma = gamma
        self.zeta = zeta
        self.group_size = group_size
        self.soft_entropy = soft_entropy
        self.l2_norm = l2_norm
        self.inv_temperature = inv_temperature
        self.persample_entropy_compute = persample_entropy_compute
        self.cb_entropy_compute = cb_entropy_compute

        assert embed_dim % group_size == 0, \
            f"embed_dim ({embed_dim}) must be divisible by group_size ({group_size})"

        self.register_buffer('basis', 2 ** torch.arange(embed_dim - 1, -1, -1))
        self.register_buffer('group_basis', 2 ** torch.arange(group_size - 1, -1, -1))

        self.num_dimensions = 2 ** embed_dim
        self.bits_per_index = embed_dim

        group_codes = torch.arange(2 ** group_size)
        group_codebook = self.indexes_to_codes(group_codes).float()[:, -group_size:]
        self.register_buffer('group_codebook', group_codebook, persistent=False)

    def indexes_to_codes(self, x):
        mask = 2 ** torch.arange(self.embed_dim - 1, -1, -1, device=x.device, dtype=torch.long)
        return ((x.unsqueeze(-1) & mask) != 0).float() * 2 - 1

    def codes_to_indexes(self, zq):
        # zq is in {-1/sqrt(d), +1/sqrt(d)}, recover signs
        assert zq.shape[-1] == self.embed_dim
        q_scale = 1. / (self.embed_dim ** 0.5) if self.l2_norm else 1.
        zb = ((zq / q_scale) + 1) / 2  # map {-1,+1} -> {0,1}
        return (zb * self.basis).sum(-1).to(torch.long)

    def codes_to_group_indexes(self, zq):
        assert zq.shape[-1] == self.embed_dim
        q_scale = 1. / (self.embed_dim ** 0.5) if self.l2_norm else 1.
        zb = ((zq / q_scale) + 1) / 2
        divided_zb = rearrange(zb, '... (g c) -> ... g c', c=self.group_size)
        return (divided_zb * self.group_basis).sum(-1).to(torch.long)

    def quantize(self, z):
        zhat = torch.where(z > 0,
                           torch.ones_like(z),
                           -torch.ones_like(z))
        return z + (zhat - z).detach()   # straight-through estimator

    def get_entropy(self, prob, dim=-1, normalize=True):
        if normalize:
            prob = prob / prob.sum(dim=dim, keepdim=True)
        entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=dim)
        return entropy

    def get_hard_per_sample_entropy(self, zb):
        B, T, D = zb.shape
        p = zb.mean(dim=1)  # (B, D)
        p = torch.stack([p, 1 - p], dim=-1)  # (B, D, 2)
        return self.get_entropy(p, dim=-1, normalize=False).sum(dim=-1).mean()

    def soft_entropy_loss(self, z):
        group_code_book = self.group_codebook / (self.embed_dim ** 0.5 if self.l2_norm else 1.)
        divided_z = rearrange(z, '... (g c) -> ... g c', c=self.group_size)
        distance = -2 * torch.einsum('... g c, d c ->... g d', divided_z, group_code_book)
        prob = (-distance * self.inv_temperature).softmax(dim=-1)

        if self.persample_entropy_compute == 'analytical':
            if self.l2_norm:
                p = torch.sigmoid(-4 * z / (self.embed_dim ** 0.5) * self.inv_temperature)
            else:
                p = torch.sigmoid(-4 * z * self.inv_temperature)
            prob_per = torch.stack([p, 1 - p], dim=-1)
            per_sample_entropy = self.get_entropy(prob_per, dim=-1, normalize=False).sum(dim=-1).mean()
        else:
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()

        avg_prob = reduce(prob, '... g d -> g d', 'mean')
        cb_entropy = self.get_entropy(avg_prob, dim=-1, normalize=False).sum()
        return per_sample_entropy, cb_entropy, avg_prob

    def forward(self, z, collect_metrics=True):
        zq = self.quantize(z)
        q_scale = 1. / (self.embed_dim ** 0.5) if self.l2_norm else 1.
        zq = zq * q_scale

        if not collect_metrics:
            return zq, zq.new_zeros(()), {}

        if self.soft_entropy:
            persample_entropy, cb_entropy, avg_prob = self.soft_entropy_loss(z)
            entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy
        else:
            zb = ((zq / q_scale) + 1) / 2
            zb_by_sample = zb.reshape(z.shape[0], -1, z.shape[-1]).float()
            persample_entropy = self.get_hard_per_sample_entropy(zb_by_sample)
            cb_entropy = codebook_entropy(zq, self.basis, self.embed_dim)
            entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy
            avg_prob = None

        commit_loss = self.beta * torch.mean(((zq.detach() - z) ** 2).sum(dim=-1))

        loss = commit_loss + self.zeta * entropy_penalty / self.inv_temperature
        metrics = {"H": cb_entropy, "commit": commit_loss,
                   "per_sample_entropy": persample_entropy, "avg_prob": avg_prob}
        return zq, loss, metrics


# ─────────────────────────────────────────────────────────────
# 3. Kronos Official: BSQuantizer (wraps BinarySphericalQuantizer
#    with coarse/fine s1/s2 split, mirrors BSQuantizer in module.py)
# ─────────────────────────────────────────────────────────────
class BSQuantizer(nn.Module):
    def __init__(self, s1_bits, s2_bits, beta, gamma0, gamma, zeta, group_size):
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        total_bits = s1_bits + s2_bits
        self.quantizer = BinarySphericalQuantizer(
            embed_dim=total_bits,
            beta=beta, gamma0=gamma0, gamma=gamma, zeta=zeta,
            group_size=group_size
        )

    def forward(self, z, half=False, collect_metrics=True):
        zq, loss, metrics = self.quantizer(z, collect_metrics=collect_metrics)

        if half:
            # Return separate (s1, s2) indices
            zq_s1 = zq[..., :self.s1_bits]
            zq_s2 = zq[..., self.s1_bits:]
            idx_s1 = self.quantizer.codes_to_indexes(
                zq_s1 * (self.quantizer.embed_dim ** 0.5) / (self.s1_bits ** 0.5)
            ) if self.quantizer.l2_norm else self.quantizer.codes_to_indexes(zq_s1)
            # simpler: use group indexes for s1 and s2
            q_scale = 1. / (self.quantizer.embed_dim ** 0.5)
            raw_s1 = (zq[..., :self.s1_bits] / q_scale + 1) / 2
            raw_s2 = (zq[..., self.s1_bits:] / q_scale + 1) / 2
            basis_s1 = 2 ** torch.arange(self.s1_bits - 1, -1, -1, device=z.device, dtype=torch.long)
            basis_s2 = 2 ** torch.arange(self.s2_bits - 1, -1, -1, device=z.device, dtype=torch.long)
            idx_s1 = (raw_s1 * basis_s1).sum(-1).long()
            idx_s2 = (raw_s2 * basis_s2).sum(-1).long()
            return loss, zq, [idx_s1, idx_s2]
        else:
            idx = self.quantizer.codes_to_indexes(zq)
            return loss, zq, idx


# ─────────────────────────────────────────────────────────────
# 4. Kronos Official: TransformerBlock (from module.py)
# ─────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x / (x.norm(dim=-1, keepdim=True) + self.eps) * self.weight * (x.shape[-1] ** 0.5)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, ffn_dropout_p=0.0,
                 attn_dropout_p=0.0, resid_dropout_p=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          dropout=attn_dropout_p,
                                          batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(ffn_dropout_p),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(resid_dropout_p),
        )

    def forward(self, x, key_padding_mask=None):
        # Pre-norm (Kronos uses pre-norm)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────
# 5. KronosTokenizer — exact official architecture
#    (from model/kronos.py) adapted for d_in=4
# ─────────────────────────────────────────────────────────────
class KronosTokenizer(nn.Module):
    def __init__(self,
                 d_in=4,           # your 4 features: log_ret_open/high/low/close
                 d_model=64,
                 n_heads=4,
                 ff_dim=128,
                 n_enc_layers=2,
                 n_dec_layers=2,
                 ffn_dropout_p=0.0,
                 attn_dropout_p=0.0,
                 resid_dropout_p=0.0,
                 s1_bits=6,
                 s2_bits=6,
                 beta=0.25,
                 gamma0=0.1,
                 gamma=1.0,
                 zeta=8.0,
                 group_size=6):
        super().__init__()
        self.d_in = d_in
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.codebook_dim = s1_bits + s2_bits

        # Input projection
        self.embed = nn.Linear(d_in, d_model)
        self.head = nn.Linear(d_model, d_in)

        # Encoder Transformer blocks (n_enc_layers - 1 because embed acts as layer 0)
        self.encoder = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, ffn_dropout_p, attn_dropout_p, resid_dropout_p)
            for _ in range(n_enc_layers - 1)
        ])
        # Decoder Transformer blocks
        self.decoder = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, ffn_dropout_p, attn_dropout_p, resid_dropout_p)
            for _ in range(n_dec_layers - 1)
        ])

        # Quantization projections (exact naming from official repo)
        self.quant_embed = nn.Linear(d_model, self.codebook_dim)
        self.post_quant_embed_pre = nn.Linear(s1_bits, d_model)   # coarse path
        self.post_quant_embed = nn.Linear(self.codebook_dim, d_model)  # full path

        # Official BSQuantizer
        self.tokenizer = BSQuantizer(s1_bits, s2_bits, beta, gamma0, gamma, zeta, group_size)

    def forward(self, x):
        """
        x: (B, T, d_in)
        Returns:
            (z_pre, z): reconstructions from coarse and full codebook
            bsq_loss:   scalar BSQ loss (commit + entropy)
            quantized:  (B, T, codebook_dim) quantized tensor
            z_indices:  list [idx_s1, idx_s2] each (B, T)
        """
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)

        z = self.quant_embed(z)                       # (B, T, codebook_dim)
        bsq_loss, quantized, z_indices = self.tokenizer(z, half=True)

        # Coarse (s1) decode path
        quantized_pre = quantized[:, :, :self.s1_bits]
        z_pre = self.post_quant_embed_pre(quantized_pre)
        for layer in self.decoder:
            z_pre = layer(z_pre)
        z_pre = self.head(z_pre)

        # Full decode path
        z_full = self.post_quant_embed(quantized)
        for layer in self.decoder:
            z_full = layer(z_full)
        z_full = self.head(z_full)

        return (z_pre, z_full), bsq_loss, quantized, z_indices

    def encode(self, x, half=True):
        """Returns [idx_s1, idx_s2] if half=True, else full_idx."""
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)
        _, _, z_indices = self.tokenizer(z, half=half, collect_metrics=False)
        return z_indices

    def indices_to_bits(self, x, half=False):
        """Convert [idx_s1, idx_s2] back to quantized float tensor."""
        codebook_dim = self.codebook_dim
        q_scale = 1. / (codebook_dim ** 0.5)
        if half:
            x1, x2 = x[0], x[1]
            mask = 2 ** torch.arange(codebook_dim // 2, device=x1.device, dtype=torch.long)
            b1 = ((x1.unsqueeze(-1) & mask) != 0).float() * 2 - 1
            b2 = ((x2.unsqueeze(-1) & mask) != 0).float() * 2 - 1
            bits = torch.cat([b1, b2], dim=-1)
        else:
            mask = 2 ** torch.arange(codebook_dim, device=x.device, dtype=torch.long)
            bits = ((x.unsqueeze(-1) & mask) != 0).float() * 2 - 1
        return bits * q_scale

    def decode(self, x, half=True):
        """Decode indices back to feature space."""
        quantized = self.indices_to_bits(x, half=half)
        z = self.post_quant_embed(quantized)
        for layer in self.decoder:
            z = layer(z)
        return self.head(z)
def prepare_ohlc_features(df):
    """
    Expects df with columns ['Open', 'High', 'Low', 'Close'].
    Returns (N, 4) array of log-returns relative to PREVIOUS close.
    """
    import numpy as np
    cols = {c.lower(): c for c in df.columns}
    o_col, h_col, l_col, c_col = cols.get('open','Open'), cols.get('high','High'), cols.get('low','Low'), cols.get('close','Close')
    close = df[c_col].values
    prev_close = np.roll(close, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        o = np.log(df[o_col].values / prev_close)
        h = np.log(df[h_col].values / prev_close)
        l = np.log(df[l_col].values / prev_close)
        c = np.log(df[c_col].values / prev_close)
    out = np.stack([o, h, l, c], axis=1)[1:]
    return np.nan_to_num(out).astype(np.float32)
