import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # z: (B, L, n_bits) or (B, n_bits)
        z = F.normalize(z, dim=-1)
        z_q = StraightThroughEstimator.apply(z)
        bits = (z_q + 1) / 2
        n_bits = bits.shape[-1]
        powers = 2 ** torch.arange(n_bits - 1, -1, -1, device=z.device)
        indices = torch.sum(bits * powers, dim=-1).long()
        return z_q, indices


class KLineTokenizer(nn.Module):
    """
    Kronos-style Tokenizer with Hierarchical Coarse/Fine BSQ split.
    input_dim=21, n_bits=12 → coarse 6 bits + fine 6 bits
    """
    def __init__(self, input_dim=21, n_bits=12, hidden_dim=256):
        super().__init__()
        assert n_bits % 2 == 0, "n_bits must be even for coarse/fine split"
        self.input_dim = input_dim
        self.n_bits = n_bits
        self.half_bits = n_bits // 2  # 6

        # Encoder: 21 → n_bits latents (all bits at once)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, n_bits),
            nn.Tanh()
        )

        self.bsq = BSQ()

        # Coarse decoder: only uses first half_bits → reconstructs input
        self.decoder_coarse = nn.Sequential(
            nn.Linear(self.half_bits, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim)
        )

        # Fine decoder: uses all n_bits → refined reconstruction
        self.decoder_fine = nn.Sequential(
            nn.Linear(n_bits, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        """
        x: (B, L, 21) or (B, 21)
        returns: (B, L) or (B,) full token indices (coarse << half_bits | fine)
        """
        z = self.encoder(x)
        z_coarse = z[..., :self.half_bits]
        z_fine   = z[..., self.half_bits:]

        _, idx_coarse = self.bsq(z_coarse)  # (B, L) values in [0, 2^6)
        _, idx_fine   = self.bsq(z_fine)    # (B, L) values in [0, 2^6)

        # Pack into a single index: coarse occupies upper bits
        full_index = (idx_coarse << self.half_bits) | idx_fine  # [0, 2^12)
        return full_index

    def encode_hierarchical(self, x):
        """
        Returns coarse and fine indices separately.
        Useful for downstream models that want to predict coarse first.
        """
        z = self.encoder(x)
        z_coarse = z[..., :self.half_bits]
        z_fine   = z[..., self.half_bits:]
        _, idx_coarse = self.bsq(z_coarse)
        _, idx_fine   = self.bsq(z_fine)
        return idx_coarse, idx_fine

    def forward(self, x):
        """
        Full forward pass. Returns:
          x_recon_coarse : coarse-only reconstruction
          x_recon_fine   : full (coarse+fine) reconstruction
          full_index     : packed token index [0, 2^n_bits)
          idx_coarse     : coarse token index [0, 2^half_bits)
          idx_fine       : fine token index   [0, 2^half_bits)
        """
        z = self.encoder(x)
        z_coarse = z[..., :self.half_bits]
        z_fine   = z[..., self.half_bits:]

        zq_coarse, idx_coarse = self.bsq(z_coarse)
        zq_fine,   idx_fine   = self.bsq(z_fine)

        x_recon_coarse = self.decoder_coarse(zq_coarse)
        x_recon_fine   = self.decoder_fine(torch.cat([zq_coarse, zq_fine], dim=-1))

        full_index = (idx_coarse << self.half_bits) | idx_fine

        return x_recon_coarse, x_recon_fine, full_index, idx_coarse, idx_fine