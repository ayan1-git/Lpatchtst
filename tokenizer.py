import torch
import torch.nn as nn
import torch.nn.functional as F

class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Identity gradient
        return grad_output

class BSQ(nn.Module):
    """
    Binary Scalar Quantization (BSQ)
    Maps continuous latent vector to discrete bits using STE.
    """
    def __init__(self):
        super().__init__()

    def forward(self, z):
        # z: (B, L, n_bits)
        z_q = StraightThroughEstimator.apply(z)
        
        # Convert signs [-1, 1] to binary [0, 1]
        bits = (z_q + 1) / 2
        
        # Calculate indices: sum(bits[i] * 2^i)
        # bits: (B, L, n_bits)
        n_bits = bits.shape[-1]
        powers = 2 ** torch.arange(n_bits - 1, -1, -1, device=z.device)
        indices = torch.sum(bits * powers, dim=-1).long()
        
        return z_q, indices

class KLineTokenizer(nn.Module):
    """
    Kronos-style Tokenizer for 21-feature K-lines.
    Uses a small MLP or Transformer Encoder to project features to latent space.
    """
    def __init__(self, input_dim=21, n_bits=12, hidden_dim=256):
        super().__init__()
        self.input_dim = input_dim
        self.n_bits = n_bits
        
        # Encoder: Map 21 features to n_bits latents
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, n_bits),
            nn.Tanh() # Helps center values around 0 for BSQ
        )
        
        self.bsq = BSQ()
        
        # Decoder: For training/reconstruction verification
        self.decoder = nn.Sequential(
            nn.Linear(n_bits, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        """
        x: (B, L, 21)
        returns: (B, L) token indices
        """
        z = self.encoder(x)
        _, indices = self.bsq(z)
        return indices

    def forward(self, x):
        """
        Full forward pass for VQ-VAE training.
        """
        z = self.encoder(x)
        z_q, indices = self.bsq(z)
        x_recon = self.decoder(z_q)
        return x_recon, indices, z_q