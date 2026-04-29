"""Geo‑INR: conditions the decoder on spherical harmonic coordinates + elevation.
Spherical harmonic basis generation from GeoFAR, adapted for a given grid.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.special import sph_harm  # require scipy>=1.8

def spherical_harmonic_basis(latitudes, longitudes, n_basis=8):
    """Generate real spherical harmonic basis Y_l^m evaluated on a lat/lon grid.
    Args:
        latitudes: 1D tensor or array of shape (H,) in degrees (from -90 to 90).
        longitudes: 1D tensor or array of shape (W,) in degrees (from 0 to 360).
        n_basis: max degree l = 0..n_basis-1, total n_basis² basis functions.
    Returns: Tensor of shape (n_basis², H, W)
    """
    # Convert to colatitude and radians
    theta = np.deg2rad(90 - latitudes)   # colatitude
    phi = np.deg2rad(longitudes)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')  # (H, W)
    basis = []
    for l in range(n_basis):
        for m in range(-l, l+1):
            # Compute real part (imaginary part removed for simplicity)
            Y = sph_harm(abs(m), l, PHI, THETA)
            if m < 0:
                Y = np.sqrt(2) * (-1)**m * Y.imag  # real spherical harmonics convention
            elif m == 0:
                Y = Y.real
            else:  # m > 0
                Y = np.sqrt(2) * (-1)**m * Y.real
            basis.append(Y.real.astype(np.float32))
    basis = np.stack(basis, axis=0)  # (n_basis², H, W)
    return torch.from_numpy(basis)

class GeoINR(nn.Module):
    """MLP that maps geographic coordinates (spherical harmonics + elevation) to FiLM parameters.
    Input: spherical harmonic basis (n_basis², H, W) and elevation map (1, H, W).
    Output: gamma, beta each of shape (out_dim, H, W) for FiLM conditioning.
    """
    def __init__(self, n_basis=8, elevation_map=None, out_dim=128, hidden_dim=256):
        super().__init__()
        self.n_basis = n_basis
        in_channels = n_basis * n_basis + 1  # harmonics + elevation
        # Register spherical harmonic basis as buffer (computed once for HR grid)
        # We'll expect the user to provide lat/lon arrays for HR grid.
        # For now, define a placeholder; actual basis will be set later.
        self.register_buffer('sh_basis', torch.zeros(n_basis*n_basis, 1, 1))  # will be overwritten
        # Elevation map at HR resolution (1, H_hr, W_hr)
        if elevation_map is not None:
            self.register_buffer('elevation', elevation_map)
        else:
            self.register_buffer('elevation', torch.zeros(1, 1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim * 2),
        )
        # Zero-init final layer for identity start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def set_grid(self, sh_basis, elevation_map):
        """Set the HR grid basis and elevation after construction."""
        self.sh_basis = sh_basis
        self.elevation = elevation_map

    def forward(self):
        """Generate FiLM parameters for the full HR grid.
        Returns:
            gamma, beta: each (1, out_dim, H_hr, W_hr)
        """
        sh = self.sh_basis  # (n_basis², H, W)
        elev = self.elevation  # (1, H, W)
        H, W = sh.shape[1], sh.shape[2]
        # Concatenate along channel dimension: (H, W, in_channels)
        feat = torch.cat([sh, elev], dim=0)  # (in_channels, H, W)
        feat = feat.permute(1, 2, 0)  # (H, W, in_channels)
        feat = feat.reshape(-1, feat.shape[-1])  # (H*W, in_channels)
        cond = self.mlp(feat)  # (H*W, out_dim*2)
        cond = cond.reshape(H, W, -1)  # (H, W, out_dim*2)
        gamma, beta = cond[..., :cond.shape[-1]//2], cond[..., cond.shape[-1]//2:]
        # Permute to (out_dim, H, W) and add batch dim
        gamma = gamma.permute(2, 0, 1).unsqueeze(0)
        beta = beta.permute(2, 0, 1).unsqueeze(0)
        return gamma, beta