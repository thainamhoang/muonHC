"""Spectral loss from your FGD, ready to plug in."""
import torch
import torch.fft

def spectral_loss(pred, target, lambda_=0.1, freq_ramp=10.0):
    """MSE + λ * frequency-weighted L1 in Fourier domain."""
    mse = torch.nn.functional.mse_loss(pred, target)
    # FFT
    pred_fft = torch.fft.rfft2(pred, norm='ortho')
    target_fft = torch.fft.rfft2(target, norm='ortho')
    # Radial frequency weights
    H, W = pred.shape[-2], pred.shape[-1]
    y, x = torch.meshgrid(torch.arange(H, device=pred.device), torch.arange(W//2+1, device=pred.device), indexing='ij')
    freq = torch.sqrt(y**2 + x**2).float()
    weight = 1.0 + freq_ramp * freq / freq.max()
    spec_l1 = (weight * torch.abs(pred_fft - target_fft)).mean()
    return mse + lambda_ * spec_l1