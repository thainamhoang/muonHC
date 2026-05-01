"""Spectral loss from your FGD, ready to plug in."""
import torch
import torch.fft

_WEIGHT_CACHE = {}


def _frequency_weight(height, width, device, dtype, freq_ramp):
    key = (height, width, str(device), dtype, float(freq_ramp))
    if key in _WEIGHT_CACHE:
        return _WEIGHT_CACHE[key]

    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width // 2 + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    freq = torch.sqrt(yy.square() + xx.square())
    weight = 1.0 + freq_ramp * freq / freq.max()
    _WEIGHT_CACHE[key] = weight
    return weight


def spectral_loss(pred, target, lambda_=0.1, freq_ramp=10.0, mse=None):
    """MSE + λ * frequency-weighted L1 in Fourier domain."""
    if mse is None:
        mse = torch.nn.functional.mse_loss(pred, target)
    # FFT
    pred_fft = torch.fft.rfft2(pred, norm='ortho')
    target_fft = torch.fft.rfft2(target, norm='ortho')
    # Radial frequency weights
    H, W = pred.shape[-2], pred.shape[-1]
    weight = _frequency_weight(H, W, pred.device, pred.real.dtype, freq_ramp)
    spec_l1 = (weight * torch.abs(pred_fft - target_fft)).mean()
    return mse + lambda_ * spec_l1
