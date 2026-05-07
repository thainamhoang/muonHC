"""Spectral and Laplacian training losses."""

import torch
import torch.fft
import torch.nn.functional as F

_WEIGHT_CACHE = {}
_LAPLACIAN_KERNEL_CACHE = {}


def _frequency_weight(height, width, device, dtype, freq_ramp):
    key = (height, width, str(device), dtype, float(freq_ramp))
    if key in _WEIGHT_CACHE:
        return _WEIGHT_CACHE[key]

    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width // 2 + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    freq = torch.sqrt(yy.square() + xx.square())
    weight = 1.0 + freq_ramp * freq / freq.max().clamp_min(1e-12)
    _WEIGHT_CACHE[key] = weight
    return weight


def spectral_loss(pred, target, lambda_=0.1, freq_ramp=10.0, mse=None):
    """MSE + lambda * frequency-weighted L1 in Fourier domain."""
    if mse is None:
        mse = F.mse_loss(pred, target)
    pred_fft = torch.fft.rfft2(pred, norm="ortho")
    target_fft = torch.fft.rfft2(target, norm="ortho")
    height, width = pred.shape[-2], pred.shape[-1]
    weight = _frequency_weight(height, width, pred.device, pred.real.dtype, freq_ramp)
    spec_l1 = (weight * torch.abs(pred_fft - target_fft)).mean()
    return mse + lambda_ * spec_l1


def _laplacian_kernel(channels, device, dtype):
    key = (int(channels), str(device), dtype)
    if key in _LAPLACIAN_KERNEL_CACHE:
        return _LAPLACIAN_KERNEL_CACHE[key]

    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    kernel = kernel.expand(channels, 1, 3, 3).contiguous()
    _LAPLACIAN_KERNEL_CACHE[key] = kernel
    return kernel


def laplacian_filter(x):
    """Apply a 3x3 Laplacian filter with reflect padding."""
    channels = x.shape[1]
    kernel = _laplacian_kernel(channels, x.device, x.dtype)
    x = F.pad(x, (1, 1, 1, 1), mode="reflect")
    return F.conv2d(x, kernel, groups=channels)


def laplacian_loss(pred, target):
    """L1 distance between Laplacian-filtered prediction and target."""
    return F.l1_loss(laplacian_filter(pred), laplacian_filter(target))


def mse_spectral_laplacian_loss(pred, target, loss_cfg=None, mse=None):
    """MSE plus optional spectral loss and spatially grounded Laplacian loss."""
    if mse is None:
        mse = F.mse_loss(pred, target)
    if loss_cfg is None:
        return mse

    spectral_lambda = float(loss_cfg.get("spectral_lambda", 0.0))
    laplacian_lambda = float(loss_cfg.get("laplacian_lambda", 0.0))

    loss = mse
    if spectral_lambda > 0.0:
        loss = spectral_loss(
            pred,
            target,
            lambda_=spectral_lambda,
            freq_ramp=float(loss_cfg.get("freq_ramp", 10.0)),
            mse=mse,
        )
    if laplacian_lambda > 0.0:
        loss = loss + laplacian_lambda * laplacian_loss(pred, target)
    return loss
