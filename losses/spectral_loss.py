"""Spectral loss from your FGD, ready to plug in."""
import torch
import torch.fft

_WEIGHT_CACHE = {}
_RADIAL_BIN_CACHE = {}


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


def _radial_bins(height, width, device, n_bins):
    key = (height, width, str(device), int(n_bins))
    if key in _RADIAL_BIN_CACHE:
        return _RADIAL_BIN_CACHE[key]

    fy = torch.fft.fftfreq(height, device=device)
    fx = torch.fft.fftfreq(width, device=device)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    radius = radius / radius.max().clamp_min(1e-12)
    bin_idx = torch.clamp((radius * n_bins).long(), max=n_bins - 1).flatten()
    counts = torch.bincount(bin_idx, minlength=n_bins).float().clamp_min(1.0)
    _RADIAL_BIN_CACHE[key] = (bin_idx, counts)
    return bin_idx, counts


def radial_spectrum(pred, n_bins=64):
    pred = pred.float()
    pred = pred - pred.mean(dim=(-2, -1), keepdim=True)
    spectrum = torch.log1p(torch.fft.fft2(pred, dim=(-2, -1)).abs())
    height, width = pred.shape[-2:]
    bin_idx, counts = _radial_bins(height, width, pred.device, n_bins)
    values = spectrum.reshape(-1, height * width)
    summed = torch.zeros(values.shape[0], n_bins, device=pred.device, dtype=values.dtype)
    summed.scatter_add_(1, bin_idx.expand(values.shape[0], -1), values)
    return (summed / counts.to(device=pred.device, dtype=values.dtype)).mean(dim=0)


def radial_spectrum_loss(pred, target, n_bins=64):
    pred_spec = radial_spectrum(pred, n_bins=n_bins)
    target_spec = radial_spectrum(target, n_bins=n_bins)
    return torch.nn.functional.l1_loss(pred_spec, target_spec)


def mse_spectral_radial_loss(pred, target, loss_cfg=None, mse=None):
    """MSE plus optional existing spectral loss and targeted radial spectrum loss."""
    if mse is None:
        mse = torch.nn.functional.mse_loss(pred, target)
    if loss_cfg is None:
        return mse

    spectral_lambda = float(loss_cfg.get("spectral_lambda", 0.0))
    radial_lambda = float(loss_cfg.get("radial_lambda", 0.0))

    loss = mse
    if spectral_lambda > 0.0:
        loss = spectral_loss(
            pred,
            target,
            lambda_=spectral_lambda,
            freq_ramp=float(loss_cfg.get("freq_ramp", 10.0)),
            mse=mse,
        )
    if radial_lambda > 0.0:
        loss = loss + radial_lambda * radial_spectrum_loss(
            pred,
            target,
            n_bins=int(loss_cfg.get("radial_bins", 64)),
        )
    return loss
