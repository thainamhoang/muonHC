"""Evaluation metrics for climate downscaling."""

import torch


def _broadcast_stat(value, reference):
    if torch.is_tensor(value):
        return value.to(device=reference.device, dtype=reference.dtype)
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def rmse_kelvin(pred_z, target_z, mean=278.45, std=21.25):
    """Convert z-score predictions/targets to Kelvin and compute RMSE.
    Args:
        pred_z, target_z: tensors in z-score space
        mean, std: normalization parameters
    Returns: RMSE in Kelvin
    """
    pred_k = pred_z * std + mean
    target_k = target_z * std + mean
    mse = torch.nn.functional.mse_loss(pred_k, target_k)
    return torch.sqrt(mse).item()

def rmse_z(pred_z, target_z):
    """RMSE in z-score space."""
    mse = torch.nn.functional.mse_loss(pred_z, target_z)
    return torch.sqrt(mse).item()

def bias_kelvin(pred_z, target_z, mean=278.45, std=21.25):
    """Mean bias in Kelvin (pred - target)."""
    pred_k = pred_z * std + mean
    target_k = target_z * std + mean
    return (pred_k - target_k).mean().item()


def log_frequency_distance(pred_z, target_z, mean=278.45, std=21.25,
                           eps=1e-12, reduction="mean"):
    """Log Frequency Distance in Kelvin.

    Inputs are expected in z-score space with shape (B, C, H, W) or
    (B, H, W). LFD is computed per sample as log(mean squared FFT error)
    over channels and spatial frequencies, then reduced across the batch.
    """
    mean = _broadcast_stat(mean, pred_z)
    std = _broadcast_stat(std, pred_z)
    pred = pred_z * std + mean
    target = target_z * std + mean

    pred_fft = torch.fft.rfft2(pred, dim=(-2, -1), norm="ortho")
    target_fft = torch.fft.rfft2(target, dim=(-2, -1), norm="ortho")
    fft_sq_error = (pred_fft - target_fft).abs().square()
    reduce_dims = tuple(range(1, fft_sq_error.ndim))
    per_sample = torch.log(fft_sq_error.mean(dim=reduce_dims).clamp_min(eps))

    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError(f"Unknown reduction: {reduction}")

def pearson_corr(pred, target):
    """Pearson correlation coefficient."""
    pred = pred.flatten()
    target = target.flatten()
    return torch.corrcoef(torch.stack([pred, target]))[0,1].item()
