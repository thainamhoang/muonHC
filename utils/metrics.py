"""Evaluation metrics for climate downscaling."""

import torch

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

def pearson_corr(pred, target):
    """Pearson correlation coefficient."""
    pred = pred.flatten()
    target = target.flatten()
    return torch.corrcoef(torch.stack([pred, target]))[0,1].item()