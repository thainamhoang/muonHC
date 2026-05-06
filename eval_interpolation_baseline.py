"""Evaluate nearest/bilinear interpolation baselines on the test split.

Examples:
    python eval_interpolation_baseline.py \
      --config configs/phase_3/t2m/cfgs_full_muon.yaml \
      --methods nearest bilinear \
      --temporal-reduction mean
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from eval_checkpoint import (  # noqa: E402
    _make_test_loader,
    _stat_to_device,
    climatelearn_rmse,
    configure_eval_backend,
)
from utils.metrics import log_frequency_distance  # noqa: E402
from utils.runtime import (  # noqa: E402
    configure_torch_performance,
    resolve_device,
    seed_everything,
)


def _resize(x, size, method):
    if method == "nearest":
        return F.interpolate(x, size=size, mode="nearest")
    if method == "bilinear":
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    raise ValueError(f"Unknown interpolation method: {method}")


def _temporal_reduce(lr_k, reduction):
    if lr_k.size(1) == 1:
        return lr_k
    if reduction == "center":
        return lr_k[:, lr_k.size(1) // 2: lr_k.size(1) // 2 + 1]
    if reduction == "mean":
        return lr_k.mean(dim=1, keepdim=True)
    raise ValueError(f"Unknown temporal reduction: {reduction}")


@torch.no_grad()
def evaluate_interpolation(method, loader, device, temporal_reduction="center"):
    dataset = loader.dataset
    lr_mean = getattr(dataset, "lr_mean")
    lr_std = getattr(dataset, "lr_std")
    hr_mean = getattr(dataset, "hr_mean")
    hr_std = getattr(dataset, "hr_std")

    sum_sq_z = torch.zeros((), device=device)
    sum_sq_k = torch.zeros((), device=device)
    cl_rmse_sum = torch.zeros((), device=device)
    lfd_sum = torch.zeros((), device=device)
    n_values = 0
    n_samples = 0

    for lr, hr in loader:
        lr = lr.to(device, non_blocking=True).float()
        hr = hr.to(device, non_blocking=True).float()

        lr_mean_t = _stat_to_device(lr_mean, lr)
        lr_std_t = _stat_to_device(lr_std, lr)
        hr_mean_t = _stat_to_device(hr_mean, hr)
        hr_std_t = _stat_to_device(hr_std, hr)

        # Interpolation baselines should operate on the physical field, because
        # LR and HR directories may have different normalization statistics.
        lr_k = lr * lr_std_t + lr_mean_t
        pred_k = _temporal_reduce(lr_k, temporal_reduction)
        pred_k = _resize(pred_k, size=hr.shape[-2:], method=method)

        hr_k = hr * hr_std_t + hr_mean_t
        pred_z = (pred_k - hr_mean_t) / (hr_std_t + 1e-8)

        sum_sq_z += (pred_z - hr).square().sum()
        sum_sq_k += (pred_k - hr_k).square().sum()
        cl_rmse_sum += climatelearn_rmse(pred_k, hr_k) * hr.size(0)
        lfd_sum += log_frequency_distance(
            pred_z,
            hr,
            mean=hr_mean,
            std=hr_std,
            crop_border=1,
            reduction="sum",
        )
        n_values += hr.numel()
        n_samples += hr.size(0)

    return {
        "rmse_k": (sum_sq_k / n_values).sqrt().item(),
        "rmse_z": (sum_sq_z / n_values).sqrt().item(),
        "cl_rmse": (cl_rmse_sum / n_samples).item(),
        "lfd": (lfd_sum / n_samples).item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to an ERA5 YAML/OmegaConf config.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["nearest", "bilinear"],
        choices=["nearest", "bilinear"],
        help="Interpolation baselines to evaluate.",
    )
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda or cuda:0.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override test batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override test dataloader workers.")
    parser.add_argument(
        "--temporal-reduction",
        choices=["center", "mean"],
        default="mean",
        help="How to turn temporal LR channels into one field before interpolation.",
    )
    parser.add_argument("--disable-cudnn", action="store_true", help="Disable cuDNN for eval fallback.")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    dataset_name = str(config.get("global_vars", {}).get("dataset", "")).lower()
    if dataset_name != "era5":
        raise ValueError(f"This baseline evaluator is intended for ERA5 configs, got {dataset_name!r}.")

    if not bool(config.data.get("temporal", False)):
        print("Warning: config data.temporal is false; baseline will run on single-frame input.", flush=True)

    seed_everything(int(config.training.get("seed", 42)))
    device = resolve_device(args.device or config.training.get("device", None))
    configure_torch_performance(config.training)
    configure_eval_backend(disable_cudnn=args.disable_cudnn)

    print(f"Config             : {args.config}", flush=True)
    print(f"Device             : {device}", flush=True)
    print(f"Temporal reduction : {args.temporal_reduction}", flush=True)

    _, test_loader = _make_test_loader(
        config,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    for method in args.methods:
        metrics = evaluate_interpolation(
            method=method,
            loader=test_loader,
            device=device,
            temporal_reduction=args.temporal_reduction,
        )
        print(f"\n{method}:")
        print(f"test/rmse_k: {metrics['rmse_k']:.6f}")
        print(f"test/rmse_z: {metrics['rmse_z']:.6f}")
        print(f"test/cl_rmse: {metrics['cl_rmse']:.6f}")
        print(f"test/lfd   : {metrics['lfd']:.6f}")


if __name__ == "__main__":
    main()
