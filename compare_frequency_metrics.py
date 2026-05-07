"""Compare frequency-domain metrics across model variants.

Example experiments JSON:
[
  {
    "name": "full_muon",
    "config_path": "/home/thahoa/muonHC/configs/phase_3/t2m/cfgs_full_muon.yaml",
    "checkpoint_path": "/home/thahoa/muonHC/output/era5/t2m_full_muon_4x_stride6_seed42/best_model.pt"
  }
]

Example:
    python compare_frequency_metrics.py \
      --experiments-json /home/thahoa/muonHC/experiments_frequency.json \
      --output-dir /home/thahoa/muonHC/output/figures/frequency_compare \
      --split val \
      --device cuda \
      --batch-size 64 \
      --num-workers 0 \
      --max-batches 10
"""

import argparse
import csv
import json
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from datasets.downscaling_dataset import DownscalingDataset
from eval_checkpoint import (
    _build_model,
    _load_model_state,
    _resolve_amp_dtype,
    _stat_to_device,
    configure_eval_backend,
)
from utils.runtime import configure_torch_performance, resolve_device, seed_everything


METRIC_COLUMNS = ["low_rmse", "mid_rmse", "high_rmse", "hf_energy_ratio"]


def denormalize_target(x, dataset):
    mean = _stat_to_device(getattr(dataset, "hr_mean", 0.0), x)
    std = _stat_to_device(getattr(dataset, "hr_std", 1.0), x)
    return x * std + mean


def ensure_nchw(x):
    x = x.float()
    if x.ndim == 3:
        return x.unsqueeze(1)
    if x.ndim != 4:
        raise ValueError(f"Expected [N, H, W] or [N, C, H, W], got {tuple(x.shape)}")
    return x


def shifted_radius_grid(height, width, device):
    fy = torch.fft.fftshift(torch.fft.fftfreq(height, device=device))
    fx = torch.fft.fftshift(torch.fft.fftfreq(width, device=device))
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    return radius / radius.max().clamp_min(1e-12)


def frequency_band_masks(height, width, device, low_cut=0.15, high_cut=0.35):
    radius = shifted_radius_grid(height, width, device)
    return {
        "low": radius < low_cut,
        "mid": (radius >= low_cut) & (radius < high_cut),
        "high": radius >= high_cut,
    }


def fft_band_rmse(pred, target, low_cut=0.15, high_cut=0.35):
    """Compute low-, mid-, and high-frequency RMSE using 2D FFT masks."""
    pred = ensure_nchw(pred)
    target = ensure_nchw(target)
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}")

    _, _, height, width = pred.shape
    error_fft = torch.fft.fftshift(
        torch.fft.fft2(pred - target, dim=(-2, -1)),
        dim=(-2, -1),
    )
    masks = frequency_band_masks(height, width, pred.device, low_cut=low_cut, high_cut=high_cut)

    metrics = {}
    for band, mask in masks.items():
        band_fft = error_fft * mask[None, None]
        band_error = torch.fft.ifft2(
            torch.fft.ifftshift(band_fft, dim=(-2, -1)),
            dim=(-2, -1),
        ).real
        metrics[f"{band}_rmse"] = torch.sqrt(torch.mean(band_error.square())).item()
    return metrics


def high_frequency_energy_ratio(pred, target, high_cut=0.35, eps=1e-8):
    """Compute high-frequency energy ratio: HF energy(pred) / HF energy(target)."""
    pred = ensure_nchw(pred)
    target = ensure_nchw(target)
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}")

    _, _, height, width = pred.shape
    pred_fft = torch.fft.fftshift(torch.fft.fft2(pred, dim=(-2, -1)), dim=(-2, -1))
    target_fft = torch.fft.fftshift(torch.fft.fft2(target, dim=(-2, -1)), dim=(-2, -1))
    radius = shifted_radius_grid(height, width, pred.device)
    high_mask = (radius >= high_cut)[None, None]
    pred_energy = torch.mean(torch.abs(pred_fft * high_mask).square())
    target_energy = torch.mean(torch.abs(target_fft * high_mask).square())
    return (pred_energy / (target_energy + eps)).item()


def _band_sse_and_count(pred, target, low_cut=0.15, high_cut=0.35):
    pred = ensure_nchw(pred)
    target = ensure_nchw(target)
    _, _, height, width = pred.shape
    error_fft = torch.fft.fftshift(
        torch.fft.fft2(pred - target, dim=(-2, -1)),
        dim=(-2, -1),
    )
    masks = frequency_band_masks(height, width, pred.device, low_cut=low_cut, high_cut=high_cut)
    out = {}
    for band, mask in masks.items():
        band_fft = error_fft * mask[None, None]
        band_error = torch.fft.ifft2(
            torch.fft.ifftshift(band_fft, dim=(-2, -1)),
            dim=(-2, -1),
        ).real
        out[band] = (band_error.square().sum().item(), band_error.numel())
    return out


def _hf_energy_sums(pred, target, high_cut=0.35):
    pred = ensure_nchw(pred)
    target = ensure_nchw(target)
    _, _, height, width = pred.shape
    pred_fft = torch.fft.fftshift(torch.fft.fft2(pred, dim=(-2, -1)), dim=(-2, -1))
    target_fft = torch.fft.fftshift(torch.fft.fft2(target, dim=(-2, -1)), dim=(-2, -1))
    radius = shifted_radius_grid(height, width, pred.device)
    high_mask = (radius >= high_cut)[None, None]
    pred_energy = torch.abs(pred_fft * high_mask).square().sum().item()
    target_energy = torch.abs(target_fft * high_mask).square().sum().item()
    return pred_energy, target_energy


def _resolve_split_preload(config, split):
    if split == "train":
        return (
            bool(config.data.get("lr_preload_train", config.data.get("lr_preload", True))),
            bool(config.data.get("hr_preload_train", config.data.get("hr_preload", False))),
        )
    return (
        bool(config.data.get("lr_preload_eval", True)),
        bool(config.data.get("hr_preload_eval", True)),
    )


def make_loader(config, device, split="val", batch_size=None, num_workers=None):
    loader_cfg = config.get("dataloader", {})
    batch_size = int(batch_size or loader_cfg.get("batch_size", 32))
    num_workers = int(num_workers if num_workers is not None else loader_cfg.get("num_workers", 0))
    pin_memory = bool(loader_cfg.get("pin_memory", str(device).startswith("cuda")))
    persistent_workers = bool(loader_cfg.get("persistent_workers", False)) if num_workers > 0 else False
    prefetch_factor = loader_cfg.get("prefetch_factor", None)
    variable_name = config.data.get(
        "var",
        config.get("global_vars", {}).get("var", "2m_temperature"),
    )
    lr_preload, hr_preload = _resolve_split_preload(config, split)

    dataset = DownscalingDataset(
        lr_dir=config.data.lr_dir,
        hr_dir=config.data.hr_dir,
        partition=split,
        temporal=bool(config.data.get("temporal", False)),
        stride=int(config.data.stride),
        lr_preload=lr_preload,
        hr_preload=hr_preload,
        variable_name=variable_name,
        lr_crop_size=config.data.get("lr_crop_size", None),
        random_crop=bool(config.data.get("random_crop_eval", False)),
        upscale=int(config.model.get("upscale", 4)),
    )

    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
    if num_workers > 0 and hasattr(dataset, "worker_init_fn"):
        kwargs["worker_init_fn"] = dataset.worker_init_fn

    return dataset, DataLoader(dataset, **kwargs)


def load_model_and_loader(
    config_path,
    checkpoint_path,
    device=None,
    split="val",
    batch_size=None,
    num_workers=None,
    disable_cudnn=False,
):
    config = OmegaConf.load(config_path)
    seed_everything(int(config.training.get("seed", 42)))
    device = resolve_device(device or config.training.get("device", None))
    configure_torch_performance(config.training)
    configure_eval_backend(disable_cudnn=disable_cudnn)

    dataset, loader = make_loader(
        config,
        device=device,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    model = _build_model(
        config,
        device=device,
        lr_shape=dataset.sample_lr_shape,
        hr_shape=dataset.sample_hr_shape,
    )
    _load_model_state(model, checkpoint_path, device=device)
    return model, loader, device, config


@torch.no_grad()
def evaluate_frequency_metrics(
    model,
    loader,
    device="cuda",
    max_batches=None,
    low_cut=0.15,
    high_cut=0.35,
    amp_enabled=False,
    amp_dtype="bfloat16",
    denormalize=True,
):
    model.eval()
    device_type = str(device).split(":", maxsplit=1)[0]
    amp_enabled = bool(amp_enabled and device_type == "cuda")
    amp_dtype = _resolve_amp_dtype(amp_dtype)
    dataset = loader.dataset

    band_sse = {"low": 0.0, "mid": 0.0, "high": 0.0}
    band_count = {"low": 0, "mid": 0, "high": 0}
    pred_hf_energy = 0.0
    target_hf_energy = 0.0

    for batch_idx, (lr, hr) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
            pred = model(lr)

        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        elif isinstance(pred, dict):
            pred = pred["pred"] if "pred" in pred else pred["output"]

        pred = pred.float()
        hr = hr.float()
        if denormalize:
            pred = denormalize_target(pred, dataset)
            hr = denormalize_target(hr, dataset)

        batch_band = _band_sse_and_count(pred, hr, low_cut=low_cut, high_cut=high_cut)
        for band, (sse, count) in batch_band.items():
            band_sse[band] += sse
            band_count[band] += count

        pred_energy, target_energy = _hf_energy_sums(pred, hr, high_cut=high_cut)
        pred_hf_energy += pred_energy
        target_hf_energy += target_energy

    if not any(band_count.values()):
        raise RuntimeError("No batches were evaluated. Check --max-batches and the selected split.")

    return {
        "low_rmse": float(np.sqrt(band_sse["low"] / max(band_count["low"], 1))),
        "mid_rmse": float(np.sqrt(band_sse["mid"] / max(band_count["mid"], 1))),
        "high_rmse": float(np.sqrt(band_sse["high"] / max(band_count["high"], 1))),
        "hf_energy_ratio": float(pred_hf_energy / max(target_hf_energy, 1e-8)),
    }


def load_experiments(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        payload = payload.get("experiments", [])
    if not isinstance(payload, list) or not payload:
        raise ValueError("Experiments JSON must be a non-empty list or {'experiments': [...]} object.")
    for exp in payload:
        for key in ("name", "config_path", "checkpoint_path"):
            if key not in exp:
                raise ValueError(f"Experiment is missing required key '{key}': {exp}")
    return payload


def save_results_csv(rows, output_dir):
    path = os.path.join(output_dir, "frequency_metrics_per_optimizer.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", *METRIC_COLUMNS])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def plot_frequency_band_rmse(rows, save_path):
    variants = [row["variant"] for row in rows]
    bands = ["low_rmse", "mid_rmse", "high_rmse"]
    x = np.arange(len(variants))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    for idx, band in enumerate(bands):
        ax.bar(
            x + (idx - 1) * width,
            [row[band] for row in rows],
            width,
            label=band.replace("_rmse", "").capitalize(),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=25, ha="right")
    ax.set_ylabel("Band RMSE")
    ax.set_title("Frequency-domain RMSE by optimizer/model variant")
    ax.legend(title="Frequency band")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_hf_energy_ratio(rows, save_path):
    variants = [row["variant"] for row in rows]
    x = np.arange(len(variants))

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    ax.bar(x, [row["hf_energy_ratio"] for row in rows], color="#4c78a8")
    ax.axhline(1.0, linestyle="--", linewidth=1.0, color="black", label="Ground-truth level")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=25, ha="right")
    ax.set_ylabel("Prediction / Ground-truth HF energy")
    ax.set_title("High-frequency energy recovery by optimizer/model variant")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument("--low-cut", type=float, default=0.15)
    parser.add_argument("--high-cut", type=float, default=0.35)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-denorm", action="store_true")
    parser.add_argument("--disable-cudnn", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    experiments = load_experiments(args.experiments_json)
    rows = []

    for exp in experiments:
        print(f"Evaluating {exp['name']}", flush=True)
        model, loader, device, config = load_model_and_loader(
            config_path=exp["config_path"],
            checkpoint_path=exp["checkpoint_path"],
            device=args.device,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            disable_cudnn=args.disable_cudnn,
        )
        amp_cfg = config.training.get("amp", {})
        metrics = evaluate_frequency_metrics(
            model=model,
            loader=loader,
            device=device,
            max_batches=args.max_batches,
            low_cut=args.low_cut,
            high_cut=args.high_cut,
            amp_enabled=args.amp,
            amp_dtype=amp_cfg.get("dtype", "bfloat16"),
            denormalize=not args.no_denorm,
        )
        row = {"variant": exp["name"], **metrics}
        rows.append(row)
        print(f"  {metrics}", flush=True)

    csv_path = save_results_csv(rows, args.output_dir)
    band_png = os.path.join(args.output_dir, "frequency_band_rmse_per_optimizer.png")
    hf_png = os.path.join(args.output_dir, "hf_energy_ratio_per_optimizer.png")
    plot_frequency_band_rmse(rows, band_png)
    plot_hf_energy_ratio(rows, hf_png)

    print(f"Saved CSV       : {csv_path}", flush=True)
    print(f"Saved band plot : {band_png}", flush=True)
    print(f"Saved HF plot   : {hf_png}", flush=True)


if __name__ == "__main__":
    main()
