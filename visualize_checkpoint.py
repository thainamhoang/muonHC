"""Create spectral and Laplacian visualizations for a checkpoint.

Example:
    python visualize_checkpoint.py \
      --config configs/phase_3/t2m/cfgs_full_muon.yaml \
      --checkpoint /home/thahoa/muonHC/output/era5/era5_t2m_full_muon_4x_stride6_seed2026/best_model.pt \
      --output-dir /home/thahoa/muonHC/output/figures/full_muon
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
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from eval_checkpoint import (
    _build_model,
    _load_model_state,
    _make_test_loader,
    _resolve_amp_dtype,
    _stat_to_device,
    configure_eval_backend,
)
from utils.runtime import configure_torch_performance, resolve_device, seed_everything


def denormalize_target(x, dataset):
    mean = _stat_to_device(getattr(dataset, "hr_mean", 0.0), x)
    std = _stat_to_device(getattr(dataset, "hr_std", 1.0), x)
    return x * std + mean


def radial_frequency_grid(height, width, device):
    fy = torch.fft.fftfreq(height, device=device)
    fx = torch.fft.fftfreq(width, device=device)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    return radius / radius.max().clamp_min(1e-12)


def radial_spectrum(x, n_bins=64):
    spectrum = torch.fft.fft2(x.float(), dim=(-2, -1)).abs().square()
    radius = radial_frequency_grid(x.shape[-2], x.shape[-1], x.device)
    bin_idx = torch.clamp((radius * n_bins).long(), max=n_bins - 1).flatten()
    values = spectrum.reshape(-1, x.shape[-2] * x.shape[-1])
    summed = torch.zeros(values.shape[0], n_bins, device=x.device)
    counts = torch.zeros(n_bins, device=x.device)
    summed.scatter_add_(1, bin_idx.expand(values.shape[0], -1), values)
    counts.scatter_add_(0, bin_idx, torch.ones_like(radius.flatten()))
    return (summed / counts.clamp_min(1)).mean(dim=0)


def band_masks(height, width, device):
    radius = radial_frequency_grid(height, width, device)
    return {
        "low": radius < 1.0 / 3.0,
        "mid": (radius >= 1.0 / 3.0) & (radius < 2.0 / 3.0),
        "high": radius >= 2.0 / 3.0,
    }


def band_limited_rmse(pred, target):
    pred_fft = torch.fft.fft2(pred.float(), dim=(-2, -1))
    target_fft = torch.fft.fft2(target.float(), dim=(-2, -1))
    masks = band_masks(pred.shape[-2], pred.shape[-1], pred.device)
    metrics = {}
    for name, mask in masks.items():
        mask = mask[None, None]
        pred_band = torch.fft.ifft2(pred_fft * mask, dim=(-2, -1)).real
        target_band = torch.fft.ifft2(target_fft * mask, dim=(-2, -1)).real
        metrics[name] = (pred_band - target_band).square().mean().sqrt()
    return metrics


def laplacian(x):
    channels = x.shape[1]
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    kernel = kernel.expand(channels, 1, 3, 3)
    return F.conv2d(x, kernel, padding=1, groups=channels)


def collect_predictions(model, loader, device, max_batches, amp_enabled, amp_dtype):
    model.eval()
    device_type = str(device).split(":", maxsplit=1)[0]
    amp_dtype = _resolve_amp_dtype(amp_dtype)
    dataset = loader.dataset
    pred_batches = []
    target_batches = []

    with torch.no_grad():
        for batch_idx, (lr, hr) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device_type,
                dtype=amp_dtype,
                enabled=bool(amp_enabled and device_type == "cuda"),
            ):
                pred = model(lr)
            pred_batches.append(denormalize_target(pred.float(), dataset).cpu())
            target_batches.append(denormalize_target(hr.float(), dataset).cpu())

    return torch.cat(pred_batches, dim=0), torch.cat(target_batches, dim=0)


def save_radial_spectra(pred, target, output_dir, n_bins):
    pred_spec = radial_spectrum(pred, n_bins=n_bins).cpu()
    target_spec = radial_spectrum(target, n_bins=n_bins).cpu()
    x = torch.linspace(0.0, 1.0, n_bins)

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(x, target_spec.clamp_min(1e-12), label="GT", linewidth=2.0)
    plt.plot(x, pred_spec.clamp_min(1e-12), label="Prediction", linewidth=2.0)
    plt.yscale("log")
    plt.xlabel("Normalized radial frequency")
    plt.ylabel("Average power spectrum")
    plt.title("Radial Spectrum: GT vs Prediction")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "radial_spectrum_gt_vs_pred.png")
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def save_band_rmse(pred, target, output_dir):
    metrics = {name: float(value.cpu()) for name, value in band_limited_rmse(pred, target).items()}

    json_path = os.path.join(output_dir, "band_rmse.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    csv_path = os.path.join(output_dir, "band_rmse.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["band", "rmse"])
        for band, value in metrics.items():
            writer.writerow([band, value])

    plt.figure(figsize=(5.5, 4.0))
    plt.bar(metrics.keys(), metrics.values(), color=["#4c78a8", "#f58518", "#e45756"])
    plt.ylabel("Band-limited RMSE")
    plt.title("Low / Mid / High Frequency RMSE")
    plt.tight_layout()
    png_path = os.path.join(output_dir, "band_rmse.png")
    plt.savefig(png_path, dpi=180)
    plt.close()
    return metrics, json_path, csv_path, png_path


def save_laplacian_maps(pred, target, output_dir, sample_index, channel):
    sample_index = min(sample_index, pred.shape[0] - 1)
    channel = min(channel, pred.shape[1] - 1)
    pred_one = pred[sample_index:sample_index + 1].to(torch.float32)
    target_one = target[sample_index:sample_index + 1].to(torch.float32)
    pred_lap = laplacian(pred_one)
    target_lap = laplacian(target_one)
    err = (pred_lap - target_lap).abs()

    panels = [
        ("Laplacian(GT)", target_lap[0, channel]),
        ("Laplacian(pred)", pred_lap[0, channel]),
        ("abs error", err[0, channel]),
    ]
    vmax = max(float(panels[0][1].abs().max()), float(panels[1][1].abs().max()), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    for ax, (title, image) in zip(axes, panels):
        if title == "abs error":
            im = ax.imshow(image.cpu(), cmap="magma")
        else:
            im = ax.imshow(image.cpu(), cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    path = os.path.join(output_dir, f"laplacian_maps_sample{sample_index}_ch{channel}.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--n-bins", type=int, default=64)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--disable-cudnn", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config = OmegaConf.load(args.config)
    seed_everything(int(config.training.get("seed", 42)))
    device = resolve_device(args.device or config.training.get("device", None))
    configure_torch_performance(config.training)
    configure_eval_backend(disable_cudnn=args.disable_cudnn)

    dataset, loader = _make_test_loader(
        config,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = _build_model(config, device=device, hr_shape=dataset.hr_shape)
    _load_model_state(model, args.checkpoint, device=device)

    amp_cfg = config.training.get("amp", {})
    pred, target = collect_predictions(
        model,
        loader,
        device=device,
        max_batches=args.max_batches,
        amp_enabled=args.amp,
        amp_dtype=amp_cfg.get("dtype", "bfloat16"),
    )

    spectra_path = save_radial_spectra(pred, target, args.output_dir, args.n_bins)
    band_metrics, band_json, band_csv, band_png = save_band_rmse(pred, target, args.output_dir)
    lap_path = save_laplacian_maps(
        pred,
        target,
        args.output_dir,
        sample_index=args.sample_index,
        channel=args.channel,
    )

    print(f"Saved radial spectra : {spectra_path}")
    print(f"Saved band RMSE plot : {band_png}")
    print(f"Saved band RMSE JSON : {band_json}")
    print(f"Saved band RMSE CSV  : {band_csv}")
    print(f"Saved Laplacian maps : {lap_path}")
    print("Band RMSE:")
    for band, value in band_metrics.items():
        print(f"  {band}: {value:.6f}")


if __name__ == "__main__":
    main()
