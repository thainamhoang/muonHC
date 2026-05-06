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
from matplotlib.colors import LinearSegmentedColormap
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


GEOFAR_BLUE = LinearSegmentedColormap.from_list(
    "geofar_blue",
    [
        "#f7fbff",
        "#d6eff2",
        "#9bd3dd",
        "#4f97bd",
        "#1f4f8b",
        "#0b1f5e",
    ],
)

BIAS_RED = LinearSegmentedColormap.from_list(
    "bias_red",
    [
        "#fff7ec",
        "#fee8c8",
        "#fdbb84",
        "#ef6548",
        "#b30000",
        "#67000d",
    ],
)


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
    x = x.float()
    x = x - x.mean(dim=(-2, -1), keepdim=True)
    spectrum = torch.log1p(torch.fft.fft2(x, dim=(-2, -1)).abs())
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
        "low": radius < 0.15,
        "mid": (radius >= 0.15) & (radius < 0.50),
        "high": radius >= 0.50,
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


def relative_band_rmse(pred, target, eps=1e-8):
    pred_fft = torch.fft.fft2(pred.float(), dim=(-2, -1))
    target_fft = torch.fft.fft2(target.float(), dim=(-2, -1))
    masks = band_masks(pred.shape[-2], pred.shape[-1], pred.device)
    metrics = {}
    for name, mask in masks.items():
        mask = mask[None, None]
        pred_band = torch.fft.ifft2(pred_fft * mask, dim=(-2, -1)).real
        target_band = torch.fft.ifft2(target_fft * mask, dim=(-2, -1)).real
        numerator = (pred_band - target_band).square().mean(dim=(-2, -1)).sqrt()
        denominator = target_band.square().mean(dim=(-2, -1)).sqrt().clamp_min(eps)
        metrics[name] = (numerator / denominator).mean()
    return metrics


def high_frequency_energy_ratio(x, high_cut=0.50, eps=1e-8):
    x = x.float()
    x = x - x.mean(dim=(-2, -1), keepdim=True)
    magnitude = torch.fft.fft2(x, dim=(-2, -1)).abs()
    radius = radial_frequency_grid(x.shape[-2], x.shape[-1], x.device)
    high_mask = radius >= high_cut
    high_energy = magnitude[..., high_mask].sum(dim=-1)
    total_energy = magnitude.flatten(start_dim=-2).sum(dim=-1).clamp_min(eps)
    return high_energy / total_energy


def laplacian(x):
    channels = x.shape[1]
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    kernel = kernel.expand(channels, 1, 3, 3)
    x = F.pad(x, (1, 1, 1, 1), mode="reflect")
    return F.conv2d(x, kernel, padding=0, groups=channels)


def robust_abs_limit(images, quantile=0.995):
    values = torch.cat([image.detach().abs().flatten() for image in images])
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return 1e-6
    return max(float(torch.quantile(values, quantile)), 1e-6)


def resolve_bias_colormap(colorway):
    if colorway == "blue":
        return GEOFAR_BLUE
    if colorway == "red":
        return BIAS_RED
    raise ValueError("colorway must be 'blue' or 'red'")


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
    spectrum_error = (pred_spec - target_spec).abs()

    csv_path = os.path.join(output_dir, "radial_spectrum.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["normalized_radial_frequency", "gt", "prediction", "abs_error"])
        for freq, gt_value, pred_value, err_value in zip(
            x.tolist(),
            target_spec.tolist(),
            pred_spec.tolist(),
            spectrum_error.tolist(),
        ):
            writer.writerow([freq, gt_value, pred_value, err_value])

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(x, target_spec, label="GT", linewidth=2.0)
    plt.plot(x, pred_spec, label="Prediction", linewidth=2.0)
    plt.xlabel("Normalized radial frequency")
    plt.ylabel("Mean log(1 + Fourier magnitude)")
    plt.title("Radial Spectrum: GT vs Prediction")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "radial_spectrum_gt_vs_pred.png")
    plt.savefig(path, dpi=180)
    plt.close()

    plt.figure(figsize=(7.0, 4.5))
    plt.plot(x, spectrum_error, linewidth=2.0, color="#d62728")
    plt.xlabel("Normalized radial frequency")
    plt.ylabel("|Prediction spectrum - GT spectrum|")
    plt.title("Radial Spectrum Error")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    error_path = os.path.join(output_dir, "radial_spectrum_error.png")
    plt.savefig(error_path, dpi=180)
    plt.close()
    return path, error_path, csv_path, float(spectrum_error.nanmean())


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


def save_relative_band_rmse(pred, target, output_dir):
    metrics = {name: float(value.cpu()) for name, value in relative_band_rmse(pred, target).items()}

    json_path = os.path.join(output_dir, "relative_band_rmse.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    csv_path = os.path.join(output_dir, "relative_band_rmse.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["band", "relative_rmse"])
        for band, value in metrics.items():
            writer.writerow([band, value])

    plt.figure(figsize=(5.8, 4.0))
    plt.bar(metrics.keys(), metrics.values(), color=["#4c78a8", "#f58518", "#e45756"])
    plt.ylabel("Relative band RMSE")
    plt.title("Relative Low / Mid / High Frequency Error")
    plt.tight_layout()
    png_path = os.path.join(output_dir, "relative_band_rmse.png")
    plt.savefig(png_path, dpi=180)
    plt.close()
    return metrics, json_path, csv_path, png_path


def save_high_frequency_energy_ratio(pred, target, output_dir):
    pred_ratio = high_frequency_energy_ratio(pred).mean()
    target_ratio = high_frequency_energy_ratio(target).mean()
    pred_over_gt = pred_ratio / target_ratio.clamp_min(1e-8)
    metrics = {
        "prediction": float(pred_ratio.cpu()),
        "gt": float(target_ratio.cpu()),
        "prediction_over_gt": float(pred_over_gt.cpu()),
    }

    json_path = os.path.join(output_dir, "high_frequency_energy_ratio.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    csv_path = os.path.join(output_dir, "high_frequency_energy_ratio.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["quantity", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])

    plt.figure(figsize=(5.8, 4.0))
    plt.bar(["GT", "Prediction"], [metrics["gt"], metrics["prediction"]], color=["#4c78a8", "#f58518"])
    plt.ylabel("High-frequency energy ratio")
    plt.title(f"High-Frequency Energy Ratio (Pred / GT = {metrics['prediction_over_gt']:.3f})")
    plt.tight_layout()
    png_path = os.path.join(output_dir, "high_frequency_energy_ratio.png")
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
    signed_vmax = robust_abs_limit([panels[0][1], panels[1][1]])
    err_vmax = robust_abs_limit([panels[2][1]])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
    for ax, (title, image) in zip(axes, panels):
        if title == "abs error":
            im = ax.imshow(image.cpu(), cmap="magma", vmin=0.0, vmax=err_vmax)
        else:
            im = ax.imshow(image.cpu(), cmap="coolwarm", vmin=-signed_vmax, vmax=signed_vmax)
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    path = os.path.join(output_dir, f"laplacian_maps_sample{sample_index}_ch{channel}.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_laplacian_bias_map(
    pred,
    target,
    output_dir,
    sample_index,
    channel,
    colorway="blue",
    mode="abs",
    percentile=99.0,
):
    sample_index = min(sample_index, pred.shape[0] - 1)
    channel = min(channel, pred.shape[1] - 1)
    pred_one = pred[sample_index:sample_index + 1].to(torch.float32)
    target_one = target[sample_index:sample_index + 1].to(torch.float32)
    diff = laplacian(pred_one) - laplacian(target_one)
    image = diff.abs()[0, channel] if mode == "abs" else diff[0, channel]

    quantile = min(max(float(percentile) / 100.0, 0.0), 1.0)
    vmax = robust_abs_limit([image], quantile=quantile)
    if mode == "abs":
        cmap = resolve_bias_colormap(colorway)
        vmin = 0.0
        cbar_label = "Laplace-filtered absolute error"
    elif mode == "signed":
        cmap = "coolwarm"
        vmin = -vmax
        cbar_label = "Laplace-filtered signed error"
    else:
        raise ValueError("mode must be 'abs' or 'signed'")

    fig, ax = plt.subplots(figsize=(10.0, 4.0), constrained_layout=True)
    im = ax.imshow(image.cpu(), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title("Prediction Bias (Laplace filtered)", fontsize=18)
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(cbar_label, fontsize=12)

    path = os.path.join(
        output_dir,
        f"laplacian_bias_{colorway}_{mode}_sample{sample_index}_ch{channel}.png",
    )
    fig.savefig(path, dpi=300, bbox_inches="tight")
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
    parser.add_argument("--laplacian-bias-colorway", choices=["blue", "red"], default="blue")
    parser.add_argument("--laplacian-bias-mode", choices=["abs", "signed"], default="abs")
    parser.add_argument("--laplacian-bias-percentile", type=float, default=99.0)
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
    model = _build_model(
        config,
        device=device,
        lr_shape=dataset.sample_lr_shape,
        hr_shape=dataset.sample_hr_shape,
    )
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

    spectra_path, spectrum_error_path, spectrum_csv, radial_spectrum_l1 = save_radial_spectra(
        pred,
        target,
        args.output_dir,
        args.n_bins,
    )
    band_metrics, band_json, band_csv, band_png = save_band_rmse(pred, target, args.output_dir)
    rel_band_metrics, rel_band_json, rel_band_csv, rel_band_png = save_relative_band_rmse(
        pred,
        target,
        args.output_dir,
    )
    hf_energy_metrics, hf_energy_json, hf_energy_csv, hf_energy_png = save_high_frequency_energy_ratio(
        pred,
        target,
        args.output_dir,
    )
    lap_path = save_laplacian_maps(
        pred,
        target,
        args.output_dir,
        sample_index=args.sample_index,
        channel=args.channel,
    )
    lap_bias_path = save_laplacian_bias_map(
        pred,
        target,
        args.output_dir,
        sample_index=args.sample_index,
        channel=args.channel,
        colorway=args.laplacian_bias_colorway,
        mode=args.laplacian_bias_mode,
        percentile=args.laplacian_bias_percentile,
    )

    print(f"Saved radial spectra : {spectra_path}")
    print(f"Saved spectrum error : {spectrum_error_path}")
    print(f"Saved spectrum CSV   : {spectrum_csv}")
    print(f"Saved band RMSE plot : {band_png}")
    print(f"Saved band RMSE JSON : {band_json}")
    print(f"Saved band RMSE CSV  : {band_csv}")
    print(f"Saved relative band  : {rel_band_png}")
    print(f"Saved relative JSON  : {rel_band_json}")
    print(f"Saved relative CSV   : {rel_band_csv}")
    print(f"Saved high-freq plot : {hf_energy_png}")
    print(f"Saved high-freq JSON : {hf_energy_json}")
    print(f"Saved high-freq CSV  : {hf_energy_csv}")
    print(f"Saved Laplacian maps : {lap_path}")
    print(f"Saved Laplacian bias : {lap_bias_path}")
    print("Band RMSE:")
    for band, value in band_metrics.items():
        print(f"  {band}: {value:.6f}")
    print("Relative band RMSE:")
    for band, value in rel_band_metrics.items():
        print(f"  {band}: {value:.6f}")
    print("High-frequency energy ratio:")
    print(f"  gt: {hf_energy_metrics['gt']:.6f}")
    print(f"  prediction: {hf_energy_metrics['prediction']:.6f}")
    print(f"  prediction_over_gt: {hf_energy_metrics['prediction_over_gt']:.6f}")
    print(f"Radial spectrum L1: {radial_spectrum_l1:.6f}")


if __name__ == "__main__":
    main()
