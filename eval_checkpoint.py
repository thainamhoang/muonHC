"""Evaluate a saved checkpoint on the test split.

Example:
    python eval_checkpoint.py \
      --config configs/phase_3/cfgs_full_muon.yaml \
      --checkpoint /home/thahoa/muonHC/output/era5/era5_t2m_full_muon_4x_stride6_seed3407/best_checkpoint.pt
"""

import argparse
import os
import sys

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from datasets.downscaling_dataset import DownscalingDataset
from models.downscaling_model import DownscalingModel
from utils.geo import setup_geo_inr_grid
from utils.metrics import log_frequency_distance
from utils.runtime import (
    configure_torch_performance,
    resolve_device,
    seed_everything,
    to_plain_container,
)


def _resolve_amp_dtype(dtype_name):
    dtype_name = str(dtype_name).lower()
    if dtype_name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_name in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"Unsupported AMP dtype: {dtype_name}")


def _make_test_loader(config, device, batch_size=None, num_workers=None):
    loader_cfg = config.get("dataloader", {})
    batch_size = int(batch_size or loader_cfg.get("batch_size", 32))
    num_workers = int(num_workers if num_workers is not None else loader_cfg.get("num_workers", 0))
    pin_memory = bool(loader_cfg.get("pin_memory", str(device).startswith("cuda")))
    persistent_workers = bool(loader_cfg.get("persistent_workers", False)) if num_workers > 0 else False
    prefetch_factor = loader_cfg.get("prefetch_factor", None)

    dataset = DownscalingDataset(
        lr_dir=config.data.lr_dir,
        hr_dir=config.data.hr_dir,
        partition="test",
        temporal=bool(config.data.get("temporal", False)),
        stride=int(config.data.stride),
        lr_preload=bool(config.data.get("lr_preload_eval", True)),
        hr_preload=bool(config.data.get("hr_preload_eval", True)),
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


def _build_model(config, device, hr_shape):
    temporal = bool(config.data.get("temporal", False))
    in_channels = int(config.model.get("in_channels", 3 if temporal else 1))
    hyperloop_kwargs = to_plain_container(config.model.get("hyperloop", None))
    geo_inr_args = to_plain_container(config.model.get("geo_inr", None))
    decoder_hidden_dim = int(config.model.decoder_hidden_dim)

    if geo_inr_args is not None:
        geo_inr_args["out_dim"] = decoder_hidden_dim

    model = DownscalingModel(
        in_channels=in_channels,
        n_coeff=int(config.model.n_coeff),
        embed_dim=int(config.model.embed_dim),
        depth=int(config.model.get("depth", 8)),
        num_heads=int(config.model.num_heads),
        upscale=int(config.model.upscale),
        decoder_hidden_dim=decoder_hidden_dim,
        backbone=config.model.get("backbone", "vit"),
        hyperloop_kwargs=hyperloop_kwargs,
        geo_inr_args=geo_inr_args,
    ).to(device)
    setup_geo_inr_grid(model, config.data.hr_dir, hr_shape, device)
    return model


def _load_model_state(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if all(key.startswith("module.") for key in state.keys()):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)


def _stat_to_device(value, reference):
    if torch.is_tensor(value):
        return value.to(device=reference.device, dtype=reference.dtype)
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled=False, amp_dtype="bfloat16"):
    model.eval()
    device_type = str(device).split(":", maxsplit=1)[0]
    amp_enabled = bool(amp_enabled and device_type == "cuda")
    amp_dtype = _resolve_amp_dtype(amp_dtype)

    dataset = loader.dataset
    target_mean = getattr(dataset, "hr_mean", 278.45)
    target_std = getattr(dataset, "hr_std", 21.25)

    sum_sq_z = torch.zeros((), device=device)
    sum_sq_k = torch.zeros((), device=device)
    lfd_sum = torch.zeros((), device=device)
    n_values = 0
    n_samples = 0

    for lr, hr in loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
            pred = model(lr)
        pred = pred.float()
        hr = hr.float()

        std = _stat_to_device(target_std, pred)
        mean = _stat_to_device(target_mean, pred)
        pred_k = pred * std + mean
        hr_k = hr * std + mean

        sum_sq_z += (pred - hr).square().sum()
        sum_sq_k += (pred_k - hr_k).square().sum()
        lfd_sum += log_frequency_distance(
            pred,
            hr,
            mean=target_mean,
            std=target_std,
            crop_border=1,
            reduction="sum",
        )
        n_values += hr.numel()
        n_samples += hr.size(0)

    return {
        "rmse_k": (sum_sq_k / n_values).sqrt().item(),
        "rmse_z": (sum_sq_z / n_values).sqrt().item(),
        "lfd": (lfd_sum / n_samples).item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML/OmegaConf config.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt or checkpoint.pt.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda or cuda:0.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override test batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override test dataloader workers.")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    seed_everything(int(config.training.get("seed", 42)))
    device = resolve_device(args.device or config.training.get("device", None))
    configure_torch_performance(config.training)

    print(f"Config     : {args.config}", flush=True)
    print(f"Checkpoint : {args.checkpoint}", flush=True)
    print(f"Device     : {device}", flush=True)

    test_dataset, test_loader = _make_test_loader(
        config,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = _build_model(config, device=device, hr_shape=test_dataset.hr_shape)
    _load_model_state(model, args.checkpoint, device=device)

    amp_cfg = config.training.get("amp", {})
    metrics = evaluate(
        model,
        test_loader,
        device=device,
        amp_enabled=bool(amp_cfg.get("enabled", False)),
        amp_dtype=amp_cfg.get("dtype", "bfloat16"),
    )

    print(f"test/rmse_k: {metrics['rmse_k']:.6f}")
    print(f"test/rmse_z: {metrics['rmse_z']:.6f}")
    print(f"test/lfd   : {metrics['lfd']:.6f}")


if __name__ == "__main__":
    main()
