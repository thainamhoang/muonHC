"""Pilot 1 training script: static or temporal downscaling from config.

Usage:
    python training.py --config configs/config_static.yaml
    python training.py --config configs/config_temporal.yaml --resume checkpoints/latest_checkpoint.pt
"""

import os
import sys
import argparse
import math
from omegaconf import OmegaConf

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from models.downscaling_model import DownscalingModel
from utils.data import build_datasets_and_loaders
from utils.geo import setup_geo_inr_grid
from utils.optimizers import build_optimizer
from utils.runtime import (
    configure_torch_performance,
    resolve_device,
    seed_everything,
    to_plain_container,
)
from utils.schedulers import build_scheduler
from utils.trainer import Trainer
from utils.wandb_utils import peek_wandb_run_id, setup_wandb


def get_spectral_lambda(config):
    if config.get('loss') is not None:
        return float(config.loss.get('spectral_lambda', 0.0))
    if config.training.get('loss') is not None:
        return float(config.training.loss.get('spectral_lambda', 0.0))
    return 0.0


def resolve_size_agnostic_model_shapes(config, lr_shape, hr_shape):
    output_size = tuple(config.model.get('output_size', hr_shape))
    input_upsample = config.model.get('input_upsample', None)
    if isinstance(input_upsample, str):
        input_upsample_size = output_size if input_upsample.lower() == 'hr' else None
    elif input_upsample:
        input_upsample_size = output_size
    else:
        input_upsample_size = None
    img_size = tuple(config.model.get('img_size', input_upsample_size or lr_shape))
    return img_size, input_upsample_size, output_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML/OmegaConf config')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    args = parser.parse_args()

    # Load with OmegaConf – supports plain YAML and richly structured configs
    config = OmegaConf.load(args.config)
    print(f"Loaded config: {args.config}", flush=True)

    # Set random seed
    seed = config.training.get('seed', 42)
    seed_everything(seed)
    print(f"Seed set: {seed}", flush=True)

    # Data configuration
    temporal = config.data.get('temporal', False)
    in_channels = config.model.get('in_channels', 3 if temporal else 1)
    device = resolve_device(config.training.get('device', None))
    configure_torch_performance(config.training)
    print(f"Building datasets/loaders on device={device}...", flush=True)
    train_dataset, train_loader, val_loader, test_loader = build_datasets_and_loaders(
        config,
        device,
    )
    print("Datasets/loaders ready.", flush=True)

    # Model
    print("Building model...", flush=True)
    hyperloop_kwargs = to_plain_container(config.model.get('hyperloop', None))
    geo_inr_args = to_plain_container(config.model.get('geo_inr', None))
    decoder_hidden_dim = int(config.model.decoder_hidden_dim)
    img_size, input_upsample_size, output_size = resolve_size_agnostic_model_shapes(
        config,
        train_dataset.sample_lr_shape,
        train_dataset.sample_hr_shape,
    )
    if geo_inr_args is not None:
        geo_inr_out_dim = geo_inr_args.get('out_dim', decoder_hidden_dim)
        if geo_inr_out_dim != decoder_hidden_dim:
            print(
                "GeoINR out_dim must match decoder_hidden_dim for FiLM; "
                f"overriding out_dim {geo_inr_out_dim} -> {decoder_hidden_dim}"
            )
            geo_inr_args['out_dim'] = decoder_hidden_dim

    model = DownscalingModel(
        in_channels=in_channels,
        n_coeff=config.model.n_coeff,
        embed_dim=config.model.embed_dim,
        depth=config.model.get('depth', 8),
        num_heads=config.model.num_heads,
        upscale=config.model.upscale,
        decoder_hidden_dim=decoder_hidden_dim,
        backbone=config.model.get('backbone', 'vit'),
        hyperloop_kwargs=hyperloop_kwargs,
        geo_inr_args=geo_inr_args,
        img_size=img_size,
        patch_size=int(config.model.get('patch_size', 1)),
        decoder_upscale=config.model.get('decoder_upscale', None),
        input_upsample_size=input_upsample_size,
        output_size=output_size,
    )
    print("Model built.", flush=True)
    print("Setting GeoINR grid if enabled...", flush=True)
    setup_geo_inr_grid(model, config.data.hr_dir, train_dataset.sample_hr_shape, device)
    print("GeoINR setup complete.", flush=True)

    total_epochs = config.training.max_epochs
    grad_accum_steps = int(config.training.get('grad_accum_steps', 1))
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / max(1, grad_accum_steps))
    print(
        f"Optimizer steps per epoch: {optimizer_steps_per_epoch} "
        f"(micro_batches={len(train_loader)}, grad_accum_steps={grad_accum_steps})",
        flush=True,
    )
    optimizer = build_optimizer(model, config.training, device=device)
    scheduler, scheduler_step_by = build_scheduler(
        optimizer,
        config.training,
        steps_per_epoch=optimizer_steps_per_epoch,
    )

    # Spectral loss lambda (optional)
    spectral_lambda = get_spectral_lambda(config)
    print(f"Spectral loss lambda: {spectral_lambda}")
    amp_cfg = config.training.get('amp', {})
    amp_enabled = bool(amp_cfg.get('enabled', False))
    amp_dtype = amp_cfg.get('dtype', 'bfloat16')
    print(f"AMP enabled: {amp_enabled} ({amp_dtype})", flush=True)
    resume_path = args.resume
    wandb_run_id = peek_wandb_run_id(resume_path)
    print("Setting up WandB...", flush=True)
    run = setup_wandb(config, run_id=wandb_run_id)
    if run is not None and config.get("wandb", {}).get("watch", False):
        run.watch(
            model,
            log=config.wandb.get("watch_log", "gradients"),
            log_freq=config.wandb.get("watch_log_freq", 200),
        )

    print("Starting trainer...", flush=True)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        max_epochs=total_epochs,
        patience=config.training.patience,
        save_dir=config.training.get('save_dir', config.training.get('ckpt_dir', 'checkpoints')),
        spectral_lambda=spectral_lambda,
        target_mean=float(config.data.get('mean', 278.45)),
        target_std=float(config.data.get('std', 21.25)),
        wandb_run=run,
        wandb_run_id=run.id if run is not None else None,
        scheduler_step_by=scheduler_step_by,
        log_interval=int(config.training.get('log_interval', 50)),
        resume_path=resume_path,
        grad_accum_steps=grad_accum_steps,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )

    best_val_rmse = trainer.train()
    mode = "temporal" if temporal else "static"
    print(f"Pilot 1 {mode} finished. Best Val RMSE: {best_val_rmse:.4f} K")
    if run is not None:
        run.finish()

if __name__ == '__main__':
    main()
