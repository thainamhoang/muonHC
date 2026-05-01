"""Pilot 1 training script: static or temporal downscaling from config.

Usage:
    python training.py --config configs/config_static.yaml
    python training.py --config configs/config_temporal.yaml --resume checkpoints/latest_checkpoint.pt
"""

import os
import sys
import argparse
from omegaconf import OmegaConf

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from models.downscaling_model import DownscalingModel
from utils.data import build_datasets_and_loaders
from utils.geo import setup_geo_inr_grid
from utils.optimizers import build_optimizer
from utils.runtime import resolve_device, seed_everything, to_plain_container
from utils.schedulers import build_scheduler
from utils.trainer import Trainer
from utils.wandb_utils import peek_wandb_run_id, setup_wandb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML/OmegaConf config')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    args = parser.parse_args()

    # Load with OmegaConf – supports plain YAML and richly structured configs
    config = OmegaConf.load(args.config)

    # Set random seed
    seed = config.training.get('seed', 42)
    seed_everything(seed)

    # Data configuration
    temporal = config.data.get('temporal', False)
    in_channels = config.model.get('in_channels', 3 if temporal else 1)
    device = resolve_device(config.training.get('device', None))
    train_dataset, train_loader, val_loader, test_loader = build_datasets_and_loaders(
        config,
        device,
    )

    # Model
    hyperloop_kwargs = to_plain_container(config.model.get('hyperloop', None))
    geo_inr_args = to_plain_container(config.model.get('geo_inr', None))
    decoder_hidden_dim = int(config.model.decoder_hidden_dim)
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
    )
    setup_geo_inr_grid(model, config.data.hr_dir, train_dataset.hr_shape, device)

    total_epochs = config.training.max_epochs
    optimizer = build_optimizer(model, config.training, device=device)
    scheduler, scheduler_step_by = build_scheduler(
        optimizer,
        config.training,
        steps_per_epoch=len(train_loader),
    )

    # Spectral loss lambda (optional)
    spectral_lambda = config.get('loss', {}).get(
        'spectral_lambda',
        config.training.get('loss', {}).get('spectral_lambda', 0.0),
    )
    resume_path = args.resume
    wandb_run_id = peek_wandb_run_id(resume_path)
    run = setup_wandb(config, run_id=wandb_run_id)
    if run is not None and config.get("wandb", {}).get("watch", False):
        run.watch(
            model,
            log=config.wandb.get("watch_log", "gradients"),
            log_freq=config.wandb.get("watch_log_freq", 200),
        )

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
    )

    best_val_rmse = trainer.train()
    mode = "temporal" if temporal else "static"
    print(f"Pilot 1 {mode} finished. Best Val RMSE: {best_val_rmse:.4f} K")
    if run is not None:
        run.finish()

if __name__ == '__main__':
    main()
