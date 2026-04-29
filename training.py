"""Pilot 1 training script: static or temporal downscaling from config.

Usage:
    python training.py --config configs/config_static.yaml
    python training.py --config configs/config_temporal.yaml --resume checkpoints/latest_checkpoint.pt
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False

try:
    import wandb
except ImportError:
    wandb = None

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from models.downscaling_model import DownscalingModel
from datasets.temporal_dataset import TemporalDownscalingDataset
from utils.trainer import Trainer


def _resolve_device(requested_device):
    if requested_device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    requested_device = str(requested_device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return requested_device


def _seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _peek_wandb_run_id(path):
    if not path or not os.path.exists(path):
        return None
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"Could not inspect checkpoint for WandB resume id: {exc}")
        return None
    return ckpt.get("wandb_run_id")


def _setup_wandb(config, run_id=None):
    wandb_configured = "wandb" in config
    wandb_cfg = config.get("wandb", {})
    if wandb_cfg.get("enabled", True) is False:
        return None
    if wandb is None:
        if not wandb_configured:
            print("wandb is not installed and no wandb config was provided; tracking disabled.")
            return None
        raise ImportError(
            "WandB tracking is enabled, but wandb is not installed. "
            "Install it with `pip install wandb` or set wandb.enabled=false."
        )

    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
    elif wandb_cfg.get("mode") is None and os.getenv("WANDB_MODE") is None:
        print("WANDB_API_KEY is not set; using WandB offline mode.")

    mode = wandb_cfg.get(
        "mode",
        os.getenv("WANDB_MODE", "online" if wandb_api_key else "offline"),
    )
    run = wandb.init(
        entity=wandb_cfg.get("entity", None),
        project=wandb_cfg.get("project", "muonHC"),
        name=wandb_cfg.get("name", None),
        id=run_id,
        resume="allow",
        mode=mode,
        config=OmegaConf.to_container(config, resolve=True),
    )
    print(f"WandB run: {run.url if run.url else run.id}")
    return run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML/OmegaConf config')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    args = parser.parse_args()

    # Load with OmegaConf – supports plain YAML and richly structured configs
    config = OmegaConf.load(args.config)

    # Set random seed
    seed = config.training.get('seed', 42)
    _seed_everything(seed)

    # Data configuration
    temporal = config.data.get('temporal', False)
    in_channels = config.model.get('in_channels', 3 if temporal else 1)
    device = _resolve_device(config.training.get('device', None))

    train_dataset = TemporalDownscalingDataset(
        root_dir=config.data.root_dir,
        split='train',
        temporal=temporal,
        stride=config.data.stride,
        normalize=True,
        mean=config.data.mean,
        std=config.data.std
    )
    val_dataset = TemporalDownscalingDataset(
        root_dir=config.data.root_dir,
        split='val',
        temporal=temporal,
        stride=config.data.stride,
        normalize=True,
        mean=config.data.mean,
        std=config.data.std
    )
    test_dataset = TemporalDownscalingDataset(
        root_dir=config.data.root_dir,
        split='test',
        temporal=temporal,
        stride=config.data.stride,
        normalize=True,
        mean=config.data.mean,
        std=config.data.std
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.get('pin_memory', device.startswith('cuda')),
        persistent_workers=(
            config.training.get('persistent_workers', False)
            if config.training.num_workers > 0 else False
        )
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.get('pin_memory', device.startswith('cuda')),
        persistent_workers=(
            config.training.get('persistent_workers', False)
            if config.training.num_workers > 0 else False
        )
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.get('pin_memory', device.startswith('cuda')),
        persistent_workers=(
            config.training.get('persistent_workers', False)
            if config.training.num_workers > 0 else False
        )
    )

    # Model
    model = DownscalingModel(
        in_channels=in_channels,
        n_coeff=config.model.n_coeff,
        embed_dim=config.model.embed_dim,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        upscale=config.model.upscale,
        hidden_dim=config.model.decoder_hidden_dim,
        geo_inr_args=None
    )

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training.lr),
        weight_decay=float(config.training.weight_decay)
    )
    total_epochs = config.training.max_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    # Spectral loss lambda (optional)
    spectral_lambda = config.get('loss', {}).get('spectral_lambda', 0.0)
    resume_path = args.resume
    wandb_run_id = _peek_wandb_run_id(resume_path)
    run = _setup_wandb(config, run_id=wandb_run_id)
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
        resume_path=resume_path,
    )

    best_val_rmse = trainer.train()
    mode = "temporal" if temporal else "static"
    print(f"Pilot 1 {mode} finished. Best Val RMSE: {best_val_rmse:.4f} K")
    if run is not None:
        run.finish()

if __name__ == '__main__':
    main()
