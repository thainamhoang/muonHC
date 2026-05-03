"""WandB setup and resume helpers."""

import os

import torch
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


def peek_wandb_run_id(path):
    if not path or not os.path.exists(path):
        return None
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"Could not inspect checkpoint for WandB resume id: {exc}")
        return None
    return ckpt.get("wandb_run_id")


def setup_wandb(config, run_id=None):
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
        tags=wandb_cfg.get("tags", None),
        config=OmegaConf.to_container(config, resolve=True),
    )
    print(f"WandB run: {run.url if run.url else run.id}")
    return run
