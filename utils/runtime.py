"""Runtime helpers shared by training entrypoints."""

import torch
from omegaconf import OmegaConf


def resolve_device(requested_device):
    if requested_device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    requested_device = str(requested_device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return requested_device


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_plain_container(value):
    if value is None:
        return None
    return OmegaConf.to_container(value, resolve=True)
