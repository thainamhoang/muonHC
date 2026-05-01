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


def configure_torch_performance(training_cfg):
    if not torch.cuda.is_available():
        return

    if bool(training_cfg.get("tf32", False)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    matmul_precision = training_cfg.get("float32_matmul_precision", None)
    if matmul_precision is not None:
        torch.set_float32_matmul_precision(str(matmul_precision))


def to_plain_container(value):
    if value is None:
        return None
    return OmegaConf.to_container(value, resolve=True)
