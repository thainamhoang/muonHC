"""Dataset and DataLoader construction helpers."""

from torch.utils.data import DataLoader

from datasets.downscaling_dataset import DownscalingDataset


def _make_loader(dataset, batch_size, num_workers, pin_memory, persistent_workers,
                 prefetch_factor, shuffle):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
    if num_workers > 0 and hasattr(dataset, "worker_init_fn"):
        kwargs["worker_init_fn"] = dataset.worker_init_fn
    return DataLoader(dataset, **kwargs)


def build_datasets_and_loaders(config, device):
    temporal = config.data.get("temporal", False)
    loader_cfg = config.get("dataloader", {})
    batch_size = loader_cfg.get("batch_size", config.training.get("batch_size", 32))
    num_workers = loader_cfg.get("num_workers", config.training.get("num_workers", 0))
    pin_memory = loader_cfg.get("pin_memory", device.startswith("cuda"))
    persistent_workers = (
        loader_cfg.get("persistent_workers", config.training.get("persistent_workers", False))
        if num_workers > 0 else False
    )
    prefetch_factor = loader_cfg.get("prefetch_factor", None)
    lr_dir = config.data.lr_dir
    hr_dir = config.data.hr_dir
    variable_name = config.data.get(
        "var",
        config.get("global_vars", {}).get("var", "2m_temperature"),
    )
    lr_crop_size = config.data.get("lr_crop_size", None)
    upscale = int(config.model.get("upscale", 4))

    print(f"LR directory : {lr_dir}")
    print(f"HR directory : {hr_dir}")
    print(f"Variable     : {variable_name}")

    train_dataset = DownscalingDataset(
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        partition="train",
        temporal=temporal,
        stride=config.data.stride,
        lr_preload=bool(config.data.get("lr_preload_train", True)),
        hr_preload=bool(config.data.get("hr_preload_train", False)),
        variable_name=variable_name,
        lr_crop_size=lr_crop_size,
        random_crop=bool(config.data.get("random_crop_train", False)),
        upscale=upscale,
    )
    val_dataset = DownscalingDataset(
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        partition="val",
        temporal=temporal,
        stride=config.data.stride,
        lr_preload=bool(config.data.get("lr_preload_eval", True)),
        hr_preload=bool(config.data.get("hr_preload_eval", True)),
        variable_name=variable_name,
        lr_crop_size=lr_crop_size,
        random_crop=bool(config.data.get("random_crop_eval", False)),
        upscale=upscale,
    )
    test_dataset = DownscalingDataset(
        lr_dir=lr_dir,
        hr_dir=hr_dir,
        partition="test",
        temporal=temporal,
        stride=config.data.stride,
        lr_preload=bool(config.data.get("lr_preload_eval", True)),
        hr_preload=bool(config.data.get("hr_preload_eval", True)),
        variable_name=variable_name,
        lr_crop_size=lr_crop_size,
        random_crop=bool(config.data.get("random_crop_eval", False)),
        upscale=upscale,
    )

    train_loader = _make_loader(
        train_dataset,
        batch_size,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor,
        shuffle=bool(loader_cfg.get("train_shuffle", True)),
    )
    val_loader = _make_loader(
        val_dataset,
        batch_size,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor,
        shuffle=False,
    )
    test_loader = _make_loader(
        test_dataset,
        batch_size,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor,
        shuffle=False,
    )
    return train_dataset, train_loader, val_loader, test_loader
