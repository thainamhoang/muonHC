"""Learning-rate scheduler construction helpers."""

import torch


def build_scheduler(optimizer, training_cfg, steps_per_epoch):
    scheduler_cfg = training_cfg.get("scheduler", {})
    scheduler_type = str(scheduler_cfg.get("type", "cosine")).lower()
    max_epochs = int(training_cfg.max_epochs)
    step_by = str(scheduler_cfg.get("step_by", "epoch")).lower()

    if scheduler_type in ("none", "disabled"):
        return None, step_by

    if scheduler_type == "cosine":
        t_max = max(1, max_epochs * steps_per_epoch) if step_by == "step" else max_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=float(scheduler_cfg.get("eta_min", scheduler_cfg.get("min_lr", 0.0))),
        )
        return scheduler, step_by

    if scheduler_type == "warmup_cosine":
        warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 5))
        warmup_start_factor = float(scheduler_cfg.get("warmup_start_factor", 0.01))
        eta_min = float(scheduler_cfg.get("eta_min", scheduler_cfg.get("min_lr", 1e-6)))

        warmup_iters = warmup_epochs * steps_per_epoch if step_by == "step" else warmup_epochs
        total_iters = max_epochs * steps_per_epoch if step_by == "step" else max_epochs
        cosine_iters = max(1, total_iters - warmup_iters)

        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_iters,
            eta_min=eta_min,
        )
        if warmup_iters <= 0:
            return cosine, step_by

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_iters,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_iters],
        )
        return scheduler, step_by

    raise ValueError(f"Unknown scheduler type: {scheduler_type}")
