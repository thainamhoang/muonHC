"""Optimizer construction helpers."""

import torch


class CombinedOptimizer(torch.optim.Optimizer):
    """Expose multiple optimizers as one optimizer for schedulers/checkpoints."""

    def __init__(self, optimizers):
        self.optimizers = optimizers
        param_groups = []
        for optimizer in optimizers:
            param_groups.extend(optimizer.param_groups)
        super().__init__(param_groups, defaults={})
        self.param_groups = param_groups

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for optimizer in self.optimizers:
            optimizer.step()
        return loss

    def state_dict(self):
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state_dict):
        for optimizer, optimizer_state in zip(
            self.optimizers,
            state_dict["optimizers"],
        ):
            optimizer.load_state_dict(optimizer_state)


def _is_muon_transformer_weight(name, param):
    if param.ndim != 2:
        return False
    if not name.startswith("vit."):
        return False
    if ".attn." in name:
        return name.endswith("in_proj_weight") or name.endswith("out_proj.weight")
    if ".mlp." in name:
        return name.endswith(".weight")
    return False


def build_optimizer(model, training_cfg, device=None):
    """Build optimizer supporting Muon+AdamW or single optimizers."""
    optimizer_cfg = training_cfg.get("optimizer", {})
    optimizer_type = str(optimizer_cfg.get("type", "adamw")).lower()

    if optimizer_type == "muon_adamw":
        from optims.muon import Muon

        muon_params = []
        adamw_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if _is_muon_transformer_weight(name, param):
                muon_params.append(param)
            else:
                adamw_params.append(param)

        print(
            f"Muon params: {len(muon_params)} tensors "
            f"({sum(p.numel() for p in muon_params):,} params)"
        )
        print(
            f"AdamW params: {len(adamw_params)} tensors "
            f"({sum(p.numel() for p in adamw_params):,} params)"
        )

        muon_lr = float(optimizer_cfg.get("muon_lr", 0.004))
        muon_wd = float(optimizer_cfg.get("muon_weight_decay", 0.05))
        muon_momentum = float(optimizer_cfg.get("muon_momentum", 0.95))
        muon_ns_steps = int(optimizer_cfg.get("muon_ns_steps", 5))
        muon_nesterov = bool(optimizer_cfg.get("muon_nesterov", True))

        adamw_lr = float(optimizer_cfg.get("adamw_lr", 1e-4))
        adamw_wd = float(optimizer_cfg.get("adamw_weight_decay", 0.1))
        betas = tuple(optimizer_cfg.get("adamw_betas", [0.9, 0.999]))

        optimizers = []
        if muon_params:
            optimizers.append(
                Muon(
                    muon_params,
                    lr=muon_lr,
                    weight_decay=muon_wd,
                    momentum=muon_momentum,
                    ns_steps=muon_ns_steps,
                    nesterov=muon_nesterov,
                )
            )
        if adamw_params:
            optimizers.append(
                torch.optim.AdamW(
                    adamw_params,
                    lr=adamw_lr,
                    weight_decay=adamw_wd,
                    betas=betas,
                )
            )
        if not optimizers:
            raise ValueError("No trainable parameters found for optimizer.")
        if len(optimizers) == 1:
            return optimizers[0]
        return CombinedOptimizer(optimizers)

    lr = float(optimizer_cfg.get("lr", training_cfg.get("lr", 2e-4)))
    weight_decay = float(optimizer_cfg.get("weight_decay", training_cfg.get("weight_decay", 1e-4)))
    betas = tuple(optimizer_cfg.get("betas", [0.9, 0.999]))

    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )
    if optimizer_type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )
    raise ValueError(f"Unknown optimizer type: {optimizer_type}")
