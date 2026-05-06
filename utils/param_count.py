"""Parameter counting helpers."""


def count_params(module, trainable_only=True):
    """Count parameters in a module."""
    params = module.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def module_param_breakdown(model, trainable_only=True):
    """Return parameter counts for the main downscaling model components."""
    total = count_params(model, trainable_only=trainable_only)
    rows = [("total", total)]

    for name in ("fck", "vit", "geo_inr", "decoder"):
        module = getattr(model, name, None)
        if module is not None:
            rows.append((name, count_params(module, trainable_only=trainable_only)))

    return rows


def reduction_percent(baseline_params, model_params):
    """Return percent reduction from baseline_params to model_params."""
    if baseline_params <= 0:
        raise ValueError("baseline_params must be positive")
    return 100.0 * (1.0 - model_params / baseline_params)


def format_param_report(model, trainable_only=True):
    """Format the component parameter breakdown for printing."""
    rows = module_param_breakdown(model, trainable_only=trainable_only)
    total = rows[0][1]
    label = "trainable" if trainable_only else "all"
    lines = [f"Parameter report ({label} params):"]

    for name, count in rows:
        if name == "total":
            lines.append(f"  total   : {count:,}")
        else:
            share = count / total if total else 0.0
            lines.append(f"  {name:<8}: {count:,} ({share:.2%})")

    return "\n".join(lines)
