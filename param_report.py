"""Print model and backbone parameter counts from muonHC configs.

Example:
    python param_report.py \
        configs/phase_3/t2m/cfgs_full_muon_wo_mhc.yaml \
        configs/phase_3/t2m/cfgs_full_muon.yaml
"""

import argparse
import os
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

try:
    from models.downscaling_model import DownscalingModel
    from utils.param_count import count_params
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    DownscalingModel = None
    count_params = None

from utils.param_count import reduction_percent


class AttrDict(dict):
    """Small OmegaConf-like wrapper for plain YAML maps."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _wrap(value):
    if isinstance(value, dict):
        return AttrDict({key: _wrap(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


def _to_plain_container(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _to_plain_container(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain_container(item) for item in value]
    return value


def _strip_comment(line):
    in_single = False
    in_double = False
    for idx, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:idx]
    return line


def _parse_scalar(value):
    value = value.strip()
    if value == "":
        return {}
    if value in ("true", "false"):
        return value == "true"
    if value in ("null", "None", "~"):
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_config(path):
    """Load the simple YAML subset used by these config files."""
    return _wrap(_fix_list_blocks(path))


def _fix_list_blocks(path):
    root = {}
    stack = [(-1, root)]
    lines = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = _strip_comment(raw_line).rstrip()
            if line.strip():
                lines.append(line)

    for idx, line in enumerate(lines):
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]

        if stripped.startswith("- "):
            if not isinstance(current, list):
                raise ValueError(f"List item without list parent in {path}: {line}")
            current.append(_parse_scalar(stripped[2:]))
            continue

        key, _, value = stripped.partition(":")
        value = value.strip()
        if value:
            current[key] = _parse_scalar(value)
            continue

        next_is_list = False
        if idx + 1 < len(lines):
            next_line = lines[idx + 1]
            next_indent = len(next_line) - len(next_line.lstrip(" "))
            next_is_list = next_indent > indent and next_line.strip().startswith("- ")
        container = [] if next_is_list else {}
        current[key] = container
        stack.append((indent, container))

    return root


def _shape_arg(value):
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("shape must be H,W or HxW")
    return tuple(int(part) for part in parts)


def _resolve_shapes(config, lr_shape, hr_shape):
    output_size = tuple(config.model.get("output_size", hr_shape))
    input_upsample = config.model.get("input_upsample", None)
    if isinstance(input_upsample, str):
        input_upsample_size = output_size if input_upsample.lower() == "hr" else None
    elif input_upsample:
        input_upsample_size = output_size
    else:
        input_upsample_size = None
    img_size = tuple(config.model.get("img_size", input_upsample_size or lr_shape))
    return img_size, input_upsample_size, output_size


def _build_model_from_config(config, lr_shape, hr_shape):
    temporal = bool(config.data.get("temporal", False))
    in_channels = int(config.model.get("in_channels", 3 if temporal else 1))
    hyperloop_kwargs = _to_plain_container(config.model.get("hyperloop", None))
    geo_inr_args = _to_plain_container(config.model.get("geo_inr", None))
    decoder_hidden_dim = int(config.model.decoder_hidden_dim)

    if geo_inr_args is not None:
        geo_inr_out_dim = int(geo_inr_args.get("out_dim", decoder_hidden_dim))
        if geo_inr_out_dim != decoder_hidden_dim:
            print(
                "GeoINR out_dim must match decoder_hidden_dim for FiLM; "
                f"counting with out_dim {geo_inr_out_dim} -> {decoder_hidden_dim}",
                flush=True,
            )
            geo_inr_args["out_dim"] = decoder_hidden_dim

    img_size, input_upsample_size, output_size = _resolve_shapes(
        config,
        lr_shape=lr_shape,
        hr_shape=hr_shape,
    )

    return DownscalingModel(
        in_channels=in_channels,
        n_coeff=int(config.model.n_coeff),
        embed_dim=int(config.model.embed_dim),
        depth=int(config.model.get("depth", 8)),
        num_heads=int(config.model.num_heads),
        upscale=int(config.model.upscale),
        decoder_hidden_dim=decoder_hidden_dim,
        backbone=config.model.get("backbone", "vit"),
        hyperloop_kwargs=hyperloop_kwargs,
        geo_inr_args=geo_inr_args,
        img_size=img_size,
        patch_size=int(config.model.get("patch_size", 1)),
        decoder_upscale=config.model.get("decoder_upscale", None),
        input_upsample_size=input_upsample_size,
        output_size=output_size,
    )


def _conv2d_params(in_channels, out_channels, kernel_size, groups=1, bias=True):
    weight = out_channels * (in_channels // groups) * kernel_size * kernel_size
    return weight + (out_channels if bias else 0)


def _transformer_block_params(dim, mlp_ratio):
    hidden_dim = int(dim * mlp_ratio)
    layer_norms = 4 * dim
    attention = 4 * dim * dim + 4 * dim
    mlp = 2 * dim * hidden_dim + hidden_dim + dim
    return layer_norms + attention + mlp


def _spatial_loop_gate_params(dim, hidden_ratio):
    hidden_dim = max(16, int(dim * hidden_ratio))
    layer_norm = 2 * dim
    first = dim * hidden_dim + hidden_dim
    second = hidden_dim + 1
    return layer_norm + first + second


def _analytic_counts_from_config(config, lr_shape, hr_shape):
    img_size, _, _ = _resolve_shapes(config, lr_shape=lr_shape, hr_shape=hr_shape)
    grid_h = img_size[0] // int(config.model.get("patch_size", 1))
    grid_w = img_size[1] // int(config.model.get("patch_size", 1))
    temporal = bool(config.data.get("temporal", False))
    in_channels = int(config.model.get("in_channels", 3 if temporal else 1))
    n_coeff = int(config.model.n_coeff)
    vit_in_channels = in_channels * n_coeff
    embed_dim = int(config.model.embed_dim)
    patch_size = int(config.model.get("patch_size", 1))
    mlp_ratio = 4
    backbone = config.model.get("backbone", "vit")

    if backbone == "hyperloop_mhc":
        hyperloop = config.model.hyperloop
        mlp_ratio = float(hyperloop.get("mlp_ratio", 4))
    block_params = _transformer_block_params(embed_dim, mlp_ratio)

    patch_embed = _conv2d_params(
        vit_in_channels,
        embed_dim,
        patch_size,
        bias=True,
    )
    pos_embed = embed_dim * grid_h * grid_w
    norm = 2 * embed_dim

    if backbone == "hyperloop_mhc":
        hyperloop = config.model.hyperloop
        begin_depth = int(hyperloop.get("begin_depth", 2))
        middle_depth = int(hyperloop.get("middle_depth", 4))
        end_depth = int(hyperloop.get("end_depth", 2))
        loops = int(hyperloop.get("K", 3))
        n_streams = int(hyperloop.get("n_streams", 2))
        use_spatial_gate = bool(hyperloop.get("use_spatial_gate", False))
        loop_hyper = n_streams + n_streams * n_streams
        if use_spatial_gate:
            loop_hyper += n_streams * _spatial_loop_gate_params(
                embed_dim,
                float(hyperloop.get("gate_hidden_ratio", 0.25)),
            )
        else:
            loop_hyper += n_streams
        loop_pos = loops * embed_dim
        backbone_params = (
            patch_embed
            + pos_embed
            + (begin_depth + middle_depth + end_depth) * block_params
            + loop_hyper
            + loop_pos
            + norm
        )
    else:
        depth = int(config.model.get("depth", 8))
        backbone_params = patch_embed + pos_embed + depth * block_params + norm

    decoder_hidden_dim = int(config.model.decoder_hidden_dim)
    decoder_upscale = int(
        config.model.get("decoder_upscale", None)
        or (int(config.model.upscale) * patch_size)
    )
    decoder_mid = decoder_hidden_dim * decoder_upscale * decoder_upscale
    decoder_params = (
        _conv2d_params(embed_dim, decoder_mid, 1)
        + _conv2d_params(decoder_hidden_dim, decoder_hidden_dim, 3)
        + _conv2d_params(decoder_hidden_dim, 1, 3)
    )

    geo_params = 0
    geo_inr_args = config.model.get("geo_inr", None)
    if geo_inr_args is not None:
        n_basis = int(geo_inr_args.get("n_basis", 8))
        hidden_dim = int(geo_inr_args.get("hidden_dim", 256))
        out_dim = decoder_hidden_dim
        geo_in = n_basis * n_basis + 1
        geo_params = geo_in * hidden_dim + hidden_dim
        geo_params += hidden_dim * (out_dim * 2) + (out_dim * 2)

    counts = {
        "fck": 0,
        "vit": backbone_params,
        "geo_inr": geo_params,
        "decoder": decoder_params,
    }
    counts["total"] = sum(counts.values())
    return counts


def _depth_summary(config):
    backbone = config.model.get("backbone", "vit")
    if backbone == "hyperloop_mhc":
        hyperloop = config.model.hyperloop
        begin_depth = int(hyperloop.get("begin_depth", 2))
        middle_depth = int(hyperloop.get("middle_depth", 4))
        end_depth = int(hyperloop.get("end_depth", 2))
        loops = int(hyperloop.get("K", 3))
        unique_depth = begin_depth + middle_depth + end_depth
        effective_depth = begin_depth + middle_depth * loops + end_depth
        return effective_depth, unique_depth
    depth = int(config.model.get("depth", 8))
    return depth, depth


def _model_name(config, path):
    mode = config.get("global_vars", {}).get("mode", None)
    if mode:
        return str(mode)
    return os.path.splitext(os.path.basename(path))[0]


def _collect_row(path, lr_shape, hr_shape, trainable_only):
    config = _load_config(path)
    counts = None
    model = None
    if DownscalingModel is not None:
        model = _build_model_from_config(config, lr_shape=lr_shape, hr_shape=hr_shape)
        total_params = count_params(model, trainable_only=trainable_only)
        backbone_params = count_params(model.vit, trainable_only=trainable_only)
    else:
        if not trainable_only:
            raise RuntimeError("--all-params requires torch so frozen FCK weights can be instantiated")
        counts = _analytic_counts_from_config(config, lr_shape=lr_shape, hr_shape=hr_shape)
        total_params = counts["total"]
        backbone_params = counts["vit"]
    effective_depth, unique_depth = _depth_summary(config)
    return {
        "path": path,
        "name": _model_name(config, path),
        "backbone": str(config.model.get("backbone", "vit")),
        "effective_depth": effective_depth,
        "unique_depth": unique_depth,
        "backbone_params": backbone_params,
        "full_params": total_params,
        "model": model,
        "counts": counts,
    }


def _print_table(rows):
    print(
        "\n"
        f"{'Model':<20} {'Backbone':<13} {'Eff':>4} {'Unique':>6} "
        f"{'Backbone params':>18} {'Full params':>14}"
    )
    print("-" * 83)
    for row in rows:
        print(
            f"{row['name']:<20} {row['backbone']:<13} "
            f"{row['effective_depth']:>4} {row['unique_depth']:>6} "
            f"{row['backbone_params']:>18,} {row['full_params']:>14,}"
        )


def _print_breakdown(row):
    model = row["model"]
    if model is None:
        counts = row["counts"]
        total = counts["total"]
        print(f"\nBreakdown: {row['name']}")
        for name in ("fck", "vit", "geo_inr", "decoder"):
            params = counts.get(name, 0)
            if params == 0 and name == "geo_inr":
                continue
            print(f"  {name:<8}: {params:>14,} ({params / total:.2%})")
        print(f"  total   : {total:>14,}")
        return

    total = count_params(model)
    print(f"\nBreakdown: {row['name']}")
    for name in ("fck", "vit", "geo_inr", "decoder"):
        module = getattr(model, name, None)
        if module is None:
            continue
        params = count_params(module)
        print(f"  {name:<8}: {params:>14,} ({params / total:.2%})")
    print(f"  total   : {total:>14,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", help="Config YAML files to compare")
    parser.add_argument("--lr-shape", type=_shape_arg, default=(32, 64), help="LR shape as H,W")
    parser.add_argument("--hr-shape", type=_shape_arg, default=(128, 256), help="HR shape as H,W")
    parser.add_argument("--all-params", action="store_true", help="Count frozen parameters too")
    parser.add_argument("--breakdown", action="store_true", help="Print component shares per model")
    args = parser.parse_args()

    trainable_only = not args.all_params
    if DownscalingModel is None:
        print("torch is not available; using analytic counts for the current repo modules.")
    rows = [
        _collect_row(path, args.lr_shape, args.hr_shape, trainable_only=trainable_only)
        for path in args.configs
    ]

    _print_table(rows)

    if len(rows) >= 2:
        baseline = rows[0]
        print(f"\nReduction vs {baseline['name']}:")
        for row in rows[1:]:
            backbone_reduction = reduction_percent(
                baseline["backbone_params"],
                row["backbone_params"],
            )
            full_reduction = reduction_percent(
                baseline["full_params"],
                row["full_params"],
            )
            print(
                f"  {row['name']}: backbone {backbone_reduction:.2f}%, "
                f"full model {full_reduction:.2f}%"
            )

    if args.breakdown:
        for row in rows:
            _print_breakdown(row)


if __name__ == "__main__":
    main()
