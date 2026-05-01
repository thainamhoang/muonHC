"""Geographic conditioning helpers for GeoINR."""

import os

import numpy as np
import torch
import torch.nn.functional as F


def resolution_root(var_dir):
    return os.path.dirname(var_dir.rstrip(os.sep))


def load_orography(hr_dir, hr_shape):
    constants_dir = resolution_root(hr_dir)
    constants_path = os.path.join(constants_dir, "constants.npz")
    if not os.path.exists(constants_path):
        raise FileNotFoundError(
            f"GeoINR requires {constants_path} with an orography/z/oro field."
        )

    data = np.load(constants_path)
    for key in ("orography", "z", "oro"):
        if key in data:
            oro_np = data[key].astype(np.float32)
            break
    else:
        raise KeyError(
            f"No orography key found in {constants_path}. "
            f"Available keys: {list(data.keys())}"
        )

    oro = torch.tensor(oro_np, dtype=torch.float32)
    while oro.dim() > 2:
        oro = oro.squeeze(0)
    if tuple(oro.shape) != tuple(hr_shape):
        oro = F.interpolate(
            oro[None, None],
            size=hr_shape,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
    return ((oro - oro.mean()) / (oro.std() + 1e-8)).unsqueeze(0)


def load_lat_lon(hr_dir, hr_shape):
    lat_path = os.path.join(hr_dir, "lat.npy")
    lon_path = os.path.join(hr_dir, "lon.npy")
    if os.path.exists(lat_path) and os.path.exists(lon_path):
        lat = np.load(lat_path).astype(np.float64)
        lon = np.load(lon_path).astype(np.float64)
        if lat.shape[0] == hr_shape[0] and lon.shape[0] == hr_shape[1]:
            return lat, lon
        print(
            f"Warning: lat/lon files have shapes {lat.shape}/{lon.shape}, "
            f"expected {hr_shape}; using regular global grid instead."
        )

    h, w = hr_shape
    lat = np.linspace(90.0 - 90.0 / h, -90.0 + 90.0 / h, h)
    lon = np.linspace(0.0, 360.0, w, endpoint=False)
    return lat, lon


def setup_geo_inr_grid(model, hr_dir, hr_shape, device):
    if getattr(model, "geo_inr", None) is None:
        return
    from models.geo_inr import spherical_harmonic_basis

    lat, lon = load_lat_lon(hr_dir, hr_shape)
    sh_basis = spherical_harmonic_basis(
        lat,
        lon,
        n_basis=model.geo_inr.n_basis,
    ).to(device=device, dtype=torch.float32)
    elevation = load_orography(hr_dir, hr_shape).to(device=device, dtype=torch.float32)
    model.geo_inr.set_grid(sh_basis, elevation)
    print(
        "GeoINR grid set: "
        f"sh_basis={tuple(sh_basis.shape)}, elevation={tuple(elevation.shape)}"
    )
