import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


def _preload_shards(shards, variable_name, partition, label):
    """Preload shards with one final array allocation instead of list+concat."""
    first = _load_npz_array(shards[0], variable_name)
    total_timesteps = first.shape[0] * len(shards)
    data = np.empty(
        (total_timesteps, *first.shape[1:]),
        dtype=first.dtype,
    )
    cursor = 0
    for shard_idx, shard_path in enumerate(shards):
        shard = first if shard_idx == 0 else _load_npz_array(shard_path, variable_name)
        next_cursor = cursor + shard.shape[0]
        data[cursor:next_cursor] = shard
        cursor = next_cursor
        if (shard_idx + 1) % 50 == 0 or shard_idx + 1 == len(shards):
            print(
                f"[{partition}] {label} preload progress: "
                f"{shard_idx + 1}/{len(shards)} shards",
                flush=True,
            )
    return data


def _load_npz_array(path, variable_name):
    archive = np.load(path)
    if variable_name in archive:
        return archive[variable_name]
    keys = list(archive.keys())
    if len(keys) == 1:
        return archive[keys[0]]
    raise KeyError(
        f"{variable_name!r} is not a file in {path}. "
        f"Available keys: {keys}"
    )


class DownscalingDataset(Dataset):
    """
    Paired (LR, HR) ERA5 downscaling dataset.

    Loading modes (controlled per resolution):
        lr_preload=True  — load all LR shards into RAM at init (recommended,
                           LR is only ~5GB total across all splits)
        hr_preload=True  — load all HR shards into RAM at init (only safe for
                           val/test splits; HR train is ~38GB)
        hr_preload=False — lazy per-worker shard cache for HR (safe for train)

    Recommended usage:
        train : lr_preload=True,  hr_preload=False  (~5GB RAM, HR read lazily)
        val   : lr_preload=True,  hr_preload=True   (~5GB LR + ~5GB HR val)
        test  : lr_preload=True,  hr_preload=True   (~5GB LR + ~5GB HR test)

    Args:
        lr_dir      : path to LR resolution directory (normalize_*.npz + partition/)
        hr_dir      : path to HR resolution directory
        partition   : "train" | "val" | "test"
        stride      : sample every Nth timestep (1=all, 6=6-hourly, 24=daily)
        lr_preload  : preload LR into RAM
        hr_preload  : preload HR into RAM
    """

    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    EPS = 1e-8

    def __init__(
        self,
        lr_dir: str,
        hr_dir: str,
        partition: str,
        stride: int = 6,
        temporal: bool = False,
        lr_preload: bool = False,
        hr_preload: bool = False,
        variable_name: str = "2m_temperature",
    ):
        self.stride = stride
        self.partition = partition
        self.temporal = temporal
        self.variable_name = variable_name

        # Resolve per-resolution preload flags
        # Explicit lr_preload / hr_preload take priority over legacy preload
        self.lr_preload = lr_preload if lr_preload is not None else False
        self.hr_preload = hr_preload if hr_preload is not None else False

        # ── Normalization stats ───────────────────────────────────────────
        self.lr_mean = torch.tensor(
            _load_npz_array(os.path.join(lr_dir, "normalize_mean.npz"), variable_name),
            dtype=torch.float32,
        ).view(1, 1, 1)
        self.lr_std = torch.tensor(
            _load_npz_array(os.path.join(lr_dir, "normalize_std.npz"), variable_name),
            dtype=torch.float32,
        ).view(1, 1, 1)
        self.hr_mean = torch.tensor(
            _load_npz_array(os.path.join(hr_dir, "normalize_mean.npz"), variable_name),
            dtype=torch.float32,
        ).view(1, 1, 1)
        self.hr_std = torch.tensor(
            _load_npz_array(os.path.join(hr_dir, "normalize_std.npz"), variable_name),
            dtype=torch.float32,
        ).view(1, 1, 1)
        self.lr_inv_std = 1.0 / (self.lr_std + self.EPS)
        self.hr_inv_std = 1.0 / (self.hr_std + self.EPS)

        # ── Shard paths ───────────────────────────────────────────────────
        lr_shards = sorted(glob.glob(os.path.join(lr_dir, partition, "*.npz")))
        hr_shards = sorted(glob.glob(os.path.join(hr_dir, partition, "*.npz")))
        lr_shards = [f for f in lr_shards if "climatology" not in f]
        hr_shards = [f for f in hr_shards if "climatology" not in f]

        assert len(lr_shards) == len(hr_shards), (
            f"Shard mismatch: LR={len(lr_shards)}, HR={len(hr_shards)}"
        )
        assert len(lr_shards) > 0, f"No shards in {lr_dir}/{partition}/"

        self.lr_shards = lr_shards
        self.hr_shards = hr_shards

        # Peek at first shard for shape info
        first_lr = _load_npz_array(lr_shards[0], variable_name)
        first_hr = _load_npz_array(hr_shards[0], variable_name)
        self.lr_shape = first_lr.shape[2:]  # [T, 1, H_lr, W_lr] → (H_lr, W_lr)
        self.hr_shape = first_hr.shape[2:]
        self.T_per_shard = first_lr.shape[0]
        total = self.T_per_shard * len(lr_shards)
        self.indices = list(range(0, total, stride))

        print(f"[{partition}] LR shape: {self.lr_shape}, HR shape: {self.hr_shape}", flush=True)
        print(
            f"[{partition}] {len(lr_shards)} shards × "
            f"{self.T_per_shard} timesteps = {total} total",
            flush=True,
        )
        print(
            f"[{partition}] {len(self.indices)} samples "
            f"(stride={stride}, lr_preload={self.lr_preload}, "
            f"hr_preload={self.hr_preload})",
            flush=True,
        )

        # ── LR preload ────────────────────────────────────────────────────
        if self.lr_preload:
            print(f"[{partition}] Preloading LR shards into RAM...", flush=True)
            self._lr_data = _preload_shards(
                lr_shards,
                variable_name,
                partition,
                "LR",
            )
            print(f"[{partition}] LR preloaded: {self._lr_data.nbytes / 1e9:.2f} GB", flush=True)
        else:
            self._lr_data = None
            self._cache_lr = None

        # ── HR preload ────────────────────────────────────────────────────
        if self.hr_preload:
            print(f"[{partition}] Preloading HR shards into RAM...", flush=True)
            self._hr_data = _preload_shards(
                hr_shards,
                variable_name,
                partition,
                "HR",
            )
            print(f"[{partition}] HR preloaded: {self._hr_data.nbytes / 1e9:.2f} GB", flush=True)
        else:
            self._hr_data = None
            self._cache_hr = None
            self._cache_shard_idx = -1

    # ── Length ────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.indices)

    # ── Raw array fetch ───────────────────────────────────────────────────

    def _get_raw(self, idx: int):
        """
        Return (lr_np, hr_np) numpy arrays for dataset index idx.

        Handles four combinations of lr_preload × hr_preload:
            (T, T) — both in RAM, direct index
            (T, F) — LR from RAM, HR from shard cache
            (F, T) — LR from shard cache, HR from RAM
            (F, F) — both from shard cache (original lazy behaviour)
        """
        real_idx = self.indices[idx]
        shard_idx = real_idx // self.T_per_shard
        t_idx = real_idx % self.T_per_shard

        # ── LR ────────────────────────────────────────────────────────────
        if self.lr_preload:
            lr = self._lr_data[real_idx]
        else:
            # Load shard if not cached — shard_idx tracked separately for LR
            if not hasattr(self, "_cache_lr_shard_idx"):
                self._cache_lr_shard_idx = -1
            if shard_idx != self._cache_lr_shard_idx:
                self._cache_lr = _load_npz_array(
                    self.lr_shards[shard_idx],
                    self.variable_name,
                )
                self._cache_lr_shard_idx = shard_idx
            lr = self._cache_lr[t_idx]

        # ── HR ────────────────────────────────────────────────────────────
        if self.hr_preload:
            hr = self._hr_data[real_idx]
        else:
            if shard_idx != self._cache_shard_idx:
                self._cache_hr = _load_npz_array(
                    self.hr_shards[shard_idx],
                    self.variable_name,
                )
                self._cache_shard_idx = shard_idx
            hr = self._cache_hr[t_idx]

        return lr, hr

    def _get_lr_raw_by_real_idx(self, real_idx: int):
        shard_idx = real_idx // self.T_per_shard
        t_idx = real_idx % self.T_per_shard
        if self.lr_preload:
            return self._lr_data[real_idx]
        if not hasattr(self, "_cache_lr_shard_idx"):
            self._cache_lr_shard_idx = -1
        if shard_idx != self._cache_lr_shard_idx:
            self._cache_lr = _load_npz_array(
                self.lr_shards[shard_idx],
                self.variable_name,
            )
            self._cache_lr_shard_idx = shard_idx
        return self._cache_lr[t_idx]

    # ── Dataset item ──────────────────────────────────────────────────────

    def __getitem__(self, idx: int):
        lr_np, hr_np = self._get_raw(idx)

        lr_raw = torch.from_numpy(lr_np).to(dtype=torch.float32)  # [1, H_lr, W_lr]
        hr_raw = torch.from_numpy(hr_np).to(dtype=torch.float32)  # [1, H_hr, W_hr]

        # ── Z-score normalize ─────────────────────────────────────────────
        if self.temporal:
            real_idx = self.indices[idx]
            total = self.T_per_shard * len(self.lr_shards)
            prev_idx = max(0, real_idx - 1)
            next_idx = min(total - 1, real_idx + 1)
            lr_prev = torch.from_numpy(self._get_lr_raw_by_real_idx(prev_idx)).to(dtype=torch.float32)
            lr_next = torch.from_numpy(self._get_lr_raw_by_real_idx(next_idx)).to(dtype=torch.float32)
            lr_norm = torch.cat(
                [
                    (lr_prev - self.lr_mean) * self.lr_inv_std,
                    (lr_raw - self.lr_mean) * self.lr_inv_std,
                    (lr_next - self.lr_mean) * self.lr_inv_std,
                ],
                dim=0,
            )
        else:
            lr_norm = (lr_raw - self.lr_mean) * self.lr_inv_std
        hr_norm = (hr_raw - self.hr_mean) * self.hr_inv_std

        return lr_norm, hr_norm

    # ── Worker init ───────────────────────────────────────────────────────

    def worker_init_fn(self, worker_id: int):
        """
        Reset per-worker shard caches.
        Call as worker_init_fn in DataLoader when num_workers > 0.
        Only has effect for lazy (non-preloaded) resolutions.
        """
        if not self.lr_preload:
            self._cache_lr = None
            self._cache_lr_shard_idx = -1
        if not self.hr_preload:
            self._cache_hr = None
            self._cache_shard_idx = -1
