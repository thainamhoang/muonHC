"""Dataset that loads ERA5 T2m with optional temporal context (±6h).
Reuses shard-based loading logic from the SSL pipeline.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

class TemporalDownscalingDataset(Dataset):
    """ERA5 T2m dataset with temporal stacking.
    Args:
        root_dir: path to shards directory (with train/val/test subdirs)
        split: 'train', 'val', or 'test'
        temporal: if True, input is stack of (t-6h, t, t+6h); else only t
        stride: interval between samples in hours (6 for 6-hourly)
        normalize: z-score using provided stats
        mean/std: normalization parameters (z-score)
    """
    def __init__(self, root_dir, split='train', temporal=False, stride=6,
                 normalize=True, mean=278.45, std=21.25):
        self.split_dir = os.path.join(root_dir, split)
        self.temporal = temporal
        self.stride = stride
        self.normalize = normalize
        self.mean = mean
        self.std = std
        # List all .npy shard files (assumed to contain 'lr' and 'hr' arrays)
        self.files = sorted([f for f in os.listdir(self.split_dir) if f.endswith('.npy')])
        # Build index: map global sample index to (file_idx, local_idx)
        self.file_lengths = []
        self.all_index_map = []  # (file_idx, local_idx) for every raw timestep
        for fidx, fname in enumerate(self.files):
            data = np.load(os.path.join(self.split_dir, fname), allow_pickle=True).item()
            n = self._num_samples(data['lr'])
            self.file_lengths.append(n)
            for i in range(n):
                self.all_index_map.append((fidx, i))
        sample_stride = max(1, int(stride))
        self.sample_indices = list(range(0, len(self.all_index_map), sample_stride))
        self.total_samples = len(self.sample_indices)

    def __len__(self):
        return self.total_samples

    @staticmethod
    def _num_samples(array):
        if array.ndim == 4:
            return array.shape[0]  # (T, C, H, W)
        if array.ndim == 3 and array.shape[0] != 1:
            return array.shape[0]  # (T, H, W)
        return 1

    @staticmethod
    def _select_sample(array, local_idx):
        if array.ndim == 4:
            return array[local_idx]
        if array.ndim == 3 and array.shape[0] != 1:
            return array[local_idx]
        return array

    def _load_sample(self, file_idx, local_idx):
        fpath = os.path.join(self.split_dir, self.files[file_idx])
        data = np.load(fpath, allow_pickle=True).item()
        lr = self._select_sample(data['lr'], local_idx)
        hr = self._select_sample(data['hr'], local_idx)
        # Ensure channel dim
        if lr.ndim == 2:
            lr = lr[np.newaxis, ...]
        if hr.ndim == 2:
            hr = hr[np.newaxis, ...]
        # Normalize to z-score
        if self.normalize:
            lr = (lr - self.mean) / self.std
            hr = (hr - self.mean) / self.std
        return lr, hr

    def __getitem__(self, idx):
        raw_idx = self.sample_indices[idx]
        if not self.temporal:
            file_idx, local_idx = self.all_index_map[raw_idx]
            lr, hr = self._load_sample(file_idx, local_idx)
            return torch.from_numpy(lr.copy()).float(), torch.from_numpy(hr.copy()).float()
        else:
            # Temporal context uses adjacent raw timesteps even when training samples are strided.
            idx_prev = max(0, raw_idx - 1)
            idx_next = min(len(self.all_index_map) - 1, raw_idx + 1)
            file_idx_prev, local_prev = self.all_index_map[idx_prev]
            file_idx_curr, local_curr = self.all_index_map[raw_idx]
            file_idx_next, local_next = self.all_index_map[idx_next]
            lr_prev, _ = self._load_sample(file_idx_prev, local_prev)
            lr_curr, hr_curr = self._load_sample(file_idx_curr, local_curr)
            lr_next, _ = self._load_sample(file_idx_next, local_next)
            lr_stack = np.concatenate([lr_prev, lr_curr, lr_next], axis=0)  # (3, H, W)
            return torch.from_numpy(lr_stack.copy()).float(), torch.from_numpy(hr_curr.copy()).float()
