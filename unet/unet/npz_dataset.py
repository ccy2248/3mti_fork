"""Minimal NPZ shard dataset for SEN12MS-CR — fast init via manifest."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class NPZ_Dataset(Dataset):
    """Lazy-loading dataset over NPZ shards with zero-init-cost manifest.

    Manifest format: each line is "<shard_name> <sample_count>".
    Example:
        train_000.npz 2000
        train_001.npz 1987
    """

    def __init__(self, root, split='train', data_range=1.0):
        self.root = root
        self.split = split
        self.data_range = data_range

        # ── Parse manifest (sample count inline, no np.load needed) ──
        manifest_path = os.path.join(root, f'{split}.manifest')
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f'Manifest not found: {manifest_path}')

        self.shard_paths = []
        self.shard_offsets = []
        total = 0
        with open(manifest_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                shard_name = parts[0]
                count = int(parts[1]) if len(parts) > 1 else self._peek_size(
                    os.path.join(root, shard_name))
                path = os.path.join(root, shard_name)
                if not os.path.isfile(path):
                    raise FileNotFoundError(f'Shard not found: {path}')
                self.shard_paths.append(path)
                self.shard_offsets.append(total)
                total += count

        self.total_samples = int(total * data_range)
        self._loaded_shard_idx = -1
        self._loaded_data = None

    def _peek_size(self, path):
        """Fallback: only used if manifest has no count. Uses np.load once."""
        with np.load(path, allow_pickle=False) as d:
            return d['s2'].shape[0]

    def __len__(self):
        return self.total_samples

    def _load_shard(self, shard_idx):
        if shard_idx == self._loaded_shard_idx:
            return
        self._loaded_data = np.load(self.shard_paths[shard_idx], allow_pickle=False)
        self._loaded_shard_idx = shard_idx

    def __getitem__(self, idx):
        shard_idx = 0
        for i, offset in enumerate(self.shard_offsets):
            if offset > idx:
                break
            shard_idx = i
        local_idx = idx - self.shard_offsets[shard_idx]
        self._load_shard(shard_idx)

        cloudy = torch.from_numpy(
            self._loaded_data['s2'][local_idx].astype('float32'))
        target = torch.from_numpy(
            self._loaded_data['label'][local_idx].astype('float32'))
        sar = torch.from_numpy(
            self._loaded_data['s1'][local_idx].astype('float32'))
        return {'cloudy': cloudy, 'target': target, 'SAR': sar}
