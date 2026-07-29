"""Minimal train/valid dataset builder for U-Net training."""

import torch
from torch.utils.data import Subset


def build_train_valid_datasets(cfg, DatasetClass):
    """Build train and validation datasets.

    For NPZ data: train split → train, test split → validation.
    Returns (train_dataset, valid_dataset, split_info_dict).
    """
    name = cfg.dataset.name
    root = cfg.dataset.root
    seed = cfg.seed
    train_ratio = cfg.dataset.train_ratio

    # ── Load full train set ──
    full_train = DatasetClass(root, split='train', data_range=cfg.dataset.data_range)

    # ── Split train into train/val ──
    n_total = len(full_train)
    n_train = int(n_total * train_ratio)
    n_val = n_total - n_train

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=generator).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = Subset(full_train, train_indices)
    valid_dataset = Subset(full_train, val_indices)

    split_info = {
        'train': n_train,
        'val': n_val,
        'total': n_total,
    }

    return train_dataset, valid_dataset, split_info
