#!/usr/bin/env python3
"""Train U-Net for cloud removal on SEN12MS-CR.

Single-stage training with:
    - L1 + SSIM on all 13 channels
    - LPIPS on RGB subset for perceptual quality
    - Multi-scale RGB supervision via decoder heads
    - EMA, gradient accumulation, random flip/rot augmentations

Usage:
    python unet/train_unet.py --data-root /path/to/SEN12MSCR --gpu 0
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ── Speed ──
torch.backends.cudnn.benchmark = True

# ── Project path ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import npz_dataset
from data_split import build_train_valid_datasets
from augment import TrainAugment
from pytorch_ssim import ssim as calc_ssim
from unet_model import UNet
from unet_config import UNetConfig

# ── Constants ──
RGB_BANDS = [3, 2, 1]
SAVE_DIR = Path(__file__).resolve().parent / 'experiments'


# ═══════════════════════════════════════════════════════════════════════════
# Parse args
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='Train U-Net for cloud removal')
    p.add_argument('--data-root', required=True, help='SEN12MSCR root (raw or NPZ)')
    p.add_argument('--dataset', choices=['sen12mscr', 'npz'], default='npz',
                    help='Dataset format: raw TIF or pre-converted NPZ shards')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--base-ch', type=int, default=64, help='Base channels')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--accumulate', type=int, default=1, help='Gradient accum steps')
    p.add_argument('--crop-size', type=int, default=128, help='Random crop size')
    p.add_argument('--no-rgb-heads', action='store_true', help='Disable RGB heads')
    p.add_argument('--no-lpips', action='store_true', help='Disable LPIPS loss entirely')
    p.add_argument('--lpips-start-epoch', type=int, default=50,
                    help='Start LPIPS loss from this epoch (0=from start)')
    p.add_argument('--amp', action='store_true', help='Enable AMP mixed precision (experimental)')
    p.add_argument('--resume', type=str, default=None, help='Resume checkpoint')
    p.add_argument('--name', type=str, default=None, help='Experiment name')
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Losses
# ═══════════════════════════════════════════════════════════════════════════

_LPIPS_MODEL = None

def _get_lpips_model(device):
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        import lpips
        _LPIPS_MODEL = lpips.LPIPS(net='vgg').eval().to(device)
        for p in _LPIPS_MODEL.parameters():
            p.requires_grad = False
    return _LPIPS_MODEL


def compute_loss(pred, target, rgb_ms=None, use_lpips=True, device='cpu'):
    """Combined loss: L1 + SSIM (13ch) + LPIPS (RGB) + multi-scale RGB."""

    # ── Main: L1 + SSIM on full 13-channel ──
    loss_l1 = F.l1_loss(pred, target)
    loss_ssim = 1.0 - calc_ssim(pred, target, size_average=True)

    loss = 0.85 * loss_l1 + 0.15 * loss_ssim

    # ── LPIPS on RGB subset ──
    if use_lpips:
        pred_rgb = pred[:, RGB_BANDS, :, :] * 2.0 - 1.0   # [0,1]→[-1,1]
        target_rgb = target[:, RGB_BANDS, :, :] * 2.0 - 1.0
        loss_lpips = _get_lpips_model(device)(pred_rgb, target_rgb).mean()
        loss = loss + 0.1 * loss_lpips

    # ── Multi-scale RGB supervision ──
    if rgb_ms is not None:
        target_rgb_full = target[:, RGB_BANDS, :, :]   # (B, 3, 256, 256)
        loss_ms = 0.0
        for i, rgb_pred in enumerate(rgb_ms):
            scale = rgb_pred.shape[-1] / target_rgb_full.shape[-1]
            gt_scaled = F.interpolate(target_rgb_full, scale_factor=scale,
                                       mode='bilinear', align_corners=False)
            loss_ms += F.l1_loss(rgb_pred, gt_scaled) + \
                       0.1 * (1.0 - calc_ssim(rgb_pred, gt_scaled, size_average=True))
        loss = loss + 0.05 * loss_ms

    return loss, {
        'L1': loss_l1.item(),
        'SSIM': 1.0 - loss_ssim.item(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# EMA
# ═══════════════════════════════════════════════════════════════════════════

class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: p.data.clone() for name, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[name])

    def restore(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[name].copy_(p.data)


# ═══════════════════════════════════════════════════════════════════════════
# Training utilities
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, loader, device, use_lpips=True):
    """Compute validation metrics."""
    model.eval()
    total_l1, total_psnr, count = 0.0, 0.0, 0
    for batch in loader:
        cloudy = batch['cloudy'].to(device)
        target = batch['target'].to(device)
        result = model(cloudy)
        pred = result[0] if isinstance(result, tuple) else result

        total_l1 += F.l1_loss(pred, target).item() * cloudy.size(0)
        mse = F.mse_loss(pred, target).item()
        total_psnr += (10 * np.log10(1.0 / mse) if mse > 0 else 100) * cloudy.size(0)
        count += cloudy.size(0)

    model.train()
    return total_l1 / count, total_psnr / count


def save_checkpoint(model, optimizer, ema, epoch, best_psnr, path):
    ema.apply()
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema_shadow': ema.shadow,
        'best_psnr': best_psnr,
    }, path)
    ema.restore()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Datasets ──
    DatasetClass = npz_dataset.NPZ_Dataset
    cfg = UNetConfig(args.data_root, crop_size=args.crop_size)
    cfg.dataset.name = args.dataset
    train_dataset, valid_dataset, split_info = build_train_valid_datasets(cfg, DatasetClass)
    print(f'Train: {len(train_dataset)}, Valid: {len(valid_dataset)}')

    dl_kwargs = dict(batch_size=args.batch_size, pin_memory=True, drop_last=True)
    if args.num_workers > 0:
        dl_kwargs.update(num_workers=args.num_workers,
                         persistent_workers=True, prefetch_factor=2)
    else:
        dl_kwargs['num_workers'] = 0
    train_loader = DataLoader(train_dataset, shuffle=True, **dl_kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=min(args.batch_size, 16),
                               shuffle=False, num_workers=2, pin_memory=True,
                               persistent_workers=True)

    # ── Model ──
    model = UNet(in_chans=13, out_chans=13, base_ch=args.base_ch,
                 rgb_heads=not args.no_rgb_heads).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {n_params / 1e6:.2f}M')

    # ── Optimizer & Scheduler ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ema = EMA(model, decay=0.999)
    train_aug = TrainAugment(crop_size=args.crop_size)
    scaler = torch.amp.GradScaler('cuda') if args.amp else None
    amp_on = scaler is not None
    print('AMP:', 'ON' if amp_on else 'OFF')

    # ── Checkpoint ──
    run_name = args.name or datetime.now().strftime('%Y%m%d_%H%M%S')
    ckpt_dir = SAVE_DIR / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(ckpt_dir / 'logs')
    print(f'Saving to: {ckpt_dir}')

    # ── Resume ──
    start_epoch, best_psnr = 0, 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        ema.shadow = ckpt['ema_shadow']
        start_epoch = ckpt['epoch'] + 1
        best_psnr = ckpt['best_psnr']
        print(f'Resumed from epoch {start_epoch}')

    # ── Training loop ──
    optimizer.zero_grad()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        total_loss = 0.0

        for step, batch in enumerate(pbar):
            # Augment
            batch = train_aug.augment(batch)
            cloudy = batch['cloudy'].to(device)
            target = batch['target'].to(device)

            # LPIPS schedule
            use_lpips = (not args.no_lpips) and (epoch >= args.lpips_start_epoch)

            # Forward
            result = model(cloudy)
            pred, rgb_ms = result if isinstance(result, tuple) else (result, None)

            # Loss
            loss, loss_dict = compute_loss(pred, target, rgb_ms,
                                           use_lpips=use_lpips, device=device)
            loss = loss / args.accumulate
            loss.backward()

            if (step + 1) % args.accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                ema.update()

            total_loss += loss.item() * args.accumulate
            pbar.set_postfix(L1=f'{loss_dict["L1"]:.4f}',
                             SSIM=f'{loss_dict["SSIM"]:.4f}',
                             lr=f'{optimizer.param_groups[0]["lr"]:.2e}')

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # ── Validation ──
        ema.apply()
        val_l1, val_psnr = validate(model, valid_loader, device,
                                    use_lpips=not args.no_lpips)
        ema.restore()

        # ── Log ──
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Loss/val_L1', val_l1, epoch)
        writer.add_scalar('Metrics/val_PSNR', val_psnr, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        print(f'  Val L1={val_l1:.4f}  PSNR={val_psnr:.2f} dB  '
              f'Best={best_psnr:.2f}')

        # ── Save ──
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(model, optimizer, ema, epoch, best_psnr,
                            ckpt_dir / 'best.pt')

        if (epoch + 1) % 50 == 0:
            save_checkpoint(model, optimizer, ema, epoch, best_psnr,
                            ckpt_dir / f'epoch_{epoch+1}.pt')

    writer.close()
    print(f'\nDone! Best PSNR: {best_psnr:.2f} dB  →  {ckpt_dir / "best.pt"}')


if __name__ == '__main__':
    main()
