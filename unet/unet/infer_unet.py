#!/usr/bin/env python3
"""Run U-Net inference on SEN12MS-CR test set, export NPZ + PNG.

Usage:
    python unet/infer_unet.py \
        --ckpt unet/experiments/<run>/best.pt \
        --data-root /path/to/SEN12MSCR \
        --output-dir /path/to/output \
        --gpu 0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.sen12mscr_dataset import SEN12MSCR_Dataset
from unet.unet_model import UNet

RGB_BANDS = [3, 2, 1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='Checkpoint path')
    p.add_argument('--data-root', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--base-ch', type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # ── Load model ──
    model = UNet(base_ch=args.base_ch).to(device)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'Loaded checkpoint from epoch {ckpt["epoch"]}')

    # ── Dataset ──
    dataset = SEN12MSCR_Dataset(args.data_root, split='test', data_range=1.0)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    print(f'{len(dataset)} test samples')

    # ── Output dirs ──
    npz_dir = Path(args.output_dir) / 'pred_npz'
    png_dir = Path(args.output_dir) / 'pred_rgb'
    npz_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    # ── Inference ──
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc='Inference')):
            cloudy = batch['cloudy'].to(device)
            result = model(cloudy)
            pred = result[0] if isinstance(result, tuple) else result
            pred_np = pred.cpu().float().numpy()

            for i in range(pred_np.shape[0]):
                global_i = idx * args.batch_size + i
                sample = pred_np[i]

                # NPZ
                np.savez_compressed(npz_dir / f'{global_i:05d}.npz',
                                    pred=sample.astype(np.float32))

                # PNG (RGB ×3.0 brightness)
                rgb = sample[RGB_BANDS].transpose(1, 2, 0)
                rgb = np.clip(rgb * 3.0, 0, 1)
                rgb = (rgb * 255).round().astype(np.uint8)
                Image.fromarray(rgb).save(png_dir / f'{global_i:05d}.png')

    print(f'Done → {args.output_dir}')


if __name__ == '__main__':
    main()
