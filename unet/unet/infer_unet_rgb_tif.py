#!/usr/bin/env python3
"""Run U-Net inference on SEN12MS-CR test set → RGB float32 TIFF.

Outputs only RGB bands (B4/B3/B2) as float32 GeoTIFF, one file per sample.
Naming convention: {season}_{roi_id}_{patch_id}_cloudy.tif

Usage:
    python unet/infer_unet_rgb_tif.py \
        --ckpt unet/experiments/20260728_165754/best.pt \
        --data-root /media/user/newMemery/worksapce/crsd/data/SEN12MSCR \
        --output-dir /media/user/newMemery/worksapce/crsd/data/SEN12MSCR_RGB_VV_VH/sen12mscr/sen12mscr_unetpre_cloudy_tif \
        --gpu 0
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.sen12mscr_dataset import SEN12MSCR_Dataset
from unet.unet_model import UNet

RGB_BANDS = [3, 2, 1]   # B4(Red), B3(Green), B2(Blue)


def parse_args():
    p = argparse.ArgumentParser(
        description='U-Net inference → RGB float32 TIFF (3-band)')
    p.add_argument('--ckpt', required=True, help='Checkpoint path (best.pt)')
    p.add_argument('--data-root', required=True,
                   help='SEN12MSCR dataset root')
    p.add_argument('--output-dir', required=True,
                   help='Directory for output RGB TIFF files')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--base-ch', type=int, default=64)
    return p.parse_args()


def parse_sample_name(s2_path: str) -> str:
    """Extract season, roi_id, patch_id from the S2 source path.

    Example:
        /path/ROIs1158_spring_s2/s2_106/ROIs1158_spring_s2_106_p30.tif
        → 'spring_106_30_cloudy'

    Args:
        s2_path: Cloud-free S2 .tif path from dataset.paths[idx]['S2'].

    Returns:
        Stem formatted as '{season}_{roi_id}_{patch_id}_cloudy'.
    """
    # Extract season: 'spring', 'summer', 'fall', 'winter'
    season_match = re.search(r'ROIs\d+_(spring|summer|fall|winter)_s2', s2_path)
    season = season_match.group(1) if season_match else 'unknown'

    # Extract ROI id: e.g. '106' from '/s2_106/'
    roi_match = re.search(r'/s2_(\d+)/', s2_path)
    roi_id = roi_match.group(1) if roi_match else '0'

    # Extract patch id: e.g. '30' from '_p30.tif'
    patch_match = re.search(r'_p(\d+)\.tif', s2_path)
    patch_id = patch_match.group(1) if patch_match else '0'

    return f'{season}_{roi_id}_{patch_id}_cloudy'


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # ── Load model ──
    model = UNet(base_ch=args.base_ch).to(device)
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'Loaded checkpoint (epoch {ckpt["epoch"]}, best PSNR {ckpt.get("best_psnr","N/A")})')

    # ── Dataset (test split, no shuffle) ──
    dataset = SEN12MSCR_Dataset(args.data_root, split='test', data_range=1.0)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    print(f'{len(dataset)} test samples')

    # ── Output directory ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect source paths for naming ──
    # SEN12MSCR_Dataset.dataset is the inner SEN12MSCR instance
    source_paths = dataset.dataset.paths   # list[dict] with S1/S2/S2_cloudy keys

    # ── Inference ──
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc='Inference')):
            cloudy = batch['cloudy'].to(device)
            result = model(cloudy)
            pred = result[0] if isinstance(result, tuple) else result
            pred_np = pred.cpu().float().numpy()   # (B, 13, H, W)

            for i in range(pred_np.shape[0]):
                global_i = batch_idx * args.batch_size + i
                sample = pred_np[i]                 # (13, H, W) float32

                # Extract RGB bands → (3, H, W)
                rgb = sample[RGB_BANDS].copy()      # float32, [0, 1]

                # Parse filename from source path
                s2_path = source_paths[global_i]['S2']
                stem = parse_sample_name(s2_path)
                tif_path = output_dir / f'{stem}.tif'

                if HAS_RASTERIO:
                    # Write as GeoTIFF (3 bands, float32)
                    with rasterio.open(
                        str(tif_path), 'w',
                        driver='GTiff',
                        height=rgb.shape[1],
                        width=rgb.shape[2],
                        count=3,
                        dtype='float32',
                    ) as dst:
                        dst.write(rgb)
                else:
                    # Fallback: memmap-based TIFF via tifffile or raw numpy
                    # Use numpy's tofile as last resort, or raise error
                    raise ImportError(
                        'rasterio is required to write TIFF files. '
                        'Install it with: pip install rasterio'
                    )

    print(f'Done! {len(dataset)} TIFF files → {output_dir}')


if __name__ == '__main__':
    main()
