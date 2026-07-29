#!/bin/bash
#SBATCH --gpus=1
#SBATCH -p gpu_4090
#SBATCH --job-name=unet-quick-test
#SBATCH --output=unet_quick_test_%j.out
#SBATCH --error=unet_quick_test_%j.err
#SBATCH --time=04:00:00

module load miniforge3/26.1
eval "$(conda shell.bash hook)"
conda activate 3mti

TMP_DIR="/tmp/unet_real_${SLURM_JOB_ID}"
rm -rf "$TMP_DIR" ; mkdir -p "$TMP_DIR"
NPZ_SRC="/data/home/scxk346/run/workspace/3mti/data/sen12mscr/npz"

echo "Step1: copy compressed NPZ to SSD"
time cp "$NPZ_SRC/train_000.npz" "$TMP_DIR/"
time cp "$NPZ_SRC/test_000.npz" "$TMP_DIR/"

echo "Step2: decompress on SSD"
python3 -c "
import numpy as np
for f in ['train_000.npz','test_000.npz']:
    d=np.load(f'$TMP_DIR/{f}',allow_pickle=False)
    np.savez(f'$TMP_DIR/{f}',s1=d['s1'],s2=d['s2'],label=d['label'])
    print(f'{f} done')
"

echo "train_000.npz 2000" > "$TMP_DIR/train.manifest"
echo "test_000.npz 2000" > "$TMP_DIR/test.manifest"

echo "=== Training START: $(date) ==="
cd /data/home/scxk346/run/workspace/3mti/unet/unet
export TORCH_HOME=/data/home/scxk346/run/checkpoints

python train_unet.py \
    --data-root "$TMP_DIR" --gpu 0 --epochs 1 --batch-size 128 \
    --crop-size 128 --lr 1e-3 --base-ch 32 --num-workers 0 \
    --no-lpips --name real_test

echo "=== Done: $(date) ==="
