#!/bin/bash
#SBATCH --gpus=1
#SBATCH -p gpu_4090
#SBATCH --job-name=3mti-quick-test
#SBATCH --output=3mti_quick_test_%j.out
#SBATCH --error=3mti_quick_test_%j.err
#SBATCH --time=00:20:00

# 加载环境
module load miniforge3/26.1
eval "$(conda shell.bash hook)"
conda activate 3mti

echo "=========================================="
echo "Quick validation test for code correctness"
echo "Job ${SLURM_JOB_ID} started at: $(date)"
echo "=========================================="

JOB_START_EPOCH=$(date +%s)

# ====== 唯一数据目录（使用 trail 小数据集） ======
TRAIN_DIR="/tmp/quick_train_${SLURM_JOB_ID}"
VAL_DIR="/tmp/quick_val_${SLURM_JOB_ID}"
rm -rf "$TRAIN_DIR" "$VAL_DIR" /tmp/train /tmp/val 2>/dev/null

mkdir -p "$TRAIN_DIR" "$VAL_DIR"
ln -s "$TRAIN_DIR" /tmp/train
ln -s "$VAL_DIR" /tmp/val

# ====== 解压 trail 小数据集 ======
echo "📦 解压 trail train.tar → $TRAIN_DIR"
tar -xf /data/home/scxk346/run/workspace/3mti/data/sen12mscr/trail_vh_rf_tif_train.tar -C "$TRAIN_DIR/" &
wait $!
echo "  train files: $(find "$TRAIN_DIR/" -name '*.tif' | wc -l)"

echo "📦 解压 trail val.tar → $VAL_DIR"
tar -xf /data/home/scxk346/run/workspace/3mti/data/sen12mscr/trail_vh_rf_tif_val.tar -C "$VAL_DIR/" &
wait $!
echo "  val files: $(find "$VAL_DIR/" -name '*.tif' | wc -l)"

# 进入项目目录
cd /data/home/scxk346/run/workspace/3mti

OUTPUT_DIR="/tmp/quick_test_output_${SLURM_JOB_ID}"
DATASET_PATH="/data/home/scxk346/run/workspace/3mti/dataset/your_dataset.json"
mkdir -p "$OUTPUT_DIR"

echo "Start Time: $(date)"
export TORCH_HOME=/data/home/scxk346/run/checkpoints
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 单卡快速验证（20步，不记录tensorboard，不eval，不checkpoint）
accelerate launch \
    --mixed_precision=bf16 \
    --num_processes=1 \
    --main_process_port=29501 \
    src/train_3MTI.py \
    --output_dir="$OUTPUT_DIR" \
    --dataset_path="$DATASET_PATH" \
    --report_to tensorboard \
    --max_train_steps=20 \
    --resolution=512 \
    --learning_rate=1e-5 \
    --train_batch_size=1 \
    --dataloader_num_workers=0 \
    --enable_xformers_memory_efficient_attention \
    --gradient_checkpointing \
    --checkpointing_steps=10000 \
    --eval_freq=10000 \
    --lambda_lpips=1.0 \
    --lambda_l2=1.0 \
    --tracker_project_name="test" \
    --tracker_run_name="quick_test" \
    --timestep=199 \
    --mv_unet \
    --no_skip \
    --set_grads_to_none

EXIT_CODE=$?
ELAPSED=$(( $(date +%s) - JOB_START_EPOCH ))

if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "✅ Quick test PASSED! (${ELAPSED}s)"
    echo "=========================================="
else
    echo "=========================================="
    echo "❌ Quick test FAILED (exit code: $EXIT_CODE)"
    echo "=========================================="
fi

# 清理
rm -rf "$TRAIN_DIR" "$VAL_DIR" /tmp/train /tmp/val "$OUTPUT_DIR"
echo "Cleaned up."
exit $EXIT_CODE
