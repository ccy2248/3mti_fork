#!/bin/bash
#SBATCH --gpus=2
#SBATCH -p gpu_4090
#SBATCH --job-name=3mti-cloud-train
#SBATCH --output=3mti_train_%j.out
#SBATCH --error=3mti_train_%j.err

# 加载环境
module load miniforge3/26.1
eval "$(conda shell.bash hook)"
conda activate 3mti

# ====== 唯一数据目录（防冲突） ======
DATA_DIR="/tmp/sen12mscr_${SLURM_JOB_ID}"
rm -rf "$DATA_DIR" /tmp/train /tmp/val 2>/dev/null

# 检查 /tmp 是否有足够空间 (trail 数据集需要 10G 以上)
AVAIL_GB=$(df --output=avail /tmp | tail -1 | awk '{print int($1/1024/1024)}')
echo "/tmp 可用空间: ${AVAIL_GB}G"
if [ "$AVAIL_GB" -lt 10 ]; then
    echo "❌ /tmp 空间不足 (需 10G, 仅 ${AVAIL_GB}G), 任务终止"
    exit 1
fi

mkdir -p "$DATA_DIR/train" "$DATA_DIR/val"
# 创建软链接，让 json 里的 /tmp/train → $DATA_DIR/train
ln -s "$DATA_DIR/train" /tmp/train
ln -s "$DATA_DIR/val" /tmp/val

# ====== 解压数据集到节点本地 ======
echo "📦 第1步: 解压 train.tar → $DATA_DIR"

START_TIME=$(date +%s)
tar -xf /data/home/scxk346/run/workspace/3mti/data/sen12mscr/trail_vh_rf_tif_train.tar -C "$DATA_DIR/" &
TAR_PID=$!

echo "  Time  |  Files Done  |  Disk Usage"
echo "  ------+-------------+------------"
while kill -0 $TAR_PID 2>/dev/null; do
    sleep 5
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TIME))
    COUNT=$(find "$DATA_DIR/train/" -name '*.tif' 2>/dev/null | wc -l)
    SIZE=$(du -sh "$DATA_DIR/train/" 2>/dev/null | cut -f1)
    printf "  %4ds |  %8d   |  %5s\n" "$ELAPSED" "$COUNT" "$SIZE"
done
wait $TAR_PID
echo "✅ train 解压完成: $(( $(date +%s) - START_TIME ))s, $(find "$DATA_DIR/train/" -name '*.tif' | wc -l) files"

echo ""
echo "📦 第2步: 解压 val.tar → $DATA_DIR"
START_TIME=$(date +%s)
tar -xf /data/home/scxk346/run/workspace/3mti/data/sen12mscr/trail_vh_rf_tif_val.tar -C "$DATA_DIR/" &
TAR_PID=$!

echo "  Time  |  Files Done  |  Disk Usage"
echo "  ------+-------------+------------"
while kill -0 $TAR_PID 2>/dev/null; do
    sleep 3
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TIME))
    COUNT=$(find "$DATA_DIR/val/" -name '*.tif' 2>/dev/null | wc -l)
    SIZE=$(du -sh "$DATA_DIR/val/" 2>/dev/null | cut -f1)
    printf "  %4ds |  %8d   |  %5s\n" "$ELAPSED" "$COUNT" "$SIZE"
done
wait $TAR_PID
echo "✅ val 解压完成: $(( $(date +%s) - START_TIME ))s, $(find "$DATA_DIR/val/" -name '*.tif' | wc -l) files"

# 进入项目目录
cd /data/home/scxk346/run/workspace/3mti

# 设置路径
OUTPUT_DIR="/data/home/scxk346/run/workspace/3mti/trained_model/sen12mscr_tif_vh"
DATASET_PATH="/data/home/scxk346/run/workspace/3mti/dataset/your_dataset.json"

# 创建输出目录
mkdir -p "$OUTPUT_DIR/checkpoints"
mkdir -p "$OUTPUT_DIR/eval"

echo "Start Time: $(date)"
export TORCH_HOME=/data/home/scxk346/run/checkpoints
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 当前配置: 双卡4090, 每卡batch=2, 有效batch=4, 8000步
accelerate launch \
    --mixed_precision=bf16 \
    --multi_gpu \
    --num_processes=2 \
    --main_process_port=29501 \
    src/train_3MTI.py \
    --output_dir="$OUTPUT_DIR" \
    --dataset_path="$DATASET_PATH" \
    --report_to tensorboard \
    --max_train_steps=8000 \
    --resolution=512 \
    --learning_rate=2e-5 \
    --train_batch_size=1 \
    --dataloader_num_workers=4 \
    --enable_xformers_memory_efficient_attention \
    --gradient_checkpointing \
    --checkpointing_steps=1000 \
    --eval_freq=1000 \
    --num_samples_eval=10 \
    --lambda_lpips=1.0 \
    --lambda_l2=1.0 \
    --tracker_project_name="3mti_sen12mscr" \
    --tracker_run_name="cloud_removal_tif_vh" \
    --timestep=199 \
    --mv_unet \
    --set_grads_to_none

# 检查结果
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Training completed successfully!"
    echo "Checkpoints saved to: $OUTPUT_DIR/checkpoints/"
    ls -la "$OUTPUT_DIR/checkpoints/"
    echo "=========================================="
else
    echo "ERROR: Training failed!"
    exit 1
fi

echo "End Time: $(date)"

# ====== 清理 ======
rm -rf "$DATA_DIR"
echo "Cleaned up local data."