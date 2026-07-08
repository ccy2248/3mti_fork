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

# 进入项目目录
cd /data/home/scxk346/run/workspace/3mti

# 设置路径
OUTPUT_DIR="/data/home/scxk346/run/workspace/3mti/trained_model/cuhk_cr2_cloud_removal_cc"
DATASET_PATH="/data/home/scxk346/run/workspace/3mti/dataset/your_dataset.json"

# 创建输出目录
mkdir -p "$OUTPUT_DIR/checkpoints"
mkdir -p "$OUTPUT_DIR/eval"

echo "Start Time: $(date)"
export TORCH_HOME=/data/home/scxk346/run/checkpoints
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
    --tracker_project_name="3mti_cuhk_cr2" \
    --tracker_run_name="cloud_removal_2gpu" \
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