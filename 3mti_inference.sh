#!/bin/bash
#SBATCH --gpus=1
#SBATCH -p gpu_4090
#SBATCH --job-name=3mti-inference
#SBATCH --output=3mti_inference_%j.out
#SBATCH --error=3mti_inference_%j.err

# 加载环境
module load miniforge3/26.1
eval "$(conda shell.bash hook)"
conda activate 3mti

# 进入项目目录
cd /data/home/scxk346/run/workspace/3mti

# 定义路径
MODEL_PATH="/data/home/scxk346/run/workspace/3mti/trained_model/cuhk_cr2_cloud_removal/checkpoints/model_8001.pkl"
INPUT_DIR="/data/home/scxk346/run/workspace/3mti/data/C-CUHK/CUHK-CR2/test/cloud"
REF_DIR="/data/home/scxk346/run/workspace/3mti/data/C-CUHK/nir/CUHK-CR2/test/label"
OUTPUT_DIR="/data/home/scxk346/run/workspace/3mti/trained_model/cuhk_cr2_cloud_removal/test_output"

mkdir -p "$OUTPUT_DIR"

# 运行推理
echo "=========================================="
echo "Starting inference..."
python src/inference_3MTI.py \
    --model_path "$MODEL_PATH" \
    --input_image "$INPUT_IR_DIR" \
    --ref_image "$REF_RGB_DIR" \
    --prompt "remove degradation" \
    --output_dir "$OUTPUT_DIR" \
    --mv_unet

echo "Inference completed."
echo "=========================================="