#!/bin/bash
#SBATCH --gpus=1
#SBATCH -p gpu_4090
#SBATCH --job-name=3mti-semantic
#SBATCH --output=3mti_semantic_%j.out
#SBATCH --error=3mti_semantic_%j.err

# 加载环境
module load miniforge3/26.1
conda activate 3mti


# 进入项目目录
cd /data/home/scxk346/run/workspace/3mti


# 运行语义提取
echo "Starting semantic extraction..."
python src/semantic_extract.py

echo "Semantic extraction completed."