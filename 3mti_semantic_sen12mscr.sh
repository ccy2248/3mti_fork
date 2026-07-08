#!/bin/bash
#SBATCH --gpus=1
#SBATCH -p gpu_4090
#SBATCH --job-name=3mti-semantic
#SBATCH --output=3mti_semantic_%j.out
#SBATCH --error=3mti_semantic_%j.err

# 加载环境
module load miniforge3/26.1
eval "$(conda shell.bash hook)"
conda activate 3mti
cd /data/home/scxk346/run/workspace/3mti

TAR_SRC="/data/home/scxk346/run/workspace/3mti/data/sen12mscr/sen12mscr_val.tar"
# 解压到 /tmp (本地 SSD)
DATA_DIR="/tmp/3mti_data"

# ====== 解压 (tar 流式读 JuiceFS → 写本地 SSD) ======
echo "📦 解压 train.tar 到 $DATA_DIR"
mkdir -p "$DATA_DIR"
START_TIME=$(date +%s)

tar -xf "$TAR_SRC" -C "$DATA_DIR/" &
TAR_PID=$!

echo ""
echo "  Time  |  Files Done  |  Disk Usage"
echo "  ------+-------------+------------"
while kill -0 $TAR_PID 2>/dev/null; do
    sleep 5
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TIME))
    COUNT=$(find "$DATA_DIR/" -name '*.png' 2>/dev/null | wc -l)
    SIZE=$(du -sh "$DATA_DIR/" 2>/dev/null | cut -f1)
    printf "  %4ds |  %8d   |  %5s\n" "$ELAPSED" "$COUNT" "$SIZE"
done
wait $TAR_PID
echo ""
echo "✅ 解压完成: $(( $(date +%s) - START_TIME ))s"

# ====== 语义提取 ======
echo "Starting semantic extraction..."
python src/semantic_extract.py
echo "Semantic extraction completed."

# ====== 清理 ======
rm -rf "$DATA_DIR"
echo "Cleaned up."
echo "Cleaned up."