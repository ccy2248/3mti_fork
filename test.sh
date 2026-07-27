#!/bin/bash
#SBATCH --gpus=2
#SBATCH -p gpu_4090
#SBATCH --job-name=storage-probe
#SBATCH --output=storage_probe_%j.out
#SBATCH --time=00:05:00

echo "=========================================="
echo "🔍 计算节点本地可写存储探测"
echo "=========================================="
echo ""

# 候选路径: 常见临时目录 + 独立挂载点
CANDIDATES="/tmp /dev/shm /scratch /local /localscratch /cache"

# 也加入所有独立 ext4/xfs 挂载点
while read -r fs type size used avail pct mp; do
    [ "$fs" = "Filesystem" ] && continue
    if echo "$type" | grep -qE "ext4|xfs|btrfs"; then
        CANDIDATES="$CANDIDATES $mp"
    fi
done < <(df -hT 2>/dev/null)

echo "  路径                   类型     可用      可写?"
echo "  ---------------------- -------- --------- -----"
BEST_PATH=""
BEST_KB=0
for path in $(echo "$CANDIDATES" | tr ' ' '\n' | sort -u); do
    [ -d "$path" ] || continue
    FSTYPE=$(df -hT "$path" 2>/dev/null | tail -1 | awk '{print $2}')
    SPACE=$(df -h "$path" --output=avail 2>/dev/null | tail -1 | tr -d ' ')
    if touch "$path/test_rw" 2>/dev/null; then
        rm -f "$path/test_rw"
        printf "  %-22s %-8s %-9s ✅\n" "$path" "$FSTYPE" "$SPACE"
        KB=$(df --output=avail "$path" 2>/dev/null | tail -1 | tr -d ' ')
        if [ "$KB" -gt "$BEST_KB" ] 2>/dev/null; then
            BEST_KB=$KB
            BEST_PATH=$path
        fi
    else
        printf "  %-22s %-8s %-9s ❌\n" "$path" "$FSTYPE" "$SPACE"
    fi
done

echo ""
echo "=========================================="
echo "📊 结论"
echo "=========================================="
if [ -n "$BEST_PATH" ]; then
    BEST_GB=$((BEST_KB / 1024 / 1024))
    FSTYPE=$(df -hT "$BEST_PATH" 2>/dev/null | tail -1 | awk '{print $2}')
    case "$FSTYPE" in
        tmpfs)  SPEED="⚡⚡⚡ RAM 速度" ;;
        ext4|xfs|btrfs) SPEED="⚡ SSD 速度" ;;
        *)      SPEED="普通" ;;
    esac
    echo "  🏆 最佳: $BEST_PATH  (可用 ${BEST_GB}G, $SPEED)"
    echo ""
    echo "=========================================="
    echo "📥 读写速度测试 (1GB → $BEST_PATH)"
    echo "=========================================="
    echo -n "  写入: "
    dd if=/dev/zero of="$BEST_PATH/perftest" bs=1M count=1024 conv=fdatasync 2>&1 | tail -1
    echo -n "  读取: "
    dd if="$BEST_PATH/perftest" of=/dev/null bs=1M count=1024 2>&1 | tail -1
    rm -f "$BEST_PATH/perftest"
else
    echo "  ❌ 没有可写的本地存储!"
fi

echo ""
echo "=========================================="
echo "📊 /tmp 内容一览 (按大小排序, Top 20)"
echo "=========================================="
du -sm /tmp/* /tmp/.* 2>/dev/null | sort -rn | head -20 | while read size path; do
    if [ "$size" -ge 1 ]; then
        printf "  %6d MB  %s\n" "$size" "$path"
    fi
done

echo ""
echo "=========================================="
echo "📊 /tmp 总览"
echo "=========================================="
echo "  已用: $(du -sh /tmp 2>/dev/null | cut -f1)"
df -h /tmp | tail -1

echo ""
echo "Done."