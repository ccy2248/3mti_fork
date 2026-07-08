#!/usr/bin/env python3
"""
重组 SEN12MS-CR 数据集为 3M-TI 训练所需格式。

原始结构:
  output_png/{season}_s1_vv/scene_{N}/patch_{M}.png
  output_png/{season}_s1_vh/scene_{N}/patch_{M}.png
  output_png/{season}_s2_cloudy_rgb/scene_{N}/patch_{M}.png
  output_png/{season}_s2_rgb/scene_{N}/patch_{M}.png

目标结构:
  data/sen12mscr/
    train/  {cloudy, rgb, vv, vh}/
    val/    {cloudy, rgb, vv, vh}/
    test/   {cloudy, rgb, vv, vh}/

配对方式: 同名文件跨 4 个模态目录匹配
  例: train/cloudy/winter_scene22_patch103.png
       train/rgb/winter_scene22_patch103.png        ← 四个目录，同名即配对
       train/vv/winter_scene22_patch103.png
       train/vh/winter_scene22_patch103.png

官方 Split:
  val:  ROIs2017_winter_s1/s1_22 → (winter, scene_22)
        ROIs1868_summer_s1/s1_19 → (summer, scene_19)
        ... 等 10 个 scene
  test: ROIs1158_spring_s1/s1_106 → (spring, scene_106)
        ... 等 10 个 scene
  其余全部为 train
"""

import os
import shutil

# ====== 配置 ======
SRC_DIR = "/data/home/scxk346/run/workspace/3mti/data/output_png"
DST_DIR = "/data/home/scxk346/run/workspace/3mti/data/sen12mscr"

# 官方分割（season, scene_number）
# 注意：只给了 s1（SAR）场景，但 s2 对应场景使用相同的 split
VAL_SCENES = {
    ("winter", "22"), ("summer", "19"), ("fall", "65"), ("spring", "17"),
    ("winter", "107"), ("summer", "80"), ("summer", "127"), ("winter", "130"),
    ("summer", "17"), ("winter", "84"),
}

TEST_SCENES = {
    ("spring", "106"), ("spring", "123"), ("spring", "140"), ("spring", "31"),
    ("spring", "44"), ("summer", "119"), ("summer", "73"), ("fall", "139"),
    ("winter", "108"), ("winter", "63"),
}

# 模态映射: 源目录后缀 → 目标子目录名
MODALITY_MAP = {
    "_s1_vv":          "vv",
    "_s1_vh":          "vh",
    "_s2_cloudy_rgb":  "cloudy",
    "_s2_rgb":         "rgb",
}


def get_split(season, scene):
    """根据 season+scene 判断属于 train/val/test"""
    key = (season, str(scene).lstrip("0") or "0")  # 去前导零，如 "022" → "22"
    # 也要尝试保留原样匹配
    key_raw = (season, scene)
    if key in TEST_SCENES or key_raw in TEST_SCENES:
        return "test"
    if key in VAL_SCENES or key_raw in VAL_SCENES:
        return "val"
    return "train"


def main():
    # 统计
    counts = {"train": 0, "val": 0, "test": 0}

    # 遍历 16 个模态-季节目录
    for src_modality_dir in sorted(os.listdir(SRC_DIR)):
        src_modality_path = os.path.join(SRC_DIR, src_modality_dir)
        if not os.path.isdir(src_modality_path):
            continue

        # 解析 season 和 modality
        # 例: "winter_s1_vv" → season="winter", modality_suffix="_s1_vv"
        matched = False
        for suffix, modality_name in MODALITY_MAP.items():
            if src_modality_dir.endswith(suffix):
                # "winter_s1_vv" → season = "winter"
                season = src_modality_dir[:-len(suffix)]  # "winter"
                modality = modality_name
                matched = True
                break
        if not matched:
            print(f"⚠️  跳过无法识别的目录: {src_modality_dir}")
            continue

        print(f"\n📂 处理: {src_modality_dir}  (season={season}, modality={modality})")

        # 遍历该目录下的所有 scene
        for scene_dir in sorted(os.listdir(src_modality_path)):
            scene_path = os.path.join(src_modality_path, scene_dir)
            if not os.path.isdir(scene_path):
                continue

            # 提取 scene 编号: "scene_22" → "22"
            scene_num = scene_dir.replace("scene_", "")

            # 确定 split
            split = get_split(season, scene_num)

            # 创建目标目录
            dst_split_dir = os.path.join(DST_DIR, split, modality)
            os.makedirs(dst_split_dir, exist_ok=True)

            # 处理该 scene 下所有 patch
            for patch_file in sorted(os.listdir(scene_path)):
                if not patch_file.endswith(".png"):
                    continue

                src_patch = os.path.join(scene_path, patch_file)
                # 目标文件名: {season}_scene{scene}_{patch}.png
                new_name = f"{season}_scene{scene_num}_{patch_file}"
                dst_patch = os.path.join(dst_split_dir, new_name)

                # 创建符号链接（省空间，瞬间完成）
                # 如果已有则跳过
                if not os.path.exists(dst_patch):
                    os.symlink(os.path.abspath(src_patch), dst_patch)
                    counts[split] += 1

    # 打印统计
    print("\n" + "=" * 60)
    print("📊 重组完成！统计：")
    print(f"  Train: {counts['train']:,} patches ({counts['train'] // 4:,} 组配对)")
    print(f"  Val:   {counts['val']:,} patches ({counts['val'] // 4:,} 组配对)")
    print(f"  Test:  {counts['test']:,} patches ({counts['test'] // 4:,} 组配对)")
    print(f"  Total: {sum(counts.values()):,} patches")
    print(f"\n📁 输出目录: {DST_DIR}/")
    for split in ["train", "val", "test"]:
        for mod in ["cloudy", "rgb", "vv", "vh"]:
            d = os.path.join(DST_DIR, split, mod)
            n = len(os.listdir(d)) if os.path.isdir(d) else 0
            print(f"  {split}/{mod}: {n} files")
    print("=" * 60)


if __name__ == "__main__":
    main()
