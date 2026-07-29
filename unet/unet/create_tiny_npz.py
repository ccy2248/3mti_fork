"""从大 NPZ shard 提取少量样本创建迷你测试集."""
import numpy as np
import sys
sys.path.insert(0, '/data/home/scxk346/run/workspace/3mti/unet/unet')

SRC = "/data/home/scxk346/run/workspace/3mti/data/sen12mscr/npz"
DST = "/data/home/scxk346/run/workspace/3mti/unet/unet/tiny_data"
import os; os.makedirs(DST, exist_ok=True)

for split, shard, n in [("train", "train_000.npz", 200), ("test", "test_000.npz", 50)]:
    print(f"Loading {shard}...")
    data = np.load(f"{SRC}/{shard}", allow_pickle=False)
    np.savez_compressed(f"{DST}/{split}_tiny.npz",
        s1=data['s1'][:n], s2=data['s2'][:n], label=data['label'][:n])
    with open(f"{DST}/{split}.manifest", "w") as f:
        f.write(f"{split}_tiny.npz\n")
    print(f"  → {DST}/{split}_tiny.npz ({n} samples)")

print("Done!")
