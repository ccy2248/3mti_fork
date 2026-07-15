import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import os
from torchvision import transforms
import random

# ====== TIF 卫星图支持 (参考 UnCRtainTS 处理方式) ======
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[WARN] rasterio not installed, .tif support disabled")

def read_tif(path):
    """rasterio 读取 tif → float32 numpy [C, H, W]"""
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)

def process_MS(img):
    """Sentinel-2 RGB/cloudy: clip [0,10000] → rescale → [0,1]"""
    img = np.clip(img, 0.0, 10000.0)
    img = img / 10000.0
    return img  # [0, 1]

def process_SAR(img, sar_type='vh'):
    """Sentinel-1 SAR VH/VV: clip dB → rescale → [0,1]"""
    if sar_type == 'vh':
        dB_min, dB_max = -32.5, 0.0   # VH 交叉极化
    else:
        dB_min, dB_max = -25.0, 0.0   # VV 同极化
    img = np.clip(img, dB_min, dB_max)
    img = (img - dB_min) / (dB_max - dB_min)
    return img  # [0, 1]

def add_combined_noise_torch(
    image,
    pseudo_mask,
    stddev=0.1,
    scale=0.3,
    sparsity=0.5,
    gan=0.4,
    poisson_weight=0.4,
    poisson_L=30,
    brightness_sensitive=True
):

    is_batched = image.dim() == 4
    if not is_batched:
        image = image.unsqueeze(0)

    B, C, H, W = image.shape
    h_small, w_small = max(1, int(H * scale)), max(1, int(W * scale))

    if brightness_sensitive:
        with torch.no_grad():
            gray = image.mean(dim=1, keepdim=True)  # [B,1,H,W]
            brightness_factor = 1.0 - gray
            brightness_factor = torch.nn.functional.interpolate(brightness_factor, size=(h_small, w_small), mode='bilinear', align_corners=False)
    else:
        brightness_factor = torch.ones((B, 1, h_small, w_small), device=image.device)

    noise = torch.randn((B, C, h_small, w_small), device=image.device) * stddev
    if sparsity is not None:
        mask = (torch.rand((B, 1, h_small, w_small), device=image.device) < sparsity).float()
        noise *= mask
    noise *= brightness_factor
    noise = torch.nn.functional.interpolate(noise, size=(H, W), mode='bilinear', align_corners=False)
    gauss_noisy = gan * noise

    poisson_input = torch.clamp(image * poisson_L, min=0)
    poisson_noise = (torch.poisson(poisson_input) / poisson_L) - image
    if brightness_sensitive:
        poisson_noise *= torch.nn.functional.interpolate(brightness_factor, size=(H, W), mode='bilinear', align_corners=False)

    

    poisson_noisy = poisson_weight * poisson_noise
    total_noise = gauss_noisy + poisson_noisy

    if pseudo_mask is not None:
        pseudo_mask = pseudo_mask.view(B, 1, 1, 1).float()  # [B,1,1,1]
        total_noise = total_noise * pseudo_mask

    noisy_image = image + total_noise
    noisy_image = torch.clamp(noisy_image, 0.0, 1.0)

    return noisy_image if is_batched else noisy_image.squeeze(0)

class RandomSubsetColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p_each=0.5):
        self.transforms = [
            transforms.ColorJitter(brightness=brightness),
            transforms.ColorJitter(contrast=contrast),
            transforms.ColorJitter(saturation=saturation),
            transforms.ColorJitter(hue=hue)
        ]
        self.p_each = p_each

    def __call__(self, img):
        selected = [t for t in self.transforms if random.random() < self.p_each]
        random.shuffle(selected)
        for t in selected:
            img = t(img)
        return img

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, height=512, width=512, tokenizer=None, prompts_file=None, use_tif=False, sar_type='vh'):
        super().__init__()

        self.use_tif = use_tif       # True: 用 rasterio 读 .tif
        self.sar_type = sar_type     # 'vh' or 'vv', 仅 tif 模式生效

        with open(dataset_path, 'r') as f:
            json_data = json.load(f)[split]

        self.image_dir = json_data['image']
        self.target_dir = json_data['target_image']
        self.ref_dir = json_data.get('ref_image', None)
        self.prompt = json_data['prompt']

        # loda prompt file
        self.extra_prompts = {}
        if prompts_file is not None:
            with open(prompts_file, "r") as f:
                for line in f:
                    if ":" in line:
                        filename, prompt = line.strip().split(":", 1)
                        self.extra_prompts[filename.strip()] = prompt.strip()

        VALID_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

        self.image_files = sorted([
            os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir)
            if f.endswith(VALID_EXTS)
        ])
        self.target_files = sorted([
            os.path.join(self.target_dir, f) for f in os.listdir(self.target_dir)
            if f.endswith(VALID_EXTS)
        ])

        if self.ref_dir:
            self.ref_files = sorted([
                os.path.join(self.ref_dir, f) for f in os.listdir(self.ref_dir)
                if f.endswith(VALID_EXTS)
            ])
        else:
            self.ref_files = [None] * len(self.image_files)
        

        assert len(self.image_files) == len(self.target_files) == len(self.ref_files), \
            "The number of input, target, and reference images must be consistent."

        self.image_size = (height, width)
        self.image_size_small = (128, 128)  # 仅 IR 超分任务使用
        self.tokenizer = tokenizer
        self.color_trans = RandomSubsetColorJitter()
        

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        input_path = self.image_files[idx]
        target_path = self.target_files[idx]
        ref_path = self.ref_files[idx]

        filename = os.path.basename(input_path)
        extra_prompt = self.extra_prompts.get(filename, "")
        if extra_prompt != "":
            final_prompt = f"{self.prompt}, {extra_prompt}"     # prompts combination
        else:
            final_prompt = self.prompt

        # if self.ref_dir2 is not None and target_path in self.target_files2:
        #     pseudo=True
        # else:
        #     pseudo=False

        # print(input_path)
        # print(target_path)
        # print(ref_path)

        pseudo = False

        try:
            if self.use_tif and HAS_RASTERIO:
                # ---- TIF 模式 (UnCRtainTS 风格归一化) ----
                cloudy_np  = read_tif(input_path)          # [3, H, W] float32
                target_np  = read_tif(target_path)         # [3, H, W] float32
                cloudy_np  = process_MS(cloudy_np)         # → [0, 1]
                target_np  = process_MS(target_np)         # → [0, 1]

                input_tensor  = torch.from_numpy(cloudy_np)
                target_tensor = torch.from_numpy(target_np)
            else:
                # ---- PNG/JPG 模式 (原有 PIL 逻辑) ----
                input_img = Image.open(input_path).convert("RGB")
                target_img = Image.open(target_path).convert("RGB")
                input_tensor  = F.to_tensor(input_img)
                target_tensor = F.to_tensor(target_img)
        except Exception as e:
            # 重试最多 100 次，防止无限递归
            retry = getattr(self, '_retry_count', 0) + 1
            self._retry_count = retry
            if retry > 100:
                raise RuntimeError(f"Failed to load any image after 100 retries. Last error: {e}")
            print(f"Error loading images: {input_path}, {target_path}, retry={retry}")
            return self.__getitem__((idx + 1) % len(self))

        if pseudo and random.random() < 0.6:
            input_tensor = add_combined_noise_torch(
            input_tensor,
            None,
            stddev=random.uniform(0.1, 0.4),
            gan=random.uniform(0.1, 0.5),
            poisson_weight=random.uniform(0.1, 0.4),
            sparsity=random.uniform(0.2, 0.5),
            scale=random.uniform(0.2, 0.5),
            brightness_sensitive=True
            )
        # 去云任务：直接 256→512 resize（不做 128 退化模拟）
        input_tensor = F.resize(input_tensor, self.image_size)

        target_tensor = F.resize(target_tensor, self.image_size)
        target_tensor = F.normalize(target_tensor, mean=[0.5], std=[0.5])

        if ref_path is not None:
            if self.use_tif and HAS_RASTERIO:
                # ---- TIF SAR 参考图 ----
                sar_np = read_tif(ref_path)                 # [1, H, W] or [H, W] float32
                sar_np = process_SAR(sar_np, self.sar_type) # → [0, 1]
                # 单通道 SAR → 复制为 3 通道（适配 VAE 输入）
                if sar_np.ndim == 2:
                    sar_np = sar_np[np.newaxis, ...]  # [1, H, W]
                if sar_np.shape[0] == 1:
                    sar_np = np.repeat(sar_np, 3, axis=0)  # [3, H, W]
                ref_tensor = torch.from_numpy(sar_np)
            else:
                ref_img = Image.open(ref_path).convert("RGB")
                ref_tensor = F.to_tensor(ref_img)
            ref_tensor = F.resize(ref_tensor, self.image_size)
            ref_tensor = F.normalize(ref_tensor, mean=[0.5], std=[0.5])

            
            target_tensor = torch.stack([target_tensor, ref_tensor], dim=0)
        else:
            input_tensor = input_tensor.unsqueeze(0)
            target_tensor = target_tensor.unsqueeze(0)

        if pseudo and random.random() < 0.1:
            input_tensor = self.color_trans(input_tensor)
        
        
        input_tensor = F.normalize(input_tensor, mean=[0.5], std=[0.5])
        input_tensor = torch.stack([input_tensor, ref_tensor], dim=0)
        out = {
            "output_pixel_values": target_tensor,
            "conditioning_pixel_values": input_tensor,
            "caption": final_prompt,
            "filename": filename,
        }

        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                final_prompt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids
            out["input_ids"] = input_ids

        return out