"""U-Net for cloud removal (13→13 channels) with RGB multi-scale supervision.

Standard encoder-decoder with skip connections. Decoder outputs at each
scale are fed through lightweight 1x1 conv heads to produce 3-channel RGB
predictions for multi-scale RGB supervision (L1 + SSIM).

Architecture:
    Encoder:  13→64→128→256→512  (×4 DoubleConv + MaxPool)
    Bottleneck: 512→1024 (DoubleConv)
    Decoder:   1024→512→256→128→64→13  (×4 UpConv + DoubleConv)
    RGB heads: 3× (1×1Conv at decoder scales 1/8, 1/4, 1/2)

Reference: Inspired by ECRformer's SDFL multi-scale projection strategy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """(Conv2d → BN → ReLU) × 2."""

    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """MaxPool → DoubleConv."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upsample → concat skip → DoubleConv."""

    def __init__(self, in_ch, out_ch, skip_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle odd dimensions
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class RGBHead(nn.Module):
    """1×1 Conv → 3-channel RGB prediction at a given scale."""

    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 3, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """U-Net for SAR-free cloud removal (cloudy 13ch → clear 13ch).

    Args:
        in_chans:   Input channels (default 13).
        out_chans:  Output channels (default 13).
        base_ch:    Base feature channels (default 64).
        rgb_heads:  Whether to add multi-scale RGB supervision heads.
    """

    def __init__(self, in_chans=13, out_chans=13, base_ch=64, rgb_heads=True):
        super().__init__()
        self.rgb_heads = rgb_heads
        ch = base_ch

        # ── Encoder ──
        self.enc1 = DoubleConv(in_chans, ch)           # 256² × ch
        self.enc2 = Down(ch, ch * 2)                    # 128² × 2ch
        self.enc3 = Down(ch * 2, ch * 4)                #  64² × 4ch
        self.enc4 = Down(ch * 4, ch * 8)                #  32² × 8ch

        # ── Bottleneck ──
        self.bottleneck = DoubleConv(ch * 8, ch * 16)   #  32² × 16ch

        # ── Decoder ──
        self.dec1 = Up(ch * 16, ch * 8, ch * 8)         #  64² × 8ch
        self.dec2 = Up(ch * 8, ch * 4, ch * 4)          # 128² × 4ch
        self.dec3 = Up(ch * 4, ch * 2, ch * 2)          # 256² × 2ch
        self.dec4 = Up(ch * 2, ch, ch)                   # 256² × ch

        # ── Final output ──
        self.final = nn.Conv2d(ch, out_chans, kernel_size=1)

        # ── Multi-scale RGB heads (lightweight) ──
        if rgb_heads:
            self.rgb_head_1 = RGBHead(ch * 8)    # decoder 64² output → 3ch RGB
            self.rgb_head_2 = RGBHead(ch * 4)    # decoder 128² output → 3ch RGB
            self.rgb_head_3 = RGBHead(ch * 2)    # decoder 256² output → 3ch RGB

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # ── Encoder ──
        e1 = self.enc1(x)                               # 256³ × ch
        e2 = self.enc2(e1)                               # 128² × 2ch
        e3 = self.enc3(e2)                               #  64² × 4ch
        e4 = self.enc4(e3)                               #  32² × 8ch

        # ── Bottleneck ──
        b = self.bottleneck(e4)                          #  32² × 16ch

        # ── Decoder ──
        d1 = self.dec1(b, e4)                            #  64² × 8ch
        d2 = self.dec2(d1, e3)                           # 128² × 4ch
        d3 = self.dec3(d2, e2)                           # 256² × 2ch
        d4 = self.dec4(d3, e1)                           # 256² × ch

        # ── Final ──
        out = self.final(d4)                              # 256² × 13ch

        if self.rgb_heads:
            rgb_ms = [
                self.rgb_head_1(d1),    # 64² × 3
                self.rgb_head_2(d2),    # 128² × 3
                self.rgb_head_3(d3),    # 256² × 3
            ]
            return out, rgb_ms

        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_unet(**kwargs):
    """Factory function compatible with ECRformer's find_model_using_name."""
    return UNet(**kwargs)
