"""U-Net cloud removal sub-project.

Model:
    UNet — 13ch→13ch encoder-decoder with multi-scale RGB supervision heads.

Usage:
    from unet.unet_model import UNet
    model = UNet(in_chans=13, out_chans=13, base_ch=64)
"""
from .unet_model import UNet, create_unet
