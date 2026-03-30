"""
model.py
--------
UNet-style encoder–decoder for 2-channel Gaussian heatmap prediction.

Architecture overview
---------------------
  Input  : (B, 3, H, W)    RGB image, normalised to [0, 1]
  Output : (B, 2, H, W)    two heatmap channels (p1 and p2 endpoints)

The backbone uses a series of down-sampling encoder blocks followed by
up-sampling decoder blocks with skip connections (classic UNet pattern).
The output is passed through a Sigmoid so values are in [0, 1].

The default capacity is deliberately modest so training is feasible on a
single consumer GPU.  Increase `base_channels` (e.g. to 64) for more capacity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Building blocks                                                               #
# --------------------------------------------------------------------------- #

class DoubleConv(nn.Module):
    """Two consecutive Conv2d → BatchNorm → ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """Max-pool then DoubleConv (encoder step)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    """Bilinear up-sample, concatenate skip, then DoubleConv (decoder step)."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad x to match skip dimensions (handles odd input sizes)
        diff_h = skip.size(2) - x.size(2)
        diff_w = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                       diff_h // 2, diff_h - diff_h // 2])
        return self.conv(torch.cat([skip, x], dim=1))


# --------------------------------------------------------------------------- #
# UNet                                                                          #
# --------------------------------------------------------------------------- #

class UNet(nn.Module):
    """
    Lightweight UNet for heatmap regression.

    Parameters
    ----------
    in_channels   : number of input image channels (3 for RGB)
    out_channels  : number of output heatmap channels (2 for two endpoints)
    base_channels : width multiplier; default=32 is fast to train on CPU/GPU
    """

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 2,
                 base_channels: int = 32):
        super().__init__()
        c = base_channels

        # Encoder
        self.enc1 = DoubleConv(in_channels, c)       # → (B, c,   H,   W)
        self.enc2 = Down(c,     c * 2)               # → (B, 2c,  H/2, W/2)
        self.enc3 = Down(c * 2, c * 4)               # → (B, 4c,  H/4, W/4)
        self.enc4 = Down(c * 4, c * 8)               # → (B, 8c,  H/8, W/8)

        # Bottleneck
        self.bottleneck = Down(c * 8, c * 16)        # → (B, 16c, H/16, W/16)

        # Decoder
        self.dec4 = Up(c * 16, c * 8,  c * 8)       # skip from enc4
        self.dec3 = Up(c * 8,  c * 4,  c * 4)       # skip from enc3
        self.dec2 = Up(c * 4,  c * 2,  c * 2)       # skip from enc2
        self.dec1 = Up(c * 2,  c,      c)            # skip from enc1

        # Output projection
        self.head = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        bn = self.bottleneck(s4)

        # Decode
        x = self.dec4(bn, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        # Predict heatmaps in [0, 1]
        return torch.sigmoid(self.head(x))


# --------------------------------------------------------------------------- #
# Convenience factory                                                           #
# --------------------------------------------------------------------------- #

def build_model(base_channels: int = 32,
                in_channels: int = 3,
                out_channels: int = 2) -> UNet:
    """Create and return a UNet model with the given capacity."""
    return UNet(in_channels=in_channels,
                out_channels=out_channels,
                base_channels=base_channels)
