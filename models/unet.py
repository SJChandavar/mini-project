"""
Standard U-Net Architecture implementation in PyTorch.

Reference:
U-Net: Convolutional Networks for Biomedical Image Segmentation (MICCAI 2015)
Ronneberger et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with MaxPool then DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False, dropout: float = 0.0):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad x1 if necessary to match x2 shape
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net for binary tooth segmentation.

    Args:
        in_channels (int): Input image channels (default: 1 for grayscale).
        out_channels (int): Output mask channels (default: 1 for binary segmentation).
        encoder_channels (list): Channel list per stage.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        encoder_channels: list = None,
        dropout: float = 0.1
    ):
        super(UNet, self).__init__()
        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512, 1024]

        c = encoder_channels
        self.inc = DoubleConv(in_channels, c[0], dropout=dropout)
        self.down1 = Down(c[0], c[1], dropout=dropout)
        self.down2 = Down(c[1], c[2], dropout=dropout)
        self.down3 = Down(c[2], c[3], dropout=dropout)
        self.down4 = Down(c[3], c[4], dropout=dropout)

        self.up1 = Up(c[4], c[3], dropout=dropout)
        self.up2 = Up(c[3], c[2], dropout=dropout)
        self.up3 = Up(c[2], c[1], dropout=dropout)
        self.up4 = Up(c[1], c[0], dropout=dropout)

        self.outc = nn.Conv2d(c[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
