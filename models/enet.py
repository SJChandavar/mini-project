"""
ENet (Efficient Neural Network) Baseline Model for Tooth Segmentation.

Reference:
ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation (arXiv 2016)
Paszke et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InitialBlock(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels - in_channels, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.bn(torch.cat([self.conv(x), self.pool(x)], dim=1)))


class Bottleneck(nn.Module):
    def __init__(
        self,
        channels: int,
        internal_scale: int = 4,
        asymmetric: bool = False,
        dilated: int = 1,
        downsampling: bool = False,
        upsampling: bool = False,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.downsampling = downsampling
        self.upsampling = upsampling
        internal_channels = channels // internal_scale

        if downsampling:
            self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=True)
            self.conv_main = nn.Conv2d(channels, internal_channels, 2, stride=2, bias=False)
        elif upsampling:
            self.unpool = nn.MaxUnpool2d(2, stride=2)
            self.conv_main = nn.Conv2d(channels, internal_channels, 1, bias=False)
        else:
            self.conv_main = nn.Conv2d(channels, internal_channels, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.prelu1 = nn.PReLU()

        if upsampling:
            self.conv_mid = nn.ConvTranspose2d(internal_channels, internal_channels, 3, stride=2, padding=1, output_padding=1)
        elif asymmetric:
            self.conv_mid = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, (5, 1), padding=(2, 0), bias=False),
                nn.Conv2d(internal_channels, internal_channels, (1, 5), padding=(0, 2), bias=False),
            )
        else:
            self.conv_mid = nn.Conv2d(internal_channels, internal_channels, 3, padding=dilated, dilation=dilated, bias=False)

        self.bn2 = nn.BatchNorm2d(internal_channels)
        self.prelu2 = nn.PReLU()

        self.conv_ext = nn.Conv2d(internal_channels, channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.prelu3 = nn.PReLU()
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x, indices=None):
        main = x
        if self.downsampling:
            main, indices = self.maxpool(x)
        elif self.upsampling:
            main = self.unpool(x, indices)

        ext = self.prelu1(self.bn1(self.conv_main(x)))
        ext = self.prelu2(self.bn2(self.conv_mid(ext)))
        ext = self.prelu3(self.bn3(self.conv_ext(ext)))
        ext = self.dropout(ext)

        diffY = main.size(2) - ext.size(2)
        diffX = main.size(3) - ext.size(3)
        if diffY != 0 or diffX != 0:
            ext = F.pad(ext, [0, diffX, 0, diffY])

        out = main + ext
        return (out, indices) if self.downsampling else out


class ENet(nn.Module):
    """
    ENet for fast real-time binary tooth segmentation.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super(ENet, self).__init__()
        self.initial = InitialBlock(in_channels, 16)

        # Stage 1
        self.b1_0 = Bottleneck(16, downsampling=True)
        self.b1_1 = Bottleneck(16)
        self.b1_2 = Bottleneck(16)
        self.b1_3 = Bottleneck(16)
        self.b1_4 = Bottleneck(16)

        # Stage 2
        self.b2_0 = Bottleneck(16, downsampling=True)
        self.b2_1 = Bottleneck(16)
        self.b2_2 = Bottleneck(16, dilated=2)
        self.b2_3 = Bottleneck(16, asymmetric=True)
        self.b2_4 = Bottleneck(16, dilated=4)
        self.b2_5 = Bottleneck(16)

        # Stage 3 (Decoder)
        self.b3_0 = Bottleneck(16, upsampling=True)
        self.b3_1 = Bottleneck(16)
        self.b3_2 = Bottleneck(16)

        self.b4_0 = Bottleneck(16, upsampling=True)
        self.b4_1 = Bottleneck(16)

        self.fullconv = nn.ConvTranspose2d(16, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x, idx1 = self.b1_0(x)
        x = self.b1_1(x)
        x = self.b1_2(x)
        x = self.b1_3(x)
        x = self.b1_4(x)

        x, idx2 = self.b2_0(x)
        x = self.b2_1(x)
        x = self.b2_2(x)
        x = self.b2_3(x)
        x = self.b2_4(x)
        x = self.b2_5(x)

        x = self.b3_0(x, idx2)
        x = self.b3_1(x)
        x = self.b3_2(x)

        x = self.b4_0(x, idx1)
        x = self.b4_1(x)

        out = self.fullconv(x)
        return out
