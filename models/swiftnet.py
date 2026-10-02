"""
SwiftNet Baseline Model Implementation for Tooth Segmentation.

Reference:
In-Place Activated BatchNorm for Memory-Efficient and Accurate Semantic Segmentation (CVPR 2019)
Orsic et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1, groups: int = 1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )


class SwiftNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=stride)
        self.conv2 = ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=1)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return F.relu(self.conv1(x) + self.shortcut(x))


class SwiftNet(nn.Module):
    """
    SwiftNet lightweight segmentation network adapted for tooth segmentation.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super(SwiftNet, self).__init__()
        self.stem = ConvBNReLU(in_channels, 32, kernel_size=3, stride=2)

        self.layer1 = SwiftNetBlock(32, 64, stride=2)
        self.layer2 = SwiftNetBlock(64, 128, stride=2)
        self.layer3 = SwiftNetBlock(128, 256, stride=2)

        # Pyramid pooling module
        self.spp = nn.Sequential(
            ConvBNReLU(256, 128, kernel_size=1),
            ConvBNReLU(128, 128, kernel_size=3),
        )

        # Upsampling decoder pathway
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fuse2 = ConvBNReLU(128 + 128, 128, kernel_size=3)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fuse1 = ConvBNReLU(128 + 64, 64, kernel_size=3)

        self.up0 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.head = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem = self.stem(x)
        l1 = self.layer1(stem)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)

        bottleneck = self.spp(l3)

        up_l2 = self.up2(bottleneck)
        fused_l2 = self.fuse2(torch.cat([up_l2, l2], dim=1))

        up_l1 = self.up1(fused_l2)
        fused_l1 = self.fuse1(torch.cat([up_l1, l1], dim=1))

        final_up = self.up0(fused_l1)

        diffY = x.size(2) - final_up.size(2)
        diffX = x.size(3) - final_up.size(3)
        if diffY != 0 or diffX != 0:
            final_up = F.pad(final_up, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        logits = self.head(final_up)
        return logits
