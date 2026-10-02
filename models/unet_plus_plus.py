"""
U-Net++ (Nested U-Net) Architecture implemented from scratch in PyTorch.

Reference:
UNet++: A Nested U-Net Architecture for Medical Image Segmentation (DLMIA 2018)
Zhou et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Standard Conv-BatchNorm-ReLU block used in U-Net++ nodes."""

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


class UNetPlusPlus(nn.Module):
    """
    U-Net++ architecture with 5 levels of dense skip pathways.

    Node grid:
    X_0,0 -> X_0,1 -> X_0,2 -> X_0,3 -> X_0,4
      ↓        ↑        ↑        ↑        ↑
    X_1,0 -> X_1,1 -> X_1,2 -> X_1,3
      ↓        ↑        ↑        ↑
    X_2,0 -> X_2,1 -> X_2,2
      ↓        ↑        ↑
    X_3,0 -> X_3,1
      ↓        ↑
    X_4,0

    Args:
        in_channels (int): Input image channels (default: 1).
        out_channels (int): Output mask channels (default: 1).
        encoder_channels (list): Channels at each level i=0..4.
        deep_supervision (bool): If True, returns outputs from multiple scale heads.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        encoder_channels: list = None,
        deep_supervision: bool = False,
        dropout: float = 0.1,
    ):
        super(UNetPlusPlus, self).__init__()
        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512, 1024]

        nb_filter = encoder_channels
        self.deep_supervision = deep_supervision
        self.dropout = dropout

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Encoder nodes (j = 0)
        self.conv0_0 = ConvBlock(in_channels, nb_filter[0], dropout=dropout)
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1], dropout=dropout)
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2], dropout=dropout)
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3], dropout=dropout)
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4], dropout=dropout)

        # Skip pathway nodes (j = 1)
        self.conv0_1 = ConvBlock(nb_filter[0] + nb_filter[1], nb_filter[0], dropout=dropout)
        self.conv1_1 = ConvBlock(nb_filter[1] + nb_filter[2], nb_filter[1], dropout=dropout)
        self.conv2_1 = ConvBlock(nb_filter[2] + nb_filter[3], nb_filter[2], dropout=dropout)
        self.conv3_1 = ConvBlock(nb_filter[3] + nb_filter[4], nb_filter[3], dropout=dropout)

        # Skip pathway nodes (j = 2)
        self.conv0_2 = ConvBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], dropout=dropout)
        self.conv1_2 = ConvBlock(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], dropout=dropout)
        self.conv2_2 = ConvBlock(nb_filter[2] * 2 + nb_filter[3], nb_filter[2], dropout=dropout)

        # Skip pathway nodes (j = 3)
        self.conv0_3 = ConvBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], dropout=dropout)
        self.conv1_3 = ConvBlock(nb_filter[1] * 3 + nb_filter[2], nb_filter[1], dropout=dropout)

        # Skip pathway nodes (j = 4)
        self.conv0_4 = ConvBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0], dropout=dropout)

        # Final 1x1 convolution segmentation heads
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # Column 0: Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Column 1
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))

        # Column 2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))

        # Column 3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))

        # Column 4
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            return output
