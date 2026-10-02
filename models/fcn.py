"""
Fully Convolutional Network (FCN-8s) Baseline Model for Tooth Segmentation.

Reference:
Fully Convolutional Networks for Semantic Segmentation (CVPR 2015)
Long, Shelhamer, Darrell.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCN8s(nn.Module):
    """
    FCN-8s architecture adapted for binary tooth segmentation.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 64):
        super(FCN8s, self).__init__()

        # ConvBlock 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        # ConvBlock 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        # ConvBlock 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        # ConvBlock 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        # ConvBlock 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        # Classifier / Bottleneck
        self.fc6 = nn.Sequential(
            nn.Conv2d(base_channels * 8, 512, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
        )
        self.fc7 = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
        )

        self.score_fr = nn.Conv2d(512, out_channels, 1)
        self.score_pool4 = nn.Conv2d(base_channels * 8, out_channels, 1)
        self.score_pool3 = nn.Conv2d(base_channels * 4, out_channels, 1)

        self.upscore2 = nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, bias=False)
        self.upscore16 = nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, bias=False)
        self.upscore32 = nn.ConvTranspose2d(out_channels, out_channels, 16, stride=8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.conv2(h)
        pool3 = self.conv3(h)
        pool4 = self.conv4(pool3)
        pool5 = self.conv5(pool4)

        h = self.fc6(pool5)
        h = self.fc7(h)
        h = self.score_fr(h)

        upscore2 = self.upscore2(h)
        score_pool4 = self.score_pool4(pool4)
        diffY = score_pool4.size(2) - upscore2.size(2)
        diffX = score_pool4.size(3) - upscore2.size(3)
        upscore2 = F.pad(upscore2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        h = upscore2 + score_pool4

        upscore16 = self.upscore16(h)
        score_pool3 = self.score_pool3(pool3)
        diffY = score_pool3.size(2) - upscore16.size(2)
        diffX = score_pool3.size(3) - upscore16.size(3)
        upscore16 = F.pad(upscore16, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        h = upscore16 + score_pool3

        h = self.upscore32(h)
        diffY = x.size(2) - h.size(2)
        diffX = x.size(3) - h.size(3)
        h = F.pad(h, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return h
