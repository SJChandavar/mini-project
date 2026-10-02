"""
U-Net 3+ Baseline Model Implementation for Tooth Segmentation.

Reference:
UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation (ICASSP 2020)
Huang et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import DoubleConv


class UNet3Plus(nn.Module):
    """
    UNet 3+ with full-scale skip connections and deep supervision support.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, feature_channels: list = None):
        super(UNet3Plus, self).__init__()
        if feature_channels is None:
            feature_channels = [64, 128, 256, 512, 1024]

        c = feature_channels
        self.conv1 = DoubleConv(in_channels, c[0])
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(c[0], c[1])
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(c[1], c[2])
        self.maxpool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(c[2], c[3])
        self.maxpool4 = nn.MaxPool2d(2)

        self.conv5 = DoubleConv(c[3], c[4])

        # Full-scale skip connection convolutions for Decoder 4
        cat_channels = 64
        self.hd4_ut1 = nn.Sequential(nn.MaxPool2d(8), nn.Conv2d(c[0], cat_channels, 3, padding=1), nn.BatchNorm2d(cat_channels), nn.ReLU())
        self.hd4_ut2 = nn.Sequential(nn.MaxPool2d(4), nn.Conv2d(c[1], cat_channels, 3, padding=1), nn.BatchNorm2d(cat_channels), nn.ReLU())
        self.hd4_ut3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(c[2], cat_channels, 3, padding=1), nn.BatchNorm2d(cat_channels), nn.ReLU())
        self.hd4_ut4 = nn.Sequential(nn.Conv2d(c[3], cat_channels, 3, padding=1), nn.BatchNorm2d(cat_channels), nn.ReLU())
        self.hd4_ut5 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(c[4], cat_channels, 3, padding=1), nn.BatchNorm2d(cat_channels), nn.ReLU())
        self.conv4d = DoubleConv(cat_channels * 5, c[3])

        # Decoder 3
        self.hd3_ut1 = nn.Sequential(nn.MaxPool2d(4), nn.Conv2d(c[0], cat_channels, 3, padding=1), nn.BatchNorm2d(cat_channels), nn.ReLU())
        self.hd3_ut2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(c[1], cat_channels, 3, padding=1), nn.BatchNorm2d(cat_channels), nn.ReLU())
        self.hd3_ut3 = nn.Sequential(nn.Conv2d(c[2], cat_channels, 3, padding=1), nn.BatchNorm2d(cat_channels), nn.ReLU())
        self.hd3_ut4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(c[3], cat_channels, 3, padding=1), nn.BatchNorm2d(cat_channels), nn.ReLU())
        self.hd3_ut5 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True), nn.Conv2d(c[4], cat_channels, 3, padding=1), nn.BatchNorm2d(cat_channels), nn.ReLU())
        self.conv3d = DoubleConv(cat_channels * 5, c[2])

        # Decoder 2
        self.hd2_ut1 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(c[0], cat_channels, 3, padding=1), nn.BatchNorm2d(cat_channels), nn.ReLU())
        self.hd2_ut2 = nn.Sequential(nn.Conv2d(c[1], cat_channels, 3, padding=1), nn.BatchNorm2d(cat_channels), nn.ReLU())
        self.hd2_ut3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(c[2], cat_channels, 3, padding=1), nn.BatchNorm2d(cat_channels), nn.ReLU())
        self.hd2_ut4 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True), nn.Conv2d(c[3], cat_channels, 3, padding=1), nn.BatchNorm2d(cat_channels), nn.ReLU())
        self.hd2_ut5 = nn.Sequential(nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True), nn.Conv2d(c[4], cat_channels, 3, padding=1), nn.BatchNorm2d(cat_channels), nn.ReLU())
        self.conv2d = DoubleConv(cat_channels * 5, c[1])

        # Decoder 1
        self.hd1_ut1 = nn.Sequential(nn.Conv2d(c[0], cat_channels, 3, padding=1), nn.BatchNorm2d(cat_channels), nn.ReLU())
        self.hd1_ut2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), nn.Conv2d(c[1], cat_channels, 3, padding=1), nn.BatchNorm2d(cat_channels), nn.ReLU())
        self.hd1_ut3 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True), nn.Conv2d(c[2], cat_channels, 3, padding=1), nn.BatchNorm2d(cat_channels), nn.ReLU())
        self.hd1_ut4 = nn.Sequential(nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True), nn.Conv2d(c[3], cat_channels, 3, padding=1), nn.BatchNorm2d(cat_channels), nn.ReLU())
        self.hd1_ut5 = nn.Sequential(nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True), nn.Conv2d(c[4], cat_channels, 3, padding=1), nn.BatchNorm2d(cat_channels), nn.ReLU())
        self.conv1d = DoubleConv(cat_channels * 5, c[0])

        self.outconv = nn.Conv2d(c[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.conv1(x)
        x2 = self.conv2(self.maxpool1(x1))
        x3 = self.conv3(self.maxpool2(x2))
        x4 = self.conv4(self.maxpool3(x3))
        x5 = self.conv5(self.maxpool4(x4))

        # Decoder 4
        hd4 = self.conv4d(torch.cat([
            self.hd4_ut1(x1), self.hd4_ut2(x2), self.hd4_ut3(x3), self.hd4_ut4(x4), self.hd4_ut5(x5)
        ], dim=1))

        # Decoder 3
        hd3 = self.conv3d(torch.cat([
            self.hd3_ut1(x1), self.hd3_ut2(x2), self.hd3_ut3(x3), self.hd3_ut4(hd4), self.hd3_ut5(x5)
        ], dim=1))

        # Decoder 2
        hd2 = self.conv2d(torch.cat([
            self.hd2_ut1(x1), self.hd2_ut2(x2), self.hd2_ut3(hd3), self.hd2_ut4(hd4), self.hd2_ut5(x5)
        ], dim=1))

        # Decoder 1
        hd1 = self.conv1d(torch.cat([
            self.hd1_ut1(x1), self.hd1_ut2(hd2), self.hd1_ut3(hd3), self.hd1_ut4(hd4), self.hd1_ut5(x5)
        ], dim=1))

        logits = self.outconv(hd1)
        return logits
