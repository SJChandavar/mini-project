"""
Proposed Architecture: ECA-Integrated U-Net++ (ECA-U-Net++)

Integrates Efficient Channel Attention (ECA) at the bottleneck layer (X_4,0) of U-Net++.
"""

import torch
import torch.nn as nn
from .eca import ECABlock
from .unet_plus_plus import ConvBlock


class ECAUNetPlusPlus(nn.Module):
    """
    Proposed Attention-Integrated U-Net++ for Tooth Segmentation.

    The ECA module is placed at the bottleneck feature representation (node X_4,0),
    enhancing cross-channel feature relationships before upsampling into the decoder dense skip pathways.

    Args:
        in_channels (int): Input image channels (default: 1).
        out_channels (int): Output mask channels (default: 1).
        encoder_channels (list): Channel list per depth level.
        deep_supervision (bool): Return multi-scale outputs if True.
        dropout (float): Dropout probability.
        use_eca (bool): Enable or disable ECA attention at the bottleneck.
        eca_gamma (float): ECA non-linear scaling parameter.
        eca_b (float): ECA scaling parameter bias.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        encoder_channels: list = None,
        deep_supervision: bool = False,
        dropout: float = 0.1,
        use_eca: bool = True,
        eca_gamma: float = 2.0,
        eca_b: float = 1.0,
    ):
        super(ECAUNetPlusPlus, self).__init__()
        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512, 1024]

        nb_filter = encoder_channels
        self.deep_supervision = deep_supervision
        self.use_eca = use_eca
        self.dropout = dropout

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Encoder nodes (j = 0)
        self.conv0_0 = ConvBlock(in_channels, nb_filter[0], dropout=dropout)
        self.conv1_0 = ConvBlock(nb_filter[0], nb_filter[1], dropout=dropout)
        self.conv2_0 = ConvBlock(nb_filter[1], nb_filter[2], dropout=dropout)
        self.conv3_0 = ConvBlock(nb_filter[2], nb_filter[3], dropout=dropout)
        self.conv4_0 = ConvBlock(nb_filter[3], nb_filter[4], dropout=dropout)

        # ECA Bottleneck Attention Layer
        if self.use_eca:
            self.eca_bottleneck = ECABlock(
                channels=nb_filter[4], gamma=eca_gamma, b=eca_b
            )
        else:
            self.eca_bottleneck = nn.Identity()

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

        # Final 1x1 convolution output segmentation head
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Forward pass with optional attention extraction.

        Args:
            x (torch.Tensor): Input tensor of shape [B, 1, H, W]
            return_attention (bool): Return tuple (logits, att_weights) if True

        Returns:
            torch.Tensor or tuple: Logits tensor or (logits, att_weights)
        """
        # Column 0: Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # Bottleneck ECA Attention Integration
        att_weights = None
        if self.use_eca and isinstance(self.eca_bottleneck, ECABlock):
            if return_attention:
                att_weights = self.eca_bottleneck.get_attention_weights(x4_0)
            x4_0 = self.eca_bottleneck(x4_0)

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
            logits = [
                self.final1(x0_1),
                self.final2(x0_2),
                self.final3(x0_3),
                self.final4(x0_4),
            ]
        else:
            logits = self.final(x0_4)

        if return_attention:
            return logits, att_weights
        return logits
