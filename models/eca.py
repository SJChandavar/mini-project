"""
Efficient Channel Attention (ECA-Net) Module for PyTorch.

Reference:
ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks (CVPR 2020)
Wang et al.
"""

import math
import torch
import torch.nn as nn


class ECABlock(nn.Module):
    """
    Efficient Channel Attention (ECA) module.

    Performs non-dimensionality-reduction 1D channel-wise convolution on
    Global Average Pooled features.

    Parameters:
    -----------
    channels : int
        Number of input feature map channels.
    gamma : float
        Non-linear mapping parameter for kernel size computation (default: 2).
    b : float
        Non-linear mapping parameter bias (default: 1).
    kernel_size : int, optional
        Explicit kernel size override for 1D convolution. If None, computed adaptively.
    """

    def __init__(self, channels: int = 512, gamma: float = 2.0, b: float = 1.0, kernel_size: int = None):
        super(ECABlock, self).__init__()
        self.channels = channels
        self.gamma = gamma
        self.b = b

        if kernel_size is None:
            # Adaptive kernel size k = | (log2(C) + b) / gamma |_odd
            t = int(abs((math.log2(max(channels, 1)) + b) / gamma))
            kernel_size = t if t % 2 == 1 else t + 1
            kernel_size = max(3, kernel_size)

        self.kernel_size = kernel_size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]

        Returns:
            torch.Tensor: Channel-attention weighted tensor of shape [B, C, H, W]
        """
        b, c, h, w = x.size()

        # Step 1: Global Average Pooling -> [B, C, 1, 1]
        y = self.avg_pool(x)

        # Step 2: Reshape for 1D Conv -> [B, 1, C]
        y = y.squeeze(-1).transpose(-1, -2)

        # Step 3: 1D Convolution across channel dimension -> [B, 1, C]
        y = self.conv(y)

        # Step 4: Reshape back -> [B, C, 1, 1]
        y = y.transpose(-1, -2).unsqueeze(-1)

        # Step 5: Sigmoid activation to obtain channel weights
        att_weights = self.sigmoid(y)

        # Step 6: Element-wise multiplication with input feature map
        return x * att_weights.expand_as(x)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract raw channel attention weights for inspection / visualization.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]

        Returns:
            torch.Tensor: Attention weights of shape [B, C]
        """
        with torch.no_grad():
            y = self.avg_pool(x)
            y = y.squeeze(-1).transpose(-1, -2)
            y = self.conv(y)
            y = y.transpose(-1, -2).squeeze(-1)
            att_weights = self.sigmoid(y)
            return att_weights
