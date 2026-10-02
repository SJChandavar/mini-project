"""
Binary Cross Entropy Loss with Logits for Tooth Segmentation.
"""

import torch
import torch.nn as nn


class BCELossWrapper(nn.Module):
    """
    Standard Binary Cross Entropy with Logits Loss.
    Primary loss function matching reference research paper.
    """

    def __init__(self, pos_weight: float = 1.0):
        super(BCELossWrapper, self).__init__()
        weight_tensor = torch.tensor([pos_weight]) if pos_weight != 1.0 else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): Unnormalized model predictions [B, 1, H, W]
            targets (torch.Tensor): Binary ground truth masks [B, 1, H, W]

        Returns:
            torch.Tensor: Scalar BCE loss value
        """
        if isinstance(logits, list):
            # Deep supervision: sum loss across all scale heads
            loss = 0.0
            for l in logits:
                loss += self.bce(l, targets)
            return loss / len(logits)
        return self.bce(logits, targets)
