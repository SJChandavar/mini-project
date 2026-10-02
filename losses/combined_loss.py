"""
Dice Loss and Combined BCE + Dice Loss implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .bce_loss import BCELossWrapper


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for Binary Segmentation.
    """

    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if isinstance(logits, list):
            probs = [torch.sigmoid(l) for l in logits]
            loss = 0.0
            for p in probs:
                intersection = (p * targets).sum(dim=(2, 3))
                cardinality = p.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
                dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
                loss += (1.0 - dice.mean())
            return loss / len(logits)

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        cardinality = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Weighted combination of Binary Cross Entropy Loss and Soft Dice Loss.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = BCELossWrapper()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_val = self.bce(logits, targets)
        dice_val = self.dice(logits, targets)
        return self.bce_weight * bce_val + self.dice_weight * dice_val


def get_loss_function(loss_name: str = "bce", config: dict = None) -> nn.Module:
    """
    Factory function for loss functions.

    Args:
        loss_name (str): Name of loss ('bce', 'dice', or 'bce_dice' / 'combined').
        config (dict, optional): Extra parameters.

    Returns:
        nn.Module: Loss module instance.
    """
    name = loss_name.lower().replace("_", "").replace("-", "")
    if name in ["bce", "binarycrossentropy"]:
        return BCELossWrapper()
    elif name in ["dice", "diceloss"]:
        return DiceLoss()
    elif name in ["bcedice", "combined", "bcediceloss"]:
        return CombinedLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}. Available: ['bce', 'dice', 'bce_dice']")
