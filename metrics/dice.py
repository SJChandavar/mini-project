"""
Dice Coefficient metric implementation.
"""

import torch
import numpy as np
from typing import Union


def compute_dice(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    smooth: float = 1e-6
) -> float:
    """
    Compute Dice Similarity Coefficient:
    Dice = 2 * TP / (2 * TP + FP + FN + smooth)

    Args:
        pred: Binary predictions (0 or 1).
        target: Binary ground truth (0 or 1).
        smooth: Small epsilon to prevent zero division.

    Returns:
        float: Dice score in range [0, 1].
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    p = (pred >= 0.5).astype(np.float32).ravel()
    t = (target >= 0.5).astype(np.float32).ravel()

    intersection = np.sum(p * t)
    cardinality = np.sum(p) + np.sum(t)

    if cardinality == 0:
        return 1.0 if np.array_equal(p, t) else 0.0

    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return float(dice)
