"""
Pixel-level Accuracy metric implementation.
"""

import torch
import numpy as np
from typing import Union


def compute_accuracy(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Compute Pixel Accuracy:
    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    Args:
        pred: Binary predictions (0 or 1).
        target: Binary ground truth (0 or 1).

    Returns:
        float: Accuracy in range [0, 1].
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    p = (pred >= 0.5).astype(np.bool_)
    t = (target >= 0.5).astype(np.bool_)

    correct = np.sum(p == t)
    total = p.size
    return float(correct / total) if total > 0 else 0.0
