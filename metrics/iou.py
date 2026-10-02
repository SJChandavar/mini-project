"""
Intersection over Union (IoU / Jaccard Index) metric implementation.
"""

import torch
import numpy as np
from typing import Union


def compute_iou(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    smooth: float = 1e-6
) -> float:
    """
    Compute Intersection over Union (IoU):
    IoU = TP / (TP + FP + FN + smooth)

    Args:
        pred: Binary predictions (0 or 1).
        target: Binary ground truth (0 or 1).
        smooth: Small epsilon to prevent zero division.

    Returns:
        float: IoU score in range [0, 1].
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    p = (pred >= 0.5).astype(np.float32).ravel()
    t = (target >= 0.5).astype(np.float32).ravel()

    intersection = np.sum(p * t)
    union = np.sum(p) + np.sum(t) - intersection

    if union == 0:
        return 1.0 if np.array_equal(p, t) else 0.0

    iou = (intersection + smooth) / (union + smooth)
    return float(iou)
