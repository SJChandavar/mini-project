"""
Peak Signal-to-Noise Ratio (PSNR) calculation for binary segmentation masks.

Reference Note:
PSNR is included because it is reported in the reference paper for mask quality evaluation.
"""

import torch
import numpy as np
from typing import Union


def compute_psnr(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    max_val: float = 1.0
) -> float:
    """
    Compute PSNR between predicted mask and ground-truth mask.

    PSNR = 10 * log10(max_val^2 / MSE)

    Args:
        pred: Predicted binary or probability mask [0, 1].
        target: Ground-truth binary mask [0, 1].
        max_val: Maximum possible pixel intensity (1.0 for normalized masks).

    Returns:
        float: PSNR value in dB.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    p = pred.astype(np.float32)
    t = target.astype(np.float32)

    mse = np.mean((p - t) ** 2)
    if mse == 0:
        return 100.0  # Perfect match convention

    psnr = 10.0 * np.log10((max_val ** 2) / mse)
    return float(psnr)
