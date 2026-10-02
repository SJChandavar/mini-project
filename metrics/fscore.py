"""
F-score, Precision, Recall, Specificity, and Confusion Matrix computation.
"""

import torch
import numpy as np
from typing import Union, Dict


def compute_confusion_matrix_elements(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray]
) -> Dict[str, int]:
    """
    Calculate TP, FP, TN, FN at pixel level.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    p = (pred >= 0.5).astype(bool).ravel()
    t = (target >= 0.5).astype(bool).ravel()

    tp = int(np.sum(p & t))
    fp = int(np.sum(p & (~t)))
    tn = int(np.sum((~p) & (~t)))
    fn = int(np.sum((~p) & t))

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def compute_fscore(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    beta: float = 1.0,
    smooth: float = 1e-6
) -> Dict[str, float]:
    """
    Compute Precision, Recall, Specificity, and F-score (F1 score).

    Args:
        pred: Predictions.
        target: Ground truth.
        beta: Weight of recall vs precision (1.0 for F1).
        smooth: Epsilon.

    Returns:
        Dict: {"precision", "recall", "specificity", "fscore"}
    """
    cm = compute_confusion_matrix_elements(pred, target)
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)

    beta_sq = beta ** 2
    fscore = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + smooth)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "fscore": float(fscore)
    }
