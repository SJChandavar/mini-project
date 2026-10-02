"""
Mean Average Precision (mAP) for Binary Tooth Segmentation.

Provides two clearly documented academic approaches:
- Option A: Connected-component tooth instance matching average precision (IoU threshold = 0.5)
- Option B: Pixel-level precision-recall Area Under Curve (AUC)
"""

import cv2
import numpy as np
import torch
from typing import Union, Dict, Any


def compute_instance_map(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    iou_threshold: float = 0.5
) -> float:
    """
    Option A: Connected Components Instance-Level Average Precision.

    1. Extract connected components (individual teeth instances) from predicted and GT masks.
    2. Compute pairwise IoU matrix between predicted instances and GT instances.
    3. Match predictions to GT objects at the specified iou_threshold.
    4. Return Precision = TP / (TP + FP + FN).
    """
    pred_bin = (pred_mask >= 0.5).astype(np.uint8)
    gt_bin = (gt_mask >= 0.5).astype(np.uint8)

    num_pred, pred_labels = cv2.connectedComponents(pred_bin)
    num_gt, gt_labels = cv2.connectedComponents(gt_bin)

    pred_instances = [pred_labels == i for i in range(1, num_pred)]
    gt_instances = [gt_labels == i for i in range(1, num_gt)]

    if not gt_instances:
        return 1.0 if not pred_instances else 0.0
    if not pred_instances:
        return 0.0

    iou_matrix = np.zeros((len(pred_instances), len(gt_instances)), dtype=np.float32)
    for i, p_inst in enumerate(pred_instances):
        for j, g_inst in enumerate(gt_instances):
            inter = np.sum(p_inst & g_inst)
            union = np.sum(p_inst | g_inst)
            iou_matrix[i, j] = inter / union if union > 0 else 0.0

    tp = 0
    fp = 0
    matched_gt = set()

    for i in range(len(pred_instances)):
        best_gt_idx = np.argmax(iou_matrix[i])
        best_iou = iou_matrix[i, best_gt_idx]
        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_instances) - len(matched_gt)
    denom = tp + fp + fn
    return float(tp / denom) if denom > 0 else 0.0


def compute_pixel_ap(
    prob_map: np.ndarray,
    gt_mask: np.ndarray,
    num_thresholds: int = 10
) -> float:
    """
    Option B: Pixel-level Precision-Recall Area Under Curve.
    """
    prob_flat = prob_map.ravel()
    gt_flat = (gt_mask >= 0.5).astype(bool).ravel()

    thresholds = np.linspace(0.05, 0.95, num_thresholds)
    precisions = []
    recalls = []

    for th in thresholds:
        p = (prob_flat >= th)
        tp = np.sum(p & gt_flat)
        fp = np.sum(p & (~gt_flat))
        fn = np.sum((~p) & gt_flat)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)

    # Calculate trapezoidal area under Precision-Recall curve
    sorted_pairs = sorted(zip(recalls, precisions), key=lambda x: x[0])
    r_arr = np.array([x[0] for x in sorted_pairs])
    p_arr = np.array([x[1] for x in sorted_pairs])

    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    if trapz_fn is not None:
        ap = trapz_fn(p_arr, r_arr)
    else:
        ap = np.sum((r_arr[1:] - r_arr[:-1]) * (p_arr[1:] + p_arr[:-1]) / 2.0)
    return float(np.clip(ap, 0.0, 1.0))


def compute_map(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    method: str = "instance",
    iou_threshold: float = 0.5
) -> float:
    """
    Compute mAP for binary tooth segmentation.

    Args:
        pred: Predicted probability or binary map.
        target: Binary ground truth mask.
        method: 'instance' (Connected components AP) or 'pixel' (PR curve AUC).
        iou_threshold: Threshold for instance matching.

    Returns:
        float: mAP score in range [0, 1].
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    if pred.ndim == 4:
        pred = pred[0, 0]
    elif pred.ndim == 3:
        pred = pred[0]

    if target.ndim == 4:
        target = target[0, 0]
    elif target.ndim == 3:
        target = target[0]

    if method == "instance":
        return compute_instance_map(pred, target, iou_threshold=iou_threshold)
    else:
        return compute_pixel_ap(pred, target)
