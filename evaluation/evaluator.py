"""
Evaluation Pipeline for testing tooth segmentation checkpoints.
"""

import os
import json
import time
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses import get_loss_function
from metrics import (
    compute_dice,
    compute_iou,
    compute_accuracy,
    compute_fscore,
    compute_psnr,
    compute_map,
    compute_confusion_matrix_elements
)
from .visualization import generate_four_panel_visualization


class Evaluator:
    """
    Evaluator class for quantitative assessment of tooth segmentation models.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = "cpu",
        threshold: float = 0.5
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.threshold = threshold
        self.criterion = get_loss_function("bce")

    def evaluate(self, save_predictions_dir: str = "./outputs/predictions") -> Dict[str, Any]:
        """
        Run test set evaluation, compute metrics, measure inference speed, and save visual predictions.
        """
        self.model.eval()
        os.makedirs(save_predictions_dir, exist_ok=True)

        running_loss = 0.0
        dices, ious, accs = [], [], []
        precisions, recalls, specificities, fscores = [], [], [], []
        psnrs, maps = [], []
        inference_times = []

        total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0

        with torch.no_grad():
            for idx, (images, masks) in enumerate(tqdm(self.test_loader, desc="[Evaluating Test Set]")):
                images = images.to(self.device)
                masks = masks.to(self.device)

                t0 = time.time()
                logits = self.model(images)
                t1 = time.time()

                batch_time = (t1 - t0) / images.size(0)
                inference_times.append(batch_time)

                loss = self.criterion(logits, masks)
                running_loss += loss.item()

                probs = torch.sigmoid(logits if not isinstance(logits, list) else logits[-1])

                for b in range(images.size(0)):
                    img_np = images[b].cpu().numpy()
                    gt_np = masks[b].cpu().numpy()
                    prob_np = probs[b].cpu().numpy()
                    pred_np = (prob_np >= self.threshold).astype(np.float32)

                    # Compute individual metrics
                    dice = compute_dice(pred_np, gt_np)
                    iou = compute_iou(pred_np, gt_np)
                    acc = compute_accuracy(pred_np, gt_np)
                    f_metrics = compute_fscore(pred_np, gt_np)
                    psnr = compute_psnr(pred_np, gt_np)
                    map_val = compute_map(prob_np, gt_np, method="instance")

                    cm = compute_confusion_matrix_elements(pred_np, gt_np)
                    total_tp += cm["tp"]
                    total_fp += cm["fp"]
                    total_tn += cm["tn"]
                    total_fn += cm["fn"]

                    dices.append(dice)
                    ious.append(iou)
                    accs.append(acc)
                    precisions.append(f_metrics["precision"])
                    recalls.append(f_metrics["recall"])
                    specificities.append(f_metrics["specificity"])
                    fscores.append(f_metrics["fscore"])
                    psnrs.append(psnr)
                    maps.append(map_val)

                    # Save first 5 sample visualizations
                    if idx == 0 and b < 5:
                        save_fig = os.path.join(save_predictions_dir, f"sample_{b + 1}_eval.png")
                        generate_four_panel_visualization(
                            img_np, gt_np, pred_np, prob_np, save_path=save_fig, title=f"Test Sample #{b + 1}"
                        )

        n_batches = max(len(self.test_loader), 1)
        metrics_summary = {
            "test_loss": float(running_loss / n_batches),
            "accuracy": float(np.mean(accs)),
            "iou": float(np.mean(ious)),
            "dice": float(np.mean(dices)),
            "fscore": float(np.mean(fscores)),
            "precision": float(np.mean(precisions)),
            "recall": float(np.mean(recalls)),
            "specificity": float(np.mean(specificities)),
            "psnr": float(np.mean(psnrs)),
            "map": float(np.mean(maps)),
            "inference_time_per_image_sec": float(np.mean(inference_times)),
            "confusion_matrix": {
                "tp": int(total_tp),
                "fp": int(total_fp),
                "tn": int(total_tn),
                "fn": int(total_fn),
            }
        }

        # Save JSON and CSV reports
        reports_dir = "./outputs/reports"
        os.makedirs(reports_dir, exist_ok=True)
        with open(os.path.join(reports_dir, "test_metrics.json"), "w") as f:
            json.dump(metrics_summary, f, indent=4)

        df_rep = pd.DataFrame([metrics_summary])
        df_rep.to_csv(os.path.join(reports_dir, "test_metrics.csv"), index=False)

        return metrics_summary
