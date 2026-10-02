"""
Full Test-Set Validation and Comprehensive Metric Reporting Script
for Tooth Segmentation Model Checkpoint.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import time
import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset.validators import scan_dataset_directory
from dataset.split import split_dataset
from dataset.dental_dataset import DentalDataset
from dataset.preprocessing import preprocess_image, preprocess_mask
from models import create_model
from inference import InferencePipeline
from metrics import (
    compute_dice,
    compute_iou,
    compute_accuracy,
    compute_fscore,
    compute_psnr,
    compute_map,
    compute_confusion_matrix_elements,
)
from metrics.map_metric import compute_instance_map, compute_pixel_ap
from evaluation.visualization import generate_overlay, generate_four_panel_visualization


def main():
    print("==========================================================")
    print(" STARTING COMPLETE TEST-SET VALIDATION")
    print("==========================================================")

    # 1. Identify Checkpoint File
    ckpt_path = "outputs/experiments/unetplusplus_eca/checkpoints/best_model.pth"
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint file not found: {ckpt_path}")
        sys.exit(1)

    print(f"1. Target Checkpoint: {ckpt_path}")

    # Load Checkpoint Metadata
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    epoch = ckpt.get("epoch", "N/A")
    ckpt_metrics = ckpt.get("metrics", {})
    print(f"   - Epochs Trained: {epoch}")
    print(f"   - Training Config: {config.get('model', {}).get('name', 'unetplusplus_eca')}")
    print(f"   - Resolution Trained: {config.get('image', {}).get('height', 512)}x{config.get('image', {}).get('width', 512)}")

    # 2. Identify Dataset
    ds_cfg = config.get("dataset", {})
    images_dir = ds_cfg.get("images", "./data/raw/images")
    masks_dir = ds_cfg.get("tooth_masks", "./data/raw/tooth_masks")
    scan_res = scan_dataset_directory(images_dir, masks_dir)
    matched_pairs = scan_res.get("matched_pairs", [])
    print(f"2. Target Dataset: Humans in the Loop Teeth Segmentation Dataset (DatasetNinja)")
    print(f"   - Total Matched Image-Mask Pairs: {len(matched_pairs)}")

    # 3. Identify Dataset Split
    split_cfg = config.get("split", {})
    train_pairs, val_pairs, test_pairs = split_dataset(
        matched_pairs,
        train_ratio=split_cfg.get("train", 0.70),
        val_ratio=split_cfg.get("validation", 0.15),
        test_ratio=split_cfg.get("test", 0.15),
        seed=config.get("project", {}).get("seed", 42),
        save_path="./outputs/reports/dataset_split.csv"
    )
    print(f"3. Dataset Split (Seed 42): Train={len(train_pairs)}, Val={len(val_pairs)}, Test={len(test_pairs)}")

    # 4. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"4. Running Inference on Device: {device}")
    pipeline = InferencePipeline(checkpoint_path=ckpt_path, model_name="unetplusplus_eca", device=device)

    # 5. Run Evaluation across ALL 90 test images
    per_image_records = []
    qualitative_dir = "./outputs/reports/qualitative_examples"
    os.makedirs(qualitative_dir, exist_ok=True)

    print(f"\n[Evaluating {len(test_pairs)} Test Set Images]...")

    for idx, (img_path, mask_path) in enumerate(test_pairs):
        img_name = os.path.basename(img_path)
        mask_name = os.path.basename(mask_path)

        # Run pipeline prediction
        t0 = time.time()
        res = pipeline.predict(img_path, threshold=0.5)
        t1 = time.time()
        inf_time = t1 - t0

        # Load real ground truth mask at original resolution
        gt_tensor = preprocess_mask(mask_path, target_shape=res["original_shape"], threshold=0.5)
        gt_np = gt_tensor.squeeze().numpy()

        pred_np = res["binary_mask"]
        prob_np = res["probability_map"]

        # Calculate metrics
        dice = compute_dice(pred_np, gt_np)
        iou = compute_iou(pred_np, gt_np)
        acc = compute_accuracy(pred_np, gt_np)
        f_metrics = compute_fscore(pred_np, gt_np)
        psnr = compute_psnr(pred_np, gt_np)
        map_instance = compute_instance_map(pred_np, gt_np, iou_threshold=0.5)
        map_pixel = compute_pixel_ap(prob_np, gt_np)
        cm = compute_confusion_matrix_elements(pred_np, gt_np)

        record = {
            "test_index": idx + 1,
            "image_name": img_name,
            "mask_name": mask_name,
            "dice": dice,
            "iou": iou,
            "accuracy": acc,
            "precision": f_metrics["precision"],
            "recall": f_metrics["recall"],
            "specificity": f_metrics["specificity"],
            "f1_score": f_metrics["fscore"],
            "psnr": psnr,
            "map_instance_iou0.5": map_instance,
            "map_pixel_pr_auc": map_pixel,
            "inference_time_sec": inf_time,
            "tp": cm["tp"],
            "fp": cm["fp"],
            "tn": cm["tn"],
            "fn": cm["fn"],
            "gt_pixel_count": int(np.sum(gt_np)),
            "pred_pixel_count": int(np.sum(pred_np))
        }
        per_image_records.append(record)

        # Save Qualitative Examples (First 15 test set images)
        if idx < 15:
            save_vis_path = os.path.join(qualitative_dir, f"test_example_{idx+1}_{os.path.splitext(img_name)[0]}.png")
            generate_four_panel_visualization(
                res["original_image"],
                gt_np,
                pred_np,
                prob_np,
                save_path=save_vis_path,
                title=f"Test Example #{idx+1}: {img_name} (Dice={dice:.4f}, IoU={iou:.4f})"
            )

    df_per_image = pd.DataFrame(per_image_records)
    df_per_image.to_csv("./outputs/reports/per_image_metrics.csv", index=False)
    print(f"\n[OK] Saved per-image metrics to outputs/reports/per_image_metrics.csv")

    # 6. Report Statistics across ENTIRE Test Set
    metrics_to_stat = [
        "dice", "iou", "accuracy", "precision", "recall",
        "specificity", "f1_score", "psnr", "map_instance_iou0.5", "map_pixel_pr_auc"
    ]

    summary_stats = {}
    for m in metrics_to_stat:
        vals = df_per_image[m].values
        summary_stats[m] = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals))
        }

    overall_metrics = {
        "num_test_images": len(test_pairs),
        "summary_statistics": summary_stats,
        "mean_metrics": {m: float(np.mean(df_per_image[m].values)) for m in metrics_to_stat},
        "median_metrics": {m: float(np.median(df_per_image[m].values)) for m in metrics_to_stat},
        "total_confusion_matrix": {
            "tp": int(df_per_image["tp"].sum()),
            "fp": int(df_per_image["fp"].sum()),
            "tn": int(df_per_image["tn"].sum()),
            "fn": int(df_per_image["fn"].sum()),
        }
    }

    # Save Final Test Metrics JSON & CSV
    with open("./outputs/reports/final_test_metrics.json", "w") as f:
        json.dump(overall_metrics, f, indent=4)

    df_summary = pd.DataFrame.from_dict(summary_stats, orient="index")
    df_summary.to_csv("./outputs/reports/final_test_metrics.csv")

    print("[OK] Saved outputs/reports/final_test_metrics.json")
    print("[OK] Saved outputs/reports/final_test_metrics.csv")

    # Display Statistical Summary Table
    print("\n==========================================================")
    print(" OVERALL TEST-SET STATISTICAL SUMMARY (N = 90)")
    print("==========================================================")
    print(f"{'Metric':<22} | {'Mean':<8} | {'Median':<8} | {'Std Dev':<8} | {'Min':<8} | {'Max':<8}")
    print("-" * 75)
    for m in metrics_to_stat:
        st = summary_stats[m]
        print(f"{m:<22} | {st['mean']:<8.4f} | {st['median']:<8.4f} | {st['std']:<8.4f} | {st['min']:<8.4f} | {st['max']:<8.4f}")
    print("==========================================================\n")


if __name__ == "__main__":
    main()
