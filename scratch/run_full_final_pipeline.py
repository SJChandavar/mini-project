"""
Complete Final Experiment Runner & Validation Script
Executes full training for ECA-U-Net++ and Baseline U-Net++, evaluates both models across the 90 test set images, computes quantitative ablation comparisons, generates qualitative examples, and outputs final reports.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import time
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from dataset.validators import scan_dataset_directory
from dataset.split import split_dataset
from dataset.dental_dataset import DentalDataset
from dataset.preprocessing import preprocess_mask
from dataset.augmentation import SynchronizedAugmenter
from models import create_model
from training.trainer import Trainer
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
from evaluation.visualization import generate_four_panel_visualization


def train_experiment(config_path: str, model_name: str, max_epochs: int = 5):
    print(f"\n==========================================================")
    print(f" TRAINING MODEL: {model_name} (Config: {config_path})")
    print(f"==========================================================")

    torch.set_num_threads(os.cpu_count() or 4)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["training"]["epochs"] = max_epochs
    config["training"]["batch_size"] = 8
    config["model"]["name"] = model_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Threads: {torch.get_num_threads()} | Resolution: {config['image']['height']}x{config['image']['width']}")

    ds_cfg = config.get("dataset", {})
    scan_res = scan_dataset_directory(ds_cfg.get("images", "./data/raw/images"), ds_cfg.get("tooth_masks", "./data/raw/tooth_masks"))
    matched_pairs = scan_res["matched_pairs"]

    train_pairs, val_pairs, test_pairs = split_dataset(
        matched_pairs,
        train_ratio=config["split"]["train"],
        val_ratio=config["split"]["validation"],
        test_ratio=config["split"]["test"],
        seed=config["project"]["seed"]
    )

    img_shape = (config["image"]["height"], config["image"]["width"])
    augmenter = SynchronizedAugmenter(config.get("augmentation", {}))

    train_ds = DentalDataset(train_pairs, target_shape=img_shape, transform=augmenter, is_train=True, cache_in_memory=True)
    val_ds = DentalDataset(val_pairs, target_shape=img_shape, transform=None, is_train=False, cache_in_memory=True)

    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=0)

    model = create_model(model_name, config)
    exp_dir = os.path.join("./outputs/experiments", model_name)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        experiment_dir=exp_dir
    )

    summary = trainer.train()
    return exp_dir, test_pairs, summary


def evaluate_model_on_test_set(model_name: str, ckpt_path: str, test_pairs: list, save_vis_dir: str = None):
    print(f"\n[Evaluating {model_name} on {len(test_pairs)} Test Set Images]...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = InferencePipeline(checkpoint_path=ckpt_path, model_name=model_name, device=device)

    records = []
    for idx, (img_path, mask_path) in enumerate(test_pairs):
        img_name = os.path.basename(img_path)
        t0 = time.time()
        res = pipeline.predict(img_path, threshold=0.5)
        t1 = time.time()

        gt_tensor = preprocess_mask(mask_path, target_shape=res["original_shape"], threshold=0.5)
        gt_np = gt_tensor.squeeze().numpy()
        pred_np = res["binary_mask"]
        prob_np = res["probability_map"]

        dice = compute_dice(pred_np, gt_np)
        iou = compute_iou(pred_np, gt_np)
        acc = compute_accuracy(pred_np, gt_np)
        f_m = compute_fscore(pred_np, gt_np)
        psnr = compute_psnr(pred_np, gt_np)
        map_inst = compute_instance_map(pred_np, gt_np, iou_threshold=0.5)
        map_pix = compute_pixel_ap(prob_np, gt_np)
        cm = compute_confusion_matrix_elements(pred_np, gt_np)

        record = {
            "test_index": idx + 1,
            "model_name": model_name,
            "image_name": img_name,
            "dice": dice,
            "iou": iou,
            "accuracy": acc,
            "precision": f_m["precision"],
            "recall": f_m["recall"],
            "specificity": f_m["specificity"],
            "f1_score": f_m["fscore"],
            "psnr": psnr,
            "map_instance_iou0.5": map_inst,
            "map_pixel_pr_auc": map_pix,
            "inference_time_sec": t1 - t0,
            "tp": cm["tp"],
            "fp": cm["fp"],
            "tn": cm["tn"],
            "fn": cm["fn"],
        }
        records.append(record)

        if save_vis_dir and idx < 15:
            os.makedirs(save_vis_dir, exist_ok=True)
            vis_file = os.path.join(save_vis_dir, f"sample_{idx+1}_{os.path.splitext(img_name)[0]}.png")
            generate_four_panel_visualization(
                res["original_image"], gt_np, pred_np, prob_np, save_path=vis_file,
                title=f"{model_name} #{idx+1}: {img_name} (Dice={dice:.4f}, IoU={iou:.4f})"
            )

    df = pd.DataFrame(records)
    return df


def main():
    print("==========================================================")
    print(" STARTING FINAL EXPERIMENT WORKFLOW")
    print("==========================================================")

    # 1. Train ECA-U-Net++ (Model B)
    exp_dir_eca, test_pairs, eca_train_summary = train_experiment("configs/final_eca.yaml", "unetplusplus_eca", max_epochs=10)
    ckpt_eca = os.path.join(exp_dir_eca, "checkpoints", "best_model.pth")

    # 2. Train Baseline U-Net++ (Model A)
    exp_dir_base, _, base_train_summary = train_experiment("configs/final_unetplusplus.yaml", "unetplusplus", max_epochs=10)
    ckpt_base = os.path.join(exp_dir_base, "checkpoints", "best_model.pth")

    # 3. Evaluate Both Models on Test Set
    vis_dir = "./outputs/reports/qualitative_examples"
    df_eca = evaluate_model_on_test_set("unetplusplus_eca", ckpt_eca, test_pairs, save_vis_dir=vis_dir)
    df_base = evaluate_model_on_test_set("unetplusplus", ckpt_base, test_pairs, save_vis_dir=None)

    # Save per-image metrics for ECA-U-Net++
    df_eca.to_csv("./outputs/reports/per_image_metrics.csv", index=False)
    print("[OK] Saved outputs/reports/per_image_metrics.csv")

    # 4. Statistical Summaries
    metrics_list = ["dice", "iou", "accuracy", "precision", "recall", "specificity", "f1_score", "psnr", "map_instance_iou0.5", "map_pixel_pr_auc", "inference_time_sec"]

    summary_eca = {}
    summary_base = {}
    ablation_records = []

    for m in metrics_list:
        v_eca = df_eca[m].values
        v_base = df_base[m].values

        summary_eca[m] = {
            "mean": float(np.mean(v_eca)),
            "median": float(np.median(v_eca)),
            "std": float(np.std(v_eca)),
            "min": float(np.min(v_eca)),
            "max": float(np.max(v_eca))
        }

        summary_base[m] = {
            "mean": float(np.mean(v_base)),
            "median": float(np.median(v_base)),
            "std": float(np.std(v_base)),
            "min": float(np.min(v_base)),
            "max": float(np.max(v_base))
        }

        mean_eca = np.mean(v_eca)
        mean_base = np.mean(v_base)
        diff = mean_eca - mean_base

        ablation_records.append({
            "Metric": m,
            "Baseline U-Net++": mean_base,
            "Proposed ECA-U-Net++": mean_eca,
            "Difference (ECA vs Baseline)": diff,
            "Percentage Gain": (diff / (mean_base + 1e-8)) * 100.0
        })

    final_test_json = {
        "num_test_images": len(test_pairs),
        "dataset_name": "Humans in the Loop Teeth Segmentation Dataset (Open-Access)",
        "model_name": "unetplusplus_eca",
        "summary_statistics": summary_eca,
        "baseline_summary_statistics": summary_base,
        "mean_metrics": {m: summary_eca[m]["mean"] for m in metrics_list},
        "median_metrics": {m: summary_eca[m]["median"] for m in metrics_list},
    }

    with open("./outputs/reports/final_test_metrics.json", "w") as f:
        json.dump(final_test_json, f, indent=4)
    print("[OK] Saved outputs/reports/final_test_metrics.json")

    df_sum_eca = pd.DataFrame.from_dict(summary_eca, orient="index")
    df_sum_eca.to_csv("./outputs/reports/final_test_metrics.csv")
    print("[OK] Saved outputs/reports/final_test_metrics.csv")

    df_ablation = pd.DataFrame(ablation_records)
    df_ablation.to_csv("./outputs/reports/ablation_comparison.csv", index=False)
    df_ablation.to_csv("./outputs/reports/model_comparison.csv", index=False)
    print("[OK] Saved outputs/reports/ablation_comparison.csv and model_comparison.csv")

    print("\n==========================================================")
    print(" ABLATION COMPARISON: BASELINE U-NET++ vs PROPOSED ECA-U-NET++")
    print("==========================================================")
    print(df_ablation.to_string(index=False))
    print("==========================================================\n")


if __name__ == "__main__":
    main()
