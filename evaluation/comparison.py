"""
Baseline Model Comparison utility for compiling multi-model metrics.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any


def compare_baseline_models(
    experiments_dir: str = "./outputs/experiments",
    save_csv: str = "./outputs/reports/model_comparison.csv",
    save_png: str = "./outputs/reports/model_comparison.png"
) -> pd.DataFrame:
    """
    Scan experiments directory, collect evaluation metrics across all trained baseline models,
    and generate summary comparison CSV table and bar charts.
    """
    records = []
    if os.path.exists(experiments_dir):
        for model_folder in sorted(os.listdir(experiments_dir)):
            folder_path = os.path.join(experiments_dir, model_folder)
            if not os.path.isdir(folder_path):
                continue

            metrics_json = os.path.join(folder_path, "metrics.json")
            test_json = os.path.join(folder_path, "test_metrics.json")

            target_file = test_json if os.path.exists(test_json) else metrics_json

            if os.path.exists(target_file):
                with open(target_file, "r") as f:
                    data = json.load(f)
                row = {"Model": model_folder}
                row.update(data)
                records.append(row)

    if not records:
        # Return empty template dataframe if no local experiments executed yet
        cols = ["Model", "val_loss", "val_acc", "val_dice", "val_iou", "precision", "recall", "fscore", "psnr", "map", "inference_time"]
        df = pd.DataFrame(columns=cols)
    else:
        df = pd.DataFrame(records)

    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        df.to_csv(save_csv, index=False)

    if save_png and not df.empty and "val_dice" in df.columns:
        plt.figure(figsize=(10, 6))
        models = df["Model"]
        dice_scores = df["val_dice"]

        bars = plt.bar(models, dice_scores, color="skyblue", edgecolor="navy")
        plt.title("Model Baseline Comparison: Dice Similarity Score")
        plt.xlabel("Architecture")
        plt.ylabel("Dice Score")
        plt.ylim(0, 1.05)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02, f"{height:.3f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_png), exist_ok=True)
        plt.savefig(save_png, dpi=300)
        plt.close()

    return df
