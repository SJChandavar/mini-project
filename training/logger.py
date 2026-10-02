"""
Training history logger for saving CSV, JSON, and plotting training progress.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any


class TrainingLogger:
    """
    Logs per-epoch metrics to CSV and JSON files, and generates training curve plots.
    """

    def __init__(self, output_dir: str = "./outputs/experiments/unetplusplus_eca"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, "training_history.csv")
        self.json_path = os.path.join(self.output_dir, "metrics.json")
        self.history: List[Dict[str, Any]] = []

        if os.path.exists(self.csv_path):
            try:
                df_existing = pd.read_csv(self.csv_path)
                if not df_existing.empty:
                    self.history = df_existing.to_dict("records")
            except Exception:
                self.history = []

    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """
        Append or update epoch metrics cleanly without duplicate epoch rows.
        """
        row = {"epoch": epoch}
        row.update(metrics)

        # Deduplicate history by filtering out any existing entry for this epoch
        self.history = [h for h in self.history if h.get("epoch") != epoch]
        self.history.append(row)
        self.history.sort(key=lambda x: x.get("epoch", 0))

        df = pd.DataFrame(self.history)
        df.to_csv(self.csv_path, index=False)

    def save_final_metrics(self, final_metrics: Dict[str, Any]):
        """
        Save final validation / test metrics to JSON.
        """
        with open(self.json_path, "w") as f:
            json.dump(final_metrics, f, indent=4)

    def plot_history(self):
        """
        Generate and save matplotlib training curves:
        - Loss curve (train vs val)
        - Dice curve
        - IoU curve
        """
        if not self.history:
            return

        df = pd.DataFrame(self.history)
        epochs = df["epoch"]

        # Plot 1: Loss Curve
        plt.figure(figsize=(8, 5))
        if "train_loss" in df.columns:
            plt.plot(epochs, df["train_loss"], label="Train Loss", color="blue", linewidth=2)
        if "val_loss" in df.columns:
            plt.plot(epochs, df["val_loss"], label="Val Loss", color="red", linewidth=2)
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "loss_curve.png"), dpi=300)
        plt.close()

        # Plot 2: Dice Score Curve
        if "val_dice" in df.columns:
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, df["val_dice"], label="Val Dice", color="green", linewidth=2)
            plt.title("Validation Dice Similarity Coefficient")
            plt.xlabel("Epoch")
            plt.ylabel("Dice Score")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "dice_curve.png"), dpi=300)
            plt.close()

        # Plot 3: IoU Score Curve
        if "val_iou" in df.columns:
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, df["val_iou"], label="Val IoU", color="purple", linewidth=2)
            plt.title("Validation Intersection over Union (IoU)")
            plt.xlabel("Epoch")
            plt.ylabel("IoU Score")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "iou_curve.png"), dpi=300)
            plt.close()
