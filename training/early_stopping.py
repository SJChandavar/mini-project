"""
Early Stopping callback for PyTorch model training.
"""

import numpy as np
import torch


class EarlyStopping:
    """
    Early Stopping monitor to stop training when metric stops improving.

    Args:
        patience (int): How many epochs to wait after last improvement (default: 10).
        monitor (str): Metric to monitor ('val_loss' or 'val_dice').
        mode (str): 'min' for loss, 'max' for metrics like Dice or accuracy.
        delta (float): Minimum change in monitored metric to qualify as improvement.
    """

    def __init__(
        self,
        patience: int = 10,
        monitor: str = "val_loss",
        mode: str = "min",
        delta: float = 1e-4
    ):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.early_stop = False
        self.best_score = None
        self.best_epoch = 0
        self.stopped_reason = ""

        if mode not in ["min", "max"]:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")

    def __call__(self, val_metric: float, epoch: int) -> bool:
        """
        Check if training should stop early.

        Args:
            val_metric (float): Current epoch value of monitored metric.
            epoch (int): Current epoch number.

        Returns:
            bool: True if early stopping criteria triggered.
        """
        score = -val_metric if self.mode == "min" else val_metric

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.stopped_reason = (
                    f"Early stopping triggered at epoch {epoch}: '{self.monitor}' did not "
                    f"improve from {abs(self.best_score):.5f} for {self.patience} consecutive epochs."
                )
                return True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0

        return False
