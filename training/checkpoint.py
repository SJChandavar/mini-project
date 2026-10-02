"""
Model Checkpoint Manager for saving and loading PyTorch weights & states.
"""

import os
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim


class CheckpointManager:
    """
    Handles saving and restoring model checkpoints with full metadata.
    """

    def __init__(self, checkpoint_dir: str = "./outputs/checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(
        self,
        filename: str,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        config: Dict[str, Any] = None,
        seed: int = 42,
        scheduler: Any = None,
        best_val_loss: float = float("inf"),
        best_val_dice: float = 0.0,
        training_history: list = None
    ) -> str:
        """
        Save complete training state dictionary.

        Args:
            filename (str): Target filename e.g. 'checkpoint_latest.pth' or 'best_model.pth'
            model: PyTorch model.
            optimizer: PyTorch optimizer.
            epoch: Current epoch integer.
            metrics: Dictionary of metric scores.
            config: Experiment configuration dictionary.
            seed: Random seed.
            scheduler: PyTorch LR scheduler (optional).
            best_val_loss: Best validation loss achieved so far.
            best_val_dice: Best validation Dice score achieved so far.
            training_history: List of historical epoch metric dictionaries.

        Returns:
            str: Absolute path of saved checkpoint file.
        """
        save_path = os.path.join(self.checkpoint_dir, filename)

        # Handle DataParallel or wrapped models
        model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

        state = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_val_loss": best_val_loss,
            "best_val_dice": best_val_dice,
            "training_history": training_history if training_history is not None else [],
            "metrics": metrics,
            "config": config,
            "seed": seed,
            "pytorch_version": torch.__version__,
        }

        torch.save(state, save_path)
        return save_path

    def load_checkpoint(
        self,
        filepath: str,
        model: nn.Module,
        optimizer: optim.Optimizer = None,
        scheduler: Any = None,
        device: str = "cpu"
    ) -> Tuple[int, Dict[str, float], Dict[str, Any], float, float, list]:
        """
        Load weights, optimizer, scheduler, and training state into model.

        Args:
            filepath (str): Path to .pth checkpoint file.
            model: PyTorch model instance.
            optimizer: PyTorch optimizer (optional).
            scheduler: PyTorch LR scheduler (optional).
            device (str): Device to map tensors to ('cpu' or 'cuda').

        Returns:
            Tuple[epoch, metrics, config, best_val_loss, best_val_dice, training_history]
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file does not exist: {filepath}")

        checkpoint = torch.load(filepath, map_location=device, weights_only=False)

        model_state = checkpoint.get("model_state_dict", {})
        clean_state = {}
        for k, v in model_state.items():
            name = k[7:] if k.startswith("module.") else k
            clean_state[name] = v

        model.load_state_dict(clean_state)

        if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        epoch = checkpoint.get("epoch", 0)
        metrics = checkpoint.get("metrics", {})
        config = checkpoint.get("config", {})
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        best_val_dice = checkpoint.get("best_val_dice", 0.0)
        training_history = checkpoint.get("training_history", [])

        return epoch, metrics, config, best_val_loss, best_val_dice, training_history
