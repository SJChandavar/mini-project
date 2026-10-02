"""
Training callbacks including learning rate scheduling handlers.
"""

import torch
import torch.optim as optim
from typing import Dict, Any, Optional


def build_lr_scheduler(
    optimizer: optim.Optimizer,
    scheduler_cfg: Dict[str, Any]
) -> Optional[Any]:
    """
    Build optional PyTorch learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer instance.
        scheduler_cfg: Configuration sub-dictionary.

    Returns:
        Scheduler instance or None if disabled.
    """
    if not scheduler_cfg.get("enabled", False):
        return None

    name = scheduler_cfg.get("name", "reduce_lr_on_plateau").lower()
    if name in ["reducelronplateau", "reduce_lr_on_plateau"]:
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_cfg.get("mode", "min"),
            factor=scheduler_cfg.get("factor", 0.5),
            patience=scheduler_cfg.get("patience", 5),
            min_lr=scheduler_cfg.get("min_lr", 1e-6)
        )
    elif name in ["cosineannealinglr", "cosine_annealing_lr"]:
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_cfg.get("t_max", 100),
            eta_min=scheduler_cfg.get("eta_min", 1e-6)
        )
    else:
        return None
