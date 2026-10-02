"""
Comprehensive & Crash-Resistant PyTorch Model Trainer for Tooth Segmentation.
"""

import os
import gc
import sys
import time
import csv
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import psutil
except ImportError:
    psutil = None

from losses import get_loss_function
from metrics import compute_dice, compute_iou, compute_accuracy
from .early_stopping import EarlyStopping
from .checkpoint import CheckpointManager
from .logger import TrainingLogger
from .callbacks import build_lr_scheduler


class Trainer:
    """
    Main training engine for dental X-ray tooth segmentation models.
    Supports gradient accumulation, resumable checkpoints, resource monitoring,
    and crash-safe memory cleanup.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: str = "cpu",
        experiment_dir: str = None,
        resume: bool = False,
        max_epochs_this_run: Optional[int] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.resume = resume
        self.max_epochs_this_run = max_epochs_this_run

        train_cfg = config.get("training", {})
        opt_cfg = config.get("optimizer", {})
        loss_cfg = config.get("loss", {})
        img_cfg = config.get("image", {})

        self.epochs = train_cfg.get("epochs", 100)
        self.gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 4)
        self.mixed_precision = train_cfg.get("mixed_precision", False) and (device == "cuda")

        # CPU Warning
        if device == "cpu" and img_cfg.get("height", 512) >= 512:
            print("\n" + "=" * 70)
            print(" WARNING:")
            print(" Full 512x512 U-Net++ + ECA training on CPU may be very slow and hardware-intensive.")
            print(" Gradient accumulation and memory management are enabled for CPU safety.")
            print("=" * 70 + "\n")

        # Loss Function
        self.criterion = get_loss_function(loss_cfg.get("name", "bce"), config=config)

        # Optimizer
        lr = opt_cfg.get("learning_rate", 0.0001)
        weight_decay = opt_cfg.get("weight_decay", 0.0)
        opt_name = opt_cfg.get("name", "adam").lower()

        if opt_name == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Scheduler
        self.scheduler = build_lr_scheduler(self.optimizer, config.get("scheduler", {}))

        # AMP Grad Scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        # Early Stopping
        patience = train_cfg.get("early_stopping_patience", 10)
        monitor = train_cfg.get("early_stopping_monitor", "val_loss")
        mode = train_cfg.get("early_stopping_mode", "min")
        self.early_stopping = EarlyStopping(patience=patience, monitor=monitor, mode=mode)

        # Output Directories & Utilities
        model_name = config.get("model", {}).get("name", "model")
        if experiment_dir is None:
            experiment_dir = os.path.join("./outputs/experiments", model_name)
        self.experiment_dir = experiment_dir
        os.makedirs(self.experiment_dir, exist_ok=True)

        checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
        self.logger = TrainingLogger(output_dir=self.experiment_dir)

        self.best_val_dice = 0.0
        self.best_val_loss = float("inf")
        self.start_epoch = 1

        # Resource Monitoring CSV
        self.resource_log_file = os.path.join("./outputs/logs", "resource_usage.csv")
        os.makedirs(os.path.dirname(self.resource_log_file), exist_ok=True)

        # Handle Automatic Resume
        if self.resume:
            self._attempt_resume()

    def _attempt_resume(self):
        """Find and load the latest valid checkpoint to resume training seamlessly."""
        checkpoint_dir = self.checkpoint_manager.checkpoint_dir
        candidates = [
            os.path.join(checkpoint_dir, "checkpoint_latest.pth"),
            os.path.join(checkpoint_dir, "last_model.pth"),
            os.path.join(checkpoint_dir, "best_model.pth"),
        ]

        target_ckpt = None
        for cand in candidates:
            if os.path.exists(cand):
                target_ckpt = cand
                break

        if target_ckpt is not None:
            print(f"\n[RESUME DETECTED] Loading checkpoint state from: {target_ckpt}")
            try:
                ep, metrics, cfg, best_loss, best_dice, history = self.checkpoint_manager.load_checkpoint(
                    filepath=target_ckpt,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    device=self.device
                )

                self.start_epoch = ep + 1
                self.best_val_loss = best_loss if best_loss != float("inf") else metrics.get("val_loss", float("inf"))
                self.best_val_dice = best_dice if best_dice != 0.0 else metrics.get("val_dice", 0.0)

                print(f" -> Resuming seamlessly from Epoch {self.start_epoch}")
                print(f" -> Preserved Best Val Loss: {self.best_val_loss:.4f} | Best Val Dice: {self.best_val_dice:.4f}")
            except Exception as e:
                print(f"❌ Error restoring checkpoint ({e}). Starting fresh from Epoch 1.")
                self.start_epoch = 1
        else:
            print("\n[RESUME NOTICE] No valid checkpoint found in directory. Starting fresh from Epoch 1.")
            self.start_epoch = 1

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Execute one training epoch with gradient accumulation and memory safety.
        """
        self.model.train()
        running_loss = 0.0
        total_acc = 0.0
        n_batches = len(self.train_loader)
        accum_steps = max(self.gradient_accumulation_steps, 1)

        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} [Train]", leave=False)
        for i, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)
                    scaled_loss = loss / accum_steps
                self.scaler.scale(scaled_loss).backward()

                if (i + 1) % accum_steps == 0 or (i + 1) == n_batches:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, masks)
                scaled_loss = loss / accum_steps
                scaled_loss.backward()

                if (i + 1) % accum_steps == 0 or (i + 1) == n_batches:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            loss_val = loss.item()
            running_loss += loss_val

            with torch.no_grad():
                probs = torch.sigmoid(logits if not isinstance(logits, list) else logits[-1])
                acc = compute_accuracy(probs, masks)
                total_acc += acc

            pbar.set_postfix({"loss": f"{loss_val:.4f}", "acc": f"{acc:.4f}"})

            # Explicit tensor deletion for RAM safety
            del images, masks, logits, probs, loss, scaled_loss

        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        avg_loss = running_loss / max(n_batches, 1)
        avg_acc = total_acc / max(n_batches, 1)
        return avg_loss, avg_acc

    def validate(self) -> Dict[str, float]:
        """
        Execute validation loop with memory safety.
        """
        self.model.eval()
        running_loss = 0.0
        total_acc = 0.0
        total_dice = 0.0
        total_iou = 0.0
        n_batches = len(self.val_loader)

        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="[Validation]", leave=False):
                images = images.to(self.device)
                masks = masks.to(self.device)

                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        logits = self.model(images)
                        loss = self.criterion(logits, masks)
                else:
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)

                running_loss += loss.item()

                probs = torch.sigmoid(logits if not isinstance(logits, list) else logits[-1])

                acc = compute_accuracy(probs, masks)
                dice = compute_dice(probs, masks)
                iou = compute_iou(probs, masks)

                total_acc += acc
                total_dice += dice
                total_iou += iou

                del images, masks, logits, probs, loss

        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return {
            "val_loss": running_loss / max(n_batches, 1),
            "val_acc": total_acc / max(n_batches, 1),
            "val_dice": total_dice / max(n_batches, 1),
            "val_iou": total_iou / max(n_batches, 1),
        }

    def _log_resource_usage(self, epoch: int, duration_sec: float):
        """Log system RAM and CPU usage to CSV."""
        cpu_pct = psutil.cpu_percent() if psutil else 0.0
        ram_pct = psutil.virtual_memory().percent if psutil else 0.0

        write_header = not os.path.exists(self.resource_log_file)
        with open(self.resource_log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["epoch", "cpu_percent", "ram_percent", "duration_seconds", "timestamp"])
            writer.writerow([epoch, cpu_pct, ram_pct, round(duration_sec, 2), time.strftime("%Y-%m-%d %H:%M:%S")])

    def train(self) -> Dict[str, Any]:
        """
        Full multi-epoch training loop with crash safety, chunking, and checkpointing.
        """
        print(f"Starting training loop on device: {self.device}")
        print(f"Start Epoch: {self.start_epoch} | Target Max Epochs: {self.epochs}")
        print(f"Batch Size: {self.train_loader.batch_size} | Gradient Accumulation Steps: {self.gradient_accumulation_steps}")

        start_time = time.time()
        epochs_executed_this_run = 0

        for epoch in range(self.start_epoch, self.epochs + 1):
            # Check max epochs per run chunk limit
            if self.max_epochs_this_run is not None and epochs_executed_this_run >= self.max_epochs_this_run:
                print(f"\n[CHUNK LIMIT REACHED] Executed {epochs_executed_this_run} epoch(s) in this run. Stopping safely.")
                print(f"To continue training, run the command again with --resume.")
                break

            t0_epoch = time.time()
            train_loss, train_acc = self.train_epoch(epoch)
            val_metrics = self.validate()
            t1_epoch = time.time()
            epoch_duration = t1_epoch - t0_epoch

            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_metrics = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_metrics["val_loss"],
                "val_acc": val_metrics["val_acc"],
                "val_dice": val_metrics["val_dice"],
                "val_iou": val_metrics["val_iou"],
                "lr": current_lr,
            }

            self.logger.log_epoch(epoch, epoch_metrics)
            self._log_resource_usage(epoch, epoch_duration)
            epochs_executed_this_run += 1

            print(
                f"Epoch {epoch:03d}/{self.epochs:03d} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | Val Acc: {val_metrics['val_acc']:.4f} | "
                f"Val Dice: {val_metrics['val_dice']:.4f} | Val IoU: {val_metrics['val_iou']:.4f} | "
                f"Time: {epoch_duration:.1f}s"
            )

            # Check LR Scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()

            # ALWAYS Save Latest Checkpoint after EVERY completed epoch
            self.checkpoint_manager.save_checkpoint(
                filename="checkpoint_latest.pth",
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics=epoch_metrics,
                config=self.config,
                seed=self.config.get("project", {}).get("seed", 42),
                scheduler=self.scheduler,
                best_val_loss=self.best_val_loss,
                best_val_dice=self.best_val_dice,
            )
            self.checkpoint_manager.save_checkpoint(
                filename="last_model.pth",
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics=epoch_metrics,
                config=self.config,
                seed=self.config.get("project", {}).get("seed", 42),
                scheduler=self.scheduler,
                best_val_loss=self.best_val_loss,
                best_val_dice=self.best_val_dice,
            )

            # Save Best Model separately ONLY if validation performance improves
            if val_metrics["val_dice"] > self.best_val_dice:
                self.best_val_dice = val_metrics["val_dice"]
                self.best_val_loss = val_metrics["val_loss"]
                self.checkpoint_manager.save_checkpoint(
                    filename="best_model.pth",
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=epoch_metrics,
                    config=self.config,
                    seed=self.config.get("project", {}).get("seed", 42),
                    scheduler=self.scheduler,
                    best_val_loss=self.best_val_loss,
                    best_val_dice=self.best_val_dice,
                )
                print(f" [*] New Best Model Saved! Best Val Dice: {self.best_val_dice:.4f}")

            # Check Early Stopping
            monitor_val = val_metrics["val_loss"] if self.early_stopping.monitor == "val_loss" else val_metrics["val_dice"]
            if self.early_stopping(monitor_val, epoch):
                print(f"\n[EARLY STOPPING] {self.early_stopping.stopped_reason}")
                break

        elapsed_time = time.time() - start_time
        print(f"\nRun session completed in {elapsed_time:.2f} seconds.")
        print(f"Best Validation Dice: {self.best_val_dice:.4f} | Best Validation Loss: {self.best_val_loss:.4f}")

        self.logger.plot_history()
        final_summary = {
            "best_val_dice": self.best_val_dice,
            "best_val_loss": self.best_val_loss,
            "last_completed_epoch": epoch if 'epoch' in locals() else self.start_epoch - 1,
            "training_time_seconds": elapsed_time,
        }
        self.logger.save_final_metrics(final_summary)

        return final_summary
