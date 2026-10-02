"""
Main Training CLI Script for Tooth Segmentation models.

Usage:
    python train.py --model unetplusplus_eca --config configs/default.yaml
"""

import argparse
import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader

from dataset.validators import scan_dataset_directory
from dataset.split import split_dataset
from dataset.dental_dataset import DentalDataset
from dataset.augmentation import SynchronizedAugmenter
from models import create_model
from training.trainer import Trainer


def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train deep learning model for tooth segmentation.")
    parser.add_argument("--model", type=str, default="unetplusplus_eca", help="Model name (e.g., unetplusplus_eca, unetplusplus, unet, fcn, enet, unet3plus, swiftnet).")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch count override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override.")
    parser.add_argument("--device", type=str, default="auto", help="Device override ('auto', 'cuda', 'cpu').")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest valid checkpoint.")
    parser.add_argument("--max-epochs-this-run", type=int, default=None, help="Maximum epochs to train in this execution session.")
    args = parser.parse_args()

    # Load configuration file
    config_path = args.config
    if not os.path.exists(config_path):
        config_path = "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.model:
        config.setdefault("model", {})["name"] = args.model
    if args.epochs:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr:
        config.setdefault("optimizer", {})["learning_rate"] = args.lr

    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("==========================================================")
    print(f" TRAINING TOOTH SEGMENTATION MODEL: {config['model']['name']}")
    print("==========================================================")
    print(f"Python Version    : {sys.version.split()[0]}")
    print(f"PyTorch Version   : {torch.__version__}")
    print(f"CUDA Available    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device Name   : {torch.cuda.get_device_name(0)}")
    print(f"Selected Device   : {device}")
    print(f"Configuration     : {config_path}")
    print(f"Random Seed       : {seed}\n")

    # Scan dataset
    ds_cfg = config.get("dataset", {})
    images_dir = ds_cfg.get("images", "./data/raw/images")
    masks_dir = ds_cfg.get("tooth_masks", "./data/raw/tooth_masks")

    scan_res = scan_dataset_directory(images_dir, masks_dir)
    matched_pairs = scan_res.get("matched_pairs", [])

    if not matched_pairs:
        print("[ERROR] No matched image-mask pairs were found in dataset directories!")
        print("\nSETUP INSTRUCTIONS:")
        print(f"1. Place your panoramic X-ray images in: {images_dir}")
        print(f"2. Place your tooth segmentation masks in: {masks_dir}")
        print("3. Run: python dataset_check.py")
        sys.exit(1)

    print(f"Found {len(matched_pairs)} matched image-mask pairs.")

    # Split dataset
    split_cfg = config.get("split", {})
    train_pairs, val_pairs, test_pairs = split_dataset(
        matched_pairs,
        train_ratio=split_cfg.get("train", 0.70),
        val_ratio=split_cfg.get("validation", 0.15),
        test_ratio=split_cfg.get("test", 0.15),
        seed=seed,
        save_path="./outputs/reports/dataset_split.csv"
    )

    print(f"Dataset Split -> Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}\n")

    # Build Datasets and DataLoaders
    img_cfg = config.get("image", {})
    target_shape = (img_cfg.get("height", 512), img_cfg.get("width", 512))

    augmenter = SynchronizedAugmenter(config.get("augmentation", {}))

    train_ds = DentalDataset(train_pairs, target_shape=target_shape, transform=augmenter, is_train=True)
    val_ds = DentalDataset(val_pairs, target_shape=target_shape, transform=None, is_train=False)

    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 4)
    num_workers = train_cfg.get("num_workers", 0)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device == "cuda")
    )

    # Build Model
    model = create_model(config["model"]["name"], config)

    # Experiment Directory
    exp_dir = os.path.join("./outputs/experiments", config["model"]["name"])

    # Instantiate Trainer & Execute Training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        experiment_dir=exp_dir,
        resume=args.resume,
        max_epochs_this_run=args.max_epochs_this_run,
    )

    trainer.train()


if __name__ == "__main__":
    main()
