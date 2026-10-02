"""
Evaluation CLI Script for evaluating trained Tooth Segmentation checkpoints.

Usage:
    python evaluate.py --model unetplusplus_eca --checkpoint outputs/experiments/unetplusplus_eca/checkpoints/best_model.pth --split test
"""

import argparse
import os
import sys
import json
import yaml
import torch
from torch.utils.data import DataLoader

from dataset.validators import scan_dataset_directory
from dataset.split import split_dataset
from dataset.dental_dataset import DentalDataset
from models import create_model
from evaluation.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained tooth segmentation model checkpoint.")
    parser.add_argument("--model", type=str, default="unetplusplus_eca", help="Model architecture name.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained .pth checkpoint.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML file.")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"], help="Data split to evaluate.")
    parser.add_argument("--device", type=str, default="auto", help="Device ('auto', 'cuda', 'cpu').")
    args = parser.parse_args()

    # Determine Checkpoint Path
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_path = os.path.join("./outputs/experiments", args.model, "checkpoints", "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint file not found at: {checkpoint_path}")
        print("Please train a model first using: python train.py")
        sys.exit(1)

    # Device Selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("==========================================================")
    print(f" EVALUATING MODEL CHECKPOINT: {args.model}")
    print("==========================================================")
    print(f"Checkpoint Path : {checkpoint_path}")
    print(f"Evaluation Split: {args.split}")
    print(f"Device          : {device}\n")

    # Load Config
    config_path = args.config if os.path.exists(args.config) else "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Scan dataset & split
    ds_cfg = config.get("dataset", {})
    images_dir = ds_cfg.get("images", "./data/raw/images")
    masks_dir = ds_cfg.get("tooth_masks", "./data/raw/tooth_masks")

    scan_res = scan_dataset_directory(images_dir, masks_dir)
    matched_pairs = scan_res.get("matched_pairs", [])

    if not matched_pairs:
        print("❌ ERROR: No dataset pairs found to evaluate.")
        sys.exit(1)

    split_cfg = config.get("split", {})
    train_pairs, val_pairs, test_pairs = split_dataset(
        matched_pairs,
        train_ratio=split_cfg.get("train", 0.70),
        val_ratio=split_cfg.get("validation", 0.15),
        test_ratio=split_cfg.get("test", 0.15),
        seed=config.get("project", {}).get("seed", 42)
    )

    split_map = {"train": train_pairs, "val": val_pairs, "test": test_pairs}
    eval_pairs = split_map[args.split]

    print(f"Evaluating {len(eval_pairs)} sample pairs on '{args.split}' split.")

    img_cfg = config.get("image", {})
    target_shape = (img_cfg.get("height", 512), img_cfg.get("width", 512))
    eval_ds = DentalDataset(eval_pairs, target_shape=target_shape, is_train=False)
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False)

    # Load Weights and Restore Config
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "config" in ckpt and ckpt["config"]:
        config = ckpt["config"]
        if "model" in config and "name" in config["model"]:
            args.model = config["model"]["name"]

    model = create_model(args.model, config)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    clean_state = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(clean_state)

    # Run Evaluator
    evaluator = Evaluator(model=model, test_loader=eval_loader, device=device, threshold=0.5)
    metrics = evaluator.evaluate(save_predictions_dir="./outputs/predictions")

    print("\n----------------------------------------------------------")
    print(" EVALUATION RESULTS SUMMARY")
    print("----------------------------------------------------------")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:28s}: {v:.4f}")
        else:
            print(f"  {k:28s}: {v}")
    print("----------------------------------------------------------\n")
    print("Results saved to: outputs/reports/test_metrics.json")


if __name__ == "__main__":
    main()
