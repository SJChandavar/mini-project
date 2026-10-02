"""
Verification Test Script for Resumable Training & Recovery System.
Executes controlled 2-epoch run, verifies checkpoint_latest.pth creation,
stops, resumes from epoch 3, and verifies history continuation.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
import torch

from models import create_model
from training.trainer import Trainer
from dataset.validators import scan_dataset_directory
from dataset.split import split_dataset
from dataset.dental_dataset import DentalDataset
import yaml


def run_resume_test():
    print("==========================================================")
    print(" STARTING RESUME SYSTEM VERIFICATION TEST")
    print("==========================================================")

    test_exp_dir = "./outputs/experiments/resume_test"
    if os.path.exists(test_exp_dir):
        import shutil
        shutil.rmtree(test_exp_dir)
    os.makedirs(test_exp_dir, exist_ok=True)

    with open("configs/safe_cpu.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["training"]["epochs"] = 4
    config["training"]["batch_size"] = 1
    config["training"]["gradient_accumulation_steps"] = 4

    ds_cfg = config.get("dataset", {})
    scan_res = scan_dataset_directory(ds_cfg["images"], ds_cfg["tooth_masks"])
    matched_pairs = scan_res["matched_pairs"]

    train_pairs, val_pairs, _ = split_dataset(
        matched_pairs,
        train_ratio=config["split"]["train"],
        val_ratio=config["split"]["validation"],
        test_ratio=config["split"]["test"],
        seed=42
    )

    img_shape = (config["image"]["height"], config["image"]["width"])
    train_ds = DentalDataset(train_pairs[:20], target_shape=img_shape, transform=None, is_train=True, cache_in_memory=True)
    val_ds = DentalDataset(val_pairs[:10], target_shape=img_shape, transform=None, is_train=False, cache_in_memory=True)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # -------------------------------------------------------------
    # PHASE 1: Fresh 2-Epoch Run
    # -------------------------------------------------------------
    print("\n--- PHASE 1: Fresh Training (2 Epochs Limit) ---")
    model1 = create_model("unetplusplus_eca", config)
    trainer1 = Trainer(
        model=model1,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device="cpu",
        experiment_dir=test_exp_dir,
        resume=False,
        max_epochs_this_run=2,
    )

    trainer1.train()

    latest_ckpt_path = os.path.join(test_exp_dir, "checkpoints", "checkpoint_latest.pth")
    hist_csv_path = os.path.join(test_exp_dir, "training_history.csv")

    assert os.path.exists(latest_ckpt_path), "❌ checkpoint_latest.pth was not created!"
    ckpt1 = torch.load(latest_ckpt_path, map_location="cpu", weights_only=False)
    print(f"\n[OK] Phase 1 Checkpoint Verified: Last Epoch = {ckpt1['epoch']}")
    assert ckpt1["epoch"] == 2, f"Expected epoch 2, got {ckpt1['epoch']}"

    df_hist1 = pd.read_csv(hist_csv_path)
    print(f"[OK] Phase 1 History CSV Verified: {len(df_hist1)} rows")

    # -------------------------------------------------------------
    # PHASE 2: Resume Run (Next 2 Epochs -> Epochs 3 and 4)
    # -------------------------------------------------------------
    print("\n--- PHASE 2: Resume Training (From Epoch 3) ---")
    model2 = create_model("unetplusplus_eca", config)
    trainer2 = Trainer(
        model=model2,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device="cpu",
        experiment_dir=test_exp_dir,
        resume=True,
        max_epochs_this_run=2,
    )

    assert trainer2.start_epoch == 3, f"❌ Expected start_epoch == 3, got {trainer2.start_epoch}"
    print(f"[OK] Resume Detection Verified: Next Epoch == {trainer2.start_epoch}")

    trainer2.train()

    ckpt2 = torch.load(latest_ckpt_path, map_location="cpu", weights_only=False)
    print(f"\n[OK] Phase 2 Checkpoint Verified: Final Epoch = {ckpt2['epoch']}")
    assert ckpt2["epoch"] == 4, f"Expected epoch 4, got {ckpt2['epoch']}"

    df_hist2 = pd.read_csv(hist_csv_path)
    print(f"[OK] Phase 2 History CSV Verified: {len(df_hist2)} rows (Epochs: {list(df_hist2['epoch'])})")
    assert len(df_hist2) == 4, f"Expected 4 history rows, got {len(df_hist2)}"

    print("\n==========================================================")
    print(" [OK] ALL RESUME SYSTEM TESTS PASSED SUCCESSFULLY!")
    print("==========================================================")


if __name__ == "__main__":
    run_resume_test()
