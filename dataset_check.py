"""
Dataset Validation CLI Script for Dental Panoramic Radiograph Dataset.

Usage:
    python dataset_check.py [--config config.yaml] [--data-dir ./data/raw]
"""

import argparse
import sys
import yaml
from dataset.validators import scan_dataset_directory


def main():
    parser = argparse.ArgumentParser(
        description="Check and validate dental panoramic radiograph dataset integrity."
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration YAML file.")
    parser.add_argument("--data-dir", type=str, default=None, help="Root raw dataset directory override.")
    parser.add_argument("--images-dir", type=str, default=None, help="Images directory override.")
    parser.add_argument("--masks-dir", type=str, default=None, help="Tooth masks directory override.")
    args = parser.parse_args()

    # Load configuration
    try:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception:
        cfg = {}

    ds_cfg = cfg.get("dataset", {})
    root_dir = args.data_dir or ds_cfg.get("root", "./data/raw")
    images_dir = args.images_dir or ds_cfg.get("images", "./data/raw/images")
    masks_dir = args.masks_dir or ds_cfg.get("tooth_masks", "./data/raw/tooth_masks")

    print("==========================================================")
    print(" DENTAL PANORAMIC RADIOGRAPH DATASET INTEGRITY CHECK")
    print("==========================================================")
    print(f"Dataset Root Dir : {root_dir}")
    print(f"Images Dir       : {images_dir}")
    print(f"Tooth Masks Dir  : {masks_dir}\n")

    res = scan_dataset_directory(images_dir, masks_dir)

    if res.get("error"):
        print(f"[FAIL] DATASET CHECK FAILED: {res['error']}")
        print("\nSETUP INSTRUCTIONS:")
        print("1. Place your panoramic X-rays in: ./data/raw/images/")
        print("2. Place the matching tooth segmentation masks in: ./data/raw/tooth_masks/")
        print("3. Ensure filenames match (e.g., 'sample_01.png' and 'sample_01.png')")
        sys.exit(1)

    print(f"Total Raw Images Found  : {res['total_images']}")
    print(f"Total Raw Masks Found   : {res['total_masks']}")
    print(f"Successfully Matched    : {res['matched_count']} pairs")
    print(f"Unmatched Images        : {len(res['unmatched_images'])}")
    print(f"Unmatched Masks         : {len(res['unmatched_masks'])}")
    print(f"Corrupted File Pairs    : {len(res['corrupted_files'])}")
    print(f"Average Foreground %    : {res['avg_foreground_ratio'] * 100:.2f}%\n")

    if res["unmatched_images"]:
        print("[WARN] Unmatched Image Samples:")
        for img in res["unmatched_images"][:5]:
            print(f"  - {img}")

    if res["unmatched_masks"]:
        print("[WARN] Unmatched Mask Samples:")
        for msk in res["unmatched_masks"][:5]:
            print(f"  - {msk}")

    if res["corrupted_files"]:
        print("[FAIL] Corrupted / Invalid File Samples:")
        for img, msk, errs in res["corrupted_files"][:5]:
            print(f"  - Pair ({img}, {msk}): {errs}")

    if res["matched_count"] > 0 and len(res["corrupted_files"]) == 0:
        print("[SUCCESS] DATASET VALIDATION PASSED SUCCESSFULLY!")
    else:
        print("[WARN] DATASET VALIDATION COMPLETED WITH WARNINGS.")


if __name__ == "__main__":
    main()
