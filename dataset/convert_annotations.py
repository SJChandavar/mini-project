"""
Dataset Annotation Converter for DatasetNinja / Humans in the Loop Teeth Dataset.

Converts zlib base64 bitmap annotations into unified 2D binary tooth masks.
Saves X-rays to data/raw/images/ and binary masks to data/raw/tooth_masks/.
Generates conversion report CSV and 5 sample visualization figures.

Usage:
    python dataset/convert_annotations.py
"""

import os
import glob
import json
import base64
import zlib
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import sys
sys.path.append(os.path.abspath("."))
from evaluation.visualization import generate_overlay


def decode_datasetninja_bitmap(bitmap_dict: dict, full_height: int, full_width: int) -> np.ndarray:
    """
    Decode zlib-compressed base64 PNG bitmap string into sub-mask array placed at origin [x, y].
    """
    if not bitmap_dict or "data" not in bitmap_dict or "origin" not in bitmap_dict:
        return np.zeros((full_height, full_width), dtype=np.uint8)

    origin_x, origin_y = bitmap_dict["origin"]
    raw_b64 = bitmap_dict["data"]

    try:
        decomp = zlib.decompress(base64.b64decode(raw_b64))
        nparr = np.frombuffer(decomp, np.uint8)
        sub_mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if sub_mask is None:
            return np.zeros((full_height, full_width), dtype=np.uint8)

        sub_h, sub_w = sub_mask.shape
        full_mask = np.zeros((full_height, full_width), dtype=np.uint8)

        y1, y2 = max(0, origin_y), min(origin_y + sub_h, full_height)
        x1, x2 = max(0, origin_x), min(origin_x + sub_w, full_width)

        crop_y = y2 - y1
        crop_x = x2 - x1

        if crop_y > 0 and crop_x > 0:
            full_mask[y1:y2, x1:x2] = (sub_mask[:crop_y, :crop_x] > 0).astype(np.uint8)

        return full_mask
    except Exception:
        return np.zeros((full_height, full_width), dtype=np.uint8)


def convert_dataset(
    source_dir: str = "./data/fallback_raw/ds",
    out_img_dir: str = "./data/raw/images",
    out_mask_dir: str = "./data/raw/tooth_masks",
    report_path: str = "./outputs/reports/conversion_report.csv",
    vis_dir: str = "./outputs/reports/conversion_samples"
):
    """
    Convert all raw X-ray images and DatasetNinja annotations to raw data folder.
    """
    img_dir = os.path.join(source_dir, "img")
    ann_dir = os.path.join(source_dir, "ann")

    if not os.path.exists(img_dir) or not os.path.exists(ann_dir):
        print(f"❌ Source directory missing: {source_dir}")
        return

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) + glob.glob(os.path.join(img_dir, "*.png")))
    print(f"Found {len(img_files)} images to convert.")

    records = []

    for idx, img_path in enumerate(tqdm(img_files, desc="[Converting Annotations]")):
        base_filename = os.path.basename(img_path)
        ann_path = os.path.join(ann_dir, base_filename + ".json")

        if not os.path.exists(ann_path):
            records.append({
                "file_id": base_filename,
                "status": "missing_annotation",
                "teeth_count": 0,
                "fg_ratio": 0.0
            })
            continue

        # Load X-ray
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            records.append({
                "file_id": base_filename,
                "status": "corrupt_image",
                "teeth_count": 0,
                "fg_ratio": 0.0
            })
            continue

        h, w = img.shape[:2]

        with open(ann_path, "r") as f:
            ann_data = json.load(f)

        full_binary_mask = np.zeros((h, w), dtype=np.uint8)
        teeth_count = 0

        for obj in ann_data.get("objects", []):
            bm = obj.get("bitmap")
            if bm:
                sub_mask = decode_datasetninja_bitmap(bm, h, w)
                full_binary_mask = np.maximum(full_binary_mask, sub_mask)
                teeth_count += 1

        fg_ratio = float(np.sum(full_binary_mask > 0) / full_binary_mask.size)

        # Standardized output paths
        stem = os.path.splitext(base_filename)[0]
        out_img_path = os.path.join(out_img_dir, f"sample_{stem}.jpg")
        out_mask_path = os.path.join(out_mask_dir, f"sample_{stem}.png")

        # Save X-ray and binary mask (0 background, 255 foreground PNG)
        cv2.imwrite(out_img_path, img)
        cv2.imwrite(out_mask_path, (full_binary_mask * 255).astype(np.uint8))

        records.append({
            "file_id": stem,
            "image_path": out_img_path,
            "mask_path": out_mask_path,
            "height": h,
            "width": w,
            "teeth_count": teeth_count,
            "fg_ratio": fg_ratio,
            "status": "success"
        })

        # Generate 5 sample visualizations
        if idx < 5:
            overlay = generate_overlay(img, full_binary_mask, color=(0, 255, 0), alpha=0.4)
            vis_canvas = np.hstack([
                cv2.cvtColor(img, cv2.COLOR_GRAY2RGB),
                cv2.cvtColor((full_binary_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB),
                overlay
            ])
            cv2.imwrite(os.path.join(vis_dir, f"sample_{stem}_vis.png"), cv2.cvtColor(vis_canvas, cv2.COLOR_RGB2BGR))

    df_rep = pd.DataFrame(records)
    df_rep.to_csv(report_path, index=False)
    print(f"\nConversion finished! Report saved to {report_path}")
    print(f"Total successfully converted: {(df_rep['status'] == 'success').sum()} pairs.")


if __name__ == "__main__":
    convert_dataset()
