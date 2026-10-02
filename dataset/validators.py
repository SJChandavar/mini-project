"""
Dataset validation helpers for checking raw images and ground truth masks.
"""

import os
import glob
import cv2
import numpy as np
from typing import Dict, List, Any, Tuple


def is_image_corrupted(file_path: str) -> bool:
    """Check if image file exists and can be decoded by OpenCV."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return True
    try:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        return img is None or img.size == 0
    except Exception:
        return True


def get_image_info(file_path: str) -> Tuple[Tuple[int, int], float]:
    """Return image dimensions (height, width) and percentage of foreground pixels if binary mask."""
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {file_path}")
    h, w = img.shape
    fg_ratio = float((img > 127).sum() / img.size)
    return (h, w), fg_ratio


def validate_dataset_pair(image_path: str, mask_path: str) -> Dict[str, Any]:
    """Validate alignment and integrity of single image/mask pair."""
    res = {
        "valid": True,
        "image_path": image_path,
        "mask_path": mask_path,
        "errors": []
    }
    if is_image_corrupted(image_path):
        res["valid"] = False
        res["errors"].append(f"Corrupted or unreadable image: {image_path}")
    if is_image_corrupted(mask_path):
        res["valid"] = False
        res["errors"].append(f"Corrupted or unreadable mask: {mask_path}")

    if res["valid"]:
        (img_h, img_w), _ = get_image_info(image_path)
        (msk_h, msk_w), fg_pct = get_image_info(mask_path)
        if (img_h, img_w) != (msk_h, msk_w):
            res["valid"] = False
            res["errors"].append(f"Dimension mismatch: Image ({img_h}x{img_w}) vs Mask ({msk_h}x{msk_w})")
        res["dimensions"] = (img_h, img_w)
        res["foreground_ratio"] = fg_pct

    return res


def scan_dataset_directory(
    images_dir: str,
    masks_dir: str,
    valid_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
) -> Dict[str, Any]:
    """
    Scan image and mask directories, finding matching pairs by stem name.
    """
    if not os.path.exists(images_dir):
        return {
            "error": f"Images directory does not exist: {images_dir}",
            "total_images": 0,
            "total_masks": 0,
            "matched_pairs": [],
            "unmatched_images": [],
            "unmatched_masks": [],
            "corrupted_files": []
        }

    if not os.path.exists(masks_dir):
        return {
            "error": f"Masks directory does not exist: {masks_dir}",
            "total_images": 0,
            "total_masks": 0,
            "matched_pairs": [],
            "unmatched_images": [],
            "unmatched_masks": [],
            "corrupted_files": []
        }

    image_files = {}
    for ext in valid_extensions:
        for p in glob.glob(os.path.join(images_dir, f"*{ext}")):
            stem = os.path.splitext(os.path.basename(p))[0]
            image_files[stem] = p

    mask_files = {}
    for ext in valid_extensions:
        for p in glob.glob(os.path.join(masks_dir, f"*{ext}")):
            stem = os.path.splitext(os.path.basename(p))[0]
            # Strip common mask suffixes like _mask or _tooth
            clean_stem = stem.replace("_mask", "").replace("_tooth", "")
            mask_files[clean_stem] = p

    image_stems = set(image_files.keys())
    mask_stems = set(mask_files.keys())

    matched_stems = image_stems.intersection(mask_stems)
    unmatched_images = [image_files[s] for s in (image_stems - matched_stems)]
    unmatched_masks = [mask_files[s] for s in (mask_stems - matched_stems)]

    matched_pairs = []
    corrupted_files = []
    fg_ratios = []

    for stem in sorted(matched_stems):
        img_p = image_files[stem]
        msk_p = mask_files[stem]
        val_res = validate_dataset_pair(img_p, msk_p)
        if val_res["valid"]:
            matched_pairs.append((img_p, msk_p))
            fg_ratios.append(val_res["foreground_ratio"])
        else:
            corrupted_files.append((img_p, msk_p, val_res["errors"]))

    return {
        "error": None,
        "total_images": len(image_files),
        "total_masks": len(mask_files),
        "matched_count": len(matched_pairs),
        "matched_pairs": matched_pairs,
        "unmatched_images": unmatched_images,
        "unmatched_masks": unmatched_masks,
        "corrupted_files": corrupted_files,
        "avg_foreground_ratio": float(np.mean(fg_ratios)) if fg_ratios else 0.0,
    }
