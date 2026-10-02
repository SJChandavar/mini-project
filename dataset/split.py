"""
Dataset splitting module with deterministic seeding and leakage prevention.
"""

import os
import random
import pandas as pd
from typing import List, Tuple, Dict, Any


def split_dataset(
    matched_pairs: List[Tuple[str, str]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    save_path: str = "./outputs/reports/dataset_split.csv",
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Split matched image-mask pairs into train, val, test subsets reproducibly.

    Args:
        matched_pairs: List of (image_path, mask_path) tuples.
        train_ratio (float): Fraction of dataset for training (default: 0.70).
        val_ratio (float): Fraction of dataset for validation (default: 0.15).
        test_ratio (float): Fraction of dataset for testing (default: 0.15).
        seed (int): Random seed.
        save_path (str): CSV filepath to save split breakdown.

    Returns:
        Tuple[train_pairs, val_pairs, test_pairs]
    """
    if not matched_pairs:
        return [], [], []

    # Normalize split ratios to sum to 1.0
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total

    # Shuffle deterministically
    pairs_copy = list(matched_pairs)
    rng = random.Random(seed)
    rng.shuffle(pairs_copy)

    n_total = len(pairs_copy)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_pairs = pairs_copy[:n_train]
    val_pairs = pairs_copy[n_train:n_train + n_val]
    test_pairs = pairs_copy[n_train + n_val:]

    # Save breakdown report to CSV
    records = []
    for img, msk in train_pairs:
        records.append({"image_path": img, "mask_path": msk, "split": "train"})
    for img, msk in val_pairs:
        records.append({"image_path": img, "mask_path": msk, "split": "val"})
    for img, msk in test_pairs:
        records.append({"image_path": img, "mask_path": msk, "split": "test"})

    df = pd.DataFrame(records)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)

    return train_pairs, val_pairs, test_pairs
