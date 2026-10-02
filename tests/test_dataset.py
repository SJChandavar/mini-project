"""
Unit tests for Dataset loading, Preprocessing, and Augmentations.
"""

import numpy as np
import pytest
import torch
from dataset.preprocessing import preprocess_image, preprocess_mask
from dataset.augmentation import SynchronizedAugmenter


def test_preprocess_image():
    """Test image preprocessing outputs float32 tensor of shape [1, 256, 256]."""
    dummy_img = np.random.randint(0, 256, (400, 600), dtype=np.uint8)
    tensor = preprocess_image(dummy_img, target_shape=(256, 256), normalize=True)
    assert tensor.shape == (1, 256, 256)
    assert tensor.dtype == torch.float32
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0


def test_preprocess_mask():
    """Test mask preprocessing outputs binary tensor [1, 256, 256] with 0 or 1."""
    dummy_mask = np.zeros((400, 600), dtype=np.uint8)
    dummy_mask[100:200, 100:200] = 255
    tensor = preprocess_mask(dummy_mask, target_shape=(256, 256), threshold=0.5)
    assert tensor.shape == (1, 256, 256)
    assert tensor.dtype == torch.float32
    unique_vals = set(torch.unique(tensor).numpy())
    assert unique_vals.issubset({0.0, 1.0})


def test_synchronized_augmentation():
    """Test spatial transformations maintain shape and binary nature of mask."""
    img = torch.rand(1, 128, 128)
    mask = (torch.rand(1, 128, 128) > 0.5).float()
    aug = SynchronizedAugmenter({"enabled": True, "horizontal_flip": True, "rotation": 15})
    aug_img, aug_mask = aug(img, mask)

    assert aug_img.shape == (1, 128, 128)
    assert aug_mask.shape == (1, 128, 128)
    assert set(torch.unique(aug_mask).numpy()).issubset({0.0, 1.0})
