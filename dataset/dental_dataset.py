"""
PyTorch Dataset class for Dental Panoramic Radiographs.
"""

import os
from typing import List, Tuple, Dict, Any, Optional
import torch
from torch.utils.data import Dataset

from .preprocessing import preprocess_image, preprocess_mask
from .augmentation import SynchronizedAugmenter


class DentalDataset(Dataset):
    """
    PyTorch Dataset for Tufts Dental Panoramic X-rays and binary tooth masks.

    Args:
        pairs (list): List of (image_path, mask_path) tuples.
        target_shape (tuple): Target image resolution (H, W).
        transform (SynchronizedAugmenter, optional): Augmentation module.
        is_train (bool): Whether dataset is in training mode.
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        target_shape: Tuple[int, int] = (512, 512),
        transform: Optional[SynchronizedAugmenter] = None,
        is_train: bool = False,
        cache_in_memory: bool = True
    ):
        self.pairs = pairs
        self.target_shape = target_shape
        self.transform = transform
        self.is_train = is_train
        self.cache_in_memory = cache_in_memory
        self._cache = {}

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache_in_memory and idx in self._cache:
            image_tensor, mask_tensor = self._cache[idx]
        else:
            img_path, mask_path = self.pairs[idx]
            image_tensor = preprocess_image(img_path, target_shape=self.target_shape, normalize=True)
            mask_tensor = preprocess_mask(mask_path, target_shape=self.target_shape, threshold=0.5)
            if self.cache_in_memory:
                self._cache[idx] = (image_tensor, mask_tensor)

        # Apply synchronized spatial / intensity augmentations if training
        if self.is_train and self.transform is not None:
            image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)

        return image_tensor, mask_tensor
