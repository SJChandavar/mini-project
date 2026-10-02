"""
Synchronized Data Augmentation pipeline for image and mask pairs.
"""

import random
import cv2
import numpy as np
import torch
from typing import Tuple, Dict, Any


class SynchronizedAugmenter:
    """
    Applies synchronized spatial transformations to both image and mask,
    and optional intensity transformations strictly to the image.
    """

    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {}
        self.enabled = config.get("enabled", True)
        self.horizontal_flip = config.get("horizontal_flip", True)
        self.rotation_deg = config.get("rotation", 10)
        self.brightness = config.get("brightness", 0.15)
        self.contrast = config.get("contrast", 0.15)
        self.gaussian_noise = config.get("gaussian_noise", 0.05)

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image (torch.Tensor): Tensor [1, H, W] in range [0, 1]
            mask (torch.Tensor): Tensor [1, H, W] binary [0, 1]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Augmented (image, mask)
        """
        if not self.enabled:
            return image, mask

        img_np = image.squeeze(0).numpy().copy()
        msk_np = mask.squeeze(0).numpy().copy()

        h, w = img_np.shape

        # 1. Synchronized Horizontal Flip
        if self.horizontal_flip and random.random() > 0.5:
            img_np = cv2.flip(img_np, 1)
            msk_np = cv2.flip(msk_np, 1)

        # 2. Synchronized Rotation and Scaling
        if self.rotation_deg > 0 and random.random() > 0.5:
            angle = random.uniform(-self.rotation_deg, self.rotation_deg)
            scale = random.uniform(0.95, 1.05)
            center = (w / 2.0, h / 2.0)
            M = cv2.getRotationMatrix2D(center, angle, scale)

            img_np = cv2.warpAffine(img_np, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            msk_np = cv2.warpAffine(msk_np, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # 3. Image-Only Brightness & Contrast Adjustment
        if self.brightness > 0 and random.random() > 0.5:
            brightness_factor = random.uniform(1.0 - self.brightness, 1.0 + self.brightness)
            img_np = img_np * brightness_factor

        if self.contrast > 0 and random.random() > 0.5:
            contrast_factor = random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
            mean_val = img_np.mean()
            img_np = (img_np - mean_val) * contrast_factor + mean_val

        # 4. Image-Only Gaussian Noise
        if self.gaussian_noise > 0 and random.random() > 0.5:
            noise = np.random.normal(0, self.gaussian_noise, img_np.shape).astype(np.float32)
            img_np = img_np + noise

        # Clip image back to [0, 1]
        img_np = np.clip(img_np, 0.0, 1.0).astype(np.float32)
        msk_np = (msk_np >= 0.5).astype(np.float32)

        return torch.from_numpy(img_np).unsqueeze(0), torch.from_numpy(msk_np).unsqueeze(0)
