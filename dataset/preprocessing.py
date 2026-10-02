"""
Image and Mask Preprocessing utilities for Dental Panoramic X-rays.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from typing import Tuple, Union


def preprocess_image(
    image_input: Union[str, np.ndarray, Image.Image],
    target_shape: Tuple[int, int] = (512, 512),
    normalize: bool = True
) -> torch.Tensor:
    """
    Preprocess dental X-ray image:
    1. Read / convert to numpy array
    2. Convert to grayscale
    3. Resize with bilinear interpolation
    4. Normalize pixel values to [0, 1]
    5. Convert to PyTorch float32 tensor of shape [1, H, W]

    Args:
        image_input: File path, numpy array, or PIL Image.
        target_shape (tuple): Target (height, width).
        normalize (bool): Scale pixel intensities to range [0, 1].

    Returns:
        torch.Tensor: Preprocessed tensor of shape [1, H, W]
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image from path: {image_input}")
    elif isinstance(image_input, Image.Image):
        img = np.array(image_input.convert("L"))
    elif isinstance(image_input, np.ndarray):
        img = image_input
        if img.ndim == 3:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            elif img.shape[2] == 1:
                img = img[:, :, 0]
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    target_h, target_w = target_shape
    if (img.shape[0], img.shape[1]) != (target_h, target_w):
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    img = img.astype(np.float32)

    if normalize:
        if img.max() > 1.0:
            img = img / 255.0

    tensor = torch.from_numpy(img).unsqueeze(0)  # Shape: [1, H, W]
    return tensor


def preprocess_mask(
    mask_input: Union[str, np.ndarray, Image.Image],
    target_shape: Tuple[int, int] = (512, 512),
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Preprocess tooth segmentation ground truth mask:
    1. Read / convert to numpy array
    2. Convert to grayscale
    3. Resize using NEAREST-NEIGHBOR interpolation to prevent boundary blurring
    4. Threshold to binary (0 background, 1 foreground)
    5. Convert to PyTorch float32 tensor of shape [1, H, W]

    Args:
        mask_input: File path, numpy array, or PIL Image.
        target_shape (tuple): Target (height, width).
        threshold (float): Binarization threshold.

    Returns:
        torch.Tensor: Preprocessed binary tensor of shape [1, H, W]
    """
    if isinstance(mask_input, str):
        mask = cv2.imread(mask_input, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask from path: {mask_input}")
    elif isinstance(mask_input, Image.Image):
        mask = np.array(mask_input.convert("L"))
    elif isinstance(mask_input, np.ndarray):
        mask = mask_input
        if mask.ndim == 3:
            mask = mask[:, :, 0]
    else:
        raise TypeError(f"Unsupported mask input type: {type(mask_input)}")

    target_h, target_w = target_shape
    if (mask.shape[0], mask.shape[1]) != (target_h, target_w):
        # CRITICAL: Always use NEAREST neighbor interpolation for segmentation masks
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    mask = mask.astype(np.float32)
    max_val = mask.max()
    if max_val > 1.0:
        mask = mask / 255.0

    binary_mask = (mask >= threshold).astype(np.float32)
    tensor = torch.from_numpy(binary_mask).unsqueeze(0)  # Shape: [1, H, W]
    return tensor
