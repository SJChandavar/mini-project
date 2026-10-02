"""
Visualization utilities for Dental Segmentation: overlays, 4-panel figures, probability heatmaps, and ECA attention weights.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Optional, Union, Tuple


def generate_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
    show_contours: bool = True
) -> np.ndarray:
    """
    Generate composite overlay image combining grayscale X-ray and colored binary mask.

    Args:
        image: Grayscale X-ray image (H, W) or (H, W, 1) in [0, 255] or [0, 1].
        mask: Binary mask (H, W) in [0, 1] or [0, 255].
        color: RGB tuple for mask overlay (default: green (0, 255, 0)).
        alpha: Overlay opacity transparency factor.
        show_contours: Draw boundary contours if True.

    Returns:
        np.ndarray: RGB overlay image (H, W, 3) uint8.
    """
    img = image.copy()
    if img.max() <= 1.0:
        img = (img * 255.0).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    if img.ndim == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 1:
        img_rgb = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img.copy()

    msk = (mask >= 0.5).astype(np.uint8)

    color_mask = np.zeros_like(img_rgb)
    color_mask[msk == 1] = color

    overlay = cv2.addWeighted(img_rgb, 1.0, color_mask, alpha, 0)

    if show_contours:
        contours, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)

    return overlay


def generate_four_panel_visualization(
    image: np.ndarray,
    gt_mask: Optional[np.ndarray],
    pred_mask: np.ndarray,
    prob_map: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Tooth Segmentation Result"
) -> plt.Figure:
    """
    Generate 4-panel visual evaluation figure:
    Panel 1: Original X-ray
    Panel 2: Ground-truth tooth mask (or message if unavailable)
    Panel 3: Predicted tooth mask
    Panel 4: Composite overlay image
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Panel 1: Original Image
    img_disp = image.squeeze() if image.ndim == 3 else image
    axes[0].imshow(img_disp, cmap="gray")
    axes[0].set_title("1. Original Dental X-Ray")
    axes[0].axis("off")

    # Panel 2: GT Mask
    if gt_mask is not None:
        gt_disp = gt_mask.squeeze() if gt_mask.ndim == 3 else gt_mask
        axes[1].imshow(gt_disp, cmap="bone")
        axes[1].set_title("2. Ground-Truth Mask")
    else:
        axes[1].text(0.5, 0.5, "Ground Truth\nNot Available", ha="center", va="center", fontsize=12)
        axes[1].set_title("2. Ground-Truth Mask")
    axes[1].axis("off")

    # Panel 3: Predicted Binary Mask
    pred_disp = (pred_mask >= 0.5).squeeze() if pred_mask.ndim == 3 else (pred_mask >= 0.5)
    axes[2].imshow(pred_disp, cmap="bone")
    axes[2].set_title("3. Predicted Tooth Mask")
    axes[2].axis("off")

    # Panel 4: Composite Overlay
    overlay = generate_overlay(img_disp, pred_disp)
    axes[3].imshow(overlay)
    axes[3].set_title("4. Segmentation Overlay")
    axes[3].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def generate_attention_plot(
    attention_weights: torch.Tensor,
    save_path: str = "./outputs/reports/eca_attention.png"
):
    """
    Visualize ECA Bottleneck Channel Attention weights distribution.
    """
    if isinstance(attention_weights, torch.Tensor):
        weights = attention_weights.detach().cpu().numpy().squeeze()
    else:
        weights = np.array(attention_weights).squeeze()

    plt.figure(figsize=(10, 4))
    plt.plot(weights, color="teal", linewidth=1.5)
    plt.fill_between(range(len(weights)), weights, color="teal", alpha=0.3)
    plt.title("Attention Weight Visualization: ECA Channel Weights at Bottleneck (X_4,0)")
    plt.xlabel("Channel Index")
    plt.ylabel("Attention Weight (Sigmoid Output)")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
