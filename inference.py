"""
Inference Pipeline Module for Automated Tooth Segmentation.
Handles checkpoint loading, device detection, model prediction, thresholding, and output rendering.
"""

import os
import time
from typing import Dict, Any, Tuple, Optional, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from dataset.preprocessing import preprocess_image
from models import create_model
from evaluation.visualization import generate_overlay, generate_four_panel_visualization


class InferencePipeline:
    """
    Production-ready Inference Pipeline for dental panoramic X-rays.

    Args:
        checkpoint_path (str, optional): Path to .pth checkpoint file.
        model_name (str): Model architecture name.
        device (str): Device override ('auto', 'cuda', or 'cpu').
        config (dict, optional): Master configuration dictionary.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_name: str = "unetplusplus_eca",
        device: str = "auto",
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or {}
        self.model_name = model_name

        # Device Selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.checkpoint_loaded = False
        self.checkpoint_path = checkpoint_path

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_weights(checkpoint_path)
        else:
            # Build default model if no checkpoint path given
            self.model = create_model(self.model_name, self.config).to(self.device)
            self.model.eval()

    def load_weights(self, checkpoint_path: str):
        """
        Load weights into PyTorch model, restoring model architecture config from checkpoint if available.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and "config" in checkpoint and checkpoint["config"]:
            self.config = checkpoint["config"]
            if "model" in self.config and "name" in self.config["model"]:
                self.model_name = self.config["model"]["name"]

        # Re-instantiate model matching checkpoint architecture
        self.model = create_model(self.model_name, self.config).to(self.device)
        self.model.eval()

        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

        clean_state = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            clean_state[name] = v

        self.model.load_state_dict(clean_state)
        self.checkpoint_loaded = True
        self.checkpoint_path = checkpoint_path

    def predict(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        threshold: float = 0.5,
        target_size: Tuple[int, int] = (512, 512)
    ) -> Dict[str, Any]:
        """
        Execute prediction pipeline on panoramic X-ray image.

        Args:
            image_input: Input X-ray (path, numpy array, or PIL image).
            threshold: Probability threshold for binary mask (default 0.5).
            target_size: Preprocessing resolution (H, W).

        Returns:
            Dict containing outputs.
        """
        if isinstance(image_input, str):
            raw_img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
            if raw_img is None:
                raise ValueError(f"Could not read image from {image_input}")
        elif isinstance(image_input, Image.Image):
            raw_img = np.array(image_input.convert("L"))
        elif isinstance(image_input, np.ndarray):
            raw_img = image_input if image_input.ndim == 2 else cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        else:
            raise TypeError("Unsupported image input type.")

        orig_h, orig_w = raw_img.shape[:2]

        # Use image config target size if present
        img_cfg = self.config.get("image", {})
        proc_h = img_cfg.get("height", target_size[0])
        proc_w = img_cfg.get("width", target_size[1])

        # Preprocess image
        img_tensor = preprocess_image(raw_img, target_shape=(proc_h, proc_w), normalize=True).unsqueeze(0).to(self.device)

        # Model forward pass
        t0 = time.time()
        attention_weights = None
        with torch.no_grad():
            if hasattr(self.model, "use_eca") and self.model.use_eca:
                logits, att_weights = self.model(img_tensor, return_attention=True)
                if att_weights is not None:
                    attention_weights = att_weights.cpu().numpy().squeeze()
            else:
                logits = self.model(img_tensor)

            if isinstance(logits, list):
                logits = logits[-1]

            prob_tensor = torch.sigmoid(logits)
        t1 = time.time()
        inference_time = t1 - t0

        prob_map_2d = prob_tensor.squeeze().cpu().numpy()

        # Resize probability map back to original input image resolution
        if (orig_h, orig_w) != (proc_h, proc_w):
            prob_map_orig = cv2.resize(prob_map_2d, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        else:
            prob_map_orig = prob_map_2d

        binary_mask_orig = (prob_map_orig >= threshold).astype(np.uint8)
        overlay_orig = generate_overlay(raw_img, binary_mask_orig, color=(0, 255, 0), alpha=0.4)

        return {
            "original_image": raw_img,
            "probability_map": prob_map_orig,
            "binary_mask": binary_mask_orig,
            "overlay": overlay_orig,
            "inference_time": inference_time,
            "attention_weights": attention_weights,
            "original_shape": (orig_h, orig_w),
            "device": str(self.device),
        }
