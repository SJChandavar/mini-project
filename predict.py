"""
Single Image Prediction CLI Script for Tooth Segmentation.

Usage:
    python predict.py --image path/to/xray.png [--model unetplusplus_eca] [--threshold 0.5]
"""

import argparse
import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

from inference import InferencePipeline


def main():
    parser = argparse.ArgumentParser(description="Predict tooth segmentation mask for a single dental panoramic X-ray.")
    parser.add_argument("--image", type=str, required=True, help="Path to input panoramic X-ray image.")
    parser.add_argument("--model", type=str, default="unetplusplus_eca", help="Model architecture name.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Segmentation probability threshold.")
    parser.add_argument("--output-dir", type=str, default="./outputs/predictions", help="Directory to save predictions.")
    parser.add_argument("--device", type=str, default="auto", help="Device ('auto', 'cuda', 'cpu').")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] Input image file does not exist: {args.image}")
        sys.exit(1)

    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_path = os.path.join("./outputs/experiments", args.model, "checkpoints", "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print(f"[WARN] Specified checkpoint not found at: {checkpoint_path}")
        print("Running prediction in uninitialized model weights mode for demonstration.")

    pipeline = InferencePipeline(
        checkpoint_path=checkpoint_path if os.path.exists(checkpoint_path) else None,
        model_name=args.model,
        device=args.device
    )

    print("==========================================================")
    print(" TOOTH SEGMENTATION INFERENCE")
    print("==========================================================")
    print(f"Input Image     : {args.image}")
    print(f"Model           : {args.model}")
    print(f"Threshold       : {args.threshold}")
    print(f"Device          : {pipeline.device}\n")

    res = pipeline.predict(args.image, threshold=args.threshold)

    # Prepare Save Outputs
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.image))[0]

    mask_path = os.path.join(args.output_dir, f"{base_name}_mask.png")
    overlay_path = os.path.join(args.output_dir, f"{base_name}_overlay.png")
    prob_path = os.path.join(args.output_dir, f"{base_name}_probability.png")

    # Save Mask (0, 255)
    cv2.imwrite(mask_path, (res["binary_mask"] * 255).astype(np.uint8))

    # Save Overlay RGB -> BGR for OpenCV write
    overlay_bgr = cv2.cvtColor(res["overlay"], cv2.COLOR_RGB2BGR)
    cv2.imwrite(overlay_path, overlay_bgr)

    # Save Probability Map as Colormap
    prob_colored = cv2.applyColorMap((res["probability_map"] * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(prob_path, prob_colored)

    print("Prediction completed successfully!\n")
    print("Outputs:")
    print(f"  - Mask        : {mask_path}")
    print(f"  - Overlay     : {overlay_path}")
    print(f"  - Probability : {prob_path}")
    print(f"\nInference Time: {res['inference_time']:.4f} seconds")


if __name__ == "__main__":
    main()
