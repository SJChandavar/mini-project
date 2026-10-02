"""
Streamlit Web Application for Automated Tooth Segmentation in Dental X-rays.

Run with:
    streamlit run app.py
"""

import os
import json
import time
import glob
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from inference import InferencePipeline
from dataset.preprocessing import preprocess_mask
from metrics import (
    compute_dice,
    compute_iou,
    compute_accuracy,
    compute_fscore,
    compute_psnr,
    compute_map,
)
from evaluation.visualization import generate_overlay, generate_attention_plot
from evaluation.report_generator import generate_architecture_diagram

# Page Configuration
st.set_page_config(
    page_title="Automated Tooth Segmentation | ECA U-Net++",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.3rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0rem; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 1.5rem; }
    .metric-card { background-color: #F3F4F6; padding: 1rem; border-radius: 8px; border-left: 5px solid #3B82F6; }
    .disclaimer-box { background-color: #FEF3C7; padding: 1rem; border-radius: 8px; border: 1px solid #F59E0B; }
</style>
""", unsafe_allow_html=True)


def find_available_checkpoints() -> list:
    """Scan outputs directory for trained .pth checkpoints, prioritizing ECA-U-Net++ best checkpoint."""
    ckpts = glob.glob("./outputs/**/*.pth", recursive=True)
    preferred = "./outputs/experiments/unetplusplus_eca/checkpoints/best_model.pth"
    preferred_norm = os.path.normpath(preferred)
    sorted_ckpts = []
    for c in ckpts:
        if os.path.normpath(c) == preferred_norm:
            sorted_ckpts.insert(0, c)
        else:
            sorted_ckpts.append(c)
    return sorted_ckpts


def main():
    st.markdown("<div class='main-header'>Automated Tooth Segmentation in Dental X-rays</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Attention-Integrated U-Net++ with Efficient Channel Attention (ECA-Net)</div>", unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.header("⚙️ Model & Inference Configuration")

    model_options = [
        "unetplusplus_eca",
        "unetplusplus",
        "unet",
        "fcn",
        "enet",
        "unet3plus",
        "swiftnet",
    ]
    selected_model = st.sidebar.selectbox("Select Model Architecture", model_options, index=0)

    available_ckpts = find_available_checkpoints()
    if available_ckpts:
        selected_ckpt = st.sidebar.selectbox("Select Checkpoint", available_ckpts, index=0)
    else:
        st.sidebar.warning("No .pth checkpoints detected in outputs/ directory.")
        selected_ckpt = None

    manual_ckpt = st.sidebar.text_input("Or Enter Custom Checkpoint Path:", value=selected_ckpt or "")

    threshold = st.sidebar.slider("Segmentation Probability Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

    device_option = st.sidebar.radio("Compute Device", ["auto", "cpu", "cuda"], index=0)

    target_res = st.sidebar.selectbox("Processing Resolution", [512, 256], index=0)

    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Project Information**\nResearch Implementation of Attention-Integrated U-Net++ for Medical Image Segmentation.")

    # Instantiate Inference Pipeline
    active_ckpt = manual_ckpt if manual_ckpt and os.path.exists(manual_ckpt) else None

    try:
        pipeline = InferencePipeline(
            checkpoint_path=active_ckpt,
            model_name=selected_model,
            device=device_option
        )
    except Exception as e:
        st.error(f"Error initializing model pipeline: {str(e)}")
        return

    # Tabs Interface
    tab_pred, tab_vis, tab_metrics, tab_arch, tab_about = st.tabs([
        "🔍 Prediction",
        "🎨 Visualization",
        "📊 Quantitative Metrics",
        "📐 Model Architecture",
        "ℹ️ About & Disclaimer",
    ])

    # -------------------------------------------------------------
    # TAB 1: PREDICTION
    # -------------------------------------------------------------
    with tab_pred:
        st.subheader("Upload Panoramic Dental X-Ray Image")

        if not active_ckpt:
            st.warning("⚠️ No trained checkpoint loaded. Predictions will be generated using uninitialized model weights for interface testing.")
            st.markdown("""
            **First-Run Setup Instructions:**
            1. Place your dental X-ray dataset in `./data/raw/images/` and `./data/raw/tooth_masks/`.
            2. Run `python dataset_check.py` to validate your dataset.
            3. Train the model using: `python train.py --model unetplusplus_eca`.
            4. Refresh this web app.
            """)

        uploaded_file = st.file_uploader(
            "Choose a dental panoramic radiograph (PNG, JPG, JPEG, TIFF)",
            type=["png", "jpg", "jpeg", "tif", "tiff"]
        )

        col_left, col_right = st.columns([1, 1])

        if uploaded_file is not None:
            # Read Image
            pil_img = Image.open(uploaded_file)
            img_np = np.array(pil_img)

            with col_left:
                st.image(img_np, caption=f"Uploaded Image: {uploaded_file.name} ({img_np.shape[1]}x{img_np.shape[0]})", use_container_width=True)

            if st.button("🚀 Segment Teeth", type="primary"):
                with st.spinner("Running deep neural network inference..."):
                    res = pipeline.predict(img_np, threshold=threshold, target_size=(target_res, target_res))
                    st.session_state["last_result"] = res
                    st.session_state["uploaded_file_name"] = uploaded_file.name

        elif st.button("🧪 Run Interface Demo Mode (Synthetic Test Sample)"):
            synth_xray = np.random.randint(50, 200, (512, 512), dtype=np.uint8)
            with st.spinner("Generating demo prediction..."):
                res = pipeline.predict(synth_xray, threshold=threshold)
                st.session_state["last_result"] = res
                st.session_state["uploaded_file_name"] = "synthetic_demo_xray.png"

        # Display Prediction Results if available
        if "last_result" in st.session_state:
            res = st.session_state["last_result"]
            st.success(f"Inference completed in {res['inference_time']:.4f} seconds on device `{res['device']}`.")

            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            with p_col1:
                st.image(res["original_image"], caption="1. Original X-Ray", use_container_width=True)
            with p_col2:
                st.image(res["binary_mask"] * 255, caption="2. Predicted Tooth Mask", use_container_width=True)
            with p_col3:
                st.image(res["overlay"], caption="3. Composite Overlay", use_container_width=True)
            with p_col4:
                st.image((res["probability_map"] * 255).astype(np.uint8), caption="4. Probability Map", use_container_width=True)

    # -------------------------------------------------------------
    # TAB 2: VISUALIZATION CONTROLS & DOWNLOADS
    # -------------------------------------------------------------
    with tab_vis:
        st.subheader("Interactive Overlay & Output Visualization Settings")

        if "last_result" in st.session_state:
            res = st.session_state["last_result"]

            v_col1, v_col2 = st.columns([1, 2])
            with v_col1:
                opacity = st.slider("Overlay Opacity (Alpha)", 0.0, 1.0, 0.4, 0.05)
                show_cont = st.checkbox("Show Boundary Contours", value=True)
                mask_color_choice = st.selectbox("Mask Overlay Color", ["Green", "Red", "Blue", "Yellow"], index=0)

                color_map_dict = {
                    "Green": (0, 255, 0),
                    "Red": (255, 0, 0),
                    "Blue": (0, 0, 255),
                    "Yellow": (255, 255, 0),
                }

                custom_overlay = generate_overlay(
                    res["original_image"],
                    res["binary_mask"],
                    color=color_map_dict[mask_color_choice],
                    alpha=opacity,
                    show_contours=show_cont
                )

            with v_col2:
                st.image(custom_overlay, caption="Customized Visualization Overlay", use_container_width=True)

            st.markdown("### 📥 Download Outputs")
            d_col1, d_col2, d_col3 = st.columns(3)

            # Convert numpy arrays to bytes for download
            is_success, mask_buf = cv2.imencode(".png", (res["binary_mask"] * 255).astype(np.uint8))
            is_success, overlay_buf = cv2.imencode(".png", cv2.cvtColor(custom_overlay, cv2.COLOR_RGB2BGR))

            with d_col1:
                st.download_button(
                    "Download Predicted Mask (.png)",
                    data=mask_buf.tobytes(),
                    file_name="predicted_tooth_mask.png",
                    mime="image/png"
                )
            with d_col2:
                st.download_button(
                    "Download Overlay Image (.png)",
                    data=overlay_buf.tobytes(),
                    file_name="tooth_segmentation_overlay.png",
                    mime="image/png"
                )
            with d_col3:
                prob_json = json.dumps({"inference_time": res["inference_time"], "shape": res["original_shape"]})
                st.download_button(
                    "Download Metadata (.json)",
                    data=prob_json,
                    file_name="prediction_metadata.json",
                    mime="application/json"
                )
        else:
            st.info("Run a prediction under the 'Prediction' tab to view interactive visualizations.")

    # -------------------------------------------------------------
    # TAB 3: QUANTITATIVE METRICS
    # -------------------------------------------------------------
    with tab_metrics:
        st.subheader("Quantitative Metric Calculation")
        st.markdown("Upload a ground-truth binary mask for the current X-ray to calculate pixel-level segmentation metrics.")

        gt_file = st.file_uploader("Upload Ground-Truth Mask (PNG/JPG)", type=["png", "jpg", "jpeg"])

        if "last_result" in st.session_state:
            res = st.session_state["last_result"]

            if gt_file is not None:
                gt_pil = Image.open(gt_file)
                gt_tensor = preprocess_mask(gt_pil, target_shape=res["original_shape"], threshold=0.5)
                gt_np = gt_tensor.squeeze().numpy()

                pred_np = res["binary_mask"]
                prob_np = res["probability_map"]

                dice_score = compute_dice(pred_np, gt_np)
                iou_score = compute_iou(pred_np, gt_np)
                acc_score = compute_accuracy(pred_np, gt_np)
                f_metrics = compute_fscore(pred_np, gt_np)
                psnr_score = compute_psnr(pred_np, gt_np)
                map_score = compute_map(prob_np, gt_np, method="instance")

                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Dice Coefficient", f"{dice_score:.4f}")
                m_col2.metric("IoU (Jaccard Index)", f"{iou_score:.4f}")
                m_col3.metric("Pixel Accuracy", f"{acc_score:.4f}")
                m_col4.metric("F1-Score / F-score", f"{f_metrics['fscore']:.4f}")

                m_col5, m_col6, m_col7, m_col8 = st.columns(4)
                m_col5.metric("Precision", f"{f_metrics['precision']:.4f}")
                m_col6.metric("Recall / Sensitivity", f"{f_metrics['recall']:.4f}")
                m_col7.metric("Specificity", f"{f_metrics['specificity']:.4f}")
                m_col8.metric("PSNR (dB)", f"{psnr_score:.2f}")

                st.metric("Mean Average Precision (mAP @ IoU 0.5)", f"{map_score:.4f}")
            else:
                st.info("ℹ️ Ground-truth mask not provided. Quantitative evaluation is unavailable for this image.")
        else:
            st.info("Perform a prediction first to evaluate quantitative metrics.")

    # -------------------------------------------------------------
    # TAB 4: MODEL ARCHITECTURE & ATTENTION
    # -------------------------------------------------------------
    with tab_arch:
        st.subheader("Model Architecture & Bottleneck ECA Attention")

        st.markdown("""
        **Architecture Overview:**
        - **Model:** Attention-Integrated U-Net++ (ECA-U-Net++)
        - **Encoder:** 5-level deep feature extractor (`[64, 128, 256, 512, 1024]` channels)
        - **Attention Integration:** Efficient Channel Attention (ECA-Net) module at bottleneck ($X_{4,0}$)
        - **Decoder:** Dense nested skip pathways ($X_{i,j}$)
        - **Output Layer:** $1 \times 1$ Convolution with Sigmoid activation for 1-channel binary segmentation.
        """)

        arch_png = "./outputs/reports/model_architecture.png"
        generate_architecture_diagram(arch_png)
        if os.path.exists(arch_png):
            st.image(arch_png, caption="Proposed ECA-U-Net++ System Pipeline", use_container_width=True)

        if "last_result" in st.session_state:
            res = st.session_state["last_result"]
            if res.get("attention_weights") is not None:
                st.markdown("### 🧠 Bottleneck Channel Attention Weights (ECA-Net)")
                att_weights = res["attention_weights"]
                att_png = "./outputs/reports/eca_attention.png"
                generate_attention_plot(att_weights, att_png)
                if os.path.exists(att_png):
                    st.image(att_png, caption="Extracted Channel Weight Distribution at Bottleneck", use_container_width=True)

    # -------------------------------------------------------------
    # TAB 5: ABOUT & MEDICAL DISCLAIMER
    # -------------------------------------------------------------
    with tab_about:
        st.subheader("About the Research Project")
        st.markdown("""
        **Title:** Automated Tooth Segmentation in X-ray Images using Attention-Integrated U-Net++ Model

        **Key Features:**
        - End-to-end deep learning workflow from dental X-ray to binary tooth mask.
        - Efficient Channel Attention (ECA) for cross-channel feature enhancement without dimensionality reduction.
        - Flexible baseline comparison with U-Net, FCN, ENet, U-Net++, U-Net 3+, and SwiftNet.
        - Full reproducibility with deterministic random seeding.
        """)

        st.markdown("""
        <div class='disclaimer-box'>
            <h4>⚠️ Medical Disclaimer</h4>
            <p>This system is a research and educational tool for automated tooth segmentation in dental X-ray images. It is not a medical diagnostic system and must not be used as a substitute for evaluation by a qualified dental or medical professional.</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
