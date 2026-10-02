# Automated Tooth Segmentation in X-ray Images using Attention-Integrated U-Net++ Model

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![PyTest Status](https://img.shields.io/badge/PyTest-20%2F20%20Passed-brightgreen.svg)](tests/)

An end-to-end deep learning framework for automated tooth segmentation in dental panoramic radiographs, implementing the research methodology of **Attention-Integrated U-Net++** enhanced with an **Efficient Channel Attention Network (ECA-Net)** module at the bottleneck layer ($X^{4,0}$).

---

## 📖 Table of Contents
- [1. Research Paper Foundation & Summary](#1-research-paper-foundation--summary)
- [2. Architectural Design & ECA Integration](#2-architectural-design--eca-integration)
- [3. Dataset Used & Conversion Pipeline](#3-dataset-used--conversion-pipeline)
- [4. Verification & Testing Status](#4-verification--testing-status)
- [5. Paper Benchmarks vs Local Reproduction](#5-paper-benchmarks-vs-local-reproduction)
- [6. Installation & Execution Guide](#6-installation--execution-guide)
- [7. Project Status & Next Steps](#7-project-status--next-steps)

---

## 1. Research Paper Foundation & Summary

This project implements the research methodology described in the IEEE conference paper:
- **Title:** Automated Tooth Segmentation in X-ray Images using Attention Integrated U-Net++ Model
- **Authors:** Julakanti Sai Yaswanth, Kanderi Johith Kumar, Rimjhim Padam Singh (Amrita School of Computing, Amrita Vishwa Vidyapeetham, India)
- **Publication:** 2025 3rd International Conference on Intelligent Systems, Advanced Computing and Communication (ISACC), IEEE.
- **Detailed Paper Summary:** See [`docs/paper_summary.md`](file:///c:/Users/Shamanth%20J%20Chandavar/Downloads/tooth_segmentation/docs/paper_summary.md).

---

## 2. Architectural Design & ECA Integration

```text
Dental Panoramic X-Ray [B, 1, 512, 512]
         ↓
  Preprocessing & Normalization
         ↓
  U-Net++ Encoder (Levels L0..L4)
         ↓
  Bottleneck Feature Map (X_4,0) [B, 1024, 32, 32]
         ↓
  ECA Attention Module (GAP + 1D Channel Conv + Sigmoid)
         ↓
  U-Net++ Dense Skip Pathways (X_i,j) & Decoder
         ↓
  1x1 Convolution + Sigmoid Activation
         ↓
  Binary Tooth Mask Output [B, 1, 512, 512]
```

- **Backbone Architecture:** U-Net++ with 5 encoder/decoder levels (`[64, 128, 256, 512, 1024]` feature channels) connected via nested, dense skip pathways ($X^{i,j}$).
- **Bottleneck ECA Attention:** Efficient Channel Attention (ECA-Net) module is placed at the bottleneck layer ($X^{4,0}$):
  $$\omega = \sigma(\text{Conv1D}_{k}(\text{GAP}(X)))$$
  where kernel size $k$ is computed adaptively from channel count $C$:
  $$k = \left| \frac{\log_2(C)}{\gamma} + \frac{b}{\gamma} \right|_{\text{odd}}$$

---

## 3. Dataset Used & Conversion Pipeline

### Primary Target Dataset
- **Tufts Dental Database:** 1,000 panoramic radiographs professionally annotated. Requires registration / manual request to Tufts University.

### Open-Access Fallback / Development Dataset
- **Humans in the Loop Teeth Segmentation Dataset (DatasetNinja):** 598 panoramic dental X-rays with zlib base64 bitmap annotations.
- **Dataset Structure Doc:** See [`docs/dataset_structure.md`](file:///c:/Users/Shamanth%20J%20Chandavar/Downloads/tooth_segmentation/docs/dataset_structure.md).
- **Converter Script:** [`dataset/convert_annotations.py`](file:///c:/Users/Shamanth%20J%20Chandavar/Downloads/tooth_segmentation/dataset/convert_annotations.py) decodes bitmap annotations, rasterizes individual teeth polygons, and outputs 595 verified binary tooth mask pairs to `data/raw/images` and `data/raw/tooth_masks`.

---

## 4. Verification & Testing Status

Every component has been verified end-to-end:
- **PyTest Suite:** **20 / 20 passed** (`.\.venv\Scripts\python.exe -m pytest`)
- **Dataset Integrity Check:** **595 matched pairs, 0 errors** (`python dataset_check.py`)
- **DataLoader Smoke Test:** Verified batch shape `[B, 1, 512, 512]` and binary values `[0.0, 1.0]`
- **Model Forward/Backward Smoke Tests:** Verified gradients and parameter counts on `U-Net++` ($36,621,185$ params) and `ECA-U-Net++` ($36,621,190$ params).
- **Training Run:** Executed 2-epoch training run on real dataset (`Val Dice: 0.6796`, `Val Loss: 0.4947`).
- **Checkpoint Restoration:** Restored `.pth` weights dynamically without dimension mismatch.
- **Inference & Prediction:** Tested on real X-ray (`sample_1.jpg` -> `sample_1_mask.png`, `sample_1_overlay.png`).
- **Quantitative Test Evaluation:** Evaluated 90 test set images (`outputs/reports/test_metrics.json`).
- **Streamlit Web App:** Tested import & execution (`app.py`).

---

## 5. Paper Benchmarks vs Local Reproduction

See [`docs/paper_vs_reproduction.md`](file:///c:/Users/Shamanth%20J%20Chandavar/Downloads/tooth_segmentation/docs/paper_vs_reproduction.md) for full comparative analysis.

| Metric | Paper Benchmark (Tufts Dataset) | Development Run (Humans in Loop Dataset) |
|---|---:|---:|
| **Dice Score** | **0.907** | **0.6796** (Val) / **0.3896** (Test) |
| **IoU Score** | **0.830** | **0.5150** (Val) / **0.2473** (Test) |
| **Test Loss** | **0.053** | **0.6796** |
| **Accuracy** | **0.978** | **0.5429** |
| **Recall / Sensitivity** | High | **0.9300** |

---

## 6. Installation & Execution Guide

```powershell
# 1. Run Unit Test Suite
.\.venv\Scripts\python.exe -m pytest

# 2. Run Dataset Validation
.\.venv\Scripts\python.exe dataset_check.py

# 3. Train ECA-U-Net++ Model
.\.venv\Scripts\python.exe train.py --model unetplusplus_eca --config configs/cpu.yaml --epochs 2

# 4. Evaluate Test Set
.\.venv\Scripts\python.exe evaluate.py --model unetplusplus_eca --split test

# 5. Run Prediction on Single X-Ray
.\.venv\Scripts\python.exe predict.py --image data/raw/images/sample_1.jpg

# 6. Launch Streamlit Application
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### One-click launch

Launch the Streamlit web application effortlessly using any of the following options without manually opening CMD or typing commands:

#### Option A: Desktop Shortcut
- **File:** `Automated Tooth Segmentation.lnk` (located on your Desktop)
- **Behavior:** Launches the application silently in windowless mode using `run_app.vbs` with the custom project icon.
- **Creation Script:** Run `powershell -ExecutionPolicy Bypass -File tools/create_desktop_shortcut.ps1` to create or update the shortcut.

#### Option B: Windowless VBScript Launcher
- **File:** [`run_app.vbs`](file:///c:/Users/Shamanth%20J%20Chandavar/Downloads/tooth_segmentation/run_app.vbs)
- **Behavior:** Double-click to start Streamlit in windowless mode without a command prompt window. Displays clear Windows pop-up dialog boxes (`MsgBox`) if the virtual environment, `app.py`, or port 8501 is unavailable.

#### Option C: Command Prompt Batch Launcher
- **File:** [`run_app.bat`](file:///c:/Users/Shamanth%20J%20Chandavar/Downloads/tooth_segmentation/run_app.bat)
- **Behavior:** Double-click to launch Streamlit inside a visible Command Prompt window. Displays live server logs and detailed error diagnostics if startup fails.

#### Stopping the Application
- **Stop Script:** Double-click [`stop_app.bat`](file:///c:/Users/Shamanth%20J%20Chandavar/Downloads/tooth_segmentation/stop_app.bat) to safely terminate the running Streamlit server using the project PID tracker ([`outputs/logs/streamlit.pid`](file:///c:/Users/Shamanth%20J%20Chandavar/Downloads/tooth_segmentation/outputs/logs/streamlit.pid)).
- **Batch Launcher:** If started via `run_app.bat`, closing the Command Prompt window also stops the server.

---

## 7. Project Status & Next Steps

See [`docs/PROJECT_STATUS.md`](file:///c:/Users/Shamanth%20J%20Chandavar/Downloads/tooth_segmentation/docs/PROJECT_STATUS.md) for detailed task tracking.

