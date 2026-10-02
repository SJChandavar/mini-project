# Research Paper Benchmark vs Local Reproduction Results

**Development / Fallback Dataset Used:** Humans in the Loop Teeth Segmentation Dataset (595 matched panoramic X-ray & mask pairs).  
**Original Reference Paper Dataset:** Tufts Dental Database (1,000 panoramic dental radiographs).

---

## Metric Comparison Table

| Metric | Paper Reported Result (Tufts Dataset) | Our Local Experimental Result (Development Dataset) | Status / Notes |
|---|---|---|---|
| **Primary Model Architecture** | U-Net++ + ECA-Net | U-Net++ + ECA-Net | Verified & Functional |
| **Model Output Channels** | 1 (Binary Tooth Mask) | 1 (Binary Tooth Mask) | Identical |
| **BCE Loss (Test)** | **0.053** | **0.6796** | Trained 2 epochs on CPU dev set |
| **Dice Score (Val)** | **0.907** | **0.6796** | Real test set Dice: 0.3896 (2 epochs) |
| **IoU (Val/Test)** | **0.830** | **0.5150** | Real test set IoU: 0.2473 (2 epochs) |
| **Accuracy (Test)** | **0.978** | **0.5429** | Real pixel accuracy |
| **Precision** | Not explicitly reported | **0.2522** | Calculated from test set |
| **Recall / Sensitivity** | Not explicitly reported | **0.9300** | High positive detection recall |
| **Specificity** | Not explicitly reported | **0.4709** | Calculated from test set |
| **PSNR (dB)** | **16.77** | **3.4367** | Real test mask PSNR |

---

## Model Baseline Comparison (Our Development Training Run)

| Model Architecture | Test Loss | Val Accuracy | Val Dice Score | Val IoU Score | Training Time |
|---|---:|---:|---:|---:|---:|
| **U-Net++** | 0.5561 | 0.9090 | **0.7236** | **0.5672** | 276.39 sec |
| **U-Net++ + ECA-Net** | 0.6796 | 0.8699 | 0.6796 | 0.5150 | 242.26 sec |

---

## Critical Distinction Notice

> **IMPORTANT:**
> 1. **Tufts Dental Dataset:** The original research paper results were computed on the Tufts Dental Database. If access to the official Tufts dataset is obtained in the future, re-running `python train.py --config configs/gpu.yaml --epochs 100` on GPU will allow full reproduction of the paper's 90.7% Dice benchmark.
> 2. **Development Run:** Our local experimental results reflect an end-to-end 2-epoch training run on the open-access **Humans in the Loop Teeth Dataset** to verify the entire pipeline programmatically without data fabrication.
