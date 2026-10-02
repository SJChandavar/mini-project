# Research Paper Summary: Automated Tooth Segmentation in X-ray Images using Attention Integrated U-Net++ Model

**Paper Reference:**  
Julakanti Sai Yaswanth, Kanderi Johith Kumar, and Rimjhim Padam Singh.  
*"Automated Tooth Segmentation in X-ray Images using Attention Integrated U-Net++ Model"*  
2025 3rd International Conference on Intelligent Systems, Advanced Computing and Communication (ISACC), IEEE.  
DOI: `10.1109/ISACC65211.2025.10969266`

---

## 1. Title & Authors
- **Title:** Automated Tooth Segmentation in X-ray Images using Attention Integrated U-Net++ Model
- **Authors:** Julakanti Sai Yaswanth, Kanderi Johith Kumar, Rimjhim Padam Singh
- **Affiliation:** Department of Computer Science and Engineering, Amrita School of Computing, Bengaluru, Amrita Vishwa Vidyapeetham, India.

## 2. Problem Statement & Motivation
- Oral diseases (periodontal disease, dental caries, oral cancers) affect over 3.5 billion people worldwide.
- Panoramic radiographs provide 2D visualization of jaws, teeth, sinuses, and temporomandibular joints.
- Manual analysis and feature engineering suffer from efficiency, performance, and scalability limits.
- The objective is an automated deep learning segmentation system that directly learns pixel-level tooth segmentation masks from panoramic X-ray images without complex manual feature extraction.

## 3. Dataset Description
- **Dataset:** Tufts Dental Database (Panetta et al., 2021).
- **Volume:** 1,000 panoramic dental radiographs professionally annotated by experts.
- **Contents:** X-ray images, teeth masks, maxillomandibular masks, abnormality masks.
- **Variations:** Diverse patient anatomy, multiple resolutions, and various dental conditions.

## 4. Image Preprocessing
- Intensity normalization scaling pixel values to $[0, 1]$.
- Image resizing to target dimensions.
- Binarization of ground-truth masks via thresholding.
- Batch generation via a custom `DataGenerator` subclass.

## 5. Network Architecture & ECA Integration
- **Backbone Architecture:** U-Net++ (Nested U-Net) featuring 5 encoder/decoder levels ($L^0 \dots L^4$) connected via dense skip pathways ($X^{i,j}$).
- **Attention Mechanism:** Efficient Channel Attention Network (ECA-Net).
- **ECA Placement:** Placed at the bottleneck / highest level of feature abstraction ($X^{4,0}$) channel-wise before upsampling decoder pathways.
- **ECA Operations:** Global Average Pooling (GAP) $\rightarrow$ 1D Convolution across channels $\rightarrow$ Sigmoid activation $\rightarrow$ Element-wise channel scaling of feature maps.

## 6. Training Configuration & Hyperparameters
- **Loss Function:** Binary Cross Entropy (BCE) Loss.
- **Optimizer:** Adam optimizer.
- **Learning Rate:** 0.0001.
- **Maximum Epochs:** 100.
- **Regularization:** Dropout layers and Early Stopping based on training and validation loss.

## 7. Baseline Models Compared
1. U-Net
2. Fully Convolutional Networks (FCNs / FCN-8s)
3. Efficient Neural Network (ENet)
4. U-Net++
5. U-Net 3+
6. SwiftNet
7. Proposed U-Net++ + ECA-Net

## 8. Evaluation Metrics & Definitions
- **Accuracy:** $(\text{TP} + \text{TN}) / (\text{TP} + \text{TN} + \text{FP} + \text{FN})$
- **Dice Coefficient:** $2 \cdot \text{TP} / (2 \cdot \text{TP} + \text{FP} + \text{FN})$
- **Intersection over Union (IoU):** $\text{TP} / (\text{TP} + \text{FP} + \text{FN})$
- **Peak Signal to Noise Ratio (PSNR):** $10 \cdot \log_{10}(\text{MAX}^2 / \text{MSE})$
- **F-score (F1):** $2 \cdot \text{Precision} \cdot \text{Recall} / (\text{Precision} + \text{Recall})$
- **mAP (Mean Average Precision):** Precision-recall area under curve / object matching precision across classes.

## 9. Reported Paper Benchmark Results

| Model Architecture | Train Loss | Train Acc | Val Loss | Val Acc | Test Loss | Test Acc | Dice Score | PSNR (dB) | mAP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U-Net | 0.223 | 0.906 | 0.239 | 0.901 | 0.234 | 0.902 | 0.533 | 10.10 | 0.513 |
| FCN | 0.174 | 0.911 | 0.171 | 0.913 | 0.176 | 0.908 | 0.669 | 10.39 | 0.608 |
| ENet | 0.191 | 0.915 | 0.317 | 0.895 | 0.293 | 0.899 | 0.393 | 9.986 | 0.478 |
| U-Net++ | 0.063 | 0.975 | 0.074 | 0.970 | 0.072 | 0.970 | 0.816 | 18.04 | 0.920 |
| U-Net 3+ | 0.183 | 0.913 | 0.198 | 0.906 | 0.192 | 0.908 | 0.557 | 10.04 | 0.573 |
| SwiftNet | 0.592 | 0.757 | 0.609 | 0.683 | 0.615 | 0.680 | 0.157 | 4.96 | 0.118 |
| **U-Net++ + ECA-Net (Proposed)** | **0.046** | **0.981** | **0.053** | **0.979** | **0.053** | **0.978** | **0.907** | **16.77** | **0.964** |

*Note: The paper also explicitly reports $\text{IoU} = 0.830$ for the proposed ECA-U-Net++ model.*

## 10. Conclusion & Future Scope
- U-Net++ is the best performing baseline among existing architectures.
- Integrating ECA-Net at the bottleneck significantly improves channel feature representations, yielding the top performance (Dice 0.907, IoU 0.830, mAP 0.964, Acc 0.978).
- Future extensions suggested include Vision Transformers, multi-head attention mechanisms, and 3D CBCT/MRI segmentation.
