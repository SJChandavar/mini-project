# Master Prompt: Automated Tooth Segmentation in X-ray Images using Attention-Integrated U-Net++ Model

You are an expert Machine Learning Engineer, Deep Learning Researcher, Computer Vision Engineer, and Full-Stack Python Developer.

Build a complete, runnable, production-quality academic project titled:

**"Automated Tooth Segmentation in X-ray Images using Attention-Integrated U-Net++ Model"**

The project must implement the methodology described in the provided research paper:

**"Automated Tooth Segmentation in X-ray Images using Attention Integrated U-Net++ Model"**

The objective is to create an end-to-end deep learning system that takes a dental panoramic X-ray image as input and automatically produces a pixel-level tooth segmentation mask using a U-Net++ architecture enhanced with an Efficient Channel Attention Network (ECA-Net) module.

> **IMPORTANT**
> - Do not create a fake demo.
> - Do not hard-code prediction masks.
> - Do not hard-code evaluation metrics.
> - Do not pretend that the model achieved the paper's reported results unless the locally trained model actually achieves them.
> - All predictions must come from the trained neural network.
> - All metrics must be calculated from real predictions and ground-truth masks.
> - The project must work with the actual dataset when the dataset path is provided.
> - The application must still start successfully when the dataset or pretrained weights are missing, but it must clearly display setup instructions rather than silently generating fake results.
> - Make the code modular, well documented, reproducible, and easy for a student/researcher to understand.

---

## 1. PROJECT OBJECTIVE

Build a complete automated dental X-ray tooth segmentation system.

### INPUT
- A panoramic dental X-ray image.

### OUTPUT
1. Original X-ray image.
2. Ground-truth mask when available.
3. Predicted tooth segmentation mask.
4. Overlay image combining X-ray and predicted mask.
5. Optional binary contour/boundary visualization.
6. Quantitative segmentation metrics when ground truth is available.
7. Model confidence/probability map.
8. Inference time.
9. A downloadable predicted mask and overlay.

The primary deep learning architecture must be:

**U-Net++ + Efficient Channel Attention Network (ECA-Net)**

The ECA module must be integrated at the bottleneck/highest-level feature representation of U-Net++.

Do not turn the core project into:
- tooth numbering,
- cavity detection,
- dental disease classification,
- object detection,
- medical diagnosis,

unless these are explicitly included later as optional extensions.

The core project is **tooth segmentation**.

---

## 2. RESEARCH BASIS

The implementation should follow the research-paper methodology as closely as practical.

The paper describes:
- Tufts Dental Dataset
- 1000 panoramic dental radiographs
- professionally annotated masks
- tooth masks
- maxillomandibular masks
- abnormality masks
- normalization and resizing
- binary segmentation
- U-Net++
- Efficient Channel Attention
- comparison with U-Net, FCN, ENet, U-Net++, U-Net3+, and SwiftNet
- Binary Cross Entropy loss
- Adam optimizer
- learning rate = 0.0001
- training for up to 100 epochs
- early stopping
- Accuracy
- IoU
- Dice coefficient
- PSNR
- F-score
- mAP

Implement the tooth segmentation task as the primary task.

---

## 3. TECHNOLOGY STACK

Use:

### Backend / Deep Learning
- Python 3.10+
- PyTorch
- torchvision
- OpenCV
- NumPy
- Pillow
- scikit-learn
- matplotlib
- pandas
- tqdm

### Model implementation
- Custom PyTorch implementation of U-Net++
- Custom Efficient Channel Attention module

### Experiment tracking
- CSV and JSON logging
- TensorBoard support
- matplotlib plots

### Frontend
- Streamlit

### Optional API
- FastAPI can be included as a separate inference API if practical.

### Environment
- `requirements.txt`
- optional `environment.yml`
- reproducible random seeds
- GPU/CPU automatic detection

Do not use TensorFlow/Keras unless there is a compelling reason. Prefer PyTorch throughout the project.

---

## 4. REQUIRED PROJECT STRUCTURE

Create the following structure:

```text
tooth-segmentation/
│
├── README.md
├── requirements.txt
├── .gitignore
├── config.yaml
├── train.py
├── evaluate.py
├── predict.py
├── app.py
├── inference.py
├── dataset_check.py
├── requirements-dev.txt
│
├── configs/
│   ├── default.yaml
│   ├── gpu.yaml
│   └── cpu.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   ├── __init__.py
│   ├── unet.py
│   ├── unet_plus_plus.py
│   ├── eca.py
│   ├── eca_unet_plus_plus.py
│   ├── fcn.py
│   ├── enet.py
│   ├── unet3plus.py
│   ├── swiftnet.py
│   └── model_factory.py
│
├── dataset/
│   ├── __init__.py
│   ├── dental_dataset.py
│   ├── preprocessing.py
│   ├── augmentation.py
│   ├── split.py
│   └── validators.py
│
├── losses/
│   ├── __init__.py
│   ├── bce_loss.py
│   └── combined_loss.py
│
├── metrics/
│   ├── __init__.py
│   ├── dice.py
│   ├── iou.py
│   ├── accuracy.py
│   ├── fscore.py
│   ├── psnr.py
│   └── map_metric.py
│
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   ├── callbacks.py
│   ├── early_stopping.py
│   ├── checkpoint.py
│   └── logger.py
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py
│   ├── visualization.py
│   ├── comparison.py
│   └── report_generator.py
│
├── outputs/
│   ├── checkpoints/
│   ├── predictions/
│   ├── overlays/
│   ├── masks/
│   ├── plots/
│   ├── reports/
│   └── logs/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
│
└── tests/
    ├── test_dataset.py
    ├── test_eca.py
    ├── test_model.py
    ├── test_metrics.py
    └── test_inference.py
```

---

## 5. DATASET HANDLING

The project must support the Tufts Dental Dataset.

Expected dataset content:
- Panoramic dental X-ray images
- Tooth segmentation masks
- Optional maxillomandibular masks
- Optional abnormality masks

Make the dataset loader flexible because users may organize the dataset differently.

Implement configurable paths in `config.yaml`.

Example:

```yaml
dataset:
  root: "./data/raw"
  images: "./data/raw/images"
  tooth_masks: "./data/raw/tooth_masks"
  jaw_masks: "./data/raw/maxillomandibular_masks"
  abnormality_masks: "./data/raw/abnormality_masks"
```

The loader must:
- find matching X-ray/mask pairs
- validate dimensions
- detect missing files
- detect corrupted images
- print useful error messages
- support PNG, JPG, JPEG, TIFF where practical
- convert images into grayscale when necessary
- convert masks into binary format
- preserve correct image-mask alignment
- never apply independent random transformations to image and mask

Create a dataset validation script:

```bash
python dataset_check.py
```

It should report:
- total images
- total masks
- matched pairs
- unmatched images
- unmatched masks
- corrupted images
- image dimensions
- mask dimensions
- class distribution
- percentage of foreground pixels
- sample visualizations

---

## 6. DATA SPLITTING

Implement configurable train/validation/test splitting.

Default:
- 70% training
- 15% validation
- 15% testing

Make the proportions configurable.

Use a fixed random seed.

> **IMPORTANT**
> Prevent data leakage.
>
> If multiple images belong to the same patient, the split should be patient-level when patient identifiers are available.

Store the split information in:

```text
outputs/reports/dataset_split.csv
```

Columns:
- image_path
- mask_path
- split

---

## 7. IMAGE PREPROCESSING

Follow the paper's preprocessing idea:
- normalize images
- resize images and masks
- threshold masks to obtain binary masks

Implement a robust preprocessing pipeline.

Default image size:

```text
512 x 512
```

Make this configurable:

```yaml
image:
  width: 512
  height: 512
  channels: 1
```

### Processing

1. Read panoramic X-ray.
2. Convert to grayscale.
3. Resize to target dimensions.
4. Normalize pixel intensities.
5. Convert to float32.
6. Scale to [0, 1].
7. Add channel dimension.

### Masks

1. Read mask.
2. Resize using nearest-neighbor interpolation.
3. Convert to grayscale.
4. Threshold.
5. Convert foreground to 1.
6. Convert background to 0.
7. Return float32 tensor.

Avoid using bilinear interpolation on segmentation masks.

---

## 8. DATA AUGMENTATION

Implement synchronized augmentation for image and mask.

Useful augmentations:
- horizontal flip
- small rotation
- small translation
- small scaling
- random crop when appropriate
- brightness adjustment
- contrast adjustment
- mild Gaussian noise

Do not apply intensity changes to the segmentation mask.

Make augmentation configurable.

Example:

```yaml
augmentation:
  enabled: true
  horizontal_flip: true
  rotation: 10
  brightness: 0.15
  contrast: 0.15
```

Ensure image and mask receive the same spatial transformations.

---

## 9. PRIMARY MODEL: U-NET++

Implement U-Net++ from scratch in PyTorch.

### Architecture requirements
- encoder path
- decoder path
- dense skip connections
- convolution blocks
- batch normalization
- ReLU activation
- max pooling for downsampling
- dropout for regularization
- transposed convolution for upsampling
- concatenation of feature maps
- final 1-channel segmentation output

The model must support input shape:

```text
[B, 1, 512, 512]
```

and output:

```text
[B, 1, 512, 512]
```

The architecture should follow the concept presented in the paper:
- five encoding/decoding levels
- progressively deeper feature representations
- dense skip pathways
- encoder-decoder structure

Use configurable feature channels, for example:

```text
[64, 128, 256, 512, 1024]
```

Make this configurable to allow smaller models on GPUs with limited memory.

---

## 10. ECA-NET ATTENTION MODULE

Implement Efficient Channel Attention manually.

The paper describes ECA as:
- global average pooling
- 1D convolution for local cross-channel interaction
- sigmoid activation
- channel-wise attention weights
- multiplication of attention weights with the input feature map

Implement:

```text
ECA(x):
```

### Steps

1. Global Average Pooling:

```text
[B, C, H, W] -> [B, C, 1, 1]
```

2. Reshape channel descriptor for 1D convolution.

3. Apply 1D convolution across channels.

4. Apply sigmoid.

5. Reshape attention weights.

6. Multiply the weights with original feature maps.

Conceptually:

```text
y = x * sigmoid(Conv1D(GAP(x)))
```

Use an adaptive kernel size based on channel count when appropriate.

Document the ECA module thoroughly.

---

## 11. INTEGRATE ECA INTO U-NET++

This is the most important architectural requirement.

Integrate ECA into U-Net++ at the bottleneck / highest-level feature representation, consistent with the research paper.

Concept:

```text
Input
  ↓
Encoder blocks
  ↓
Dense U-Net++ skip connections
  ↓
Bottleneck
  ↓
ECA Attention
  ↓
Decoder
  ↓
Segmentation Head
  ↓
Predicted Tooth Mask
```

The ECA block must operate on the bottleneck feature representation.

Do not place it arbitrarily in the input layer.

Implement:

```text
ECABlock
      ↓
U-Net++ bottleneck
      ↓
Decoder
```

Make the attention layer modular so the model can be switched between:

1. Standard U-Net++
2. U-Net++ + ECA

through configuration.

---

## 12. OUTPUT ACTIVATION

Use one output channel for binary tooth segmentation.

The model should return logits.

During inference:

```python
probability = sigmoid(logits)
```

Then:

```python
binary_mask = probability >= threshold
```

Default threshold:

```text
0.5
```

Make threshold configurable.

---

## 13. LOSS FUNCTION

The paper specifically uses Binary Cross Entropy loss.

Implement `BCEWithLogitsLoss`.

Default:

```yaml
loss:
  name: "bce"
  type: "binary_cross_entropy"
```

Also implement optional Dice Loss as an experimental alternative.

Allow:
- BCE
- Dice
- BCE + Dice

But the default experiment must use BCE to match the paper.

---

## 14. OPTIMIZER

Default optimizer:

**Adam**

Learning rate:

```text
0.0001
```

Config:

```yaml
optimizer:
  name: adam
  learning_rate: 0.0001
  weight_decay: 0.0
```

Make the optimizer configurable.

---

## 15. TRAINING

Default:

```text
epochs: 100
```

Use early stopping to avoid overfitting.

Track:
- training loss
- validation loss
- training accuracy
- validation accuracy
- validation Dice
- validation IoU

Save the best checkpoint based on validation Dice.

Save:
- last checkpoint
- best checkpoint

Checkpoint filename:

```text
outputs/checkpoints/best_eca_unetplusplus.pth
```

Checkpoint must contain:
- model state dict
- optimizer state dict
- epoch
- validation loss
- validation Dice
- config
- random seed

---

## 16. LEARNING RATE SCHEDULER

Implement optional learning rate scheduling.

Default can be disabled to maintain closer consistency with the paper.

If enabled, support:
- ReduceLROnPlateau
- CosineAnnealingLR

Never silently change the paper's default configuration.

---

## 17. EARLY STOPPING

Implement configurable early stopping.

Parameters:

```yaml
patience: 10
monitor: val_loss
mode: min
```

Also support monitoring validation Dice.

When early stopping occurs:
- print reason
- restore best model
- save training history

---

## 18. REPRODUCIBILITY

Set random seeds for:
- Python
- NumPy
- PyTorch

Support:

```yaml
seed: 42
```

Enable deterministic behavior when practical.

Print:
- Python version
- PyTorch version
- CUDA availability
- GPU name
- image size
- batch size
- learning rate
- number of epochs
- model name
- parameter count

---

## 19. BATCH SIZE

Default:

```text
batch_size: 4
```

Make it configurable because panoramic images can consume significant GPU memory.

Automatically provide a helpful message if CUDA out-of-memory occurs.

---

## 20. METRICS

Implement the following metrics:

1. Accuracy
2. Intersection over Union (IoU)
3. Dice Coefficient
4. F-score
5. PSNR
6. mAP

Definitions should be implemented correctly.

### Dice

```text
Dice = 2 * TP / (2 * TP + FP + FN)
```

### IoU

```text
IoU = TP / (TP + FP + FN)
```

### Accuracy

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### F1/F-score

```text
F1 = 2 * Precision * Recall / (Precision + Recall)
```

Also calculate:
- Precision
- Recall
- Specificity

Handle zero denominators safely.

---

## 21. IMPORTANT NOTE ABOUT mAP

Because the primary problem is semantic/binary segmentation, clearly document how mAP is being computed.

Do **not** simply copy the research paper's mAP value without implementing a legitimate metric.

Provide one of these clearly documented approaches:

### Option A
Convert connected components of predicted tooth masks into object instances and compute average precision over matched tooth regions.

### Option B
Treat segmentation probability maps using an appropriate pixel-level AP implementation, while clearly naming it segmentation AP rather than object-detection mAP.

The implementation must explicitly document:
- IoU threshold
- matching method
- precision-recall calculation
- averaging procedure

Do not hide this definition.

---

## 22. PSNR

Implement PSNR between:
- ground-truth binary mask
- predicted binary mask

Clearly document that PSNR is included because it is reported in the reference paper.

---

## 23. MODEL BASELINE COMPARISON

Implement baseline models where practical:

1. U-Net
2. FCN
3. ENet
4. U-Net++
5. U-Net3+
6. SwiftNet
7. U-Net++ + ECA-Net

All models must use:
- same dataset split
- same preprocessing
- same training protocol where architecture allows
- same evaluation metrics

This is necessary for a fair comparison.

Provide configuration:

```yaml
experiment:
  models:
    - unet
    - fcn
    - enet
    - unetplusplus
    - unet3plus
    - swiftnet
    - unetplusplus_eca
```

If a baseline architecture is too large or unavailable on the target environment, provide a documented compatible implementation rather than silently excluding it.

---

## 24. EXPERIMENT MANAGEMENT

Each model training run should create a separate directory:

```text
outputs/
  experiments/
    unet/
    fcn/
    enet/
    unetplusplus/
    unet3plus/
    swiftnet/
    unetplusplus_eca/
```

Inside each experiment:
- `best_model.pth`
- `last_model.pth`
- `training_history.csv`
- `metrics.json`
- `config.yaml`
- `loss_curve.png`
- `dice_curve.png`
- `iou_curve.png`
- `sample_predictions/`

---

## 25. TRAINING CURVES

Generate plots for:

1. Training loss vs epoch
2. Validation loss vs epoch
3. Training accuracy vs epoch
4. Validation accuracy vs epoch
5. Dice vs epoch
6. IoU vs epoch
7. Learning rate vs epoch

Use matplotlib.

Do not use fake values.

The curves must be generated from actual training history.

---

## 26. EVALUATION PIPELINE

Create:

```bash
python evaluate.py \
    --model unetplusplus_eca \
    --checkpoint outputs/checkpoints/best_eca_unetplusplus.pth \
    --split test
```

Output:
- average loss
- accuracy
- IoU
- Dice
- F-score
- precision
- recall
- specificity
- PSNR
- mAP
- inference time

Save:

```text
outputs/reports/test_metrics.json
```

and

```text
outputs/reports/test_metrics.csv
```

---

## 27. VISUAL EVALUATION

For several test images, generate a 4-panel visualization:

### Panel 1
Original X-ray

### Panel 2
Ground-truth tooth mask

### Panel 3
Predicted tooth mask

### Panel 4
Overlay

Overlay:
- original X-ray in grayscale
- predicted segmentation overlaid transparently

Also generate:
- probability map
- binary mask
- contours

Save results to:

```text
outputs/predictions/
```

---

## 28. INFERENCE SCRIPT

Create:

```bash
python predict.py --image path/to/xray.png
```

It should:

1. Load trained checkpoint.
2. Detect CPU/GPU.
3. Read image.
4. Preprocess image.
5. Run inference.
6. Apply sigmoid.
7. Threshold prediction.
8. Restore/resize output to original image dimensions.
9. Save mask.
10. Save overlay.
11. Save probability map.
12. Print inference time.

Example output:

```text
Prediction completed

Input:
sample_xray.png

Output:
outputs/predictions/sample_xray_mask.png
outputs/predictions/sample_xray_overlay.png
outputs/predictions/sample_xray_probability.png

Inference time:
0.XXX seconds
```

---

## 29. STREAMLIT WEB APPLICATION

Create a clean research/demo interface using Streamlit.

Run with:

```bash
streamlit run app.py
```

Application title:

**Automated Tooth Segmentation in Dental X-rays**

---

## 30. STREAMLIT SIDEBAR

Sidebar should contain:

### Model
- U-Net++
- U-Net++ + ECA

### Checkpoint
- automatically detect available checkpoints
- allow manual checkpoint selection

### Threshold
- 0.1 to 0.9 slider
- default 0.5

### Device
- Auto
- CPU
- CUDA

### Image preprocessing
- target size
- normalization

---

## 31. IMAGE UPLOAD

Allow:

**Upload dental panoramic X-ray**

Supported:
- PNG
- JPG
- JPEG
- TIFF

Show:
- filename
- dimensions
- grayscale preview

---

## 32. PREDICTION UI

After clicking:

**"Segment Teeth"**

show:

- Original X-ray
- Predicted Mask
- Overlay
- Probability Map

Use tabs:

1. Prediction
2. Visualization
3. Metrics
4. Model Information
5. About

---

## 33. METRICS UI

When a ground-truth mask is uploaded, calculate:
- Dice
- IoU
- Accuracy
- Precision
- Recall
- F-score
- PSNR
- mAP

Display them as metric cards.

Do not display metrics when there is no ground truth.

Instead show:

> "Ground-truth mask not provided. Quantitative evaluation is unavailable for this image."

---

## 34. VISUALIZATION UI

Provide controls for:
- overlay opacity
- probability threshold
- show/hide contours
- mask color
- side-by-side vs overlay view

Also provide a download button for:
- predicted mask
- overlay
- probability map
- evaluation JSON

---

## 35. MODEL INFORMATION PAGE

Display:

**Model:**
U-Net++ + Efficient Channel Attention

**Architecture:**
Encoder-decoder with dense skip connections

**Attention:**
Efficient Channel Attention at bottleneck

**Input:**
1-channel grayscale dental X-ray

**Output:**
1-channel binary tooth mask

**Activation:**
Sigmoid during inference

**Loss:**
Binary Cross Entropy

**Optimizer:**
Adam

**Learning rate:**
0.0001

**Maximum epochs:**
100

**Early stopping:**
Enabled

Also show:
- parameter count
- trainable parameters
- model file
- checkpoint date if available

---

## 36. MODEL ARCHITECTURE VISUALIZATION

Create a visual architecture diagram in the application and/or generate:

```text
outputs/reports/model_architecture.png
```

The diagram should show:

```text
Dental X-ray
      ↓
Preprocessing
      ↓
U-Net++ Encoder
      ↓
Dense Skip Connections
      ↓
Bottleneck
      ↓
ECA Attention
      ↓
U-Net++ Decoder
      ↓
1x1 Convolution
      ↓
Sigmoid
      ↓
Tooth Segmentation Mask
```

Clearly label the ECA module.

---

## 37. MODEL SUMMARY

Provide:
- number of parameters
- input shape
- output shape
- layer summary

Use `torchinfo` if practical.

Example:

```python
from torchinfo import summary

summary(
    model,
    input_size=(1, 1, 512, 512)
)
```

Save the output into:

```text
outputs/reports/model_summary.txt
```

---

## 38. CONFUSION MATRIX

For pixel-level segmentation, calculate:
- TP
- TN
- FP
- FN

Generate a confusion matrix.

Save:

```text
outputs/reports/confusion_matrix.png
```

---

## 39. THRESHOLD ANALYSIS

Implement optional threshold analysis.

Evaluate thresholds:

```text
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
```

For every threshold calculate:
- Dice
- IoU
- Precision
- Recall
- F-score

Generate:

```text
threshold_vs_dice.png
threshold_vs_iou.png
```

Do not select a threshold based on the test set unless explicitly identified as an experiment.

Default production threshold should remain 0.5.

---

## 40. ABLATION STUDY

Include an optional experiment comparing:

### Experiment A
U-Net++

### Experiment B
U-Net++ + ECA

Calculate:
- loss
- Dice
- IoU
- accuracy
- precision
- recall
- F-score
- PSNR
- mAP
- inference time

Generate:

```text
ablation_comparison.csv
ablation_comparison.png
```

The purpose is to measure the contribution of ECA.

---

## 41. BASELINE COMPARISON REPORT

Create a summary table:

```text
Model | Loss | Accuracy | Dice | IoU | F-score | PSNR | mAP | Inference Time
```

Do not manually fill this table.

Populate it from actual experiment results.

Save:

```text
outputs/reports/model_comparison.csv
```

and

```text
outputs/reports/model_comparison.png
```

---

## 42. EXPECTED REFERENCE RESULTS

The research paper reports approximately:

| Model | Loss | Dice | PSNR | mAP | Accuracy |
|---|---:|---:|---:|---:|---:|
| U-Net | 0.234 | 0.533 | 10.10 | 0.513 | 0.902 |
| FCN | 0.171 | 0.669 | 10.39 | 0.608 | 0.908 |
| ENet | 0.317 | 0.393 | 9.986 | 0.478 | 0.899 |
| U-Net++ | 0.054 | 0.816 | 18.04 | 0.920 | 0.970 |
| U-Net3+ | 0.198 | 0.557 | 10.04 | 0.573 | 0.908 |
| SwiftNet | 0.609 | 0.157 | 4.96 | 0.118 | 0.680 |
| U-Net++ + ECA-Net | 0.053 | 0.907 | 16.77 | 0.964 | 0.978 |

The paper's conclusion also reports IoU = 0.830 for the proposed model.

> **IMPORTANT**
>
> These are reference values from the paper only.
>
> Do not hard-code them into the application.
>
> Use them only in:
> - README as "reported reference results"
> - documentation
> - optional comparison/reference table
>
> Clearly label them as:
>
> **"Reported in the reference paper"**
>
> and distinguish them from locally reproduced results.

---

## 43. TRAINING COMMANDS

README must provide exact commands.

Install:

```bash
python -m venv .venv
```

Activate the environment.

Then:

```bash
pip install -r requirements.txt
```

Dataset validation:

```bash
python dataset_check.py
```

Train proposed model:

```bash
python train.py \
    --model unetplusplus_eca \
    --config configs/default.yaml
```

Train baseline:

```bash
python train.py \
    --model unetplusplus \
    --config configs/default.yaml
```

Evaluate:

```bash
python evaluate.py \
    --model unetplusplus_eca \
    --checkpoint outputs/checkpoints/best_eca_unetplusplus.pth \
    --split test
```

Predict:

```bash
python predict.py \
    --model unetplusplus_eca \
    --checkpoint outputs/checkpoints/best_eca_unetplusplus.pth \
    --image sample.png
```

Start application:

```bash
streamlit run app.py
```

---

## 44. CONFIGURATION FILE

Create a complete `config.yaml`.

Example:

```yaml
project:
  name: "Automated Tooth Segmentation"
  seed: 42

dataset:
  root: "./data/raw"
  images: "./data/raw/images"
  masks: "./data/raw/tooth_masks"

split:
  train: 0.70
  validation: 0.15
  test: 0.15

image:
  width: 512
  height: 512
  channels: 1

training:
  epochs: 100
  batch_size: 4
  num_workers: 2
  early_stopping_patience: 10

optimizer:
  name: "adam"
  learning_rate: 0.0001
  weight_decay: 0.0

loss:
  name: "bce"

model:
  name: "unetplusplus_eca"
  encoder_channels:
    - 64
    - 128
    - 256
    - 512
    - 1024
  dropout: 0.1
  eca:
    enabled: true

inference:
  threshold: 0.5

augmentation:
  enabled: true
```

---

## 45. ERROR HANDLING

The project must gracefully handle:
- missing dataset
- missing mask
- invalid image
- corrupt image
- unsupported image format
- mismatched image/mask dimensions
- missing checkpoint
- CUDA unavailable
- CUDA out-of-memory
- empty dataset
- invalid configuration
- incorrect image channels

Give useful error messages.

Do not crash with cryptic stack traces when a user can be given a clear explanation.

---

## 46. GPU SUPPORT

Automatically select:

**CUDA if available, otherwise CPU.**

Display:

```text
Using device: NVIDIA GPU / CPU
```

Support mixed precision training when CUDA is available:

```python
torch.cuda.amp
```

or the modern PyTorch equivalent.

Make mixed precision configurable.

---

## 47. MEMORY EFFICIENCY

Because 512x512 panoramic images can be large, implement:
- configurable batch size
- configurable image size
- optional mixed precision
- `torch.no_grad()` during inference
- optional gradient accumulation
- efficient DataLoader
- pin_memory when CUDA is available

---

## 48. UNIT TESTS

Create tests for:

### Dataset
- image loading
- mask loading
- shape validation
- transformations

### ECA
- input/output dimensions
- attention values
- gradient flow

### Model
- forward pass
- output shape
- backward pass

### Metrics
- Dice
- IoU
- Accuracy
- Precision
- Recall
- F-score
- PSNR

### Inference
- checkpoint loading
- prediction shape
- thresholding

Run:

```bash
pytest
```

---

## 49. README REQUIREMENTS

Create a detailed README containing:

1. Project title
2. Problem statement
3. Motivation
4. Research objective
5. Dataset description
6. Architecture
7. U-Net++ explanation
8. ECA explanation
9. Why ECA is used
10. Preprocessing
11. Training
12. Evaluation metrics
13. Baseline models
14. Installation
15. Dataset setup
16. Training commands
17. Evaluation commands
18. Prediction commands
19. Streamlit usage
20. Folder structure
21. Expected outputs
22. Troubleshooting
23. Limitations
24. Ethical/medical disclaimer
25. Reference paper information

---

## 50. MEDICAL DISCLAIMER

The application must clearly state:

> "This system is a research and educational tool for automated tooth segmentation in dental X-ray images. It is not a medical diagnostic system and must not be used as a substitute for evaluation by a qualified dental or medical professional."

Do not claim that segmentation results constitute diagnosis.

---

## 51. ACADEMIC REPORT GENERATION

Create:

```text
outputs/reports/final_report.md
```

It should automatically summarize the actual experiment.

### Sections
1. Abstract
2. Introduction
3. Problem Statement
4. Dataset
5. Methodology
6. U-Net++ Architecture
7. ECA Attention Mechanism
8. Data Preprocessing
9. Training Configuration
10. Experimental Setup
11. Baseline Models
12. Evaluation Metrics
13. Results
14. Ablation Study
15. Qualitative Results
16. Limitations
17. Conclusion
18. Future Work

Do not fabricate results.

If a model was not trained, write:

> "Experiment not executed."

rather than inventing values.

---

## 52. FUTURE EXTENSIONS

Document possible future enhancements without implementing them unless practical:
- transformer-based segmentation
- multi-head attention
- stronger augmentation
- instance-level tooth segmentation
- tooth numbering
- cavity segmentation
- restoration segmentation
- CBCT support
- multi-modal dental imaging
- explainable AI
- uncertainty estimation

These must remain clearly separated from the core research implementation.

---

## 53. CODE QUALITY

Follow these principles:
- modular architecture
- type hints
- docstrings
- clear variable names
- no unnecessary global state
- configuration-driven experiments
- logging instead of excessive print statements
- reusable functions
- separation of dataset/model/training/evaluation/UI
- PEP8-compatible Python
- meaningful comments
- no duplicated code

---

## 54. DOCUMENTATION OF ARCHITECTURE

Every major component must be documented.

### For U-Net++
Explain:
- encoder
- decoder
- dense skip connections
- bottleneck
- upsampling
- segmentation head

### For ECA
Explain:
- global average pooling
- channel descriptor
- 1D convolution
- sigmoid attention
- channel-wise multiplication

Include mathematical formulas in documentation.

---

## 55. VISUALIZATION OF ATTENTION

Add an optional research visualization that shows attention behavior.

At minimum:
- visualize ECA channel attention values
- save an attention plot
- optionally display attention statistics in Streamlit

Output:

```text
outputs/reports/eca_attention.png
```

Do not claim the visualization proves clinical interpretability.

Call it:

**"Attention Weight Visualization"**

---

## 56. PERFORMANCE REPORT

Measure:
- preprocessing time
- model inference time
- total inference time
- images per second when appropriate
- GPU memory usage when CUDA is available

Save:

```text
outputs/reports/performance.json
```

---

## 57. CLI EXPERIENCE

All scripts must provide:

```bash
python train.py --help
python evaluate.py --help
python predict.py --help
python dataset_check.py --help
```

Use `argparse`.

---

## 58. AUTOMATIC DIRECTORY CREATION

The code should automatically create missing directories:

```text
data/
outputs/
checkpoints/
reports/
predictions/
plots/
logs/
```

Do not require the user to manually create these directories.

---

## 59. NO FAKE DATA

Never create synthetic dental X-rays and present them as real dataset samples.

For unit tests only, synthetic tensors may be generated internally to test model functionality.

Clearly label them as test data.

---

## 60. NO HARDCODED PAPER RESULTS

This is critical.

Do not do:

```python
dice = 0.907
```

or:

```python
accuracy = 0.978
```

or:

```python
mAP = 0.964
```

unless those numbers come from the actual evaluation pipeline.

Paper values may appear only in documentation as reference results.

---

## 61. REPRODUCIBILITY

Create a reproducibility section that records:
- git commit if available
- random seed
- dataset split
- image size
- batch size
- optimizer
- learning rate
- number of epochs
- early stopping patience
- model configuration
- Python version
- PyTorch version
- CUDA version if available

---

## 62. FIRST-RUN EXPERIENCE

When the user runs:

```bash
streamlit run app.py
```

and no trained checkpoint exists:

Do not fail.

Instead show:

> "No trained model checkpoint was found."

Then provide setup instructions:

1. Put the dataset in the configured directory.
2. Run `dataset_check.py`.
3. Train the model.
4. Return to the application.

Also provide an optional demo mode using an internally generated random tensor **ONLY** to verify that the interface loads, but explicitly label this as:

> "Interface test mode - not a dental prediction."

Do not present random output as a valid segmentation result.

---

## 63. APPLICATION DESIGN

The Streamlit UI should look like a polished academic/research application.

Main page:

### Title
**Automated Tooth Segmentation in Dental X-rays**

### Subtitle
**Attention-Integrated U-Net++ with Efficient Channel Attention**

Sections:

```text
Upload X-ray
↓
Select Model
↓
Configure Threshold
↓
Segment Teeth
↓
View Results
↓
Evaluate if Ground Truth is Available
```

Use columns for:
- Original Image
- Predicted Mask
- Overlay

Use tabs for:
- Results
- Metrics
- Architecture
- Attention
- About

---

## 64. EXPECTED USER WORKFLOW

A normal researcher should be able to:

1. Clone/open the project.
2. Install requirements.
3. Place Tufts Dental Dataset in `data/raw`.
4. Run dataset validation.
5. Train U-Net++ baseline.
6. Train U-Net++ + ECA.
7. Evaluate models.
8. Compare models.
9. Upload a new X-ray.
10. Generate segmentation.
11. View mask and overlay.
12. Download outputs.
13. View metrics for images with ground truth.
14. Generate experiment reports.

---

## 65. FINAL DELIVERABLE

Generate the entire project.

Do not merely provide snippets.

Create:
- all Python files
- configuration files
- README
- requirements.txt
- model implementations
- dataset loader
- preprocessing
- augmentation
- training pipeline
- evaluation pipeline
- metric implementations
- visualization utilities
- Streamlit application
- tests
- report generation
- architecture diagram generation

All imports must be correct.

All file paths must be relative/configurable.

All modules must work together.

Before presenting the result, perform an internal consistency check:

1. Can the dataset loader return an image/mask pair?
2. Does the model accept `[B,1,H,W]`?
3. Does model output `[B,1,H,W]`?
4. Is the ECA block actually connected to the U-Net++ bottleneck?
5. Does `backward()` work?
6. Does one training epoch work?
7. Does validation work?
8. Does checkpoint saving work?
9. Does checkpoint loading work?
10. Does prediction work?
11. Are mask transformations synchronized?
12. Are metrics computed from actual predictions?
13. Can Streamlit run without a checkpoint?
14. Are missing files handled gracefully?
15. Are paper-reported values never used as fake runtime results?

---

## 66. FINAL RESPONSE FROM THE AI BUILDING THE PROJECT

After creating the project, provide:

1. Complete folder tree
2. Installation instructions
3. Dataset placement instructions
4. Training command
5. Evaluation command
6. Prediction command
7. Streamlit command
8. Explanation of U-Net++
9. Explanation of ECA
10. Explanation of the experimental methodology
11. List of generated outputs
12. Known limitations
13. Any components that could not be reproduced exactly from the research paper

Be explicit about any assumptions.

Do not claim successful reproduction of the research paper unless the actual experiment was run and the results support that claim.

---

## PRIMARY SUCCESS CRITERION

The finished application must provide a genuine end-to-end implementation of:

```text
Dental X-ray
→ preprocessing
→ U-Net++
→ ECA attention at bottleneck
→ decoder
→ binary tooth segmentation
→ evaluation
→ visualization
→ web interface
```

The primary model is:

**U-Net++ + Efficient Channel Attention (ECA-Net)**

The project must be suitable for:
- academic project demonstration
- research experimentation
- dissertation/project documentation
- model comparison
- dental image segmentation experimentation

Build the project completely and ensure that all components are integrated rather than merely described.
