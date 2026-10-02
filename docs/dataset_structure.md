# Dataset Structure Documentation: Humans in the Loop Teeth Segmentation Dataset

**Dataset Type:** Fallback / Development Dataset (Humans in the Loop / DatasetNinja)  
**Official Page:** [Humans in the Loop Teeth Segmentation](https://humansintheloop.org/resources/datasets/teeth-segmentation-dataset/)  
**Kaggle Source:** [DatasetNinja Teeth Segmentation](https://www.kaggle.com/datasets/humansintheloop/teeth-segmentation-on-dental-x-ray-images)

> **IMPORTANT DISCLAIMER:**  
> This dataset is used as an open-access **FALLBACK / DEVELOPMENT DATASET** to build, verify, and evaluate the deep learning pipeline end-to-end. It is **NOT** the Tufts Dental Dataset.

---

## 1. Raw Dataset Layout

Extracted raw directory: `data/fallback_raw/`

```text
data/fallback_raw/
├── meta.json                     # Class definitions & metadata
├── LICENSE.md
├── README.md
└── ds/
    ├── img/                      # 598 Panoramic dental radiograph images (.jpg)
    │   ├── 1.jpg
    │   ├── 2.jpg
    │   └── ...
    └── ann/                      # 598 JSON annotation files (.jpg.json)
        ├── 1.jpg.json
        ├── 2.jpg.json
        └── ...
```

---

## 2. Quantitative Summary

| Parameter | Count / Specification |
|---|---|
| **Total Images** | 598 panoramic X-rays (`.jpg`) |
| **Total Annotation Files** | 598 JSON files (`.jpg.json`) |
| **Image Dimensions** | High-resolution (~1024 x 2041 pixels) |
| **Color Channels** | Single-channel grayscale / RGB dental radiographs |
| **Annotation Format** | Supervisely / DatasetNinja zlib-compressed base64 bitmap masks with origin offset $[x, y]$ |
| **Classes** | Individual tooth instance categories (Tooth IDs 11..48) |
| **Average Tooth Mask Ratio** | ~8% - 15% foreground pixels per image |

---

## 3. Annotation Schema Example (`1.jpg.json`)

```json
{
  "description": "",
  "tags": [],
  "size": {
    "height": 1024,
    "width": 2041
  },
  "objects": [
    {
      "id": 16982858,
      "classId": 33132,
      "geometryType": "bitmap",
      "classTitle": "15",
      "bitmap": {
        "data": "eJwBBAL7/YlQTkcNChoKAAAADUlIRFIAAACJ...",
        "origin": [1411, 401]
      }
    }
  ]
}
```

---

## 4. Conversion Pipeline Requirements

To use this dataset for binary tooth segmentation:
1. Decode each zlib base64 bitmap string into a 2D numpy mask.
2. Offset the sub-mask by `origin` `[x, y]`.
3. Combine all tooth instance objects into a single binary mask:
   - `0` = background
   - `1` = tooth foreground
4. Save standardized X-rays to `data/raw/images/sample_XXX.jpg`.
5. Save standardized binary masks to `data/raw/tooth_masks/sample_XXX.png`.
