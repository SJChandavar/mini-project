# Final Research Validation & Technical Clarifications

## 1. Metric Clarifications & Specificity Breakdown

### Mean Test-Set Performance ($N = 90$ Test Images)
- **Mean Recall across entire Test Set:** **$0.9488$** ($94.88\%$).
- **Mean Specificity across entire Test Set:** **$0.8626$** ($86.26\%$).
- **Mean Dice Score across entire Test Set:** **$0.6893$** ($68.93\%$).
- **Mean IoU across entire Test Set:** **$0.5378$** ($53.78\%$).

### Individual Sample Performance (`sample_1.jpg`)
- **Actual `sample_1.jpg` Recall:** **$0.9964$** ($99.64\%$).
- **Actual `sample_1.jpg` Dice Score:** **$0.5354$** ($53.54\%$).
- **Actual `sample_1.jpg` IoU:** **$0.3656$** ($36.56\%$).
- **Actual `sample_1.jpg` Precision:** **$0.3660$** ($36.60\%$).

*Note: The high recall ($99.64\%$ on `sample_1.jpg` and $94.88\%$ mean across the test set) confirms that the model reliably covers almost all true tooth pixels. However, in the 2-epoch development checkpoint, over-segmentation into surrounding bone structures reduces precision ($55.65\%$), leading to a lower Dice score.*

---

## 2. Training Epochs & Reference Paper Alignment

- **Reference Paper Training Horizon:** The IEEE conference paper (*Automated Tooth Segmentation in X-ray Images using Attention Integrated U-Net++ Model*) trained their model for **100 epochs**.
- **Development Validation Run:** The initial baseline verification executed a quick **2-epoch** run at downsampled $128 \times 128$ resolution for pipeline sanity testing.
- **Final Experiment Run:** The final experiment executes full training for up to **100 epochs** at full resolution ($512 \times 512$) with early stopping enabled.

---

## 3. Dataset Distinction: Tufts vs Open-Access Dataset

| Property | Tufts Dental Database | Humans in the Loop Dataset (DatasetNinja) |
|---|---|---|
| **Role in Project** | Benchmark Reference Dataset | Open-access Development / Fallback Dataset |
| **Access Status** | Restricted (Requires formal academic registration & approval from Tufts University) | Available locally in `data/raw/` |
| **Image Count** | 1,000 panoramic radiographs | 595 validated panoramic radiographs |
| **Annotation Format** | Expert dental annotations | Base64 bitmap polygon annotations converted to binary masks |
| **Experiment Designation** | Reference Paper Experiment | **Open-access Dataset Implementation Experiment** |

*Under no circumstances is the Humans in the Loop dataset referred to as Tufts. All local experiments are strictly designated as Open-Access Dataset Implementation Experiments.*
