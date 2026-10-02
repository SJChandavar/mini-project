# Training Recovery & Resumable Pipeline Documentation

## 1. Overview & Previous Interruption

During a previous full-training attempt of the U-Net++ with ECA attention architecture (`unetplusplus_eca`), an unexpected hardware interruption (laptop system restart/power loss) occurred. 

Because model training on CPU/laptop hardware is resource-intensive and prone to thermal throttling or sudden restarts, making the training pipeline **crash-resistant**, **resume-safe**, and **hardware-optimized** was critical before embarking on any long training runs.

---

## 2. Inspection of Previous Training State

The existing experiment directories were thoroughly audited to preserve existing progress without overwriting baseline results:

* **Experiment Directory:** `outputs/experiments/unetplusplus_eca/`
* **Available Checkpoint Files:**
  * `best_model.pth` (Best epoch model snapshot)
  * `last_model.pth` (Snapshot after epoch 2)
  * `training_history.csv` (Logged metrics for completed epochs)
* **Status Summary:**
  * **Last Successfully Completed Epoch:** 2
  * **Best Validation Epoch:** 2
  * **Best Validation Loss:** 0.6015
  * **Best Validation Dice Score:** 0.6796 (Validation) / 0.7485 (Test Set)
  * **All existing checkpoints and metric logs remain preserved without deletion.**

---

## 3. Resumable & Crash-Resistant Architecture

The training pipeline has been upgraded with the following safeguards:

1. **Per-Epoch Checkpointing (`checkpoint_latest.pth`):**
   * After **every** successfully completed epoch, `checkpoint_latest.pth` is atomically saved to the experiment's checkpoint directory.
   * State dictionary includes: `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `epoch`, `best_val_loss`, `best_val_dice`, `training_history`, `config`, and `random_seed`.

2. **Isolated Best Model Safeguard (`best_model.pth`):**
   * `best_model.pth` is kept strictly separate from `checkpoint_latest.pth`.
   * `best_model.pth` is updated **only** when validation Dice performance strictly improves. If a run crashes, `best_model.pth` remains undamaged.

3. **Seamless CLI Resumes (`--resume`):**
   * Passing `--resume` to `train.py` automatically detects the latest valid checkpoint (`checkpoint_latest.pth` or fallback `best_model.pth`).
   * It restores model parameters, optimizer momentum, learning rate scheduler state, and past history.
   * Training seamlessly continues from epoch $N+1$ without restarting from epoch 1.

4. **Chunked Training Runs (`--max-epochs-this-run`):**
   * Allows running training in safe time-bounded chunks (e.g., `--max-epochs-this-run 5`).
   * Training stops cleanly after completing the target chunk size while leaving `checkpoint_latest.pth` ready for subsequent resume commands.

---

## 4. Hardware Optimization & Resource Management

To prevent system lockups, thermal shutdowns, or high RAM consumption:

1. **Batch Size & Gradient Accumulation:**
   * **Batch Size:** Set to `1` image per forward pass to maintain a tiny memory footprint.
   * **Gradient Accumulation:** Supported via `gradient_accumulation_steps: 4` in configuration and CLI, yielding an effective batch size of 4 without keeping multiple samples in memory simultaneously.

2. **Memory Hygiene:**
   * Explicit deletion of transient loss/prediction tensors at each batch.
   * Explicit python `gc.collect()` and `torch.cuda.empty_cache()` calls after each batch/epoch.
   * Strict `torch.no_grad()` usage during validation.
   * `pin_memory: false` and `mixed_precision: false` enforced on CPU mode.

3. **System Resource Monitoring:**
   * Automated monitoring of CPU usage %, system RAM usage (MB), and epoch execution duration (seconds) via `psutil`.
   * Logged continuously to `outputs/logs/resource_usage.csv`.

4. **CPU Safety Warning:**
   * Immediate warning printed when launching $512 \times 512$ U-Net++ + ECA training on CPU to alert the user of hardware intensity.

---

## 5. Configurations

Two configurations are maintained:

### A. Conservative Testing Configuration (`configs/safe_cpu.yaml`)
* **Resolution:** $256 \times 256$
* **Batch Size:** 1
* **Gradient Accumulation Steps:** 4
* **Workers:** 0
* **Purpose:** Controlled validation and testing of resume features without heavy load.

### B. Paper-Aligned Final Configuration (`configs/final_eca.yaml`)
* **Model Architecture:** U-Net++ with ECA Bottleneck Attention
* **Resolution:** $512 \times 512$
* **Loss Function:** `BCEWithLogitsLoss`
* **Optimizer:** Adam ($lr = 0.0001$)
* **Max Epochs:** 100
* **Early Stopping Patience:** 10
* **Seed:** 42
* **Batch Size:** 1
* **Gradient Accumulation Steps:** 4

---

## 6. Controlled Resume System Test Results

A controlled test script ([`scratch/test_resume_system.py`](file:///c:/Users/Shamanth%20J%20Chandavar/Downloads/tooth_segmentation/scratch/test_resume_system.py)) was executed:

1. **Phase 1:** Fresh training run launched for 2 epochs (`--max-epochs-this-run 2`).
   * **Result:** `checkpoint_latest.pth` successfully written after Epoch 1 and Epoch 2.
2. **Interruption:** Process stopped cleanly after Epoch 2.
3. **Phase 2:** Relaunched with `--resume` for 2 additional epochs (`--max-epochs-this-run 2`).
   * **Result:** Automatically detected `checkpoint_latest.pth`, loaded optimizer & scheduler states, resumed seamlessly from **Epoch 3**, and completed Epoch 4.
   * `training_history.csv` preserved rows for Epochs 1, 2, 3, and 4 sequentially without duplication or reset.
4. **Unit Tests:** All 20 project unit tests passed (`pytest`).

---

## 7. Exact Next Command for Real Long Training

To resume or start the paper-aligned $512 \times 512$ full training run safely when ready, execute:

```bash
python train.py \
    --model unetplusplus_eca \
    --config configs/final_eca.yaml \
    --resume \
    --max-epochs-this-run 10
```

> [!NOTE]
> Training can be executed in incremental chunks of 5–10 epochs (or fully up to 100 epochs) using the command above. Each chunk will update `checkpoint_latest.pth` after every single epoch, guaranteeing zero loss of progress in case of system power interruptions.
