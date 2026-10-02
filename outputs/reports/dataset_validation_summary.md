# Dataset Validation Summary Report

**Execution Command:** `python dataset_check.py`  
**Dataset Path:** `./data/raw`  
**Validation Status:** `[SUCCESS] PASSED`

---

## Quantitative Validation Metrics

| Metric | Result |
|---|---|
| **Total Raw Images Found** | 595 |
| **Total Raw Masks Found** | 595 |
| **Successfully Matched Pairs** | 595 |
| **Unmatched Images** | 0 |
| **Unmatched Masks** | 0 |
| **Corrupted Files** | 0 |
| **Average Image Height** | Variable (~1024 px) |
| **Average Image Width** | Variable (~2041 px) |
| **Average Tooth Foreground %** | **14.55%** |

---

## Validation Findings
- 100% of raw X-ray images have a perfectly corresponding binary tooth mask with identical resolution.
- Mask values are strictly binary: `0` for background and `1` for foreground tooth structures.
- No corrupted files, truncated byte streams, or zero-byte images were detected.
