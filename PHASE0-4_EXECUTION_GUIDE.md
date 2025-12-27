# Phase 0-4 Minimal Delivery Package
## Exact Files & Folders Needed to Complete All Phases

**Purpose:** Send to another person's PC to execute phases 0→4 without the entire project  
**Total Package Size:** ~5-8 GB (mostly data)  
**Timeline:** 5-6 days compute time  

---

## 📦 FOLDER STRUCTURE (Minimal)

```
ids-compression-benchmark/
├── checkpoints/
│   └── pytorch_fixed_full/              ← REQUIRED: Base model checkpoint
├── data/
│   └── splits/                          ← REQUIRED: Training/validation data
├── models/
│   └── lstm_ids_v2.py                  ← NEW: Phase 2 LSTM architecture
├── scripts/
│   ├── phase0_per_class_analysis.py    ← PHASE 0: Diagnostic analysis
│   ├── phase1_improvements.py           ← PHASE 1: Improvement utilities
│   ├── train_phase1_improved.py         ← PHASE 1: Training script
│   ├── train_phase2_lstm.py             ← PHASE 2: LSTM training
│   ├── phase3_ensemble.py               ← PHASE 3: Ensemble utilities
│   ├── train_phase3_ensemble.py         ← PHASE 3: Ensemble optimization
│   └── orchestrate_training.py          ← OPTIONAL: Run all phases at once
├── configs/
│   ├── kd_config.yaml                  ← Config files (optional but recommended)
│   ├── canonical_schema.yaml
│   └── class_weights.json
├── results/                             ← OUTPUT: Will be created automatically
├── environment.yml                      ← Python environment
├── requirements.txt                     ← Python dependencies
└── PHASE0-4_EXECUTION_GUIDE.md         ← Instructions (this file)
```

---

## 📋 FILES TO COPY (Detailed Checklist)

### ✅ **MUST-COPY SECTION**

#### 1. **Checkpoints Directory** `checkpoints/pytorch_fixed_full/`
These are the base model and training data for all phases.

```
checkpoints/pytorch_fixed_full/
├── best_model.pt              ✓ REQUIRED - Base CNN checkpoint (88.17% F1)
├── validation_data.pt         ✓ REQUIRED - Validation set for phase 0 analysis
└── (optional: checkpoint_epoch_*.pt if you want all intermediate checkpoints)
```

**Size:** ~500 MB  
**Purpose:** Phase 0 loads this model to analyze per-class failures

---

#### 2. **Data Splits Directory** `data/splits/`
These are the training and validation datasets for phases 1-3.

```
data/splits/
├── train.parquet              ✓ REQUIRED - Training set (~76M samples)
├── val.parquet                ✓ REQUIRED - Validation set (~balanced)
└── split_manifest.json        ✓ REQUIRED - Metadata about splits
```

**Size:** ~4-5 GB  
**Purpose:** Phases 1, 2, 3 use this for retraining and evaluation  
**NOTE:** These files are LARGE. If bandwidth limited, ask for compressed archive.

---

#### 3. **Script Files** `scripts/`
The Python scripts for each phase.

```
scripts/
├── phase0_per_class_analysis.py         ✓ REQUIRED
├── phase1_improvements.py               ✓ REQUIRED
├── train_phase1_improved.py             ✓ REQUIRED
├── train_phase2_lstm.py                 ✓ REQUIRED
├── phase3_ensemble.py                   ✓ REQUIRED
├── train_phase3_ensemble.py             ✓ REQUIRED
└── orchestrate_training.py              ✓ OPTIONAL (convenience script)
```

**Size:** ~1-2 MB  
**Purpose:** Phase 0-4 execution scripts

---

#### 4. **Models Directory** `models/`
The LSTM architecture (new in phases 2+).

```
models/
├── lstm_ids_v2.py                       ✓ REQUIRED
└── __init__.py                          ✓ REQUIRED (just copy as-is)
```

**Size:** <1 MB  
**Purpose:** LSTM model definition for phase 2

---

#### 5. **Config Files** `configs/`
Optional but recommended for consistency.

```
configs/
├── kd_config.yaml                       ✓ RECOMMENDED
├── canonical_schema.yaml                ✓ RECOMMENDED
├── class_weights.json                   ✓ RECOMMENDED
└── (others can be skipped)
```

**Size:** <1 MB  
**Purpose:** Reference configurations

---

#### 6. **Environment & Dependencies**
```
├── requirements.txt                     ✓ REQUIRED - Python packages
├── environment.yml                      ✓ REQUIRED - Conda environment (if using conda)
```

**Purpose:** Install dependencies with `pip install -r requirements.txt`

---

### ⚠️ **DO NOT COPY** (Will Be Created)

The following will be generated automatically - no need to copy:

```
results/                                 ← Auto-created, will store outputs
├── phase0_analysis/                    ← Phase 0 analysis results
├── phase1_improved_model/               ← Phase 1 trained model
├── phase2_lstm_model/                   ← Phase 2 trained model
└── phase3_ensemble/                     ← Phase 3 ensemble config
```

---

## 🚀 EXECUTION SEQUENCE

### **PHASE 0: Diagnostic Analysis** (4 hours)
```bash
python scripts/phase0_per_class_analysis.py
```
**Input:** `checkpoints/pytorch_fixed_full/best_model.pt` + `validation_data.pt`  
**Output:** Per-class metrics, confusion matrix, failure patterns  
**Expected:** Identifies which classes drag down F1

---

### **PHASE 1: CNN Improvements** (1-2 days)
```bash
python scripts/train_phase1_improved.py
```
**Input:** Base CNN checkpoint + training data  
**Output:** Improved CNN checkpoint (88.9-89.5% F1)  
**Expected:** Mixup + Temperature scaling + Threshold tuning

---

### **PHASE 2: LSTM Training** (1-2 days)
```bash
python scripts/train_phase2_lstm.py
```
**Input:** Training data (LSTM expects same format as CNN)  
**Output:** LSTM checkpoint (~89.0-89.8% F1)  
**Expected:** Bidirectional LSTM with temporal feature learning  
**Note:** Can run in parallel with Phase 1 if using separate GPU

---

### **PHASE 3: Ensemble Optimization** (2-3 hours)
```bash
python scripts/train_phase3_ensemble.py
```
**Input:** Phase 1 CNN checkpoint + Phase 2 LSTM checkpoint  
**Output:** Ensemble configuration (90.0-91.0% F1) ✓ TARGET  
**Expected:** Optimized weights and temperatures for both models

---

### **PHASE 4: Statistical Validation** (1 hour, optional)
```bash
# To be executed if phase4_statistical_validation.py is created
# Provides: Bootstrap confidence intervals, statistical tests
```

---

## 📊 EXPECTED RESULTS

| Phase | Expected F1 | Gain | Output |
|-------|------------|------|--------|
| Baseline (Phase 0) | 88.17% | — | Analysis report |
| Phase 1 | 88.9-89.5% | +0.73-1.33% | Improved CNN checkpoint |
| Phase 2 | 89.0-89.8% | +0.83-1.63% | LSTM checkpoint |
| Phase 3 | 90.0-91.0% | +0.83-2.83% | Ensemble config ✓ |
| Phase 4 | — | — | Statistical tables |

---

## 💾 DATASET DETAILS

### **Training Data** `data/splits/train.parquet`
- **Samples:** ~76 million (balanced, 5000 per class × 15+ classes)
- **Features:** 41 NetFlow features
- **Format:** Parquet (compressed)
- **Size:** ~3-4 GB

### **Validation Data** `data/splits/val.parquet`
- **Samples:** ~Balanced (1000+ per class)
- **Features:** 41 NetFlow features (same as training)
- **Format:** Parquet
- **Size:** ~1-2 GB

### **Also in Checkpoint**
- `checkpoints/pytorch_fixed_full/validation_data.pt` - Pre-loaded validation set for Phase 0
- **Size:** ~1 GB (same data, different format for faster loading)

---

## ⚙️ SYSTEM REQUIREMENTS

### **Minimum**
- **RAM:** 16 GB (for data loading)
- **GPU:** Optional but strongly recommended (10+ GB VRAM)
  - Without GPU: Phase 1-3 will take 3-5x longer
  - With GPU (RTX 3090 or better): 1-2 days total
- **Disk:** 20 GB free (for model checkpoints + results)
- **Python:** 3.8+

### **Recommended**
- **RAM:** 32 GB
- **GPU:** RTX 3090 / A100 (20+ GB VRAM)
- **Disk:** 50 GB free

---

## 📥 QUICK SETUP CHECKLIST

1. **Copy all folders** from above checklist to new PC
   ```
   checkpoints/
   data/
   models/
   scripts/
   configs/
   requirements.txt
   environment.yml
   ```

2. **Create Python environment**
   ```bash
   # Option A: Using pip
   pip install -r requirements.txt
   
   # Option B: Using conda
   conda env create -f environment.yml
   conda activate ids-benchmark
   ```

3. **Verify checkpoint exists**
   ```bash
   ls checkpoints/pytorch_fixed_full/best_model.pt
   ls data/splits/train.parquet
   ```

4. **Run Phase 0**
   ```bash
   cd scripts
   python phase0_per_class_analysis.py
   ```

5. **Follow console output** - should see progress bars and results

---

## 🔧 TROUBLESHOOTING

### **"best_model.pt not found"**
- Ensure `checkpoints/pytorch_fixed_full/` directory is copied
- Check file permissions (should be readable)

### **"train.parquet not found"**
- Ensure `data/splits/` directory is copied
- This file is ~3-4 GB, may take time to transfer

### **CUDA out of memory**
- Reduce `BATCH_SIZE` in phase scripts (line ~40-50)
- Change from 256 → 128 or 64
- Or run on CPU (slower but will work)

### **"Module not found" errors**
- Run `pip install -r requirements.txt` again
- Verify installation: `python -c "import torch; print(torch.__version__)"`

---

## 📞 SUPPORT

If issues arise, provide:
1. **Error message** (full traceback)
2. **System specs** (GPU type, RAM, Python version)
3. **Phase number** (0, 1, 2, 3, or 4)
4. **Output from** `python scripts/phase0_per_class_analysis.py` (first 50 lines)

---

## 📝 FILE CHECKSUMS (Optional Verification)

If bandwidth concerns, verify downloads:

```bash
# After copying, verify critical files
ls -lh checkpoints/pytorch_fixed_full/best_model.pt
ls -lh data/splits/train.parquet
ls -lh data/splits/val.parquet

# Check total size
du -sh checkpoints/
du -sh data/
du -sh models/
du -sh scripts/
```

Expected:
- `best_model.pt`: ~500 MB
- `train.parquet`: ~3-4 GB
- `val.parquet`: ~1-2 GB
- Total: ~5-8 GB

---

## ✅ FINAL CHECKLIST BEFORE SENDING

- [ ] `checkpoints/pytorch_fixed_full/best_model.pt` copied
- [ ] `checkpoints/pytorch_fixed_full/validation_data.pt` copied
- [ ] `data/splits/train.parquet` copied
- [ ] `data/splits/val.parquet` copied
- [ ] `data/splits/split_manifest.json` copied
- [ ] `scripts/phase0_per_class_analysis.py` copied
- [ ] `scripts/phase1_improvements.py` copied
- [ ] `scripts/train_phase1_improved.py` copied
- [ ] `scripts/train_phase2_lstm.py` copied
- [ ] `scripts/phase3_ensemble.py` copied
- [ ] `scripts/train_phase3_ensemble.py` copied
- [ ] `scripts/orchestrate_training.py` copied (optional)
- [ ] `models/lstm_ids_v2.py` copied
- [ ] `models/__init__.py` copied
- [ ] `requirements.txt` copied
- [ ] `environment.yml` copied
- [ ] `configs/kd_config.yaml` copied (recommended)

---

**Status:** Ready to Ship ✅  
**Created:** December 27, 2025  
**Phases Covered:** 0 → 1 → 2 → 3 → 4

