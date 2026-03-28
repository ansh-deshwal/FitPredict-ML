# FitPredict-ML 🧬

**Multi-Modal Protein Fitness Prediction using Deep Learning**

[![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)](https://github.com/ansh-deshwal/FitPredict-ML)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Model](https://img.shields.io/badge/Model-ESM--2%20(650M)-green)](https://github.com/facebookresearch/esm)
[![Data](https://img.shields.io/badge/Data-ProteinGym-purple)](https://proteingym.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> FitPredict-ML predicts the functional effects of protein point mutations by combining **sequence embeddings** (ESM-2), **structural features** (PDB), and (eventually) **evolutionary information** (MSA).

---

## 🎯 Project Overview

This project predicts DMS fitness scores for single-point mutants of β-lactamase TEM-1. The current pipeline implements three modalities:

- ✅ **Sequence Embeddings** — ESM-2 (650M parameters), 1280-d per variant
- ✅ **Structural Features** — 11 features from PDB 1M40: burial depth, DSSP secondary structure, rASA, backbone dihedral angles, contact count
- 🔜 **Evolutionary Conservation** — MSA features, not yet implemented

### Dataset
- **Protein**: β-lactamase (BLAT_ECOLX / TEM-1)
- **Source**: Stiffler et al. 2015
- **Mutations**: 4,996 single-point variants
- **Structure**: PDB ID 1M40 (0.85 Å resolution)
- **Label**: DMS fitness score (continuous)

---

## 📊 Project Progress

| Stage | Component | Status | Description |
|-------|-----------|--------|-------------|
| **Stage 1** | Data Prep | ✅ Done | Loaded and validated β-lactamase DMS dataset |
| **Stage 2** | Sequence Branch | ✅ Done | Extracted 1280-d ESM-2 embeddings |
| **Stage 3** | Baseline Models | ✅ Done | Ridge (80/20 split) and MLP (70/15/15 split) on embeddings |
| **Stage 4** | Structure Branch | ✅ Done | Extracted 11 structural features from PDB 1M40 |
| **Stage 5** | Multi-Modal Fusion | ✅ Done | Sequence + structure fusion model with residual blocks |
| **Stage 6** | Evolutionary Branch | ⏳ Pending | MSA features |

> **Note on reported numbers:** All ρ values in this README (ρ = 0.719 for MLP, ρ = 0.768 for Fusion) were produced before recent code fixes. The models need to be retrained to produce valid updated numbers.

---

## 📈 Current Results

All metrics below are **stale** — produced before the current split and path fixes. They are included for reference only. Retraining will update them.

### Sequence-Only Baselines

| Model | Split | Test Spearman ρ | Test Pearson r | Test R² | Parameters |
|-------|-------|----------------|----------------|---------|------------|
| Ridge Regression | 80/20 | ~0.500 | ~0.500 | ~0.250 | 1,281 |
| **MLP (3-layer)** | 80/20 (old) | **0.719** | 0.702 | 0.399 | 721,665 |

### MLP Architecture

```
Input (1280) → Linear(512) → ReLU → Dropout(0.3)
             → Linear(128) → ReLU → Dropout(0.3)
             → Linear(1)

Optimizer:   Adam (lr=0.001, weight_decay=1e-5)
Loss:        MSE
Batch size:  32
Epochs:      50 (fixed, no early stopping)
Scheduler:   ReduceLROnPlateau (factor=0.5, patience=5)
Split:       70/15/15 (current code); old metrics used 80/20
Checkpoint:  best val Spearman ρ saved and reloaded for final eval
```

### Fusion Model Results

| Model | Split | Test Spearman ρ |
|-------|-------|----------------|
| Multi-modal Fusion | pre-fix | 0.768 (stale) |

See fusion architecture details in the [Methodology](#-methodology) section.

---

## 📂 Repository Structure

```
FitPredict-ML/
│
├── Data/
│   ├── BLAT_ECOLX_Stiffler_2015.csv          # Primary DMS dataset (4,996 mutations) — used by all scripts
│   ├── BLAT_ECOLX_Deng_2012.csv              # Alternative DMS dataset — unused
│   ├── BLAT_ECOLX_Firnberg_2014.csv          # Alternative DMS dataset — unused
│   ├── BLAT_ECOLX_Jacquier_2013.csv          # Alternative DMS dataset — unused
│   └── 1M40.pdb                              # β-lactamase 3D structure (PDB ID 1M40)
│
├── Scripts/
│   ├── extract_embeddings.py                 # ESM-2 embeddings → Data/beta_lactamase_esm2_embeddings.npy
│   ├── extract_structure_features.py         # PDB structure features → Results/beta_lactamase_structure_features.npy
│   ├── train_baseline.py                     # Ridge Regression (80/20 split)
│   ├── train_mlp.py                          # 3-layer MLP (70/15/15 split)
│   └── train_fusion.py                       # Multi-modal fusion: sequence + structure (70/15/15 split)
│
├── Results/
│   ├── baseline_plot.png                     # Ridge scatter plots
│   ├── baseline_predictions.csv              # Ridge test predictions
│   ├── mlp_baseline_plot.png                 # MLP loss curve + scatter plots
│   ├── mlp_predictions.csv                   # MLP test predictions
│   ├── fusion_plot.png                       # Fusion loss curve + scatter plots
│   ├── fusion_predictions.csv                # Fusion test predictions
│   ├── beta_lactamase_structure_features.npy # Structural feature matrix (4,996 × 11) — committed
│   └── beta_lactamase_structure_features.csv # Same, human-readable — committed
│
├── ProteinForge Initial.pdf                  # Project proposal document
├── proteinfit_progress1.docx                 # Progress notes
├── requirements.txt
├── CLAUDE.md
└── README.md
```

> **Note:** `Data/beta_lactamase_esm2_embeddings.npy` (~25 MB, float32) and model checkpoints (`best_mlp_model.pt`, `best_fusion_model.pt`) are excluded from version control via `.gitignore`. The structural feature files in `Results/` are committed.

---

## 🚀 Quick Start

### 1️⃣ Clone and install

```bash
git clone https://github.com/ansh-deshwal/FitPredict-ML.git
cd FitPredict-ML
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2️⃣ Extract features

**Sequence embeddings** (one-time, ~30 min on GPU, ~2 h on CPU):
```bash
python Scripts/extract_embeddings.py
```
Output: `Data/beta_lactamase_esm2_embeddings.npy` (4,996 × 1280, float32, ~25 MB)

**Structural features** (one-time, ~1 min — requires `mkdssp` for full output):
```bash
python Scripts/extract_structure_features.py
```
Output: `Results/beta_lactamase_structure_features.npy` (4,996 × 11). Already committed; re-run only if you change the extraction logic.

### 3️⃣ Train models

```bash
python Scripts/train_baseline.py   # Ridge regression
python Scripts/train_mlp.py        # MLP
python Scripts/train_fusion.py     # Multi-modal fusion
```

All scripts resolve paths via `Path(__file__)` and can be run from any directory.

---

## 🛠️ Installation & DSSP Setup

DSSP computes secondary structure (helix/sheet/coil), rASA, and backbone dihedral angles. Without it, those 8 features (indices 2–9) default to 0; only `burial_score`, `is_buried`, and `contact_count` are populated.

### Linux / WSL
```bash
sudo apt-get update && sudo apt-get install dssp
which mkdssp   # should print /usr/bin/mkdssp
```

### macOS
```bash
brew install brewsci/bio/dssp
```

### Windows (native)
No Windows binary is available from the `salilab` conda channel. Use WSL (above) or the pure-Python fallback:
```bash
pip install pydssp
```
`pydssp` is not the reference Kabsch & Sander implementation and may produce minor numerical differences, but should not materially affect model training.

---

## 🔬 Methodology

### Phase 1: Sequence-Only Baseline

**Feature extraction (`extract_embeddings.py`):**
- ESM-2 650M parameter transformer (layer 33)
- Mean pooling over non-special tokens (excludes CLS at index 0 and EOS): `token_embeddings[j, 1 : seq_len + 1].mean(0)`
- Output: 1280-d float32 vector per variant, saved to `Data/`

**Ridge regression (`train_baseline.py`):**
- `sklearn.linear_model.Ridge(alpha=1.0)`
- 80/20 train/test split, no validation set
- Outputs: `Results/baseline_predictions.csv`, `Results/baseline_plot.png`

**MLP (`train_mlp.py`):**
- Architecture: `Linear(1280→512) → ReLU → Dropout(0.3) → Linear(512→128) → ReLU → Dropout(0.3) → Linear(128→1)`
- 721,665 trainable parameters
- Adam (lr=0.001, weight_decay=1e-5), MSE loss, batch size 32, 50 epochs
- `ReduceLROnPlateau` (factor=0.5, patience=5) on val loss
- Gradient clipping: max_norm=1.0
- 70/15/15 train/val/test split (random_state=42); best val Spearman ρ checkpoint reloaded for final eval
- Outputs: `Results/mlp_predictions.csv`, `Results/mlp_baseline_plot.png`, `Results/best_mlp_model.pt`

### Phase 2: Structural Features

**Extraction (`extract_structure_features.py`):**
1. Downloads PDB 1M40 if not present in `Data/`
2. Parses chain A Cα coordinates
3. Builds pairwise Cα–Cα contact map (8 Å threshold)
4. Runs DSSP via Biopython for secondary structure, rASA, and backbone dihedrals
5. For each mutation (e.g., `H24C`), extracts the residue number and looks it up in the PDB by residue sequence number. Residues not present in the PDB structure (1M40 covers residues 26–290) get all-zero rows.

**11 features per mutation (in order):**

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 0 | `burial_score` | int | Cα neighbours within 10 Å (excludes self) |
| 1 | `is_buried` | binary | 1 if burial_score > 20, else 0 |
| 2 | `ss_helix` | binary | 1 if DSSP assigns helix (H) |
| 3 | `ss_sheet` | binary | 1 if DSSP assigns strand (E) |
| 4 | `ss_coil` | binary | 1 if DSSP assigns coil/other |
| 5 | `rASA` | float [0,1] | Relative accessible surface area |
| 6 | `sin_phi` | float [-1,1] | sin of backbone φ dihedral |
| 7 | `cos_phi` | float [-1,1] | cos of backbone φ dihedral |
| 8 | `sin_psi` | float [-1,1] | sin of backbone ψ dihedral |
| 9 | `cos_psi` | float [-1,1] | cos of backbone ψ dihedral |
| 10 | `contact_count` | int | Cα contacts within 8 Å |

Features 2–9 require `mkdssp` and are zeroed if DSSP is unavailable. Features 0, 1, and 10 require only Cα coordinates.

### Phase 3: Multi-Modal Fusion

**Architecture (`train_fusion.py`):**

```
Sequence (1280-d)  ──┐
                      ├──> Concat (1291-d) → Linear(512) → BN → ReLU → Dropout(0.3)
Structure (11-d)   ──┘
                                ↓
                        ResidualBlock(512)
                                ↓
                        ResidualBlock(512)
                                ↓
                        Linear(512→128) → BN → ReLU → Dropout(0.2) → Linear(128→1)
```

Each `ResidualBlock(dim)` contains:
```
block: BN(dim) → Linear(dim,dim) → ReLU → Dropout(0.2) → BN(dim) → Linear(dim,dim) → Dropout(0.2)
output: ReLU(block(x) + x)
```
The skip connection is added before the outer ReLU. `struct_dim` is read from `S.shape[1]` at runtime.

**Training configuration:**
- Optimizer: Adam (lr=0.001, weight_decay=1e-4)
- Loss: MSE
- Batch size: 32
- Max epochs: 100 with early stopping (patience=15, monitors val Spearman ρ)
- Scheduler: ReduceLROnPlateau on val loss (factor=0.5, patience=5)
- Gaussian noise augmentation on embeddings during training (std=0.01)
- Gradient clipping: max_norm=1.0
- Structural features: z-score normalised per feature before splitting
- Split: 70/15/15 train/val/test (random_state=42); best val Spearman ρ checkpoint reloaded
- Outputs: `Results/fusion_predictions.csv`, `Results/fusion_plot.png`, `Results/best_fusion_model.pt`

---

## 📊 Benchmark Comparison

> All ρ values below are stale (produced before current code fixes). Retrain all models to get valid numbers.

| Approach | Spearman ρ | Modality | Notes |
|----------|-----------|----------|-------|
| Random baseline | 0.000 | None | No signal |
| Ridge (Linear) | ~0.500 | Sequence | Linear baseline, 80/20 split |
| ESM-1v (zero-shot) | ~0.650 | Sequence | Literature estimate |
| **MLP (3-layer)** | **0.719** | Sequence | Pre-fix, 80/20 split |
| **Multi-modal Fusion** | **0.768** | Seq + Struct | Pre-fix; needs retraining |
| SOTA Literature | ~0.75–0.80 | Seq + Struct + Evol | Published benchmarks |

---

## 🔮 Roadmap

### Completed ✅
- [x] Data loading and validation
- [x] ESM-2 sequence embedding extraction (1280-d)
- [x] Ridge regression baseline
- [x] 3-layer MLP on sequence embeddings
- [x] Structural feature extraction from PDB 1M40 (11 features)
- [x] Multi-modal fusion network (sequence + structure, residual blocks)
- [x] 70/15/15 train/val/test split in MLP and Fusion training
- [x] Best-checkpoint save/reload in MLP and Fusion

### In Progress 🔄
- [ ] Retrain MLP and Fusion under current code and report updated numbers (Ridge is already correct)
- [ ] Ablation: sequence-only vs. structure-only vs. fusion
- [ ] Cross-validation framework

### Planned 📋
- [ ] Evolutionary features from MSA
- [ ] Hyperparameter optimisation (Optuna/Ray Tune)
- [ ] K-fold cross-validation
- [ ] Extend to additional ProteinGym datasets
- [ ] Model interpretability

---

## 📖 Key Concepts

### Deep Mutational Scanning (DMS)
High-throughput technique that measures the functional effect of thousands of protein mutations simultaneously. Each variant receives a fitness score (log-enrichment ratio) relative to wild-type.

### ESM-2 Protein Language Model
Transformer trained on ~250M protein sequences. Outputs per-residue representations that capture evolutionary constraints and structural context without explicit 3D supervision.

### Burial Depth
Number of Cα neighbours within 10 Å. Residues with more than 20 neighbours are classified as buried (protein core); others are surface-exposed.

### DSSP
The Kabsch & Sander (1983) algorithm for assigning secondary structure from 3D coordinates using hydrogen bond geometry. Also computes solvent-accessible surface area and backbone dihedrals. Requires the `mkdssp` binary.

### Relative Accessible Surface Area (rASA)
Fraction of a residue's solvent-accessible surface area relative to its theoretical maximum in an extended Ala-X-Ala tripeptide. 0 = fully buried, 1 = fully exposed.

### Backbone Dihedral Angles (φ, ψ)
Local backbone conformation angles encoded as (sin, cos) pairs to avoid the ±180° discontinuity.

### Contact Map
Binary N×N matrix where entry (i, j) = 1 if Cα atoms i and j are within 8 Å. Per-residue contact count (row sum) captures structural centrality.

### Spearman Rank Correlation (ρ)
Non-parametric measure of monotonic association. Used as the primary metric because correct *ranking* of mutations by severity is more important for prioritisation than absolute accuracy.

---

## 🛠️ Troubleshooting

### GPU not detected
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Reinstall with CUDA support:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### PDB download fails
```bash
wget https://files.rcsb.org/download/1M40.pdb -P Data/
```

### DSSP not found
On Windows, use WSL (`sudo apt-get install dssp`) or install the pure-Python fallback (`pip install pydssp`).

### Out of memory during embedding extraction
Reduce `batch_size` in `extract_embeddings.py` (default 4; try 2 or 1 on 4 GB VRAM).

### Out of memory during training
Reduce `batch_size` / `BATCH_SIZE` in the training script (default 32; try 16 or 8).

---

## 📚 References

1. **ESM-2**: Lin et al. (2023) — "Evolutionary-scale prediction of atomic-level protein structure with a language model" — *Science* 379(6628)
2. **ProteinGym**: Notin et al. (2023) — "ProteinGym: Large-scale benchmarks for protein fitness prediction" — *NeurIPS*
3. **Dataset**: Stiffler et al. (2015) — "Evolvability as a function of purifying selection in TEM-1 β-lactamase" — *Cell* 160(5)
4. **PDB structure**: Minasov et al. (2002) — "An ultrahigh resolution structure of TEM-1 β-lactamase" — *J. Am. Chem. Soc.* 124(19)
5. **DSSP**: Kabsch & Sander (1983) — "Dictionary of protein secondary structure" — *Biopolymers* 22(12)

---

## 🤝 Contributing

Areas of active interest:
- Evolutionary / MSA features
- Alternative fusion architectures
- Extended benchmarks on other ProteinGym proteins

**Process:** fork → feature branch → PR.

---

## 📝 License

MIT — see [LICENSE](LICENSE).

---

## 👤 Authors

**Ansh** — Baseline models · [ansh-deshwal](https://github.com/ansh-deshwal)

**Ansh Jain** — Multi-modal fusion · [ataylus](https://github.com/ataylus)

**Anshita Sharma** — Data analysis · [anshita-3](https://github.com/anshita-3)

**Arjun Sharma** — Structure feature extraction · [arjsh16](https://github.com/arjsh16)

---

## 🙏 Acknowledgments

Meta AI (ESM-2), ProteinGym team, RCSB PDB, Stiffler Lab, Biopython community.
