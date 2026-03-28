# FitPredict-ML

**Multi-Modal Protein Fitness Prediction using Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Model](https://img.shields.io/badge/Model-ESM--2%20(650M)-green)](https://github.com/facebookresearch/esm)
[![Data](https://img.shields.io/badge/Data-ProteinGym-purple)](https://proteingym.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Predicts DMS fitness scores for single-point mutants of β-lactamase TEM-1 by combining sequence embeddings (ESM-2), structural features (PDB 1M40), and a multi-modal fusion network.

---

## Project Overview

- **Protein**: β-lactamase TEM-1 (BLAT_ECOLX)
- **Dataset**: Stiffler et al. 2015 — 4,996 single-point variants, continuous DMS fitness score
- **Structure**: PDB ID 1M40 (0.85 Å resolution)
- **Modalities implemented**: sequence embeddings (ESM-2 1280-d) + structural features (11-d from PDB)
- **Modality not yet implemented**: evolutionary / MSA features

---

## Pipeline

The scripts must be run in this order:

```
1. extract_embeddings.py        → Data/beta_lactamase_esm2_embeddings.npy
2. extract_structure_features.py → Results/beta_lactamase_structure_features.npy
3. train_baseline.py             (no GPU required)
   train_mlp.py
   train_fusion.py
   train_fusion_v2.py            (steps 3a–3d can run in any order)
```

---

## Project Status

| Stage | Script | Status | Notes |
|-------|--------|--------|-------|
| 1 — Data prep | — | Done | CSV loaded and validated |
| 2 — Sequence embeddings | `extract_embeddings.py` | Done | 4,996 × 1280 ESM-2 vectors |
| 3 — Structural features | `extract_structure_features.py` | Done | 4,996 × 11, committed to repo |
| 4 — Ridge baseline | `train_baseline.py` | Done | 80/20 split |
| 5 — MLP baseline | `train_mlp.py` | Done | 70/15/15 split |
| 6 — Fusion v1 | `train_fusion.py` | Done | seq + raw struct concat |
| 7 — Fusion v2 | `train_fusion_v2.py` | Done (not yet run) | StructureEncoder + no-leakage norm |
| 8 — Evolutionary features | — | Not started | MSA features |

> **All reported ρ values are stale** — produced before recent split and path fixes. Models need to be retrained to produce valid numbers.

---

## Current Results

| Model | Split | Test Spearman ρ | Test Pearson r | Test R² |
|-------|-------|----------------|----------------|---------|
| Ridge Regression | 80/20 | 0.6435 | 0.6065 | 0.269 |
| MLP (3-layer) | 70/15/15 | 0.7318 | 0.7289 | 0.404 |
| **Fusion v1** | 70/15/15 | **0.8248** | **0.8256** | **0.680** |
| Fusion v2 | 70/15/15 | 0.8140 | 0.8062 | 0.637 |

---

## Repository Structure

```
FitPredict-ML/
│
├── Data/
│   ├── BLAT_ECOLX_Stiffler_2015.csv          # Primary dataset — used by all scripts
│   ├── BLAT_ECOLX_Deng_2012.csv              # Unused
│   ├── BLAT_ECOLX_Firnberg_2014.csv          # Unused
│   ├── BLAT_ECOLX_Jacquier_2013.csv          # Unused
│   └── 1M40.pdb                              # β-lactamase structure (downloaded by extract_structure_features.py)
│
├── Scripts/
│   ├── extract_embeddings.py                 # ESM-2 → Data/beta_lactamase_esm2_embeddings.npy
│   ├── extract_structure_features.py         # PDB + DSSP → Results/beta_lactamase_structure_features.npy
│   ├── train_baseline.py                     # Ridge regression, 80/20 split
│   ├── train_mlp.py                          # 3-layer MLP, 70/15/15 split
│   ├── train_fusion.py                       # Fusion v1: raw concat(seq+struct), 70/15/15
│   └── train_fusion_v2.py                    # Fusion v2: StructureEncoder(11→64), 70/15/15
│
├── Results/
│   ├── beta_lactamase_structure_features.npy # (4,996 × 11) — committed to git
│   ├── beta_lactamase_structure_features.csv # Same, human-readable — committed to git
│   ├── baseline_plot.png                     # Ridge: train + test scatter (2 panels)
│   ├── baseline_predictions.csv              # Ridge test-set predictions
│   ├── mlp_baseline_plot.png                 # MLP: loss curve + train + test scatter (3 panels)
│   ├── mlp_predictions.csv                   # MLP test-set predictions
│   ├── fusion_plot.png                       # Fusion v1: loss curve + scatter (3 panels)
│   ├── fusion_predictions.csv                # Fusion v1 test-set predictions
│   ├── fusion_v2_plot.png                    # Fusion v2: loss curve + scatter — generated on first run
│   └── fusion_v2_predictions.csv             # Fusion v2 test-set predictions — generated on first run
│
├── Models/                                   # Created at runtime by train_fusion_v2.py
│   └── best_fusion_v2.pt                     # Best checkpoint — gitignored (*.pt)
│
├── ProteinForge Initial.pdf
├── proteinfit_progress1.docx
├── requirements.txt
├── CLAUDE.md
└── README.md
```

> `Data/beta_lactamase_esm2_embeddings.npy` (~25 MB, float32) and all `*.pt` checkpoints
> (`Results/best_mlp_model.pt`, `Results/best_fusion_model.pt`, `Models/best_fusion_v2.pt`)
> are excluded by `.gitignore`. The structural feature `.npy` and `.csv` files in `Results/` are committed.

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/ansh-deshwal/FitPredict-ML.git
cd FitPredict-ML
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Extract sequence embeddings (one-time, ~30 min on GPU)

```bash
python Scripts/extract_embeddings.py
```

Output: `Data/beta_lactamase_esm2_embeddings.npy` (4,996 × 1280, float32, ~25 MB)

### 3. Extract structural features (one-time, ~1 min)

```bash
python Scripts/extract_structure_features.py
```

Output: `Results/beta_lactamase_structure_features.npy` (4,996 × 11).
Already committed — re-run only if you change the extraction logic.
Requires `mkdssp` for full output; see [DSSP Setup](#dssp-setup) below.

### 4. Train models

```bash
python Scripts/train_baseline.py      # Ridge — CPU only, seconds
python Scripts/train_mlp.py           # MLP — ~2 min on GPU
python Scripts/train_fusion.py        # Fusion v1 — ~5 min on GPU
python Scripts/train_fusion_v2.py     # Fusion v2 — ~5 min on GPU
```

All scripts resolve paths via `Path(__file__)` and can be run from any directory.

---

## DSSP Setup

DSSP computes secondary structure (helix/sheet/coil), rASA, and backbone dihedral angles. Without it, features 2–9 are zeroed; only `burial_score`, `is_buried`, and `contact_count` are populated.

**Linux / WSL**
```bash
sudo apt-get update && sudo apt-get install dssp
which mkdssp    # should print /usr/bin/mkdssp
```

**macOS**
```bash
brew install brewsci/bio/dssp
```

**Windows (native) — no binary available**
```bash
pip install pydssp
```
`pydssp` is not the reference Kabsch & Sander implementation and may produce minor numerical differences.

---

## Methodology

### Stage 1: Sequence Embeddings (`extract_embeddings.py`)

- Loads `mutated_sequence` column from `BLAT_ECOLX_Stiffler_2015.csv`
- Model: ESM-2 650M (`esm2_t33_650M_UR50D`), eval mode, layer 33
- Batch size: 4 (reduce to 2 or 1 on 4 GB VRAM)
- Mean pooling over sequence positions: `token_embeddings[j, 1 : seq_len + 1].mean(0)` — excludes CLS (index 0) and EOS (index seq_len + 1)
- Output: 1280-d float32 vector per variant

### Stage 2: Structural Features (`extract_structure_features.py`)

- Downloads PDB 1M40 to `Data/` if not present
- Parses chain A Cα coordinates
- Burial: counts Cα neighbours within 10 Å (excludes self); buried if count > 20
- Contact map: binary Cα–Cα contacts within 8 Å; per-residue count = row sum
- DSSP (if `mkdssp` available): secondary structure (H/E/other→C), rASA, backbone φ/ψ → (sin, cos) pairs; undefined angles (360°) → (0, 0)
- Residue number extracted from mutant string (e.g., `H24C` → 24); residues with unresolvable position or absent from PDB Cα dict get all-zero rows

**11 features per variant (in order):**

| # | Name | Type | Description |
|---|------|------|-------------|
| 0 | `burial_score` | int | Cα neighbours within 10 Å |
| 1 | `is_buried` | binary | 1 if burial_score > 20 |
| 2 | `ss_helix` | binary | 1 if DSSP assigns H |
| 3 | `ss_sheet` | binary | 1 if DSSP assigns E |
| 4 | `ss_coil` | binary | 1 if DSSP assigns other (mapped to C) |
| 5 | `rASA` | float [0,1] | Relative accessible surface area |
| 6 | `sin_phi` | float [-1,1] | sin of backbone φ |
| 7 | `cos_phi` | float [-1,1] | cos of backbone φ |
| 8 | `sin_psi` | float [-1,1] | sin of backbone ψ |
| 9 | `cos_psi` | float [-1,1] | cos of backbone ψ |
| 10 | `contact_count` | int | Cα contacts within 8 Å |

Features 2–9 require `mkdssp` and are zeroed if unavailable.

### Stage 3a: Ridge Regression (`train_baseline.py`)

- `sklearn.linear_model.Ridge(alpha=1.0)`
- Split: 80/20 train/test (random_state=42), no validation set
- Outputs: `Results/baseline_predictions.csv`, `Results/baseline_plot.png`

### Stage 3b: MLP (`train_mlp.py`)

**Architecture:**
```
Linear(1280 → 512) → ReLU → Dropout(0.3)
Linear(512 → 128)  → ReLU → Dropout(0.3)
Linear(128 → 1)
```
721,665 trainable parameters.

**Training:**
- Split: 70/15/15 train/val/test (random_state=42)
  - `train_test_split(..., test_size=0.15)` → trainval + test
  - `train_test_split(..., test_size=0.1765)` → train + val  (0.1765 × 0.85 ≈ 0.15 of total)
- Optimizer: Adam (lr=0.001, weight_decay=1e-5)
- Loss: MSE, batch size 32, **50 fixed epochs** (no early stopping)
- Scheduler: ReduceLROnPlateau on val loss (factor=0.5, patience=5)
- Gradient clipping: max_norm=1.0
- Best val Spearman ρ checkpoint saved to `Results/best_mlp_model.pt`; reloaded for final eval
- Outputs: `Results/mlp_predictions.csv`, `Results/mlp_baseline_plot.png`, `Results/best_mlp_model.pt` (gitignored)

### Stage 3c: Fusion v1 (`train_fusion.py`)

**Architecture:**
```
Sequence (1280-d)  ──┐
                      ├──> Concat (1291-d) → Linear(512) → BN → ReLU → Dropout(0.3)
Structure (11-d)   ──┘                              ↓
                                            ResidualBlock(512)
                                                    ↓
                                            ResidualBlock(512)
                                                    ↓
                                    Linear(512→128) → BN → ReLU → Dropout(0.2)
                                                    ↓
                                              Linear(128→1)
```

Each `ResidualBlock(dim)`:
```
block:  BN(dim) → Linear(dim,dim) → ReLU → Dropout(0.2) → BN(dim) → Linear(dim,dim) → Dropout(0.2)
output: ReLU(block(x) + x)
```
`struct_dim` is read from `S.shape[1]` at runtime.

**Training:**
- Split: same 70/15/15 as MLP
- Structural features: z-score normalised using **train-split mean/std only** (fitted after split, applied to val and test)
- Optimizer: Adam (lr=0.001, weight_decay=1e-4)
- Loss: MSE, batch size 32, max 100 epochs
- Gaussian noise on embeddings during training (std=0.01)
- Gradient clipping: max_norm=1.0
- Scheduler: ReduceLROnPlateau on val loss (factor=0.5, patience=5)
- Early stopping: patience=15 on val Spearman ρ
- Best val Spearman ρ checkpoint saved to `Results/best_fusion_model.pt`; reloaded for final eval
- Outputs: `Results/fusion_predictions.csv`, `Results/fusion_plot.png`, `Results/best_fusion_model.pt` (gitignored)

### Stage 3d: Fusion v2 (`train_fusion_v2.py`)

Replaces the raw 11-d structural concatenation with a learned `StructureEncoder` that projects structural features to a 64-d representation before fusion.

**Architecture:**
```
Sequence (1280-d)  ──────────────────────────────────┐
                                                       ├──> Concat (1344-d) → Linear(512) → BN → ReLU → Dropout(0.3)
Structure (11-d) → StructureEncoder (64-d)  ──────────┘              ↓
                                                               ResidualBlock(512)
                                                                       ↓
                                                               ResidualBlock(512)
                                                                       ↓
                                               Linear(512→128) → BN → ReLU → Dropout(0.2)
                                                                       ↓
                                                                 Linear(128→1)
```

`StructureEncoder`:
```
Linear(11→64) → BN → ReLU → Dropout(0.2) → Linear(64→64) → ReLU
```

`ResidualBlock` is identical to v1.

**Training:** identical to Fusion v1, except:
- Creates `Models/` directory at runtime
- Best checkpoint saved to `Models/best_fusion_v2.pt`
- Outputs: `Results/fusion_v2_predictions.csv`, `Results/fusion_v2_plot.png`, `Models/best_fusion_v2.pt` (gitignored)

---

## Benchmark

| Model | Modality | Split | Spearman ρ |
|-------|----------|-------|-----------|
| Ridge | Sequence only | 80/20 | 0.6435 |
| MLP | Sequence only | 70/15/15 | 0.7318 |
| **Fusion v1** | Seq + Struct | 70/15/15 | **0.8248** |
| Fusion v2 | Seq + Struct | 70/15/15 | 0.8140 |

---

## Roadmap

**Done**
- [x] ESM-2 embedding extraction
- [x] Structural feature extraction from PDB 1M40 (11 features)
- [x] Ridge regression baseline (80/20)
- [x] 3-layer MLP on sequence embeddings (70/15/15)
- [x] Multi-modal fusion v1: raw concat + residual blocks (70/15/15, no-leakage normalisation)
- [x] Multi-modal fusion v2: StructureEncoder + no-leakage normalisation
- [x] Full pipeline run — all models trained and evaluated (ρ: Ridge 0.64 → MLP 0.73 → Fusion v1 0.82)

**Pending**
- [ ] Evolutionary / MSA features (Stage 8)
- [ ] Ablation: sequence-only vs structure-only vs fusion
- [ ] Hyperparameter optimisation
- [ ] Extend to additional ProteinGym datasets

---

## Key Concepts

**Deep Mutational Scanning (DMS):** High-throughput technique measuring functional effects of thousands of protein mutations. Each variant receives a fitness score (log-enrichment ratio) relative to wild-type.

**ESM-2:** Transformer trained on ~250M protein sequences. Outputs per-residue representations capturing evolutionary and structural context without explicit 3D supervision.

**Burial depth:** Number of Cα neighbours within 10 Å. Residues with more than 20 neighbours are classified as buried (protein core).

**DSSP:** Kabsch & Sander (1983) algorithm assigning secondary structure from hydrogen bond geometry. Also computes rASA and backbone dihedrals. Requires `mkdssp`.

**rASA:** Relative accessible surface area — fraction of a residue's solvent exposure relative to its theoretical maximum. 0 = fully buried, 1 = fully exposed.

**Backbone dihedrals (φ, ψ):** Encoded as (sin, cos) pairs to avoid the ±180° discontinuity.

**Spearman ρ:** Primary evaluation metric. Measures monotonic rank correlation — correct *ranking* of mutations is more useful for experimental prioritisation than absolute accuracy.

---

## Troubleshooting

**GPU not detected**
```bash
python -c "import torch; print(torch.cuda.is_available())"
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**PDB download fails**
```bash
wget https://files.rcsb.org/download/1M40.pdb -P Data/
```

**DSSP not found** — use WSL (`sudo apt-get install dssp`) or `pip install pydssp`.

**Out of memory during embedding extraction** — reduce `batch_size` in `extract_embeddings.py` (default 4; try 1 or 2).

**Out of memory during training** — reduce `BATCH_SIZE` / `batch_size` in the training script (default 32; try 16 or 8).

---

## References

1. Lin et al. (2023) — ESM-2: "Evolutionary-scale prediction of atomic-level protein structure with a language model" — *Science* 379(6628)
2. Notin et al. (2023) — ProteinGym: "Large-scale benchmarks for protein fitness prediction" — *NeurIPS*
3. Stiffler et al. (2015) — "Evolvability as a function of purifying selection in TEM-1 β-lactamase" — *Cell* 160(5)
4. Minasov et al. (2002) — PDB 1M40: "An ultrahigh resolution structure of TEM-1 β-lactamase" — *J. Am. Chem. Soc.* 124(19)
5. Kabsch & Sander (1983) — DSSP: "Dictionary of protein secondary structure" — *Biopolymers* 22(12)

---

## Authors

**Ansh** — Baseline models · [ansh-deshwal](https://github.com/ansh-deshwal)

**Ansh Jain** — Multi-modal fusion · [ataylus](https://github.com/ataylus)

**Anshita Sharma** — Data analysis · [anshita-3](https://github.com/anshita-3)

**Arjun Sharma** — Structure feature extraction · [arjsh16](https://github.com/arjsh16)

---

## License

MIT — see [LICENSE](LICENSE).
