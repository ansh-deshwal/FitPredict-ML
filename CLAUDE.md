# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Multi-modal protein fitness prediction for beta-lactamase TEM-1 (BLAT_ECOLX). Given a single-point mutant sequence, the model predicts its DMS fitness score. The pipeline has three stages that must be run in order:

1. `Scripts/extract_embeddings.py` — ESM-2 650M → `Data/beta_lactamase_esm2_embeddings.npy` (4996 × 1280)
2. `Scripts/extract_structure_features.py` — PDB 1M40 + DSSP → `Results/beta_lactamase_structure_features.npy` (4996 × 11)
3. Training scripts (any order): `train_baseline.py`, `train_mlp.py`, `train_fusion.py`, `train_fusion_v2.py` → outputs in `Results/` and `Models/`

## Running the scripts

```bash
# Requires: torch, esm, biopython, scikit-learn, scipy, matplotlib, pandas, numpy, tqdm
# DSSP must be installed: sudo apt-get install dssp  (or: conda install -c salilab dssp)

python Scripts/extract_embeddings.py          # ~30 min on CPU, needs CUDA for speed
python Scripts/extract_structure_features.py  # requires mkdssp on PATH
python Scripts/train_baseline.py              # Ridge regression, no GPU needed
python Scripts/train_mlp.py                   # PyTorch MLP, ~2 min on GPU
python Scripts/train_fusion.py                # Fusion v1, ~5 min on GPU
python Scripts/train_fusion_v2.py             # Fusion v2 (StructureEncoder), ~5 min on GPU
```

All scripts use `Path(__file__)` for paths — run them from any directory.

## Architecture

**Data flow:**
```
BLAT_ECOLX_Stiffler_2015.csv (mutant, mutated_sequence, DMS_score)
        │
        ├─ extract_embeddings.py ──► Data/beta_lactamase_esm2_embeddings.npy  [4996×1280]
        └─ extract_structure_features.py ──► Results/beta_lactamase_structure_features.npy  [4996×11]
                                                        │
                    ┌───────────────────────────────────┼────────────────────────┐
               train_baseline.py          train_mlp.py  │  train_fusion.py  train_fusion_v2.py
               (Ridge, 80/20)         (MLP, 70/15/15)   │  (Fusion v1)      (Fusion v2)
               ρ=0.6435               ρ=0.7318           │  ρ=0.8248         ρ=0.8140
                                                         │  (70/15/15)       (70/15/15)
```

**Fusion v1** (`train_fusion.py`): concatenates ESM-2 embeddings (1280-d) + z-score normalised structure features (11-d) → 1291-d concat → linear projection to 512-d → two `ResidualBlock`s → head (512→128→1). `struct_dim` is read from `S.shape[1]` at runtime.

**Fusion v2** (`train_fusion_v2.py`): adds a `StructureEncoder` (Linear(11→64) → BN → ReLU → Dropout(0.3) → Linear(64→64) → ReLU) before fusion → 1344-d concat → same residual architecture. Same training config as v1.

**Train/val/test split:**
- `train_baseline.py`: 80/20 train/test, no validation set
- All other training scripts: 70% / 15% / 15% train/val/test, `random_state=42`. Val set drives `ReduceLROnPlateau` and early stopping; test set used only for final evaluation.

**Normalisation:** structural features are z-score normalised using train-split mean/std only (fitted after split, applied to val and test — no leakage).

## Structural features (11 columns, in order)

`burial_score`, `is_buried`, `ss_helix`, `ss_sheet`, `ss_coil`, `rASA`, `sin_phi`, `cos_phi`, `sin_psi`, `cos_psi`, `contact_count`

Residues missing from the PDB structure get all-zero rows. DSSP-dependent features (`ss_*`, `rASA`, angles) are zeroed if `mkdssp` is unavailable.

## Things in progress / known gaps

- Three alternative CSV datasets in `Data/` (`Deng_2012`, `Firnberg_2014`, `Jacquier_2013`) are present but unused.
- Stage 8 (evolutionary/MSA features) is not yet implemented.
