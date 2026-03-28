# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Multi-modal protein fitness prediction for beta-lactamase TEM-1 (BLAT_ECOLX). Given a single-point mutant sequence, the model predicts its DMS fitness score. The pipeline has three stages that must be run in order:

1. `Scripts/extract_embeddings.py` — ESM-2 650M → `Data/beta_lactamase_esm2_embeddings.npy` (4996 × 1280)
2. `Scripts/extract_structure_features.py` — PDB 1M40 + DSSP → `Results/beta_lactamase_structure_features.npy` (4996 × 11)
3. Training scripts (any order): `train_baseline.py`, `train_mlp.py`, `train_fusion.py` → outputs in `Results/`

## Running the scripts

```bash
# Requires: torch, esm, biopython, scikit-learn, scipy, matplotlib, pandas, numpy, tqdm
# DSSP must be installed: sudo apt-get install dssp  (or: conda install -c salilab dssp)

cd Scripts
python extract_embeddings.py          # ~30 min on CPU, needs CUDA for speed
python extract_structure_features.py  # requires mkdssp on PATH
python train_baseline.py              # Ridge regression, no GPU needed
python train_mlp.py                   # PyTorch MLP, ~2 min on GPU
python train_fusion.py                # Fusion model, ~5 min on GPU
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
                              ┌─────────────────────────┴──────────────────────┐
                         train_baseline.py          train_mlp.py         train_fusion.py
                         (Ridge on embeds)      (MLP on embeds)     (embeds + struct feats)
                           ρ≈0.50 (stale)        ρ=0.719 (stale)      ρ=0.768 (stale)
```

**Fusion model** (`train_fusion.py`): concatenates ESM-2 embeddings + z-score normalised structure features → linear projection to 512-d → two `ResidualBlock`s (BatchNorm → Linear → ReLU → Dropout × 2, with skip connection) → head (512→128→1). `struct_dim` is read from `S.shape[1]` at runtime.

**Train/val/test split:** 70% / 15% / 15%, `random_state=42`. The val set drives `ReduceLROnPlateau` and early stopping; the test set is only used for final evaluation.

## Structural features (11 columns, in order)

`burial_score`, `is_buried`, `ss_helix`, `ss_sheet`, `ss_coil`, `rASA`, `sin_phi`, `cos_phi`, `sin_psi`, `cos_psi`, `contact_count`

Residues missing from the PDB structure get all-zero rows. DSSP-dependent features (`ss_*`, `rASA`, angles) are zeroed if `mkdssp` is unavailable.

## Things in progress / known gaps

- Three alternative CSV datasets in `Data/` (`Deng_2012`, `Firnberg_2014`, `Jacquier_2013`) are present but unused.
- Stage 6 (evolutionary/MSA features) is not yet implemented.
