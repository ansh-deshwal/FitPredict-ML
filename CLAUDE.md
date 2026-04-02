# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Multi-modal protein fitness prediction for beta-lactamase TEM-1 (BLAT_ECOLX). Given a single-point mutant sequence, the model predicts its DMS fitness score. The pipeline has five extraction stages followed by training scripts:

1. `Scripts/extract_embeddings.py` — ESM-2 650M → `Data/beta_lactamase_esm2_embeddings.npy` (4996 × 1280)
2. `Scripts/extract_structure_features.py` — PDB 1M40 + DSSP → `Results/beta_lactamase_structure_features.npy` (4996 × 11)
3. `Scripts/extract_esm1v_scores.py` — ESM-1v masked-marginal → `Results/beta_lactamase_esm1v_scores.npy` (4996,)
4. `Scripts/extract_evolutionary_features.py` — MSA (BLAT_ECOLX_MSA.a2m) → `Results/beta_lactamase_evolutionary_features.npy` (4996 × 22)
5. Training scripts (any order): `train_baseline.py`, `train_mlp.py`, `train_fusion.py`, `train_fusion_v2.py`, `train_fusion_v3.py`, `train_fusion_v4.py` → outputs in `Results/` and `Models/`

## Running the scripts

```bash
# Requires: torch, esm, biopython, scikit-learn, scipy, matplotlib, pandas, numpy, tqdm
# DSSP must be installed: sudo apt-get install dssp  (or: conda install -c salilab dssp)

python Scripts/extract_embeddings.py              # ~30 min on CPU, needs CUDA for speed
python Scripts/extract_structure_features.py      # requires mkdssp on PATH
python Scripts/extract_esm1v_scores.py            # ESM-1v masked-marginal, ~5 min on GPU
python Scripts/extract_evolutionary_features.py   # MSA features, needs Data/BLAT_ECOLX_MSA.a2m
python Scripts/train_baseline.py                  # Ridge regression, no GPU needed
python Scripts/train_mlp.py                       # PyTorch MLP, ~2 min on GPU
python Scripts/train_fusion.py                    # Fusion v1, ~5 min on GPU
python Scripts/train_fusion_v2.py                 # Fusion v2 (StructureEncoder), ~5 min on GPU
python Scripts/train_fusion_v3.py                 # Fusion v3 (+ ESM-1v score), ~5 min on GPU
python Scripts/train_fusion_v4.py                 # Fusion v4 (seq + struct + MSA evol), ~5 min on GPU
```

All scripts use `Path(__file__)` for paths — run them from any directory.

### Getting the MSA file (required for extract_evolutionary_features.py)

```bash
curl -o DMS_msa_files.zip \
  "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_msa_files.zip"
unzip -j DMS_msa_files.zip "DMS_msa_files/BLAT_ECOLX_full_11-26-2021_b02.a2m" -d Data/
mv Data/BLAT_ECOLX_full_11-26-2021_b02.a2m Data/BLAT_ECOLX_MSA.a2m
rm DMS_msa_files.zip
```

Note: the zip is ~1.5 GB. The extracted MSA is 65 MB (209,644 sequences).

## Architecture

**Data flow:**
```
BLAT_ECOLX_Stiffler_2015.csv (mutant, mutated_sequence, DMS_score)
        │
        ├─ extract_embeddings.py ──────► Data/beta_lactamase_esm2_embeddings.npy      [4996×1280]
        ├─ extract_structure_features.py ► Results/beta_lactamase_structure_features.npy [4996×11]
        ├─ extract_esm1v_scores.py ────► Results/beta_lactamase_esm1v_scores.npy       [4996]
        └─ extract_evolutionary_features.py ► Results/beta_lactamase_evolutionary_features.npy [4996×22]
                        │
          ┌─────────────┬─────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
   train_baseline   train_mlp   train_fusion  train_fusion  train_fusion  train_fusion
                                              _v2           _v3           _v4
   Ridge 80/20    MLP 70/15/15  Fusion v1     Fusion v2     Fusion v3     Fusion v4
   ρ=0.6435       ρ=0.7318      ρ=0.8248      ρ=0.8140      ρ=0.8774      ρ=0.8294
```

**Fusion v1** (`train_fusion.py`): ESM-2 (1280-d) + z-scored struct (11-d) → 1291-d concat → Linear(1291→512) → 2×ResidualBlock → head (512→128→1).

**Fusion v2** (`train_fusion_v2.py`): adds `StructureEncoder` (11→64) before fusion → 1344-d concat → same residual architecture.

**Fusion v3** (`train_fusion_v3.py`): extends v2 with z-scored ESM-1v score (1-d) → [seq(1280) | struct_enc(64) | esm1v(1)] = 1345-d. Best model (ρ=0.8774). ESM-1v grouped by position: ~263 forward passes for 4996 variants.

**Fusion v4 / 3-modal** (`train_fusion_v4.py`): seq (1280) + `StructureEncoder` (11→64) + `EvolEncoder` (22→64) → 1408-d. Replaces ESM-1v with explicit MSA features. Authors: Anshita Sharma, Ansh Jain. Note: 27% of mutations fall at BLAT_ECOLX insertion positions (no MSA column) and receive zero evolutionary features.

**Train/val/test split:**
- `train_baseline.py`: 80/20 train/test, no validation set
- All other training scripts: 70% / 15% / 15% train/val/test, `random_state=42`. Val set drives `ReduceLROnPlateau` and early stopping; test set used only for final evaluation.

**Normalisation:** structural and evolutionary features are z-score normalised using train-split mean/std only (no leakage).

## Structural features (11 columns, in order)

`burial_score`, `is_buried`, `ss_helix`, `ss_sheet`, `ss_coil`, `rASA`, `sin_phi`, `cos_phi`, `sin_psi`, `cos_psi`, `contact_count`

Residues missing from the PDB structure get all-zero rows. DSSP-dependent features (`ss_*`, `rASA`, angles) are zeroed if `mkdssp` is unavailable.

## Evolutionary features (22 columns, in order)

`entropy`, `gap_frac`, `freq_A`, `freq_C`, `freq_D`, `freq_E`, `freq_F`, `freq_G`, `freq_H`, `freq_I`, `freq_K`, `freq_L`, `freq_M`, `freq_N`, `freq_P`, `freq_Q`, `freq_R`, `freq_S`, `freq_T`, `freq_V`, `freq_W`, `freq_Y`

Computed from 209,644 sequences in the ProteinGym MSA. The BLAT_ECOLX reference has 215 match-state columns out of ~280 HMM states; 71 residue positions are insertions relative to the HMM and receive all-zero rows.

## Things in progress / known gaps

- Three alternative CSV datasets in `Data/` (`Deng_2012`, `Firnberg_2014`, `Jacquier_2013`) are present but unused.
- ESM-1v uses only ensemble member 1; averaging all 5 members could improve score quality slightly.
- Fusion v5 (combining all modalities: seq + struct + ESM-1v + MSA evol = 1409-d) is the natural next step.
