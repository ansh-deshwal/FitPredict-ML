# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Multi-modal protein fitness prediction for beta-lactamase TEM-1 (BLAT_ECOLX). Given a single-point mutant sequence, the model predicts its DMS fitness score. The pipeline has five extraction stages followed by training scripts:

1. `Scripts/extract_embeddings.py` — ESM-2 650M → `Data/beta_lactamase_esm2_embeddings.npy` (4996 × 1280)
2. `Scripts/extract_structure_features.py` — PDB 1M40 + DSSP → `Results/beta_lactamase_structure_features.npy` (4996 × 11)
3. `Scripts/extract_esm1v_scores.py` — ESM-1v masked-marginal → `Results/beta_lactamase_esm1v_scores.npy` (4996,)
4. `Scripts/extract_esm1v_scores_ensemble.py` — ESM-1v all-5-member ensemble average → `Results/beta_lactamase_esm1v_ensemble_scores.npy` (4996,)
5. `Scripts/extract_evolutionary_features.py` — MSA (BLAT_ECOLX_MSA.a2m) → `Results/beta_lactamase_evolutionary_features.npy` (4996 × 22)
6. Training scripts (any order): `train_baseline.py`, `train_mlp.py`, `train_fusion.py`, `train_fusion_v2.py`, `train_fusion_v3.py`, `train_fusion_v4.py`, `train_fusion_v5.py`, `train_multidataset.py` → outputs in `Results/` and `Models/`
7. `Scripts/evaluate_generalization.py` — zero-shot transfer evaluation on Deng 2012, Firnberg 2014, Jacquier 2013

## Running the scripts

```bash
# Requires: torch, esm, biopython, scikit-learn, scipy, matplotlib, pandas, numpy, tqdm
# DSSP must be installed: sudo apt-get install dssp  (or: conda install -c salilab dssp)

python Scripts/extract_embeddings.py                  # ~30 min on CPU, needs CUDA for speed
python Scripts/extract_structure_features.py          # requires mkdssp on PATH
python Scripts/extract_esm1v_scores.py                # ESM-1v member 1 only, ~5 min on GPU
python Scripts/extract_esm1v_scores_ensemble.py       # ESM-1v all 5 members averaged, ~25 min on GPU
python Scripts/extract_evolutionary_features.py       # MSA features, needs Data/BLAT_ECOLX_MSA.a2m
python Scripts/train_baseline.py                      # Ridge regression, no GPU needed
python Scripts/train_mlp.py                           # PyTorch MLP, ~2 min on GPU
python Scripts/train_fusion.py                        # Fusion v1, ~5 min on GPU
python Scripts/train_fusion_v2.py                     # Fusion v2 (StructureEncoder), ~5 min on GPU
python Scripts/train_fusion_v3.py                     # Fusion v3 (+ ESM-1v score), ~5 min on GPU
python Scripts/train_fusion_v4.py                     # Fusion v4 (seq + struct + MSA evol), ~5 min on GPU
python Scripts/train_fusion_v5.py                     # Fusion v5 (all 4 modalities), ~5 min on GPU
python Scripts/evaluate_generalization.py             # cross-dataset transfer eval, ~15 min on GPU (first run)
python Scripts/train_multidataset.py                  # multi-assay training, ~10 min on GPU
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
        ├─ extract_esm1v_scores.py ────► Results/beta_lactamase_esm1v_scores.npy          [4996]
        ├─ extract_esm1v_scores_ensemble.py ► Results/beta_lactamase_esm1v_ensemble_scores.npy [4996]
        └─ extract_evolutionary_features.py ► Results/beta_lactamase_evolutionary_features.npy [4996×22]
                        │
          ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
   train_  train_    train_    train_    train_    train_    train_    train_
   baseline mlp      fusion    fusion_v2 fusion_v3 fusion_v4 fusion_v5 multidataset
   Ridge   MLP       Fusion v1 Fusion v2 Fusion v3 Fusion v4 Fusion v5 Multi-assay
   80/20   70/15/15  70/15/15  70/15/15  70/15/15  70/15/15  70/15/15  80/20 pooled
   ρ=0.64  ρ=0.73    ρ=0.82    ρ=0.81    ρ=0.88    ρ=0.83    ρ=0.88    Deng ρ=TBD
```

**Fusion v1** (`train_fusion.py`): ESM-2 (1280-d) + z-scored struct (11-d) → 1291-d concat → Linear(1291→512) → 2×ResidualBlock → head (512→128→1).

**Fusion v2** (`train_fusion_v2.py`): adds `StructureEncoder` (11→64) before fusion → 1344-d concat → same residual architecture.

**Fusion v3** (`train_fusion_v3.py`): extends v2 with z-scored ESM-1v score (1-d) → [seq(1280) | struct_enc(64) | esm1v(1)] = 1345-d. ρ=0.8774. ESM-1v grouped by position: ~263 forward passes for 4996 variants.

**Fusion v4 / 3-modal** (`train_fusion_v4.py`): seq (1280) + `StructureEncoder` (11→64) + `EvolEncoder` (22→64) → 1408-d. Replaces ESM-1v with explicit MSA features. ρ=0.8294. Note: 27% of mutations fall at BLAT_ECOLX insertion positions (no MSA column) and receive zero evolutionary features.

**Fusion v5 / 4-modal** (`train_fusion_v5.py`): all four modalities — [seq(1280) | struct_enc(64) | esm1v(1) | evol_enc(64)] = 1409-d. Best single-assay model (ρ=0.8821). Same residual architecture as v3/v4.

**Multi-dataset** (`train_multidataset.py`): trains on Stiffler 2015 + Firnberg 2014 + Jacquier 2013 simultaneously using rank-normalised fitness targets [0,1] per dataset. Eliminates cross-assay scale mismatch. Evaluates on Deng 2012 as fully held-out generalization test. Uses same FourModalFusion architecture as v5. ESM-2 embeddings for non-Stiffler datasets are cached in `Results/` after first run of `evaluate_generalization.py`.

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

## Generalization results (cross-dataset, no retraining)

Fusion v3 evaluated on three held-out datasets using position-based feature lookup:

| Dataset | N | Spearman ρ | R² | Notes |
|---|---|---|---|---|
| Stiffler 2015 (test set) | 749 | 0.8774 | — | trained on this |
| Deng 2012 | 4996 | 0.5702 | -0.55 | different assay scale |
| Firnberg 2014 | 4783 | 0.8838 | -16.6 | good ranking, wrong scale |
| Jacquier 2013 | 989 | 0.7802 | 0.48 | good transfer |

R² is negative for Deng and Firnberg because each dataset uses a completely different fitness scale — the model predicts in the wrong absolute range. `train_multidataset.py` addresses this with rank normalisation.

## Things in progress / known gaps

- `train_multidataset.py` has been written but not yet run — Deng 2012 generalization ρ (currently 0.57) is TBD.
- ESM-1v ensemble scores (`extract_esm1v_scores_ensemble.py`) have been extracted but no training script uses them yet — plugging them into `train_fusion_v5.py` is a straightforward swap.
- ProteinGym-style 5-fold CV evaluation (for a directly comparable published benchmark number) is not yet implemented.
