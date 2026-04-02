# FitPredict-ML

**Multi-Modal Protein Fitness Prediction using Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Model](https://img.shields.io/badge/Model-ESM--2%20(650M)-green)](https://github.com/facebookresearch/esm)
[![Data](https://img.shields.io/badge/Data-ProteinGym-purple)](https://proteingym.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Predicts DMS fitness scores for single-point mutants of β-lactamase TEM-1 by fusing four biological modalities: ESM-2 sequence embeddings, structural features (PDB 1M40 + DSSP), ESM-1v zero-shot scores, and MSA evolutionary frequencies.

---

## Project Overview

- **Protein**: β-lactamase TEM-1 (BLAT_ECOLX) — the enzyme bacteria use to destroy penicillin-class antibiotics
- **Task**: Predict continuous DMS fitness scores for single-point mutants
- **Primary dataset**: Stiffler et al. 2015 — 4,996 single-point variants
- **Structure**: PDB ID 1M40 (0.85 Å resolution)
- **Best single-assay model**: Fusion v5 — Spearman ρ = **0.8821** on held-out test set
- **Modalities**: sequence (ESM-2 1280-d) · structural (11-d PDB/DSSP) · ESM-1v zero-shot score (1-d) · MSA evolutionary frequencies (22-d)

---

## Benchmark Results

| Model | Modalities | Split | Spearman ρ | Pearson r | R² |
|---|---|---|---|---|---|
| Ridge Regression | Sequence | 80/20 | 0.6435 | 0.6065 | 0.269 |
| MLP (3-layer) | Sequence | 70/15/15 | 0.7318 | 0.7289 | 0.404 |
| Fusion v1 | Seq + Struct (raw) | 70/15/15 | 0.8248 | 0.8256 | 0.680 |
| Fusion v2 | Seq + StructEncoder | 70/15/15 | 0.8140 | 0.8062 | 0.637 |
| Fusion v3 | Seq + Struct + ESM-1v | 70/15/15 | 0.8774 | — | — |
| Fusion v4 | Seq + Struct + MSA evol | 70/15/15 | 0.8294 | — | — |
| **Fusion v5** | **Seq + Struct + ESM-1v + MSA evol** | **70/15/15** | **0.8821** | **0.8952** | **0.800** |

All models trained and evaluated on Stiffler 2015 only. Spearman ρ is the primary metric — it measures whether the model correctly *ranks* mutations by fitness.

### Cross-dataset generalization (Fusion v3, no retraining)

| Dataset | N | Spearman ρ | Notes |
|---|---|---|---|
| Stiffler 2015 (test) | 749 | 0.8774 | trained on this |
| Firnberg 2014 | 4783 | 0.8838 | good ranking, different scale |
| Jacquier 2013 | 989 | 0.7802 | good transfer |
| Deng 2012 | 4996 | 0.5702 | hardest case — different assay |

---

## Pipeline

Run extraction scripts first, then training scripts in any order:

```bash
# 1. Sequence embeddings (~30 min on GPU)
python Scripts/extract_embeddings.py

# 2. Structural features (~1 min, requires mkdssp)
python Scripts/extract_structure_features.py

# 3a. ESM-1v scores — single member (~5 min on GPU)
python Scripts/extract_esm1v_scores.py

# 3b. ESM-1v scores — 5-member ensemble average (~25 min on GPU)
python Scripts/extract_esm1v_scores_ensemble.py

# 4. MSA evolutionary features (~2 min, requires Data/BLAT_ECOLX_MSA.a2m)
python Scripts/extract_evolutionary_features.py

# Training (any order, each ~5 min on GPU)
python Scripts/train_baseline.py       # Ridge, CPU only
python Scripts/train_mlp.py            # MLP
python Scripts/train_fusion.py         # Fusion v1
python Scripts/train_fusion_v2.py      # Fusion v2
python Scripts/train_fusion_v3.py      # Fusion v3
python Scripts/train_fusion_v4.py      # Fusion v4
python Scripts/train_fusion_v5.py      # Fusion v5 (best)

# Generalization evaluation (~15 min on GPU, first run)
python Scripts/evaluate_generalization.py

# Multi-dataset training (~10 min on GPU)
python Scripts/train_multidataset.py
```

All scripts resolve paths via `Path(__file__)` — run from any directory.

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

### 2. Get the MSA file (required for evolutionary features only)

```bash
curl -o DMS_msa_files.zip \
  "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_msa_files.zip"
unzip -j DMS_msa_files.zip "DMS_msa_files/BLAT_ECOLX_full_11-26-2021_b02.a2m" -d Data/
mv Data/BLAT_ECOLX_full_11-26-2021_b02.a2m Data/BLAT_ECOLX_MSA.a2m
rm DMS_msa_files.zip
```

The zip is ~1.5 GB. The extracted MSA is 65 MB (209,644 sequences).

### 3. DSSP setup

DSSP computes secondary structure, rASA, and backbone dihedrals. Without it, structural features 2–9 are zeroed.

```bash
# Linux / WSL
sudo apt-get update && sudo apt-get install dssp
which mkdssp    # should print /usr/bin/mkdssp

# macOS
brew install brewsci/bio/dssp

# Windows (native) — no binary available
pip install pydssp
```

---

## Architecture

### Fusion v5 (best model)

```
ESM-2 sequence embedding  (1280-d) ─────────────────────────────────────┐
Structural features (11-d) → StructEncoder → (64-d) ────────────────────┤
ESM-1v zero-shot score     (1-d, z-scored) ─────────────────────────────┤→ Concat (1409-d)
MSA evolutionary features (22-d) → EvolEncoder → (64-d) ────────────────┘
                                                                          ↓
                                                            Linear(1409→512) → BN → ReLU → Dropout(0.3)
                                                                          ↓
                                                                 ResidualBlock(512)
                                                                          ↓
                                                                 ResidualBlock(512)
                                                                          ↓
                                                     Linear(512→128) → BN → ReLU → Dropout(0.2)
                                                                          ↓
                                                                   Linear(128→1)
```

**StructEncoder / EvolEncoder:** `Linear(in→64) → BN → ReLU → Dropout(0.3) → Linear(64→64) → ReLU`

**ResidualBlock:** `BN → Linear → ReLU → Dropout(0.2) → BN → Linear → Dropout(0.2)`, with skip connection + ReLU

**Training:** Adam (lr=0.001, weight_decay=1e-4) · MSE loss · batch 32 · Gaussian noise on embeddings (σ=0.01) · gradient clipping (max_norm=1.0) · ReduceLROnPlateau (factor=0.5, patience=5) · early stopping (patience=15 on val ρ)

### Multi-dataset model

Same architecture as Fusion v5. Key difference: DMS scores from each dataset are **rank-normalised independently** to [0,1] before pooling, eliminating cross-assay scale mismatch. Trained on Stiffler + Firnberg + Jacquier (~14,768 variants), evaluated on Deng 2012 as held-out test.

---

## Feature Details

### Structural features (11-d)

Computed from PDB 1M40, chain A. Each variant uses features from its mutation *position* in the WT structure (structure is WT regardless of which amino acid is substituted).

| # | Feature | Description |
|---|---|---|
| 0 | `burial_score` | Cα neighbours within 10 Å |
| 1 | `is_buried` | 1 if burial_score > 20 |
| 2 | `ss_helix` | 1 if DSSP = H |
| 3 | `ss_sheet` | 1 if DSSP = E |
| 4 | `ss_coil` | 1 if DSSP = other |
| 5 | `rASA` | Relative accessible surface area [0,1] |
| 6–7 | `sin_phi`, `cos_phi` | Backbone φ dihedral |
| 8–9 | `sin_psi`, `cos_psi` | Backbone ψ dihedral |
| 10 | `contact_count` | Cα contacts within 8 Å |

Features 2–9 require `mkdssp`. Residues absent from PDB get all-zero rows.

### ESM-1v score (1-d)

Masked-marginal score: `log P(mut_aa | masked position) − log P(wt_aa | masked position)`. Computed with ~263 forward passes (one per unique position). Default uses ensemble member 1; `extract_esm1v_scores_ensemble.py` averages all 5 members.

### Evolutionary features (22-d)

From ProteinGym MSA (209,644 sequences): `entropy`, `gap_frac`, and amino acid frequencies for all 20 residues. 71 positions are insertions relative to the HMM and receive all-zero rows.

---

## Repository Structure

```
FitPredict-ML/
│
├── Data/
│   ├── BLAT_ECOLX_Stiffler_2015.csv          # Primary dataset (4,996 variants)
│   ├── BLAT_ECOLX_Deng_2012.csv              # Cross-dataset eval
│   ├── BLAT_ECOLX_Firnberg_2014.csv          # Cross-dataset eval
│   ├── BLAT_ECOLX_Jacquier_2013.csv          # Cross-dataset eval
│   ├── 1M40.pdb                              # β-lactamase crystal structure
│   ├── BLAT_ECOLX_MSA.a2m                   # ProteinGym MSA (gitignored, ~65 MB)
│   └── beta_lactamase_esm2_embeddings.npy   # ESM-2 embeddings (gitignored, ~25 MB)
│
├── Scripts/
│   ├── extract_embeddings.py                 # ESM-2 → Data/beta_lactamase_esm2_embeddings.npy
│   ├── extract_structure_features.py         # PDB/DSSP → Results/beta_lactamase_structure_features.npy
│   ├── extract_esm1v_scores.py               # ESM-1v member 1 → Results/beta_lactamase_esm1v_scores.npy
│   ├── extract_esm1v_scores_ensemble.py      # ESM-1v 5-member avg → Results/beta_lactamase_esm1v_ensemble_scores.npy
│   ├── extract_evolutionary_features.py      # MSA → Results/beta_lactamase_evolutionary_features.npy
│   ├── train_baseline.py                     # Ridge regression (80/20)
│   ├── train_mlp.py                          # MLP on sequence only (70/15/15)
│   ├── train_fusion.py                       # Fusion v1: seq + struct raw concat
│   ├── train_fusion_v2.py                    # Fusion v2: + StructureEncoder
│   ├── train_fusion_v3.py                    # Fusion v3: + ESM-1v score
│   ├── train_fusion_v4.py                    # Fusion v4: seq + struct + MSA evol
│   ├── train_fusion_v5.py                    # Fusion v5: all 4 modalities (best)
│   ├── evaluate_generalization.py            # Cross-dataset transfer eval (no retraining)
│   └── train_multidataset.py                 # Multi-assay training with rank normalisation
│
├── Results/                                  # Plots, predictions, cached embeddings
├── Models/                                   # Model checkpoints (*.pt, gitignored)
├── requirements.txt
├── CLAUDE.md
└── README.md
```

---

## Roadmap

**Done**
- [x] ESM-2 embedding extraction
- [x] Structural feature extraction from PDB 1M40 (11 features)
- [x] MSA evolutionary feature extraction (22 features)
- [x] ESM-1v zero-shot scoring (single member + 5-member ensemble)
- [x] Ridge regression baseline
- [x] MLP on sequence embeddings
- [x] Fusion v1–v5 (incremental modality ablation)
- [x] Cross-dataset generalization evaluation (Deng 2012, Firnberg 2014, Jacquier 2013)
- [x] Multi-dataset training with rank normalisation

**Pending**
- [ ] Run `train_multidataset.py` — Deng 2012 generalization ρ TBD
- [ ] Train Fusion v5 with ESM-1v ensemble scores (swap `.npy` path)
- [ ] ProteinGym-style 5-fold CV for direct comparison with published methods
- [ ] Multi-protein generalization

---

## Key Concepts

**Deep Mutational Scanning (DMS):** High-throughput technique measuring functional effects of thousands of protein mutations simultaneously. Each variant receives a fitness score (log-enrichment ratio) relative to wild-type.

**ESM-2:** Transformer trained on ~250M protein sequences. Outputs 1280-d per-sequence representations capturing evolutionary and structural context.

**ESM-1v:** Protein language model for variant effect prediction. Uses masked-marginal scoring: mask a position, ask the model what amino acid it expects, compute log-odds of mutant vs wild-type.

**Spearman ρ:** Primary evaluation metric. Measures monotonic rank correlation — correctly *ranking* mutations is more useful for experimental prioritisation than predicting absolute values.

**Rank normalisation:** Converting each dataset's fitness scores to percentile ranks [0,1] before cross-dataset training. Eliminates scale mismatch between assays that use different readouts and measurement ranges.

---

## References

1. Lin et al. (2023) — ESM-2: "Evolutionary-scale prediction of atomic-level protein structure with a language model" — *Science* 379(6628)
2. Meier et al. (2021) — ESM-1v: "Language models enable zero-shot prediction of the effects of mutations on protein function" — *NeurIPS*
3. Notin et al. (2023) — ProteinGym: "Large-scale benchmarks for protein fitness prediction" — *NeurIPS*
4. Stiffler et al. (2015) — "Evolvability as a function of purifying selection in TEM-1 β-lactamase" — *Cell* 160(5)
5. Minasov et al. (2002) — PDB 1M40: "An ultrahigh resolution structure of TEM-1 β-lactamase" — *J. Am. Chem. Soc.* 124(19)
6. Kabsch & Sander (1983) — DSSP: "Dictionary of protein secondary structure" — *Biopolymers* 22(12)

---

## Authors

**Ansh** — Baseline models · [ansh-deshwal](https://github.com/ansh-deshwal)

**Ansh Jain** — Multi-modal fusion · [ataylus](https://github.com/ataylus)

**Anshita Sharma** — Data analysis · [anshita-3](https://github.com/anshita-3)

**Arjun Sharma** — Structure feature extraction · [arjsh16](https://github.com/arjsh16)

---

## License

MIT — see [LICENSE](LICENSE).
