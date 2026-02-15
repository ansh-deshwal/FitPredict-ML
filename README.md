# FitPredict-ML 🧬

**Multi-Modal Protein Fitness Prediction using Deep Learning**

[![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)](https://github.com/ansh-deshwal/FitPredict-ML)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Model](https://img.shields.io/badge/Model-ESM--2%20(650M)-green)](https://github.com/facebookresearch/esm)
[![Data](https://img.shields.io/badge/Data-ProteinGym-purple)](https://proteingym.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> FitPredict-ML integrates **Sequence (ESM-2)**, **Structure (PDB)**, and **Evolutionary** data to predict protein mutation effects, aiming to accelerate drug discovery without expensive wet-lab experiments.

---

## 🎯 Project Overview

This project predicts the functional effects of protein mutations using a multi-modal deep learning approach. By combining:
- ✅ **Sequence Embeddings** from ESM-2 (650M parameters)
- ✅ **Structural Information** from PDB (burial depth, surface exposure)
- 🔜 **Evolutionary Conservation** (MSA data)

We achieve strong correlation with experimental Deep Mutational Scanning (DMS) data for predicting mutation fitness.

### 🔬 Current Dataset
- **Protein**: β-lactamase (BLAT_ECOLX / TEM-1)
- **Source**: Stiffler et al. 2015
- **Mutations**: 4,996 single-point variants
- **Structure**: PDB ID 1M40 (0.85Å resolution)
- **Metric**: DMS fitness scores

---

## 📊 Project Progress

| Stage | Component | Status | Performance | Description |
|-------|-----------|--------|-------------|-------------|
| **Stage 1** | Data Prep | ✅ **Done** | - | Cleaned & processed β-lactamase dataset |
| **Stage 2** | Sequence Branch | ✅ **Done** | ρ = 0.719 | Extracted 1280-d ESM-2 embeddings |
| **Stage 3** | Baseline Models | ✅ **Done** | ρ = 0.719 | Ridge & MLP trained on sequence only |
| **Stage 4** | Structure Branch | ✅ **Done** | - | Extracted burial depth & surface features |
| **Stage 5** | Multi-Modal Fusion | 🔄 **Active** | TBD | Combining sequence + structure |
| **Stage 6** | Evolutionary Branch | ⏳ **Pending** | - | MSA features integration |

---

## 📈 Current Results

### Baseline Performance (Sequence-Only Models)

| Model | Architecture | Test Spearman ρ | Test Pearson r | Test R² | Parameters |
|-------|-------------|----------------|----------------|---------|------------|
| Ridge Regression | Linear | ~0.500 | ~0.500 | ~0.250 | 1,280 |
| **MLP (3-layer)** | Deep | **0.7190** ⭐ | 0.7020 | 0.3993 | 721,665 |

### MLP Architecture Details
```
Input (1280) → Dense(512) → ReLU → Dropout(0.3)
              → Dense(128) → ReLU → Dropout(0.3)
              → Dense(1) → Output

Training Configuration:
  - Optimizer: Adam (lr=0.001, weight_decay=1e-5)
  - Loss: MSE
  - Batch Size: 32
  - Epochs: 50
  - Scheduler: ReduceLROnPlateau
```

### Training Characteristics
- ✅ **No overfitting** - Test Spearman (0.719) > Train Spearman (0.703)
- ✅ **Smooth convergence** - Learning rate reduced at epoch ~22
- ✅ **Strong rank correlation** - Model captures mutation severity ordering
- 📊 **Performance gap** - ~5-10% below SOTA (ρ ~ 0.75-0.80)

![MLP Results](results/mlp_baseline_plot.png)

### Structural Features (New! 🎉)

**Extracted from PDB 1M40 (Ultra-high resolution: 0.85Å)**

| Feature | Type | Range | Mean | Description |
|---------|------|-------|------|-------------|
| Burial Score | Continuous | 0-33 | 19.30 | Number of Cα atoms within 10Å radius |
| Buried Flag | Binary | 0/1 | 0.426 | 1 if burial score > 20, else 0 |

**Key Findings:**
- 42.6% of mutations are in buried (core) regions
- 57.4% of mutations are surface-exposed
- Burial depth correlates with structural stability
- Surface mutations may affect binding interfaces

**Distribution:**
```
Buried residues (core):    2,128 mutations (42.6%)
Surface residues:          2,868 mutations (57.4%)
```

This structural context is crucial because:
- **Buried mutations** often destabilize protein folding → lower fitness
- **Surface mutations** may affect function without destabilizing structure
- **Burial depth** provides orthogonal information to sequence embeddings

---

## 📂 Repository Structure

```
FitPredict-ML/
│
├── data/
│   ├── BLAT_ECOLX_Stiffler_2015.csv              # Raw DMS Data
│   ├── 1M40.pdb                                  # β-lactamase 3D structure
│   ├── beta_lactamase_esm2_embeddings.npy        # Sequence embeddings (1280-d)
│   └── beta_lactamase_structure_features.npy     # Structural features (2-d)
│
├── scripts/
│   ├── extract_embeddings.py                     # ESM-2 Feature Extractor
│   ├── extract_structure_features.py             # PDB Structure Parser
│   ├── train_baseline.py                         # Ridge Regression
│   ├── train_mlp.py                              # MLP Neural Network
│   └── train_multimodal.py                       # Multi-modal Fusion (coming)
│
├── results/
│   ├── baseline_plot.png                         # Ridge results
│   ├── baseline_predictions.csv                  # Ridge predictions
│   ├── mlp_baseline_plot.png                     # MLP results (3-panel)
│   ├── mlp_predictions.csv                       # MLP predictions
│   └── structure_analysis.png                    # Burial depth visualization
│
├── models/
│   ├── best_mlp_model.pt                         # Trained MLP weights
│   └── multimodal_fusion.pt                      # Fusion model (coming)
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/ansh-deshwal/FitPredict-ML.git
cd FitPredict-ML
```

### 2️⃣ Install Dependencies
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install packages
pip install torch fair-esm pandas numpy scikit-learn scipy matplotlib biopython
```

Or use requirements file:
```bash
pip install -r requirements.txt
```

### 3️⃣ Extract Features

**A) Sequence Embeddings (one-time, ~30 min on GPU):**
```bash
python scripts/extract_embeddings.py
```
Output: `beta_lactamase_esm2_embeddings.npy` (1280 features × 4,996 mutations)

**B) Structural Features (one-time, ~1 min):**
```bash
python scripts/extract_structure_features.py
```
Output: `beta_lactamase_structure_features.npy` (2 features × 4,996 mutations)

### 4️⃣ Train Models

**Ridge Regression Baseline:**
```bash
python scripts/train_baseline.py
```

**MLP Neural Network:**
```bash
python scripts/train_mlp.py
```

**Multi-Modal Fusion (coming soon):**
```bash
python scripts/train_multimodal.py
```

---

## 🧰 Technical Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Language** | Python | 3.10+ |
| **Deep Learning** | PyTorch | 2.0+ |
| **Protein LM** | ESM-2 (fair-esm) | 650M params |
| **Structure** | Biopython | 1.80+ |
| **ML Framework** | scikit-learn | 1.3+ |
| **Data Processing** | pandas, NumPy | - |
| **Visualization** | Matplotlib | 3.7+ |

---

## 🔬 Methodology

### Phase 1: Sequence-Only Baseline ✅ Complete

**Feature Extraction:**
- ESM-2 650M parameter transformer model
- Mean pooling over sequence tokens (excluding CLS/EOS)
- Output: 1280-dimensional dense embeddings per variant

**Model Training:**
1. **Ridge Regression** - Linear baseline for comparison
2. **3-Layer MLP** - Non-linear model with dropout regularization
   - Architecture: 1280 → 512 → 128 → 1
   - Adam optimizer with LR scheduling
   - 80/20 train-test split (random seed 42)

**Evaluation:**
- Primary: Spearman rank correlation (ρ)
- Secondary: Pearson r, MSE, R²

### Phase 2: Structural Features ✅ Complete

**PDB Structure Processing:**
1. Download ultra-high resolution structure (PDB: 1M40, 0.85Å)
2. Extract Cα (alpha-carbon) coordinates for all residues
3. For each mutation position, calculate:
   - **Burial Score**: Count of neighboring Cα atoms within 10Å
   - **Buried Flag**: Binary indicator (buried if >20 neighbors)

**Biological Rationale:**
- Buried residues are critical for protein core stability
- Surface residues affect binding and interactions
- Burial depth is independent of sequence similarity
- Provides spatial context missing from sequence alone

### Phase 3: Multi-Modal Fusion 🔄 In Progress

**Architecture Plan:**
```
Sequence Branch (1280-d)  ──┐
                             ├──> Attention Fusion ──> Prediction
Structure Branch (2-d)    ──┘

Components:
  - Separate encoders for sequence/structure
  - Cross-attention mechanism for feature integration
  - Gated fusion for learned weighting
  - Dropout for regularization
```

**Expected Improvements:**
- Capture sequence-structure interactions
- Better generalization across mutation types
- Improved performance on buried vs. surface variants

---

## 📊 Benchmark Comparison

| Approach | Spearman ρ | Feature Modality | Notes |
|----------|-----------|------------------|-------|
| Random Baseline | 0.000 | None | No predictive power |
| Ridge (Linear) | ~0.500 | Sequence only | Linear baseline |
| **MLP (Current)** | **0.719** | Sequence only | Current best |
| ESM-1v (zero-shot) | ~0.650 | Sequence only | Direct LM predictions |
| Multi-modal (Target) | ~0.750+ | Seq + Struct | Under development |
| SOTA Literature | ~0.75-0.80 | Seq + Struct + Evol | Published benchmarks |

**Progress to SOTA:**
- Current: 71.9% of perfect correlation
- Target: 75-80% of perfect correlation
- Gap: ~3-8 percentage points

---

## 🔮 Roadmap

### Completed ✅
- [x] Data preprocessing and quality control
- [x] ESM-2 sequence embedding extraction (1280-d)
- [x] Ridge regression baseline (ρ: ~0.5)
- [x] MLP neural network baseline (ρ: 0.719)
- [x] PDB structure download and parsing
- [x] Burial depth feature extraction (2-d)
- [x] Comprehensive evaluation framework

### In Progress 🔄
- [ ] Multi-modal fusion network architecture
- [ ] Attention-based feature integration
- [ ] Cross-validation framework
- [ ] Ablation studies (sequence vs. structure contribution)

### Planned 📋
- [ ] Add more structural features (DSSP, ASA, contact maps)
- [ ] Evolutionary features from MSA
- [ ] Hyperparameter optimization (Optuna/Ray Tune)
- [ ] K-fold cross-validation
- [ ] Extend to additional ProteinGym datasets
- [ ] Model interpretability (attention visualization)
- [ ] API deployment for predictions

---

## 📖 Key Concepts

### Deep Mutational Scanning (DMS)
High-throughput experimental technique measuring functional effects of thousands of protein mutations. Each variant receives a fitness score (log-enrichment ratio) indicating activity relative to wild-type.

### ESM-2 Protein Language Model
Transformer model trained on 250M protein sequences. Learns evolutionary patterns, structural constraints, and functional motifs without supervision. Enables zero-shot predictions and rich embeddings.

### Burial Depth
Number of neighboring residues within a spatial threshold (10Å). Indicates whether a residue is:
- **Buried** (>20 neighbors): Core/interior, critical for stability
- **Surface** (<20 neighbors): Exterior, often functional sites

Higher burial correlates with structural importance and mutation intolerance.

### Spearman Rank Correlation
Non-parametric measure of monotonic relationship. Evaluates whether model correctly **ranks** mutations by severity, not just linear prediction accuracy. Critical for drug discovery prioritization.

### Multi-Modal Learning
Combining heterogeneous data types (sequence, structure, evolution) to capture complementary information. Aims to exceed single-modality performance through learned feature fusion.

---

## 📝 Code Review & Findings

### Structure Extraction Script Analysis

**Author**: Arjun Sharma (adapted)  
**File**: `extract_structure_features.py`

**Strengths** ✅:
1. **Ultra-high resolution PDB** (0.85Å) ensures accurate atomic positions
2. **Simple but effective features** - burial depth is well-established predictor
3. **Efficient computation** - vectorized distance calculations
4. **Automatic download** - fetches PDB if missing
5. **Clean code** - well-documented with clear steps

**Potential Improvements** 💡:
1. **Feature diversity**: Currently only 2 features
   - Could add: DSSP secondary structure, solvent accessibility, B-factors
2. **Distance threshold**: 10Å is reasonable but could be optimized
   - Could try: 8Å, 12Å, or learned threshold
3. **Burial cutoff**: 20 neighbors is heuristic
   - Could use: percentile-based threshold or continuous feature only
4. **Chain assumption**: Assumes chain 'A' exists
   - Could add: validation or multi-chain support
5. **Error handling**: Missing residues return 0
   - Could add: interpolation or explicit missing value flags

**Findings** 📊:
- **Good coverage**: Features extracted for all 4,996 mutations
- **Balanced distribution**: 42.6% buried / 57.4% surface
- **No missing data**: All positions mapped successfully (some have 0 neighbors for edge cases)
- **Reasonable range**: 0-33 neighbors matches typical protein geometry

**Next Steps** 🎯:
1. Visualize burial depth vs. fitness correlation
2. Test structure-only model performance
3. Build fusion architecture combining sequence + structure
4. Compare buried vs. surface mutation prediction accuracy

---

## 🛠️ Troubleshooting

### GPU Not Detected
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Reinstall PyTorch with CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### PDB Download Fails
```bash
# Manual download
wget https://files.rcsb.org/download/1M40.pdb

# Or use browser
# Visit: https://www.rcsb.org/structure/1M40
```

### Out of Memory
Reduce batch size in training scripts:
```python
batch_size = 16  # or 8 for 4GB VRAM
```

### Biopython Installation Issues
```bash
# On Windows
pip install biopython --upgrade

# On Linux/Mac
pip install biopython
```

---

## 📚 References

### Core Papers
1. **ESM-2**: Lin et al. (2023) - "Evolutionary-scale prediction of atomic-level protein structure with a language model" - *Science* 379(6628)
2. **ProteinGym**: Notin et al. (2023) - "ProteinGym: Large-scale benchmarks for protein fitness prediction" - *NeurIPS* 
3. **Dataset**: Stiffler et al. (2015) - "Evolvability as a function of purifying selection in TEM-1 β-lactamase" - *Cell* 160(5)

### Structure & Methods
4. **PDB Structure**: Minasov et al. (2002) - "An ultrahigh resolution structure of TEM-1 β-lactamase" - *J. Am. Chem. Soc.* 124(19)
5. **Burial Depth**: Varrazzo et al. (2005) - "Three-dimensional computation of atom depth in complex molecular structures" - *Bioinformatics* 21(12)

### Related Work
6. **Deep Learning for Proteins**: AlQuraishi (2019) - "End-to-end differentiable learning of protein structure" - *Cell Systems* 8(4)
7. **Mutation Effects**: Rao et al. (2021) - "MSA Transformer enables reference-free variant effect prediction" - *bioRxiv*

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional structural features (DSSP, contact maps)
- Evolutionary features (MSA statistics)
- Alternative fusion architectures
- Hyperparameter optimization
- Extended benchmarks on other proteins

**Process:**
1. Fork the repository
2. Create feature branch (`git checkout -b feature/NewFeature`)
3. Commit changes (`git commit -m 'Add NewFeature'`)
4. Push to branch (`git push origin feature/NewFeature`)
5. Open Pull Request

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 👤 Authors

**Ansh** - Primary Developer
- GitHub: ansh-deshwal(https://github.com/ansh-deshwal)
- Project: [FitPredict-ML](https://github.com/ansh-deshwal/FitPredict-ML)

**Arjun Sharma** - Structure Feature Extraction
  (https://github.com/arjsh16)
- Contributed structure parsing and burial depth calculation

---

## 🙏 Acknowledgments

- **Meta AI** - ESM-2 model and fair-esm library
- **ProteinGym Team** - Curated benchmarks and evaluation standards
- **RCSB PDB** - Protein structure database
- **Stiffler Lab** - β-lactamase DMS dataset
- **Biopython Community** - Structure parsing tools

---

## 📊 Project Timeline

- **Week 1**: Data exploration and preprocessing ✅
- **Week 2**: ESM-2 embedding extraction ✅
- **Week 3**: Baseline models (Ridge, MLP) ✅
- **Week 4**: Structural feature extraction ✅ (Current)
- **Week 5**: Multi-modal fusion implementation 🔄
- **Week 6**: Hyperparameter tuning and evaluation 📋
- **Week 7**: Extended benchmarks and analysis 📋
- **Week 8**: Final evaluation and documentation 📋

---

<div align="center">

### 🎯 Current Milestone

**Stage 4 Complete**: Structural features extracted  
**Next**: Multi-modal fusion network

---

**⭐ Star this repo if you find it useful!**

Made with ❤️ for advancing protein engineering through AI

**Latest Achievement**: MLP baseline **Spearman ρ = 0.719** on sequence-only prediction

**Current Status**: Integrating structure + sequence for improved predictions

</div>
