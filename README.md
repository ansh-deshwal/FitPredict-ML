# FitPredict-ML 🧬

**Multi-Modal Protein Fitness Prediction using Deep Learning**

[![Status](https://img.shields.io/badge/Status-In%20Progress-orange)](https://github.com/ansh-deshwal/FitPredict-ML)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Model](https://img.shields.io/badge/Model-ESM--2%20(650M)-green)](https://github.com/facebookresearch/esm)
[![Data](https://img.shields.io/badge/Data-ProteinGym-purple)](https://proteingym.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> FitPredict-ML integrates **Sequence (ESM-2)**, **Structure**, and **Evolutionary** data to predict protein mutation effects, aiming to accelerate drug discovery without expensive wet-lab experiments.

---

## 🎯 Project Overview

This project aims to predict the functional effects of protein mutations using a multi-modal deep learning approach. By combining:
- **Sequence Embeddings** from ESM-2 (650M parameters)
- **Structural Information** (DSSP, Contact Maps) - *Coming Soon*
- **Evolutionary Conservation** (MSA data) - *Coming Soon*

We can predict mutation fitness scores that correlate with experimental Deep Mutational Scanning (DMS) data.

### 🔬 Current Dataset
- **Protein**: β-lactamase (BLAT_ECOLX)
- **Source**: Stiffler et al. 2015
- **Mutations**: ~5,000 single-point variants
- **Metric**: DMS fitness scores

---

## 📊 Project Progress

| Stage | Component | Status | Description |
|-------|-----------|--------|-------------|
| **Stage 1** | Data Prep | ✅ **Done** | Cleaned & processed β-lactamase dataset |
| **Stage 2** | Sequence Branch | ✅ **Done** | Extracted 1280-d embeddings (ESM-2 650M) |
| **Stage 3** | Baseline Models | ✅ **Done** | Ridge Regression & MLP baselines trained |
| **Stage 4** | Structure Branch | ⏳ **Pending** | Integrating DSSP & Contact maps |
| **Stage 5** | Fusion Network | ⏳ **Pending** | Attention mechanism implementation |

---

## 📈 Current Results

### Baseline Model Performance

| Model | Test Spearman ρ | Test Pearson r | Test R² | Test MSE |
|-------|----------------|----------------|---------|----------|
| **Ridge Regression** | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| **MLP (3-layer)** | **0.7190** ⭐ | 0.7020 | 0.3993 | 0.8027 |

### MLP Architecture
```
Input (1280) → Dense(512) → ReLU → Dropout(0.3)
           → Dense(128) → ReLU → Dropout(0.3)
           → Dense(1) → Output
Total Parameters: 721,665
```

### Training Metrics
- **No overfitting observed** - Test performance exceeds training
- **Training Spearman ρ**: 0.7032
- **Test Spearman ρ**: 0.7190
- **Training converged** after 50 epochs with learning rate scheduling

![MLP Results](results/mlp_baseline_plot.png)

**Key Findings:**
- Strong rank correlation indicates model captures mutation severity ordering
- Performance ~5-10% below state-of-the-art (ρ ~ 0.75-0.80)
- Solid baseline for multi-modal extensions

---

## 📂 Repository Structure
```
FitPredict-ML/
│
├── data/
│   ├── BLAT_ECOLX_Stiffler_2015.csv          # Raw DMS Data
│   └── beta_lactamase_esm2_embeddings.npy    # Pre-computed Embeddings (1280-d)
│
├── scripts/
│   ├── extract_embeddings.py                 # ESM-2 Feature Extractor
│   ├── train_baseline.py                     # Ridge Regression Baseline
│   └── train_mlp.py                          # MLP Neural Network
│
├── results/
│   ├── baseline_plot.png                     # Ridge Regression Results
│   ├── baseline_predictions.csv              # Ridge Predictions
│   ├── mlp_baseline_plot.png                 # MLP Results (3-panel)
│   └── mlp_predictions.csv                   # MLP Predictions
│
├── models/
│   └── best_mlp_model.pt                     # Trained MLP Weights
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
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install torch fair-esm pandas numpy scikit-learn scipy matplotlib tqdm
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### 3️⃣ Extract Sequence Embeddings (One-time)
```bash
python scripts/extract_embeddings.py
```
⏱️ **Runtime**: ~30 minutes on GPU (RTX 2050/3050)  
💾 **Output**: `beta_lactamase_esm2_embeddings.npy` (~25 MB)

### 4️⃣ Train Baseline Models

**Ridge Regression:**
```bash
python scripts/train_baseline.py
```

**MLP Neural Network:**
```bash
python scripts/train_mlp.py
```

📈 **Output**: 
- Console metrics (MSE, Spearman ρ, Pearson r, R²)
- Visualization plots
- Prediction CSVs

---

## 🧰 Technical Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.10+ |
| **Deep Learning** | PyTorch 2.0+ |
| **ML Framework** | scikit-learn |
| **Protein LM** | ESM-2 (650M) from Meta AI |
| **Data Processing** | pandas, NumPy |
| **Visualization** | Matplotlib |
| **Evaluation** | scipy (Spearman), sklearn (MSE, R²) |

---

## 🔬 Methodology

### Phase 1: Sequence-Only Baseline (✅ Complete)

1. **Feature Extraction**
   - ESM-2 650M parameter model
   - Mean pooling over sequence tokens
   - 1280-dimensional embeddings per variant

2. **Model Training**
   - Ridge regression (linear baseline)
   - 3-layer MLP with dropout regularization
   - Adam optimizer with learning rate scheduling
   - 80/20 train-test split

3. **Evaluation Metrics**
   - Spearman rank correlation (primary)
   - Pearson correlation
   - Mean Squared Error (MSE)
   - R² coefficient of determination

### Phase 2: Multi-Modal Integration (🔜 Coming Soon)

- **Structural Features**: DSSP secondary structure, solvent accessibility
- **Contact Maps**: Predicted residue-residue contacts
- **Evolutionary**: Multiple sequence alignment (MSA) statistics
- **Fusion**: Attention-based multi-modal integration

---

## 📊 Benchmark Comparison

| Approach | Spearman ρ | Notes |
|----------|-----------|-------|
| Random Baseline | 0.000 | No predictive power |
| Ridge Regression | ~0.500 | Linear model |
| **Our MLP** | **0.719** | Current best |
| ESM-1v (zero-shot) | ~0.650 | Direct LM predictions |
| State-of-the-art | ~0.750-0.800 | Multi-modal models |

---

## 🔮 Roadmap

### Completed ✅
- [x] Data preprocessing and cleaning
- [x] ESM-2 embedding extraction
- [x] Ridge regression baseline (Spearman ρ: ~0.5)
- [x] MLP neural network (Spearman ρ: 0.719)
- [x] Comprehensive evaluation framework

### In Progress 🔄
- [ ] Add structural features (DSSP)
- [ ] Implement contact map prediction
- [ ] Build multi-modal fusion network

### Planned 📋
- [ ] Hyperparameter optimization (Optuna)
- [ ] K-fold cross-validation framework
- [ ] Extend to additional ProteinGym datasets
- [ ] Model interpretability (attention visualization)
- [ ] Deploy model via API/web interface

---

## 📖 Key Concepts

### What is Deep Mutational Scanning (DMS)?
DMS is an experimental technique that measures the functional effect of thousands of protein mutations simultaneously. Each variant is assigned a fitness score indicating its activity relative to wild-type.

### Why ESM-2?
ESM-2 (Evolutionary Scale Modeling) is a protein language model trained on 250M protein sequences. It captures evolutionary patterns and structural constraints without explicit structure input, achieving strong zero-shot performance on various tasks.

### Why Spearman Correlation?
Unlike Pearson correlation (linear relationships), Spearman ρ measures **rank correlation** - whether the model correctly orders mutations by severity. This is crucial for drug discovery where we care about identifying the most/least harmful variants.

### Multi-Modal Learning
By combining sequence, structure, and evolutionary signals, we aim to achieve better generalization than single-modality approaches, capturing complementary information about mutation effects.

---

## 🛠️ Troubleshooting

### GPU Not Detected
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory Errors
Reduce batch size in training scripts:
```python
batch_size = 16  # or even 8 for 4GB VRAM
```

### Slow Training on CPU
Expected - MLP trains in ~20 minutes on CPU vs ~2-5 minutes on GPU.

---

## 📚 References

1. **ESM-2**: Lin et al. (2023) - "Evolutionary-scale prediction of atomic-level protein structure with a language model" - *Science*
2. **ProteinGym**: Notin et al. (2023) - "ProteinGym: Large-scale benchmarks for protein fitness prediction" - *NeurIPS*
3. **Dataset**: Stiffler et al. (2015) - "Evolvability as a function of purifying selection in TEM-1 β-lactamase" - *Cell*
4. **Deep Learning for Proteins**: AlQuraishi (2019) - "End-to-end differentiable learning of protein structure" - *Cell Systems*

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Ansh**
- GitHub: [@yourusername](https://github.com/ansh-deshwal)
- Project Link: [https://github.com/yourusername/FitPredict-ML](https://github.com/ansh-deshwal/FitPredict-ML)

---

## 🙏 Acknowledgments

- Meta AI for the ESM-2 model and fair-esm library
- ProteinGym team for curated benchmarks and evaluation framework
- Stiffler et al. for the β-lactamase DMS dataset
- PyTorch team for the deep learning framework

---

## 📊 Project Timeline

- **Week 1**: Data exploration and preprocessing ✅
- **Week 2**: ESM-2 embedding extraction ✅
- **Week 3**: Baseline model development ✅ (Current)
- **Week 4-5**: Multi-modal feature integration 🔄
- **Week 6-7**: Fusion network implementation 📋
- **Week 8**: Final evaluation and deployment 📋

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Made with ❤️ for advancing protein engineering through AI

**Current Status**: MLP baseline achieving **Spearman ρ = 0.719** on test set

</div>
