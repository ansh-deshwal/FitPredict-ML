# FitPredict-ML 🧬

**Multi-Modal Protein Fitness Prediction using Deep Learning**

[![Status](https://img.shields.io/badge/Status-In%20Progress-orange)](https://github.com/yourusername/FitPredict-ML)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Model](https://img.shields.io/badge/Model-ESM--2%20(650M)-green)](https://github.com/facebookresearch/esm)
[![Data](https://img.shields.io/badge/Data-ProteinGym-purple)](https://proteingym.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> FitPredict-ML integrates **Sequence (ESM-2)**, **Structure**, and **Evolutionary** data to predict protein mutation effects, aiming to accelerate drug discovery without expensive wet-lab experiments.

---

## 🎯 Project Overview

This project aims to predict the functional effects of protein mutations using a multi-modal deep learning approach. By combining:
- **Sequence Embeddings** from ESM-2 (650M parameters)
- **Structural Information** (DSSP, Contact Maps)
- **Evolutionary Conservation** (MSA data)

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
| **Stage 3** | Baseline Model | 🔄 **Active** | Training Ridge Regression baseline |
| **Stage 4** | Structure Branch | ⏳ **Pending** | Integrating DSSP & Contact maps |
| **Stage 5** | Fusion Network | ⏳ **Pending** | Attention mechanism implementation |

---

## 📂 Repository Structure
```
FitPredict-ML/
│
├── data/
│   ├── BLAT_ECOLX_Stiffler_2015.csv          # Raw DMS Data
│   └── beta_lactamase_esm2_embeddings.npy    # Pre-computed Embeddings
│
├── scripts/
│   ├── extract_embeddings.py                 # ESM-2 Feature Extractor
│   └── train_baseline.py                     # Ridge Regression Model
│
├── results/
│   ├── baseline_plot.png                     # Performance Visualization
│   └── baseline_predictions.csv              # Test Set Predictions
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/FitPredict-ML.git
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

### 3️⃣ Extract Sequence Embeddings
```bash
python scripts/extract_embeddings.py
```
⏱️ **Runtime**: ~30 minutes on GPU (RTX 2050/3050)  
💾 **Output**: `beta_lactamase_esm2_embeddings.npy` (~25 MB)

### 4️⃣ Train Baseline Model
```bash
python scripts/train_baseline.py
```
📈 **Output**: 
- Console metrics (MSE, Spearman ρ, Pearson r)
- `baseline_plot.png` visualization
- `baseline_predictions.csv` with test predictions

---

## 📈 Current Results

### Baseline Model Performance
```
Training Set:
  MSE:                0.XXXX
  R² Score:           0.XXXX
  Spearman ρ:         0.XXXX
  Pearson r:          0.XXXX

Test Set:
  MSE:                0.XXXX
  R² Score:           0.XXXX
  Spearman ρ:         0.XXXX
  Pearson r:          0.XXXX
```

![Baseline Results](results/baseline_plot.png)

---

## 🧰 Technical Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.10+ |
| **ML Framework** | PyTorch, scikit-learn |
| **Protein LM** | ESM-2 (650M) from Meta AI |
| **Data Processing** | pandas, NumPy |
| **Visualization** | Matplotlib, seaborn |
| **Evaluation** | scipy (Spearman), sklearn (MSE, R²) |

---

## 🔮 Roadmap

- [x] Data preprocessing and cleaning
- [x] ESM-2 embedding extraction
- [x] Ridge regression baseline
- [ ] Add structural features (DSSP)
- [ ] Implement contact map prediction
- [ ] Build multi-modal fusion network
- [ ] Hyperparameter optimization
- [ ] Cross-validation framework
- [ ] Extend to additional ProteinGym datasets
- [ ] Deploy model via API/web interface

---

## 📖 Key Concepts

### What is Deep Mutational Scanning (DMS)?
DMS is an experimental technique that measures the functional effect of thousands of protein mutations simultaneously. Each variant is assigned a fitness score indicating its activity relative to wild-type.

### Why ESM-2?
ESM-2 (Evolutionary Scale Modeling) is a protein language model trained on 250M protein sequences. It captures evolutionary patterns and structural constraints without explicit structure input.

### Multi-Modal Learning
By combining sequence, structure, and evolutionary signals, we aim to achieve better generalization than single-modality approaches.

---

## 📚 References

1. **ESM-2**: Lin et al. (2023) - "Evolutionary-scale prediction of atomic-level protein structure with a language model"
2. **ProteinGym**: Notin et al. (2023) - "ProteinGym: Large-scale benchmarks for protein fitness prediction"
3. **Dataset**: Stiffler et al. (2015) - "Evolvability as a function of purifying selection in TEM-1 β-lactamase"

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
- GitHub: [ansh-deshwal](https://github.com/ansh-deshwal)
- Email: anshdeshwal2608@gmail.com

---

## 🙏 Acknowledgments

- Meta AI for the ESM-2 model
- ProteinGym team for curated benchmarks
- Stiffler et al. for the β-lactamase dataset

---

<div align="center">

**⭐ Star this repo if you find it useful!**

Made with ❤️ for advancing protein engineering through AI

</div>
