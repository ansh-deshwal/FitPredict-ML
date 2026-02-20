"""Central configuration file"""
import torch
from pathlib import Path

# Directories
BASE_DIR = Path(r"C:\Personal\College\SEM\DL\FitPredict-ML")
DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Files
CSV_FILE = DATA_DIR / "BLAT_ECOLX_Stiffler_2015.csv"
ESM2_EMBEDDINGS_FILE = DATA_DIR / "beta_lactamase_esm2_embeddings.npy"
STRUCTURE_FEATURES_FILE = DATA_DIR / "beta_lactamase_structure_features.npy"
PDB_FILE = DATA_DIR / "1M40.pdb"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Model configs
AUTOENCODER_CONFIG = {
    'input_dim': 1280,
    'latent_dim': 64,
    'dropout': 0.2,
}

BILSTM_CONFIG = {
    'input_dim': 64,
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.2,
}

CNN_CONFIG = {
    'output_dim': 64,
}

FUSION_CONFIG = {
    'seq_features_dim': 128,
    'struct_features_dim': 64,
    'hidden_dim': 256,
    'num_residual_blocks': 2,
    'dropout': 0.2,
}

# Training configs
TRAINING_CONFIG = {
    'test_size': 0.2,
    'random_seed': 42,
    'lr': 0.001,
    'weight_decay': 1e-5,
    'batch_size': 32,
    'epochs': 100,
    'gradient_clip_norm': 1.0,
    'early_stopping_patience': 15,
}

AUTOENCODER_TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 64,
    'lr': 0.001,
    'weight_decay': 1e-5,
}

# Contact map settings
CONTACT_MAP_CONFIG = {
    'distance_threshold': 8.0,
    'map_size': 256,
}

# Sequence settings
SEQUENCE_CONFIG = {
    'max_seq_length': 286,
}