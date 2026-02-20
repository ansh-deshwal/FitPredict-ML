"""Utility functions"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

def extract_mutation_positions(df):
    """Extract positions from mutant column (e.g., M182T -> 182)"""
    positions = df['mutant'].str[1:-1].apply(
        lambda x: int(x) if x.isdigit() else 0
    ).values
    return positions

def to_tensor(data, device='cpu'):
    """Convert numpy to torch tensor"""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return torch.FloatTensor(data).to(device)

def set_seed(seed=42):
    """Set random seeds"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_model(model, save_dir, name):
    """Save model weights"""
    save_path = Path(save_dir) / f"{name}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"✓ Saved: {save_path}")

def load_model(model, save_dir, name):
    """Load model weights"""
    load_path = Path(save_dir) / f"{name}.pt"
    model.load_state_dict(torch.load(load_path))
    print(f"✓ Loaded: {load_path}")
    return model

def compute_metrics(y_true, y_pred):
    """Compute all metrics"""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'spearman_rho': spearmanr(y_true, y_pred)[0],
        'pearson_r': pearsonr(y_true, y_pred)[0],
    }

def print_metrics(metrics, title="Metrics"):
    """Print metrics"""
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    print(f"  MSE:         {metrics['mse']:.4f}")
    print(f"  R²:          {metrics['r2']:.4f}")
    print(f"  Spearman ρ:  {metrics['spearman_rho']:.4f}")
    print(f"  Pearson r:   {metrics['pearson_r']:.4f}")
    print(f"{'='*50}")

def plot_learning_curves(train_losses, val_losses, val_rhos, save_path):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Train')
    ax1.plot(epochs, val_losses, 'r-', linewidth=2, label='Val')
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Curves', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, val_rhos, 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Spearman ρ', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Performance', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

def plot_predictions(y_true, y_pred, save_path, title="Predictions"):
    """Plot predictions vs true"""
    metrics = compute_metrics(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.4, s=20)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    ax.set_xlabel('True', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_title(f"{title}\nρ={metrics['spearman_rho']:.3f}, R²={metrics['r2']:.3f}", 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {save_path}")

class EarlyStopping:
    """Early stopping"""
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

def load_prepared_data(data_dir):
    """Load train/test data from npz"""
    data_dir = Path(data_dir)
    
    train_file = data_dir / "train_data.npz"
    train_npz = np.load(train_file)
    train_data = {
        'indices': train_npz['indices'],
        'esm2_embeddings': train_npz['esm2_embeddings'],
        'labels': train_npz['labels'],
        'positions': train_npz['positions']
    }
    
    test_file = data_dir / "test_data.npz"
    test_npz = np.load(test_file)
    test_data = {
        'indices': test_npz['indices'],
        'esm2_embeddings': test_npz['esm2_embeddings'],
        'labels': test_npz['labels'],
        'positions': test_npz['positions']
    }
    
    return train_data, test_data