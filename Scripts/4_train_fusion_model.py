"""
Step 4: Train Multi-Modal Fusion Network

This script:
1. Loads pre-trained autoencoder
2. Loads contact maps
3. Trains complete fusion network
4. Saves best model
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add Scripts directory to path
SCRIPTS_DIR = Path(__file__).parent
sys.path.append(str(SCRIPTS_DIR))

import config
from models import (
    ProteinAutoencoder,
    MultiModalFusionNetwork,
    print_model_summary
)
from utils import (
    to_tensor,
    save_model,
    load_model,
    compute_metrics,
    print_metrics,
    plot_learning_curves,
    plot_predictions,
    EarlyStopping,
    set_seed
)

# =====================================================================
# DATA LOADING
# =====================================================================

def load_training_data():
    """Load all training data"""
    print("Loading training data...")
    
    # Load prepared splits
    train_data = np.load(config.DATA_DIR / "train_data.npy", allow_pickle=True).item()
    test_data = np.load(config.DATA_DIR / "test_data.npy", allow_pickle=True).item()
    
    # Load contact maps
    contact_maps_full = np.load(config.DATA_DIR / "contact_maps_full.npy")
    
    # Split contact maps
    train_contact_maps = contact_maps_full[train_data['indices']]
    test_contact_maps = contact_maps_full[test_data['indices']]
    
    print(f"  Train samples: {len(train_data['labels'])}")
    print(f"  Test samples: {len(test_data['labels'])}")
    
    return train_data, test_data, train_contact_maps, test_contact_maps


# =====================================================================
# TRAINING FUNCTION
# =====================================================================

def train_fusion_model(model, train_data, train_contact_maps, 
                       test_data, test_contact_maps, device):
    """
    Train fusion network with early stopping
    """
    # Optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.TRAINING_CONFIG['lr'],
        weight_decay=config.TRAINING_CONFIG['weight_decay']
    )
    
    # Loss
    criterion = nn.MSELoss()
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.TRAINING_CONFIG['early_stopping_patience']
    )
    
    # Convert to tensors
    X_train = to_tensor(train_data['esm2_embeddings'], device)
    y_train = to_tensor(train_data['labels'], device).reshape(-1, 1)
    pos_train = torch.LongTensor(train_data['positions']).to(device)
    cmap_train = to_tensor(train_contact_maps, device).unsqueeze(1)  # Add channel dim
    
    X_test = to_tensor(test_data['esm2_embeddings'], device)
    y_test = to_tensor(test_data['labels'], device).reshape(-1, 1)
    pos_test = torch.LongTensor(test_data['positions']).to(device)
    cmap_test = to_tensor(test_contact_maps, device).unsqueeze(1)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rho': []
    }
    
    best_rho = -1
    batch_size = config.TRAINING_CONFIG['batch_size']
    epochs = config.TRAINING_CONFIG['epochs']
    
    print("\nTraining Multi-Modal Fusion Network...")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {config.TRAINING_CONFIG['lr']}")
    
    for epoch in range(epochs):
        # ===== TRAINING =====
        model.train()
        
        # Shuffle training data
        perm = torch.randperm(len(X_train))
        
        epoch_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(X_train), batch_size):
            indices = perm[i:i+batch_size]
            
            batch_X = X_train[indices]
            batch_y = y_train[indices]
            batch_pos = pos_train[indices]
            batch_cmap = cmap_train[indices]
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch_X, batch_cmap, batch_pos)
            
            # Loss
            loss = criterion(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_loss / num_batches
        history['train_loss'].append(avg_train_loss)
        
        # ===== VALIDATION =====
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_test, cmap_test, pos_test)
            val_loss = criterion(val_predictions, y_test).item()
            history['val_loss'].append(val_loss)
            
            # Compute Spearman correlation
            from scipy.stats import spearmanr
            val_preds_np = val_predictions.cpu().numpy().flatten()
            val_true_np = y_test.cpu().numpy().flatten()
            val_rho, _ = spearmanr(val_true_np, val_preds_np)
            history['val_rho'].append(val_rho)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val ρ: {val_rho:.4f}")
        
        # Save best model
        if val_rho > best_rho:
            best_rho = val_rho
            save_model(model, config.MODELS_DIR, "fusion_model_best")
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    return model, history


# =====================================================================
# EVALUATION
# =====================================================================

def evaluate_model(model, test_data, test_contact_maps, device):
    """
    Evaluate model on test set
    """
    print("\nEvaluating model on test set...")
    
    model.eval()
    
    # Convert to tensors
    X_test = to_tensor(test_data['esm2_embeddings'], device)
    y_test = test_data['labels']
    pos_test = torch.LongTensor(test_data['positions']).to(device)
    cmap_test = to_tensor(test_contact_maps, device).unsqueeze(1)
    
    # Predict
    with torch.no_grad():
        predictions = model(X_test, cmap_test, pos_test)
        predictions = predictions.cpu().numpy().flatten()
    
    # Compute metrics
    metrics = compute_metrics(y_test, predictions)
    print_metrics(metrics, "Test Set")
    
    return metrics, predictions


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("="*70)
    print("STEP 4: TRAIN MULTI-MODAL FUSION NETWORK")
    print("="*70)
    
    # Set random seed
    set_seed(config.TRAINING_CONFIG['random_seed'])
    
    # Load data
    train_data, test_data, train_contact_maps, test_contact_maps = load_training_data()
    
    # Load pre-trained autoencoder
    print("\nLoading pre-trained autoencoder...")
    autoencoder = ProteinAutoencoder(
        input_dim=config.AUTOENCODER_CONFIG['input_dim'],
        latent_dim=config.AUTOENCODER_CONFIG['latent_dim'],
        dropout=config.AUTOENCODER_CONFIG['dropout']
    )
    autoencoder = load_model(autoencoder, config.MODELS_DIR, "autoencoder")
    autoencoder.to(config.DEVICE)
    autoencoder.eval()
    
    # Initialize fusion network
    print("\nInitializing fusion network...")
    fusion_model = MultiModalFusionNetwork(autoencoder, config).to(config.DEVICE)
    
    print_model_summary(fusion_model, "Multi-Modal Fusion Network")
    
    # Train
    fusion_model, history = train_fusion_model(
        fusion_model,
        train_data,
        train_contact_maps,
        test_data,
        test_contact_maps,
        config.DEVICE
    )
    
    # Load best model
    print("\nLoading best model...")
    fusion_model = load_model(fusion_model, config.MODELS_DIR, "fusion_model_best")
    
    # Evaluate
    test_metrics, test_predictions = evaluate_model(
        fusion_model,
        test_data,
        test_contact_maps,
        config.DEVICE
    )
    
    # Save results
    print("\nSaving results...")
    
    # Save training history
    np.save(config.MODELS_DIR / "fusion_training_history.npy", history)
    
    # Save predictions
    import pandas as pd
    results_df = pd.DataFrame({
        'true_fitness': test_data['labels'],
        'predicted_fitness': test_predictions,
        'residual': test_data['labels'] - test_predictions
    })
    results_df.to_csv(config.METRICS_DIR / "test_predictions.csv", index=False)
    
    # Save metrics
    import json
    with open(config.METRICS_DIR / "test_metrics.json", 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Learning curves
    plot_learning_curves(
        history['train_loss'],
        history['val_loss'],
        history['val_rho'],
        config.FIGURES_DIR / "learning_curves.png"
    )
    
    # Predictions scatter plot
    plot_predictions(
        test_data['labels'],
        test_predictions,
        config.FIGURES_DIR / "test_predictions.png",
        title="Multi-Modal Fusion Model"
    )
    
    print("\n" + "="*70)
    print("✓ STEP 4 COMPLETE")
    print("="*70)
    
    print(f"\nFinal Test Results:")
    print(f"  Spearman ρ: {test_metrics['spearman_rho']:.4f}")
    print(f"  Pearson r:  {test_metrics['pearson_r']:.4f}")
    print(f"  R²:         {test_metrics['r2']:.4f}")
    print(f"  MSE:        {test_metrics['mse']:.4f}")
    
    print(f"\nSaved to:")
    print(f"  Model: {config.MODELS_DIR / 'fusion_model_best.pt'}")
    print(f"  Metrics: {config.METRICS_DIR / 'test_metrics.json'}")
    print(f"  Predictions: {config.METRICS_DIR / 'test_predictions.csv'}")
    print(f"  Figures: {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()