"""Step 2: Train autoencoder"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
sys.path.append(str(SCRIPTS_DIR))

import config
from models import ProteinAutoencoder, print_model_summary
from utils import to_tensor, save_model, set_seed

def train_autoencoder(model, X_full, device, epochs=100):
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.AUTOENCODER_TRAINING_CONFIG['lr'],
        weight_decay=config.AUTOENCODER_TRAINING_CONFIG['weight_decay']
    )
    criterion = nn.MSELoss()
    X_tensor = to_tensor(X_full, device)
    batch_size = config.AUTOENCODER_TRAINING_CONFIG['batch_size']
    
    print(f"\nTraining: {len(X_full)} samples, batch={batch_size}, epochs={epochs}")
    
    history = {'losses': []}
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tensor))
        
        epoch_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(X_tensor), batch_size):
            indices = perm[i:i+batch_size]
            batch_X = X_tensor[indices]
            
            optimizer.zero_grad()
            encoded, decoded = model(batch_X)
            loss = criterion(decoded, batch_X)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        history['losses'].append(avg_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}] Loss: {avg_loss:.4f}")
    
    return model, history

def main():
    print("="*70)
    print("STEP 2: TRAIN AUTOENCODER")
    print("="*70)
    
    set_seed(config.TRAINING_CONFIG['random_seed'])
    
    print(f"\nLoading: {config.ESM2_EMBEDDINGS_FILE}")
    try:
        X_full = np.load(config.ESM2_EMBEDDINGS_FILE)
    except:
        X_full = np.load(config.ESM2_EMBEDDINGS_FILE, mmap_mode='r')
        X_full = np.array(X_full)
    print(f"  Shape: {X_full.shape}")
    
    autoencoder = ProteinAutoencoder(
        config.AUTOENCODER_CONFIG['input_dim'],
        config.AUTOENCODER_CONFIG['latent_dim'],
        config.AUTOENCODER_CONFIG['dropout']
    ).to(config.DEVICE)
    
    print_model_summary(autoencoder, "Autoencoder")
    
    autoencoder, history = train_autoencoder(
        autoencoder, X_full, config.DEVICE,
        config.AUTOENCODER_TRAINING_CONFIG['epochs']
    )
    
    save_model(autoencoder, config.MODELS_DIR, "autoencoder")
    np.save(config.MODELS_DIR / "autoencoder_history.npy", history)
    
    autoencoder.eval()
    with torch.no_grad():
        X_test = to_tensor(X_full[:100], config.DEVICE)
        encoded, decoded = autoencoder(X_test)
        recon_error = nn.MSELoss()(decoded, X_test).item()
        print(f"\nReconstruction MSE: {recon_error:.4f}")
    
    print("\n" + "="*70)
    print("✓ STEP 2 COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()