import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

# 1. Setup GPU and Paths
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

data_dir    = Path(__file__).parent.parent / "Data"
results_dir = Path(__file__).parent.parent / "Results"
results_dir.mkdir(exist_ok=True)

csv_path = data_dir / "BLAT_ECOLX_Stiffler_2015.csv"
emb_path = data_dir / "beta_lactamase_esm2_embeddings.npy"

# 2. Load Data
print("Loading data...")
try:
    df = pd.read_csv(csv_path)
    X = np.load(emb_path)
    y = df["DMS_score"].values.astype(float)
    print(f"Dataset loaded: {len(X)} samples, {X.shape[1]} features")
except FileNotFoundError as e:
    print(f"Error: Could not find data files. {e}")
    exit()

# 3. Split Data (70% Train, 15% Val, 15% Test)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, shuffle=True
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1765, random_state=42, shuffle=True
)  # 0.1765 * 0.85 ≈ 0.15 of total
print(f"Training samples:   {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples:       {len(X_test)}")

# Convert to PyTorch Tensors
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
X_val_t   = torch.FloatTensor(X_val).to(device)
y_val_t   = torch.FloatTensor(y_val).reshape(-1, 1).to(device)
X_test_t  = torch.FloatTensor(X_test).to(device)
y_test_t  = torch.FloatTensor(y_test).reshape(-1, 1).to(device)

# 4. Define the MLP Model
class ProteinMLP(nn.Module):
    def __init__(self):
        super(ProteinMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        return self.network(x)

model = ProteinMLP().to(device)
print("\nModel Architecture:")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# 5. Optimizer and Scheduler
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.MSELoss()

# Scheduler (without verbose to avoid compatibility issues)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# 6. Training Loop
print("\n" + "="*50)
print("Starting Training")
print("="*50)

batch_size = 32
epochs = 50
train_losses = []
val_losses = []
val_rhos = []
best_val_rho = -1.0

for epoch in range(epochs):
    # ===== TRAINING =====
    model.train()
    permutation = torch.randperm(X_train_t.size(0))
    
    epoch_loss = 0
    num_batches = 0
    
    for i in range(0, X_train_t.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train_t[indices], y_train_t[indices]
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        epoch_loss += loss.item()
        num_batches += 1
        
    avg_train_loss = epoch_loss / num_batches
    train_losses.append(avg_train_loss)
    
    # ===== VALIDATION =====
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_loss = criterion(val_outputs, y_val_t).item()
        val_losses.append(val_loss)

        val_preds = val_outputs.cpu().numpy().flatten()
        val_rho, _ = spearmanr(y_val, val_preds)
        val_rhos.append(val_rho)
    
    # Step scheduler
    scheduler.step(val_loss)

    # Save best checkpoint
    if val_rho > best_val_rho:
        best_val_rho = val_rho
        torch.save(model.state_dict(), results_dir / "best_mlp_model.pt")

    # Print progress
    if (epoch+1) % 5 == 0 or epoch == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:3d}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val ρ: {val_rho:.4f} | "
              f"LR: {current_lr:.6f}")

# 7. Final Evaluation (best checkpoint)
print("\n" + "="*50)
print("Final Evaluation  (best checkpoint)")
print("="*50)

model.load_state_dict(torch.load(results_dir / "best_mlp_model.pt"))
model.eval()
with torch.no_grad():
    train_preds = model(X_train_t).cpu().numpy().flatten()
    test_preds = model(X_test_t).cpu().numpy().flatten()

# Calculate comprehensive metrics
train_mse = mean_squared_error(y_train, train_preds)
test_mse = mean_squared_error(y_test, test_preds)

train_r2 = r2_score(y_train, train_preds)
test_r2 = r2_score(y_test, test_preds)

train_rho, _ = spearmanr(y_train, train_preds)
test_rho, _ = spearmanr(y_test, test_preds)

train_pearson, _ = pearsonr(y_train, train_preds)
test_pearson, _ = pearsonr(y_test, test_preds)

print(f"\nTraining Set:")
print(f"  MSE:            {train_mse:.4f}")
print(f"  R²:             {train_r2:.4f}")
print(f"  Spearman ρ:     {train_rho:.4f}")
print(f"  Pearson r:      {train_pearson:.4f}")

print(f"\nTest Set:")
print(f"  MSE:            {test_mse:.4f}")
print(f"  R²:             {test_r2:.4f}")
print(f"  Spearman ρ:     {test_rho:.4f}")
print(f"  Pearson r:      {test_pearson:.4f}")
print("="*50)

# 8. Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Training curves
axes[0].plot(train_losses, label='Train Loss', linewidth=2)
axes[0].plot(val_losses, label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Training predictions
axes[1].scatter(y_train, train_preds, alpha=0.3, s=10)
min_val = min(min(y_train), min(train_preds))
max_val = max(max(y_train), max(train_preds))
axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
axes[1].set_xlabel('True Fitness')
axes[1].set_ylabel('Predicted Fitness')
axes[1].set_title(f'Training Set (Spearman ρ = {train_rho:.3f})')
axes[1].grid(True, alpha=0.3)

# Plot 3: Test predictions
axes[2].scatter(y_test, test_preds, alpha=0.3, s=10, color='purple')
min_val = min(min(y_test), min(test_preds))
max_val = max(max(y_test), max(test_preds))
axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
axes[2].set_xlabel('True Fitness')
axes[2].set_ylabel('Predicted Fitness')
axes[2].set_title(f'Test Set (Spearman ρ = {test_rho:.3f})')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / "mlp_baseline_plot.png", dpi=300, bbox_inches='tight')
print(f"\nSaved plot to Results/mlp_baseline_plot.png")

# Save predictions
results_df = pd.DataFrame({
    'true_fitness': y_test,
    'predicted_fitness': test_preds,
    'residual': y_test - test_preds
})
results_df.to_csv(results_dir / "mlp_predictions.csv", index=False)
print(f"Saved predictions to Results/mlp_predictions.csv")

plt.show()
