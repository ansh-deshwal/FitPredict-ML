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

# ── Setup ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# Points to the Data/ and Results/ folders regardless of where you run from
script_dir  = Path(__file__).parent.parent / "Data"
results_dir = Path(__file__).parent.parent / "Results"
results_dir.mkdir(exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(script_dir / "BLAT_ECOLX_Stiffler_2015.csv")
X  = np.load(script_dir  / "beta_lactamase_esm2_embeddings.npy")       # (4996, 1280)
S  = np.load(results_dir / "beta_lactamase_structure_features.npy")    # (4996, 11)
y  = df["DMS_score"].values.astype(float)

print(f"Embeddings:      {X.shape}")
print(f"Struct features: {S.shape}")
print(f"Labels:          {y.shape}")

# Normalize structural features
S_mean = S.mean(axis=0)
S_std  = S.std(axis=0) + 1e-8
S_norm = (S - S_mean) / S_std

# ── Train/Val/Test Split (70% / 15% / 15%) ────────────────────────────────────
X_trainval, X_test, S_trainval, S_test, y_trainval, y_test = train_test_split(
    X, S_norm, y, test_size=0.15, random_state=42, shuffle=True
)
X_train, X_val, S_train, S_val, y_train, y_val = train_test_split(
    X_trainval, S_trainval, y_trainval, test_size=0.1765, random_state=42, shuffle=True
)  # 0.1765 * 0.85 ≈ 0.15 of total
print(f"\nTraining samples:   {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples:       {len(X_test)}")

# Convert to tensors
X_train_t = torch.FloatTensor(X_train).to(device)
S_train_t = torch.FloatTensor(S_train).to(device)
y_train_t = torch.FloatTensor(y_train).to(device)
X_val_t   = torch.FloatTensor(X_val).to(device)
S_val_t   = torch.FloatTensor(S_val).to(device)
y_val_t   = torch.FloatTensor(y_val).to(device)
X_test_t  = torch.FloatTensor(X_test).to(device)
S_test_t  = torch.FloatTensor(S_test).to(device)
y_test_t  = torch.FloatTensor(y_test).to(device)


# ── Model ─────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Unit 4: Residual connection + BatchNorm."""
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)  # skip connection


class MultiModalFusionModel(nn.Module):
    """
    Fuses ESM-2 sequence embeddings (1280) + structural features (11).

    Syllabus coverage:
      Unit 1 — Custom feedforward architecture
      Unit 2 — Dropout, L2 weight decay, Gaussian noise, early stopping
      Unit 4 — Residual blocks, BatchNorm1d
    """
    def __init__(self, seq_dim=1280, struct_dim=11, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(seq_dim + struct_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.res_block1 = ResidualBlock(hidden_dim, dropout=0.2)
        self.res_block2 = ResidualBlock(hidden_dim, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, seq_emb, struct_feat):
        x = torch.cat([seq_emb, struct_feat], dim=-1)  # (batch, seq_dim + struct_dim)
        x = self.projection(x)                          # (batch, 512)
        x = self.res_block1(x)                          # (batch, 512)
        x = self.res_block2(x)                          # (batch, 512)
        return self.head(x).squeeze(-1)                 # (batch,)


model = MultiModalFusionModel(struct_dim=S.shape[1]).to(device)
print("\nModel Architecture:")
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")


# ── Training Config ───────────────────────────────────────────────────────────
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5)

EPOCHS     = 100
BATCH_SIZE = 32
NOISE_STD  = 0.01   # Unit 2: Gaussian noise augmentation
PATIENCE   = 15     # Unit 2: Early stopping


# ── Training Loop ─────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("Starting Training")
print("="*55)

train_losses, val_losses, val_rhos = [], [], []
best_val_rho     = -1.0
patience_counter = 0

for epoch in range(EPOCHS):
    # Train
    model.train()
    permutation = torch.randperm(X_train_t.size(0))
    epoch_loss  = 0
    num_batches = 0

    for i in range(0, X_train_t.size(0), BATCH_SIZE):
        idx     = permutation[i : i + BATCH_SIZE]
        batch_x = X_train_t[idx]
        batch_s = S_train_t[idx]
        batch_y = y_train_t[idx]

        # Unit 2: Gaussian noise on embeddings during training only
        batch_x = batch_x + torch.randn_like(batch_x) * NOISE_STD

        optimizer.zero_grad()
        preds = model(batch_x, batch_s)
        loss  = criterion(preds, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss  += loss.item()
        num_batches += 1

    avg_train_loss = epoch_loss / num_batches
    train_losses.append(avg_train_loss)

    # Validate
    model.eval()
    with torch.no_grad():
        val_preds_t = model(X_val_t, S_val_t)
        val_loss    = criterion(val_preds_t, y_val_t).item()
        val_preds   = val_preds_t.cpu().numpy()
        val_rho, _  = spearmanr(y_val, val_preds)

    val_losses.append(val_loss)
    val_rhos.append(val_rho)
    scheduler.step(val_loss)

    is_best = val_rho > best_val_rho
    if (epoch + 1) % 5 == 0 or epoch == 0 or is_best:
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val ρ: {val_rho:.4f} | "
              f"LR: {lr:.6f}"
              + (" ← best" if is_best else ""))

    # Unit 2: Early stopping
    if is_best:
        best_val_rho = val_rho
        torch.save(model.state_dict(), results_dir / "best_fusion_model.pt")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
            break


# ── Final Evaluation ──────────────────────────────────────────────────────────
print("\n" + "="*55)
print("Final Evaluation  (best checkpoint)")
print("="*55)

model.load_state_dict(torch.load(results_dir / "best_fusion_model.pt"))
model.eval()
with torch.no_grad():
    train_preds_final = model(X_train_t, S_train_t).cpu().numpy()
    test_preds_final  = model(X_test_t,  S_test_t).cpu().numpy()

train_rho, _ = spearmanr(y_train, train_preds_final)
test_rho,  _ = spearmanr(y_test,  test_preds_final)
test_r,    _ = pearsonr(y_test,   test_preds_final)
test_r2      = r2_score(y_test,   test_preds_final)
test_mse     = mean_squared_error(y_test, test_preds_final)

print(f"\nTraining Set:")
print(f"  Spearman ρ : {train_rho:.4f}")
print(f"\nTest Set:")
print(f"  MSE        : {test_mse:.4f}")
print(f"  R²         : {test_r2:.4f}")
print(f"  Spearman ρ : {test_rho:.4f}")
print(f"  Pearson r  : {test_r:.4f}")
print(f"\n{'─'*55}")
print(f"  Fusion model ρ   : {test_rho:.4f}")
print(f"  (retrain Ridge/MLP under 70/15/15 split for valid baseline comparison)")
print(f"{'─'*55}")


# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(train_losses, label='Train Loss', linewidth=2)
axes[0].plot(val_losses,   label='Val Loss',   linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_train, train_preds_final, alpha=0.3, s=10)
lo, hi = y_train.min(), y_train.max()
axes[1].plot([lo, hi], [lo, hi], 'r--', lw=2)
axes[1].set_xlabel('True Fitness')
axes[1].set_ylabel('Predicted Fitness')
axes[1].set_title(f'Train Set  (ρ = {train_rho:.3f})')
axes[1].grid(True, alpha=0.3)

axes[2].scatter(y_test, test_preds_final, alpha=0.3, s=10, color='green')
lo, hi = y_test.min(), y_test.max()
axes[2].plot([lo, hi], [lo, hi], 'r--', lw=2)
axes[2].set_xlabel('True Fitness')
axes[2].set_ylabel('Predicted Fitness')
axes[2].set_title(f'Test Set  (ρ = {test_rho:.3f})')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / "fusion_plot.png", dpi=300, bbox_inches='tight')
print(f"\nSaved plot to Results/fusion_plot.png")

pd.DataFrame({
    'true_fitness':      y_test,
    'predicted_fitness': test_preds_final,
    'residual':          y_test - test_preds_final
}).to_csv(results_dir / "fusion_predictions.csv", index=False)
print("Saved predictions to Results/fusion_predictions.csv")

plt.show()
