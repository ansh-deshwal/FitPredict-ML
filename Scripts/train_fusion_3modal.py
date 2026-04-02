'''
Stage 7 — Three-Modal Fusion: Sequence + Structure + Evolutionary
=================================================================
Combines all three modalities:
    - ESM-2 sequence embeddings  (1280-d)
    - Structural features        (11-d)
    - Evolutionary MSA features  (22-d)

Architecture
------------
    SeqBranch    : 1280 → (pass-through)
    StructEncoder: 11   → 64
    EvolEncoder  : 22   → 64
    Concat       : 1280 + 64 + 64 = 1408
    Projection   : 1408 → 512
    ResBlock x2  : 512  → 512
    Head         : 512  → 128 → 1

Prerequisites (run these first)
--------------------------------
    python3 Scripts/extract_embeddings.py
    python3 Scripts/extract_structure_features.py
    python3 Scripts/extract_evolutionary_features.py
'''

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

BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"
MODELS_DIR  = BASE_DIR / "Models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_DIR    / "BLAT_ECOLX_Stiffler_2015.csv")
X  = np.load(DATA_DIR        / "beta_lactamase_esm2_embeddings.npy")         # (4996, 1280)
S  = np.load(RESULTS_DIR     / "beta_lactamase_structure_features.npy")      # (4996, 11)
E  = np.load(RESULTS_DIR     / "beta_lactamase_evolutionary_features.npy")   # (4996, 22)
y  = df["DMS_score"].values.astype(float)

print(f"Sequence embeddings  : {X.shape}")
print(f"Structural features  : {S.shape}   ← expecting (4996, 11)")
print(f"Evolutionary features: {E.shape}   ← expecting (4996, 22)")
print(f"Labels               : {y.shape}")

assert S.shape[1] == 11, f"Expected 11 structural features, got {S.shape[1]}"
assert E.shape[1] == 22, f"Expected 22 evolutionary features, got {E.shape[1]}"

SEQ_DIM    = X.shape[1]    # 1280
STRUCT_DIM = S.shape[1]    # 11
EVOL_DIM   = E.shape[1]    # 22

# ── Train / Val / Test Split (70/15/15, same seed as all other models) ────────
X_trainval, X_test, S_trainval, S_test, E_trainval, E_test, y_trainval, y_test = \
    train_test_split(X, S, E, y, test_size=0.15, random_state=42, shuffle=True)

X_train, X_val, S_train, S_val, E_train, E_val, y_train, y_val = \
    train_test_split(X_trainval, S_trainval, E_trainval, y_trainval,
                     test_size=0.1765, random_state=42, shuffle=True)

# ── Normalize structural + evolutionary features (train stats only) ────────────
def normalize(train, val, test):
    mean = train.mean(axis=0)
    std  = train.std(axis=0) + 1e-8
    return (train - mean) / std, (val - mean) / std, (test - mean) / std

S_train, S_val, S_test = normalize(S_train, S_val, S_test)
E_train, E_val, E_test = normalize(E_train, E_val, E_test)

print(f"\nTraining samples   : {len(X_train)}")
print(f"Validation samples : {len(X_val)}")
print(f"Test samples       : {len(X_test)}")

# ── Tensors ───────────────────────────────────────────────────────────────────
def to_tensor(*arrays):
    return [torch.FloatTensor(a).to(device) for a in arrays]

X_train_t, S_train_t, E_train_t, y_train_t = to_tensor(X_train, S_train, E_train, y_train)
X_val_t,   S_val_t,   E_val_t,   y_val_t   = to_tensor(X_val,   S_val,   E_val,   y_val)
X_test_t,  S_test_t,  E_test_t,  y_test_t  = to_tensor(X_test,  S_test,  E_test,  y_test)


# ── Model ─────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Residual connection with BatchNorm."""
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
        return self.relu(self.block(x) + x)


class ModalityEncoder(nn.Module):
    """
    Small MLP encoder for a single low-dimensional modality.
    Used for both structural (11-d) and evolutionary (22-d) features.
    Projects to a shared 64-d space before fusion.
    """
    def __init__(self, in_dim, out_dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class ThreeModalFusionModel(nn.Module):
    """
    Fuses ESM-2 (1280-d) + structural (11-d) + evolutionary (22-d) features.

    Architecture
    ------------
    1. StructEncoder : 11  → 64
    2. EvolEncoder   : 22  → 64
    3. Concat        : [seq(1280) | struct(64) | evol(64)] → 1408
    4. Projection    : 1408 → 512
    5. ResBlock x2   : 512 → 512
    6. Head          : 512 → 128 → 1
    """
    def __init__(self, seq_dim=1280, struct_dim=11, evol_dim=22,
                 enc_dim=64, hidden_dim=512, dropout=0.3):
        super().__init__()

        self.struct_encoder = ModalityEncoder(struct_dim, enc_dim, dropout)
        self.evol_encoder   = ModalityEncoder(evol_dim,  enc_dim, dropout)

        fused_dim = seq_dim + enc_dim + enc_dim   # 1280 + 64 + 64 = 1408

        self.projection = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
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

    def forward(self, seq, struct, evol):
        s_enc = self.struct_encoder(struct)            # (batch, 64)
        e_enc = self.evol_encoder(evol)                # (batch, 64)
        x = torch.cat([seq, s_enc, e_enc], dim=-1)    # (batch, 1408)
        x = self.projection(x)                         # (batch, 512)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return self.head(x).squeeze(-1)                # (batch,)


model = ThreeModalFusionModel(
    seq_dim=SEQ_DIM, struct_dim=STRUCT_DIM, evol_dim=EVOL_DIM
).to(device)

print("\nModel Architecture:")
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")


# ── Training Config ───────────────────────────────────────────────────────────
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)

EPOCHS          = 100
BATCH_SIZE      = 32
NOISE_STD       = 0.01   # Gaussian noise on sequence embeddings only
PATIENCE        = 15
MODEL_SAVE_PATH = MODELS_DIR / "best_fusion_3modal.pt"


# ── Training Loop ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Training — Seq(1280) + Struct(11) + Evol(22) → 3-modal fusion")
print("=" * 65)

train_losses, val_losses, val_rhos = [], [], []
best_val_rho     = -1.0
patience_counter = 0

for epoch in range(EPOCHS):

    # ── Train ──────────────────────────────────────────────────────────────
    model.train()
    permutation = torch.randperm(X_train_t.size(0))
    epoch_loss  = 0
    num_batches = 0

    for i in range(0, X_train_t.size(0), BATCH_SIZE):
        idx     = permutation[i : i + BATCH_SIZE]
        batch_x = X_train_t[idx]
        batch_s = S_train_t[idx]
        batch_e = E_train_t[idx]
        batch_y = y_train_t[idx]

        # Gaussian noise on sequence embeddings only
        batch_x = batch_x + torch.randn_like(batch_x) * NOISE_STD

        optimizer.zero_grad()
        preds = model(batch_x, batch_s, batch_e)
        loss  = criterion(preds, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss  += loss.item()
        num_batches += 1

    avg_train_loss = epoch_loss / num_batches
    train_losses.append(avg_train_loss)

    # ── Validate ───────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        val_preds_t = model(X_val_t, S_val_t, E_val_t)
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

    if is_best:
        best_val_rho = val_rho
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
            break


# ── Final Evaluation ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Final Evaluation  (best checkpoint)")
print("=" * 65)

model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()
with torch.no_grad():
    train_preds_final = model(X_train_t, S_train_t, E_train_t).cpu().numpy()
    test_preds_final  = model(X_test_t,  S_test_t,  E_test_t).cpu().numpy()

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

print(f"\n{'─' * 65}")
print(f"  Complete Results Table")
print(f"{'─' * 65}")
print(f"  Ridge (seq only)              ρ = 0.6435")
print(f"  MLP   (seq only)              ρ = 0.7318")
print(f"  Fusion v1 (seq + struct)      ρ = 0.8248")
print(f"  3-modal   (seq+struct+evol)   ρ = {test_rho:.4f}  ({test_rho - 0.8248:+.4f} vs fusion v1)")
print(f"{'─' * 65}")


# ── Plots ─────────────────────────────────────────────────────────────────────
RESULTS_DIR.mkdir(exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f"3-Modal Fusion — ESM-2(1280) + Structure(11) + Evolution(22) | Test ρ = {test_rho:.4f}",
    fontsize=12
)

axes[0].plot(train_losses, label="Train Loss", linewidth=2)
axes[0].plot(val_losses,   label="Val Loss",   linewidth=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].set_title("Training and Validation Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_train, train_preds_final, alpha=0.3, s=10)
lo, hi = y_train.min(), y_train.max()
axes[1].plot([lo, hi], [lo, hi], "r--", lw=2)
axes[1].set_xlabel("True Fitness")
axes[1].set_ylabel("Predicted Fitness")
axes[1].set_title(f"Train Set  (ρ = {train_rho:.3f})")
axes[1].grid(True, alpha=0.3)

axes[2].scatter(y_test, test_preds_final, alpha=0.3, s=10, color="green")
lo, hi = y_test.min(), y_test.max()
axes[2].plot([lo, hi], [lo, hi], "r--", lw=2)
axes[2].set_xlabel("True Fitness")
axes[2].set_ylabel("Predicted Fitness")
axes[2].set_title(f"Test Set  (ρ = {test_rho:.3f})")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = RESULTS_DIR / "fusion_3modal_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"\nSaved plot      → {plot_path}")

pd.DataFrame({
    "true_fitness"     : y_test,
    "predicted_fitness": test_preds_final,
    "residual"         : y_test - test_preds_final,
}).to_csv(RESULTS_DIR / "fusion_3modal_predictions.csv", index=False)
print(f"Saved predictions → Results/fusion_3modal_predictions.csv")

plt.show()
