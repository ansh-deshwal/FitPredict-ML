"""
train_fusion_v3.py
==================
Multi-modal fusion: ESM-2 sequence embeddings + structural features
+ ESM-1v zero-shot score.

What's new vs v2
-----------------
* Adds a single ESM-1v masked-marginal score per variant as an extra input.
  This score encodes evolutionary plausibility — a strong prior that is
  orthogonal to the learned sequence and structure representations.
* The score is z-score normalised (train split only) and concatenated after
  the StructureEncoder output: [seq(1280) | struct_enc(64) | esm1v(1)] → 1345-d.

Requires
--------
Results/beta_lactamase_esm1v_scores.npy  (run extract_esm1v_scores.py first)

Architecture
------------
  StructureEncoder : Linear(11→64) → BN → ReLU → Dropout(0.3)
                   → Linear(64→64) → ReLU
  Concat           : [seq(1280) | struct_enc(64) | esm1v(1)] → 1345-d
  Projection       : Linear(1345→512) → BN → ReLU → Dropout(0.3)
  ResidualBlock × 2: 512 → 512
  Head             : Linear(512→128) → BN → ReLU → Dropout(0.2) → Linear(128→1)

Split   : 70 / 15 / 15  train / val / test  (random_state=42)
Outputs : Results/fusion_v3_plot.png
          Results/fusion_v3_predictions.csv
          Models/best_fusion_v3.pt
"""

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

# ── Paths ──────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"
MODELS_DIR  = BASE_DIR / "Models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data …")
df  = pd.read_csv(DATA_DIR    / "BLAT_ECOLX_Stiffler_2015.csv")
X   = np.load(DATA_DIR        / "beta_lactamase_esm2_embeddings.npy")       # (N, 1280)
S   = np.load(RESULTS_DIR     / "beta_lactamase_structure_features.npy")    # (N, 11)
E   = np.load(RESULTS_DIR     / "beta_lactamase_esm1v_scores.npy")          # (N,)
y   = df["DMS_score"].values.astype(float)

E   = E.reshape(-1, 1)   # (N, 1)

print(f"ESM-2 embeddings : {X.shape}")
print(f"Struct features  : {S.shape}")
print(f"ESM-1v scores    : {E.shape}   range [{E.min():.3f}, {E.max():.3f}]")
print(f"Labels           : {y.shape}")

STRUCT_DIM = S.shape[1]   # 11

# ── Split first, then normalise (no leakage) ──────────────────────────────────
X_tv,  X_test,  S_tv,  S_test,  E_tv,  E_test,  y_tv,  y_test = train_test_split(
    X, S, E, y, test_size=0.15, random_state=42, shuffle=True
)
X_train, X_val, S_train, S_val, E_train, E_val, y_train, y_val = train_test_split(
    X_tv, S_tv, E_tv, y_tv, test_size=0.1765, random_state=42, shuffle=True
)   # 0.1765 × 0.85 ≈ 0.15 of total

# Structural features
S_mean, S_std = S_train.mean(0), S_train.std(0) + 1e-8
S_train = (S_train - S_mean) / S_std
S_val   = (S_val   - S_mean) / S_std
S_test  = (S_test  - S_mean) / S_std

# ESM-1v score
E_mean, E_std = E_train.mean(), E_train.std() + 1e-8
E_train = (E_train - E_mean) / E_std
E_val   = (E_val   - E_mean) / E_std
E_test  = (E_test  - E_mean) / E_std

print(f"\nTrain : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}")

# ── Tensors ────────────────────────────────────────────────────────────────────
def to_t(arr):
    return torch.FloatTensor(arr).to(device)

X_train_t, S_train_t, E_train_t, y_train_t = to_t(X_train), to_t(S_train), to_t(E_train), to_t(y_train)
X_val_t,   S_val_t,   E_val_t,   y_val_t   = to_t(X_val),   to_t(S_val),   to_t(E_val),   to_t(y_val)
X_test_t,  S_test_t,  E_test_t,  y_test_t  = to_t(X_test),  to_t(S_test),  to_t(E_test),  to_t(y_test)


# ── Model ──────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """BN → Linear → ReLU → Dropout → BN → Linear → Dropout, then ReLU(x + skip)."""
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


class StructureEncoder(nn.Module):
    def __init__(self, struct_dim=11, out_dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(struct_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
            nn.ReLU(),
        )

    def forward(self, s):
        return self.net(s)


class MultiModalFusionV3(nn.Module):
    """
    ESM-2 (1280-d) + StructureEncoder (11→64-d) + ESM-1v score (1-d) → 1345-d
    → Linear projection → 2 × ResidualBlock → regression head.
    """
    def __init__(self, seq_dim=1280, struct_dim=11,
                 struct_enc_dim=64, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.struct_encoder = StructureEncoder(struct_dim, struct_enc_dim, dropout)
        fusion_dim = seq_dim + struct_enc_dim + 1   # +1 for ESM-1v scalar
        self.projection = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
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

    def forward(self, seq_emb, struct_feat, esm1v_score):
        s = self.struct_encoder(struct_feat)              # (batch, 64)
        x = torch.cat([seq_emb, s, esm1v_score], dim=-1) # (batch, 1345)
        x = self.projection(x)                            # (batch, 512)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return self.head(x).squeeze(-1)                   # (batch,)


model = MultiModalFusionV3(struct_dim=STRUCT_DIM).to(device)
print("\nModel Architecture:")
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")


# ── Training config ────────────────────────────────────────────────────────────
EPOCHS          = 100
BATCH_SIZE      = 32
NOISE_STD       = 0.01
PATIENCE        = 15
MODEL_SAVE_PATH = MODELS_DIR / "best_fusion_v3.pt"

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)


# ── Training loop ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Training  —  seq(1280) + struct(11→64) + ESM-1v score(1)")
print("=" * 65)

train_losses, val_losses, val_rhos = [], [], []
best_val_rho     = -1.0
patience_counter = 0

for epoch in range(EPOCHS):

    model.train()
    perm        = torch.randperm(X_train_t.size(0))
    epoch_loss  = 0
    num_batches = 0

    for i in range(0, X_train_t.size(0), BATCH_SIZE):
        idx = perm[i : i + BATCH_SIZE]
        bx  = X_train_t[idx] + torch.randn(len(idx), X_train_t.size(1),
                                            device=device) * NOISE_STD
        bs  = S_train_t[idx]
        be  = E_train_t[idx]
        by  = y_train_t[idx]

        optimizer.zero_grad()
        loss = criterion(model(bx, bs, be), by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss  += loss.item()
        num_batches += 1

    avg_train_loss = epoch_loss / num_batches
    train_losses.append(avg_train_loss)

    model.eval()
    with torch.no_grad():
        val_preds_t = model(X_val_t, S_val_t, E_val_t)
        val_loss    = criterion(val_preds_t, y_val_t).item()
        val_rho, _  = spearmanr(y_val, val_preds_t.cpu().numpy())

    val_losses.append(val_loss)
    val_rhos.append(val_rho)
    scheduler.step(val_loss)

    is_best = val_rho > best_val_rho
    if (epoch + 1) % 5 == 0 or epoch == 0 or is_best:
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | "
              f"Train {avg_train_loss:.4f} | Val {val_loss:.4f} | "
              f"Val ρ {val_rho:.4f} | "
              f"LR {optimizer.param_groups[0]['lr']:.2e}"
              + (" ← best" if is_best else ""))

    if is_best:
        best_val_rho = val_rho
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}.")
            break


# ── Final evaluation ───────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Final Evaluation  (best checkpoint)")
print("=" * 65)

model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
model.eval()
with torch.no_grad():
    train_preds = model(X_train_t, S_train_t, E_train_t).cpu().numpy()
    test_preds  = model(X_test_t,  S_test_t,  E_test_t).cpu().numpy()

train_rho, _ = spearmanr(y_train, train_preds)
test_rho,  _ = spearmanr(y_test,  test_preds)
test_r,    _ = pearsonr(y_test,   test_preds)
test_r2      = r2_score(y_test,   test_preds)
test_mse     = mean_squared_error(y_test, test_preds)

print(f"\nTrain  Spearman ρ : {train_rho:.4f}")
print(f"\nTest   MSE        : {test_mse:.4f}")
print(f"       R²          : {test_r2:.4f}")
print(f"       Spearman ρ  : {test_rho:.4f}")
print(f"       Pearson r   : {test_r:.4f}")


# ── Plots ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f"Fusion v3  —  ESM-2 (1280) + StructureEncoder (11→64) + ESM-1v score  "
    f"|  Test ρ = {test_rho:.3f}",
    fontsize=11,
)

axes[0].plot(train_losses, label="Train", linewidth=2)
axes[0].plot(val_losses,   label="Val",   linewidth=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].set_title("Loss Curves")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

for ax, y_true, y_pred, title, color in [
    (axes[1], y_train, train_preds, f"Train  (ρ = {train_rho:.3f})", "steelblue"),
    (axes[2], y_test,  test_preds,  f"Test   (ρ = {test_rho:.3f})",  "seagreen"),
]:
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color=color)
    lo, hi = y_true.min(), y_true.max()
    ax.plot([lo, hi], [lo, hi], "r--", lw=2)
    ax.set_xlabel("True Fitness")
    ax.set_ylabel("Predicted Fitness")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = RESULTS_DIR / "fusion_v3_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"\nSaved plot        → {plot_path}")

pd.DataFrame({
    "true_fitness"     : y_test,
    "predicted_fitness": test_preds,
    "residual"         : y_test - test_preds,
}).to_csv(RESULTS_DIR / "fusion_v3_predictions.csv", index=False)
print("Saved predictions → Results/fusion_v3_predictions.csv")
print(f"Saved model       → {MODEL_SAVE_PATH}")

plt.show()
