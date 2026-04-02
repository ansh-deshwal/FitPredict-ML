"""
train_fusion_v5.py
==================
Four-modal fusion: ESM-2 sequence embeddings + structural features
+ ESM-1v zero-shot score + MSA evolutionary features.

What's new vs v3 / v4
----------------------
* Combines all four modalities in one model:
    seq    (1280-d)  — ESM-2 per-sequence embedding
    struct (11→64-d) — DSSP structural features via StructEncoder
    esm1v  (1-d)     — ESM-1v masked-marginal score (z-scored)
    evol   (22→64-d) — MSA position frequencies + entropy via EvolEncoder
* Concat: [seq(1280) | struct_enc(64) | esm1v(1) | evol_enc(64)] → 1409-d

Requires
--------
Data/beta_lactamase_esm2_embeddings.npy         (run extract_embeddings.py)
Results/beta_lactamase_structure_features.npy   (run extract_structure_features.py)
Results/beta_lactamase_esm1v_scores.npy         (run extract_esm1v_scores.py)
Results/beta_lactamase_evolutionary_features.npy (run extract_evolutionary_features.py)

Architecture
------------
  StructEncoder : Linear(11→64) → BN → ReLU → Dropout(0.3)
               → Linear(64→64) → ReLU
  EvolEncoder   : Linear(22→64) → BN → ReLU → Dropout(0.3)
               → Linear(64→64) → ReLU
  Concat        : [seq(1280) | struct_enc(64) | esm1v(1) | evol_enc(64)] → 1409-d
  Projection    : Linear(1409→512) → BN → ReLU → Dropout(0.3)
  ResidualBlock × 2: 512 → 512
  Head          : Linear(512→128) → BN → ReLU → Dropout(0.2) → Linear(128→1)

Split   : 70 / 15 / 15  train / val / test  (random_state=42)
Outputs : Results/fusion_v5_plot.png
          Results/fusion_v5_predictions.csv
          Models/best_fusion_v5.pt
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
df        = pd.read_csv(DATA_DIR    / "BLAT_ECOLX_Stiffler_2015.csv")
X         = np.load(DATA_DIR        / "beta_lactamase_esm2_embeddings.npy")        # (N, 1280)
S         = np.load(RESULTS_DIR     / "beta_lactamase_structure_features.npy")     # (N, 11)
E_esm1v   = np.load(RESULTS_DIR     / "beta_lactamase_esm1v_scores.npy")           # (N,)
E_evol    = np.load(RESULTS_DIR     / "beta_lactamase_evolutionary_features.npy")  # (N, 22)
y         = df["DMS_score"].values.astype(float)

E_esm1v = E_esm1v.reshape(-1, 1)   # (N, 1)

assert S.shape[1]      == 11, f"Expected 11 structural features, got {S.shape[1]}"
assert E_evol.shape[1] == 22, f"Expected 22 evolutionary features, got {E_evol.shape[1]}"

print(f"ESM-2 embeddings   : {X.shape}")
print(f"Struct features    : {S.shape}")
print(f"ESM-1v scores      : {E_esm1v.shape}   range [{E_esm1v.min():.3f}, {E_esm1v.max():.3f}]")
print(f"Evol features      : {E_evol.shape}")
print(f"Labels             : {y.shape}")

STRUCT_DIM = S.shape[1]       # 11
EVOL_DIM   = E_evol.shape[1]  # 22

# ── Split first, then normalise (no leakage) ──────────────────────────────────
(X_tv, X_test,
 S_tv, S_test,
 Esm1v_tv, Esm1v_test,
 Evol_tv, Evol_test,
 y_tv, y_test) = train_test_split(
    X, S, E_esm1v, E_evol, y,
    test_size=0.15, random_state=42, shuffle=True
)

(X_train, X_val,
 S_train, S_val,
 Esm1v_train, Esm1v_val,
 Evol_train, Evol_val,
 y_train, y_val) = train_test_split(
    X_tv, S_tv, Esm1v_tv, Evol_tv, y_tv,
    test_size=0.1765, random_state=42, shuffle=True
)   # 0.1765 × 0.85 ≈ 0.15 of total

def normalize(train, val, test):
    mean = train.mean(axis=0)
    std  = train.std(axis=0) + 1e-8
    return (train - mean) / std, (val - mean) / std, (test - mean) / std

S_train,     S_val,     S_test     = normalize(S_train,     S_val,     S_test)
Esm1v_train, Esm1v_val, Esm1v_test = normalize(Esm1v_train, Esm1v_val, Esm1v_test)
Evol_train,  Evol_val,  Evol_test  = normalize(Evol_train,  Evol_val,  Evol_test)

print(f"\nTrain : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}")

# ── Tensors ────────────────────────────────────────────────────────────────────
def to_t(arr):
    return torch.FloatTensor(arr).to(device)

X_train_t     = to_t(X_train);      X_val_t     = to_t(X_val);      X_test_t     = to_t(X_test)
S_train_t     = to_t(S_train);      S_val_t     = to_t(S_val);      S_test_t     = to_t(S_test)
Esm1v_train_t = to_t(Esm1v_train);  Esm1v_val_t = to_t(Esm1v_val);  Esm1v_test_t = to_t(Esm1v_test)
Evol_train_t  = to_t(Evol_train);   Evol_val_t  = to_t(Evol_val);   Evol_test_t  = to_t(Evol_test)
y_train_t     = to_t(y_train);      y_val_t     = to_t(y_val);      y_test_t     = to_t(y_test)


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


class ModalityEncoder(nn.Module):
    """Small MLP encoder for a single low-dimensional modality (11-d or 22-d → 64-d)."""
    def __init__(self, in_dim, out_dim=64, dropout=0.3):
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


class FourModalFusionV5(nn.Module):
    """
    Fuses all four modalities:
        ESM-2 (1280-d) + StructEncoder (11→64-d)
        + ESM-1v score (1-d) + EvolEncoder (22→64-d)
    → [seq(1280) | struct_enc(64) | esm1v(1) | evol_enc(64)] = 1409-d
    → Linear projection → 2 × ResidualBlock → regression head.
    """
    def __init__(self, seq_dim=1280, struct_dim=11, evol_dim=22,
                 enc_dim=64, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.struct_encoder = ModalityEncoder(struct_dim, enc_dim, dropout)
        self.evol_encoder   = ModalityEncoder(evol_dim,  enc_dim, dropout)

        fusion_dim = seq_dim + enc_dim + 1 + enc_dim   # 1280 + 64 + 1 + 64 = 1409

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

    def forward(self, seq_emb, struct_feat, esm1v_score, evol_feat):
        s = self.struct_encoder(struct_feat)                          # (batch, 64)
        e = self.evol_encoder(evol_feat)                              # (batch, 64)
        x = torch.cat([seq_emb, s, esm1v_score, e], dim=-1)          # (batch, 1409)
        x = self.projection(x)                                        # (batch, 512)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return self.head(x).squeeze(-1)                               # (batch,)


model = FourModalFusionV5(struct_dim=STRUCT_DIM, evol_dim=EVOL_DIM).to(device)
print("\nModel Architecture:")
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")


# ── Training config ────────────────────────────────────────────────────────────
EPOCHS          = 100
BATCH_SIZE      = 32
NOISE_STD       = 0.01
PATIENCE        = 15
MODEL_SAVE_PATH = MODELS_DIR / "best_fusion_v5.pt"

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)


# ── Training loop ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Training  —  seq(1280) + struct(11→64) + ESM-1v(1) + evol(22→64)")
print("=" * 70)

train_losses, val_losses, val_rhos = [], [], []
best_val_rho     = -1.0
patience_counter = 0

for epoch in range(EPOCHS):

    model.train()
    perm        = torch.randperm(X_train_t.size(0))
    epoch_loss  = 0
    num_batches = 0

    for i in range(0, X_train_t.size(0), BATCH_SIZE):
        idx  = perm[i : i + BATCH_SIZE]
        bx   = X_train_t[idx] + torch.randn(len(idx), X_train_t.size(1),
                                             device=device) * NOISE_STD
        bs   = S_train_t[idx]
        be1v = Esm1v_train_t[idx]
        bev  = Evol_train_t[idx]
        by   = y_train_t[idx]

        optimizer.zero_grad()
        loss = criterion(model(bx, bs, be1v, bev), by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss  += loss.item()
        num_batches += 1

    avg_train_loss = epoch_loss / num_batches
    train_losses.append(avg_train_loss)

    model.eval()
    with torch.no_grad():
        val_preds_t = model(X_val_t, S_val_t, Esm1v_val_t, Evol_val_t)
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
print("\n" + "=" * 70)
print("Final Evaluation  (best checkpoint)")
print("=" * 70)

model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
model.eval()
with torch.no_grad():
    train_preds = model(X_train_t, S_train_t, Esm1v_train_t, Evol_train_t).cpu().numpy()
    test_preds  = model(X_test_t,  S_test_t,  Esm1v_test_t,  Evol_test_t).cpu().numpy()

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

print(f"\n{'─' * 70}")
print(f"  Full Results Table")
print(f"{'─' * 70}")
print(f"  Ridge     (seq only)                           ρ = 0.6435")
print(f"  MLP       (seq only)                           ρ = 0.7318")
print(f"  Fusion v1 (seq + struct)                       ρ = 0.8248")
print(f"  Fusion v2 (seq + StructEncoder)                ρ = 0.8140")
print(f"  Fusion v3 (seq + struct + ESM-1v)              ρ = 0.8774  ← prev best")
print(f"  Fusion v4 (seq + struct + evol MSA)            ρ = 0.8294")
print(f"  Fusion v5 (seq + struct + ESM-1v + evol MSA)   ρ = {test_rho:.4f}  ({test_rho - 0.8774:+.4f} vs v3)")
print(f"{'─' * 70}")


# ── Plots ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f"Fusion v5  —  ESM-2(1280) + Struct(11→64) + ESM-1v(1) + Evol(22→64)  "
    f"|  Test ρ = {test_rho:.4f}",
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
plot_path = RESULTS_DIR / "fusion_v5_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"\nSaved plot        → {plot_path}")

pd.DataFrame({
    "true_fitness"     : y_test,
    "predicted_fitness": test_preds,
    "residual"         : y_test - test_preds,
}).to_csv(RESULTS_DIR / "fusion_v5_predictions.csv", index=False)
print("Saved predictions → Results/fusion_v5_predictions.csv")
print(f"Saved model       → {MODEL_SAVE_PATH}")

plt.show()
