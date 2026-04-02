"""
train_multidataset.py
=====================
Multi-dataset fusion training with rank-normalised fitness targets.

Problem solved
--------------
Each DMS dataset uses a completely different fitness scale:
  Stiffler 2015 : [-3.74,  0.36]  mean=-1.26
  Deng     2012 : [-6.06, +2.46]  mean=-2.72
  Firnberg 2014 : [ 0.00, +2.90]  mean= 0.51
  Jacquier 2013 : [-5.32, +1.00]  mean=-1.56

A model trained on Stiffler's absolute values cannot generalise — it predicts
in the wrong range for every other dataset. The fix: rank-normalise each
dataset's scores independently to [0, 1] before pooling. The model then
learns relative fitness ordering (which mutations are better than others),
not absolute values. This is exactly what Spearman ρ measures.

Training data : Stiffler 2015 + Firnberg 2014 + Jacquier 2013  (pooled, rank-normalised)
Held-out test : Deng 2012  (the hardest generalisation case, never seen during training)
In-dist val   : 20% of pooled training data (stratified by dataset)

Architecture
------------
Reuses FourModalFusionV5 (same 4 modalities as train_fusion_v5.py):
  StructEncoder  : 11 → 64
  EvolEncoder    : 22 → 64
  Concat         : [seq(1280) | struct_enc(64) | esm1v(1) | evol_enc(64)] = 1409-d
  Projection     : 1409 → 512
  ResBlock × 2   : 512 → 512
  Head           : 512 → 128 → 1

Feature sourcing (same lookup strategy as evaluate_generalization.py)
----------------------------------------------------------------------
ESM-2 embeddings : precomputed Stiffler file + cached files from evaluate_generalization.py
Struct features  : position lookup from Stiffler's structure_features.npy
ESM-1v scores    : (pos, mut_aa) lookup from Stiffler's esm1v_scores.npy
Evol features    : position lookup from Stiffler's evolutionary_features.npy
All zero-padded for variants whose position is absent in the Stiffler lookups.

Normalisation
-------------
Struct / ESM-1v / Evol features : z-scored using pooled train-split stats
DMS targets                      : rank-normalised per dataset → [0,1] (no further norm)

Outputs
-------
Models/best_multidataset.pt
Results/multidataset_plot.png
Results/multidataset_generalization_deng.csv
Results/generalization_results_multidataset.csv   (full summary table)
"""

import re
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, rankdata
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}\n")

BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"
MODELS_DIR  = BASE_DIR / "Models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ── Mutation parser ────────────────────────────────────────────────────────────
def parse_mutation(mutant_str):
    m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mutant_str.strip())
    if m is None:
        raise ValueError(f"Cannot parse mutation: {mutant_str!r}")
    return m.group(1), int(m.group(2)), m.group(3)   # wt_aa, pos, mut_aa


# ── Build Stiffler lookup tables ───────────────────────────────────────────────
print("Building feature lookup tables from Stiffler 2015 …")
df_stiff   = pd.read_csv(DATA_DIR    / "BLAT_ECOLX_Stiffler_2015.csv")
S_stiff    = np.load(RESULTS_DIR     / "beta_lactamase_structure_features.npy")    # (4996,11)
E1v_stiff  = np.load(RESULTS_DIR     / "beta_lactamase_esm1v_scores.npy")          # (4996,)
Ev_stiff   = np.load(RESULTS_DIR     / "beta_lactamase_evolutionary_features.npy") # (4996,22)

pos_to_struct = {}
pos_to_evol   = {}
pos_mut_to_esm1v = {}

for i, row in df_stiff.iterrows():
    _, pos, mut_aa = parse_mutation(row["mutant"])
    if pos not in pos_to_struct:
        pos_to_struct[pos] = S_stiff[i]
        pos_to_evol[pos]   = Ev_stiff[i]
    pos_mut_to_esm1v[(pos, mut_aa)] = float(E1v_stiff[i])

print(f"  Struct/Evol lookup : {len(pos_to_struct)} positions")
print(f"  ESM-1v lookup      : {len(pos_mut_to_esm1v)} entries")


# ── Feature builder for any dataset ───────────────────────────────────────────
def build_features(df, embed_path):
    """Returns X (ESM-2), S (struct), E1v (esm1v), Ev (evol), y_raw (DMS scores)."""
    X = np.load(embed_path).astype(np.float32)          # (N, 1280)

    N = len(df)
    S   = np.zeros((N, 11), dtype=np.float32)
    E1v = np.zeros((N,  1), dtype=np.float32)
    Ev  = np.zeros((N, 22), dtype=np.float32)

    for local_i, (_, row) in enumerate(df.iterrows()):
        _, pos, mut_aa = parse_mutation(row["mutant"])
        if pos in pos_to_struct:
            S[local_i]  = pos_to_struct[pos]
            Ev[local_i] = pos_to_evol[pos]
        if (pos, mut_aa) in pos_mut_to_esm1v:
            E1v[local_i, 0] = pos_mut_to_esm1v[(pos, mut_aa)]

    y_raw = df["DMS_score"].values.astype(np.float32)
    return X, S, E1v, Ev, y_raw


# ── Load all datasets ──────────────────────────────────────────────────────────
print("\nLoading datasets …")

TRAINING_DATASETS = [
    ("Stiffler_2015", DATA_DIR    / "BLAT_ECOLX_Stiffler_2015.csv",
                      DATA_DIR    / "beta_lactamase_esm2_embeddings.npy"),
    ("Firnberg_2014", DATA_DIR    / "BLAT_ECOLX_Firnberg_2014.csv",
                      RESULTS_DIR / "Firnberg_2014_esm2_embeddings.npy"),
    ("Jacquier_2013", DATA_DIR    / "BLAT_ECOLX_Jacquier_2013.csv",
                      RESULTS_DIR / "Jacquier_2013_esm2_embeddings.npy"),
]

HELD_OUT = ("Deng_2012", DATA_DIR    / "BLAT_ECOLX_Deng_2012.csv",
                         RESULTS_DIR / "Deng_2012_esm2_embeddings.npy")

# Check all embedding files exist
missing = []
for name, csv, emb in TRAINING_DATASETS + [HELD_OUT]:
    if not emb.exists():
        missing.append(str(emb))
if missing:
    sys.exit(f"ERROR: Missing embedding files (run evaluate_generalization.py first):\n" +
             "\n".join(missing))

# Build features and rank-normalise targets per dataset
all_X, all_S, all_E1v, all_Ev, all_y, all_labels = [], [], [], [], [], []

for name, csv_path, emb_path in TRAINING_DATASETS:
    df = pd.read_csv(csv_path)
    X, S, E1v, Ev, y_raw = build_features(df, emb_path)

    # Rank-normalise: [0, 1] preserving order
    y_rank = (rankdata(y_raw) / len(y_raw)).astype(np.float32)

    all_X.append(X);    all_S.append(S);     all_E1v.append(E1v)
    all_Ev.append(Ev);  all_y.append(y_rank)
    all_labels.extend([name] * len(df))
    print(f"  {name:<16s}  {len(df):>4d} variants  y_rank [{y_rank.min():.3f}, {y_rank.max():.3f}]")

X_pool  = np.concatenate(all_X,   axis=0)
S_pool  = np.concatenate(all_S,   axis=0)
E1v_pool = np.concatenate(all_E1v, axis=0)
Ev_pool  = np.concatenate(all_Ev,  axis=0)
y_pool   = np.concatenate(all_y,   axis=0)
labels   = np.array(all_labels)
print(f"\n  Pooled training set: {len(X_pool)} variants")

# Load Deng 2012 (held-out) — rank-normalised for fair metric comparison
df_deng = pd.read_csv(HELD_OUT[1])
X_deng, S_deng, E1v_deng, Ev_deng, y_deng_raw = build_features(df_deng, HELD_OUT[2])
y_deng_rank = (rankdata(y_deng_raw) / len(y_deng_raw)).astype(np.float32)
print(f"  {'Deng_2012 (held-out)':<20s}  {len(df_deng):>4d} variants")


# ── Train / val split (stratified by dataset) ──────────────────────────────────
idx = np.arange(len(X_pool))
idx_train, idx_val = train_test_split(idx, test_size=0.20, random_state=42,
                                      stratify=labels)

X_train, X_val   = X_pool[idx_train],   X_pool[idx_val]
S_train, S_val   = S_pool[idx_train],   S_pool[idx_val]
E1v_train,E1v_val= E1v_pool[idx_train], E1v_pool[idx_val]
Ev_train, Ev_val = Ev_pool[idx_train],  Ev_pool[idx_val]
y_train, y_val   = y_pool[idx_train],   y_pool[idx_val]

print(f"\n  Train : {len(X_train)} | Val : {len(X_val)}")


# ── Normalise features (train stats only, no leakage) ─────────────────────────
def normalize(train, val, *others):
    mean = train.mean(axis=0)
    std  = train.std(axis=0) + 1e-8
    return [(a - mean) / std for a in [train, val, *others]]

S_train,   S_val,   S_deng   = normalize(S_train,   S_val,   S_deng)
E1v_train, E1v_val, E1v_deng = normalize(E1v_train, E1v_val, E1v_deng)
Ev_train,  Ev_val,  Ev_deng  = normalize(Ev_train,  Ev_val,  Ev_deng)


# ── Tensors ────────────────────────────────────────────────────────────────────
def to_t(a):
    return torch.FloatTensor(a).to(device)

Xt, St, E1t, Evt, yt      = to_t(X_train),  to_t(S_train),  to_t(E1v_train), to_t(Ev_train),  to_t(y_train)
Xv, Sv, E1v_, Evv, yv     = to_t(X_val),    to_t(S_val),    to_t(E1v_val),   to_t(Ev_val),    to_t(y_val)
Xd, Sd, E1d, Evd, yd_rank = to_t(X_deng),   to_t(S_deng),   to_t(E1v_deng),  to_t(Ev_deng),   to_t(y_deng_rank)


# ── Model (identical to FourModalFusionV5) ─────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim), nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.BatchNorm1d(dim), nn.Linear(dim, dim), nn.Dropout(dropout),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)


class ModalityEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, out_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class FourModalFusion(nn.Module):
    def __init__(self, seq_dim=1280, struct_dim=11, evol_dim=22,
                 enc_dim=64, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.struct_encoder = ModalityEncoder(struct_dim, enc_dim, dropout)
        self.evol_encoder   = ModalityEncoder(evol_dim,  enc_dim, dropout)
        fusion_dim = seq_dim + enc_dim + 1 + enc_dim   # 1409
        self.projection = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.res_block1 = ResidualBlock(hidden_dim, dropout=0.2)
        self.res_block2 = ResidualBlock(hidden_dim, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1),
        )

    def forward(self, seq, struct, esm1v, evol):
        s = self.struct_encoder(struct)
        e = self.evol_encoder(evol)
        x = torch.cat([seq, s, esm1v, e], dim=-1)
        x = self.projection(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return self.head(x).squeeze(-1)


model = FourModalFusion().to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel: FourModalFusion  |  {total_params:,} trainable parameters")


# ── Training config ────────────────────────────────────────────────────────────
EPOCHS          = 100
BATCH_SIZE      = 32
NOISE_STD       = 0.01
PATIENCE        = 15
MODEL_SAVE_PATH = MODELS_DIR / "best_multidataset.pt"

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)


# ── Training loop ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Training  —  Stiffler + Firnberg + Jacquier  (rank-normalised targets)")
print("=" * 70)

train_losses, val_losses, val_rhos = [], [], []
best_val_rho, patience_counter = -1.0, 0

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(Xt.size(0))
    epoch_loss, num_batches = 0, 0

    for i in range(0, Xt.size(0), BATCH_SIZE):
        idx  = perm[i : i + BATCH_SIZE]
        bx   = Xt[idx] + torch.randn(len(idx), Xt.size(1), device=device) * NOISE_STD
        optimizer.zero_grad()
        loss = criterion(model(bx, St[idx], E1t[idx], Evt[idx]), yt[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item(); num_batches += 1

    avg_train_loss = epoch_loss / num_batches
    train_losses.append(avg_train_loss)

    model.eval()
    with torch.no_grad():
        vp = model(Xv, Sv, E1v_, Evv)
        val_loss = criterion(vp, yv).item()
        val_rho, _ = spearmanr(y_val, vp.cpu().numpy())

    val_losses.append(val_loss); val_rhos.append(val_rho)
    scheduler.step(val_loss)

    is_best = val_rho > best_val_rho
    if (epoch + 1) % 5 == 0 or epoch == 0 or is_best:
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | Train {avg_train_loss:.4f} | "
              f"Val {val_loss:.4f} | Val ρ {val_rho:.4f} | "
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

model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True, map_location=device))
model.eval()

with torch.no_grad():
    train_preds = model(Xt, St, E1t, Evt).cpu().numpy()
    val_preds   = model(Xv, Sv, E1v_, Evv).cpu().numpy()
    deng_preds  = model(Xd, Sd, E1d, Evd).cpu().numpy()

train_rho, _ = spearmanr(y_train,     train_preds)
val_rho,   _ = spearmanr(y_val,       val_preds)
deng_rho,  _ = spearmanr(y_deng_rank, deng_preds)
deng_r,    _ = pearsonr(y_deng_rank,  deng_preds)
deng_r2      = r2_score(y_deng_rank,  deng_preds)
deng_mse     = mean_squared_error(y_deng_rank, deng_preds)

# Also evaluate each training dataset separately
print(f"\nTrain  ρ (pooled)   : {train_rho:.4f}")
print(f"Val    ρ (pooled)   : {val_rho:.4f}")
print(f"\nDeng 2012 (held-out, never seen during training):")
print(f"  Spearman ρ : {deng_rho:.4f}   (single-assay baseline: 0.5702)")
print(f"  Pearson  r : {deng_r:.4f}")
print(f"  R²         : {deng_r2:.4f}")
print(f"  MSE        : {deng_mse:.4f}")

print(f"\n{'─' * 70}")
print(f"  Generalisation comparison  (Deng 2012 Spearman ρ)")
print(f"{'─' * 70}")
print(f"  Single-assay Fusion v3  (trained on Stiffler only)  : 0.5702")
print(f"  Multi-dataset           (trained on 3 assays)       : {deng_rho:.4f}  "
      f"({deng_rho - 0.5702:+.4f})")
print(f"{'─' * 70}")


# ── Plots ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f"Multi-dataset Fusion — Stiffler+Firnberg+Jacquier → Deng 2012  "
    f"|  Deng ρ = {deng_rho:.4f}",
    fontsize=11,
)

axes[0].plot(train_losses, label="Train", linewidth=2)
axes[0].plot(val_losses,   label="Val",   linewidth=2)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss")
axes[0].set_title("Loss Curves"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_train, train_preds, alpha=0.2, s=8, c="steelblue")
lo, hi = 0, 1
axes[1].plot([lo, hi], [lo, hi], "r--", lw=2)
axes[1].set_xlabel("True Rank Score"); axes[1].set_ylabel("Predicted Rank Score")
axes[1].set_title(f"Train (pooled)  ρ = {train_rho:.3f}"); axes[1].grid(True, alpha=0.3)

axes[2].scatter(y_deng_rank, deng_preds, alpha=0.3, s=10, c="seagreen")
axes[2].plot([lo, hi], [lo, hi], "r--", lw=2)
axes[2].set_xlabel("True Rank Score (Deng 2012)"); axes[2].set_ylabel("Predicted Rank Score")
axes[2].set_title(f"Deng 2012 (held-out)  ρ = {deng_rho:.3f}"); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = RESULTS_DIR / "multidataset_plot.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"\nSaved plot        → {plot_path}")

pd.DataFrame({
    "mutant":            df_deng["mutant"].values,
    "true_rank":         y_deng_rank,
    "predicted_rank":    deng_preds,
    "true_dms_raw":      y_deng_raw,
}).to_csv(RESULTS_DIR / "multidataset_generalization_deng.csv", index=False)

pd.DataFrame([
    {"dataset": "Stiffler+Firnberg+Jacquier (train ρ)", "spearman_rho": train_rho},
    {"dataset": "Stiffler+Firnberg+Jacquier (val ρ)",   "spearman_rho": val_rho},
    {"dataset": "Deng_2012 (held-out)",                  "spearman_rho": deng_rho,
     "pearson_r": deng_r, "r2": deng_r2, "mse": deng_mse},
]).to_csv(RESULTS_DIR / "generalization_results_multidataset.csv", index=False)

print(f"Saved predictions → Results/multidataset_generalization_deng.csv")
print(f"Saved summary     → Results/generalization_results_multidataset.csv")
print(f"Saved model       → {MODEL_SAVE_PATH}")

plt.show()
