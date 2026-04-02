"""
evaluate_generalization.py
==========================
Zero-shot cross-dataset generalization evaluation for Fusion v3.

Evaluates how well the Fusion v3 model (trained on Stiffler 2015) predicts
fitness scores in three held-out experimental datasets — all measuring
TEM-1 beta-lactamase fitness but via different assays:

  • BLAT_ECOLX_Deng_2012      (4996 single-point mutants)
  • BLAT_ECOLX_Firnberg_2014  (4783 single-point mutants)
  • BLAT_ECOLX_Jacquier_2013  (  989 single-point mutants)

No retraining is performed. This is a pure transfer test.

Feature sourcing
----------------
Structural (11-d):
  Position-indexed lookup from Results/beta_lactamase_structure_features.npy
  using Stiffler 2015 mutation annotations. Because structural features come
  from the WT PDB (position-specific, assay-independent), the same lookup
  applies to all datasets.

ESM-1v score (1-d):
  (position, mut_aa) lookup from Results/beta_lactamase_esm1v_scores.npy.
  Variants whose mutation is not in Stiffler receive score 0 (rare edge case).

ESM-2 embeddings (1280-d):
  Computed on-the-fly per dataset, then cached to Results/ for reuse.
  Requires ESM-2 650M (~2.5 GB); skip or run on GPU for speed.

Normalisation
-------------
z-score stats are computed from the full Stiffler 2015 dataset (a close
proxy for train-split stats; the model was never exposed to the other three
datasets).

Requires
--------
Data/BLAT_ECOLX_Stiffler_2015.csv
Data/beta_lactamase_esm2_embeddings.npy
Results/beta_lactamase_structure_features.npy
Results/beta_lactamase_esm1v_scores.npy
Models/best_fusion_v3.pt

Outputs
-------
Results/generalization_results.csv    — per-dataset metrics
Console table comparing Stiffler test ρ vs each new dataset ρ
"""

import re
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"
MODELS_DIR  = BASE_DIR / "Models"

STIFFLER_CSV      = DATA_DIR    / "BLAT_ECOLX_Stiffler_2015.csv"
STIFFLER_EMBED    = DATA_DIR    / "beta_lactamase_esm2_embeddings.npy"
STRUCT_FEAT_PATH  = RESULTS_DIR / "beta_lactamase_structure_features.npy"
ESM1V_SCORES_PATH = RESULTS_DIR / "beta_lactamase_esm1v_scores.npy"
MODEL_PATH        = MODELS_DIR  / "best_fusion_v3.pt"

EVAL_DATASETS = [
    ("Deng_2012",     DATA_DIR / "BLAT_ECOLX_Deng_2012.csv"),
    ("Firnberg_2014", DATA_DIR / "BLAT_ECOLX_Firnberg_2014.csv"),
    ("Jacquier_2013", DATA_DIR / "BLAT_ECOLX_Jacquier_2013.csv"),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")


# ── Model definition (must match train_fusion_v3.py) ──────────────────────────

class ResidualBlock(nn.Module):
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
    def __init__(self, seq_dim=1280, struct_dim=11,
                 struct_enc_dim=64, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.struct_encoder = StructureEncoder(struct_dim, struct_enc_dim, dropout)
        fusion_dim = seq_dim + struct_enc_dim + 1
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
        s = self.struct_encoder(struct_feat)
        x = torch.cat([seq_emb, s, esm1v_score], dim=-1)
        x = self.projection(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return self.head(x).squeeze(-1)


# ── Load model ────────────────────────────────────────────────────────────────
print("Loading Fusion v3 checkpoint …")
if not MODEL_PATH.exists():
    sys.exit(f"ERROR: {MODEL_PATH} not found. Run train_fusion_v3.py first.")
model = MultiModalFusionV3().to(device)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=device))
model.eval()
print(f"  Loaded {MODEL_PATH.name}")


# ── Build Stiffler feature lookups ────────────────────────────────────────────
print("\nBuilding lookup tables from Stiffler 2015 data …")

def parse_mutation(mutant_str):
    m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mutant_str.strip())
    if m is None:
        raise ValueError(f"Cannot parse mutation: {mutant_str!r}")
    return m.group(1), int(m.group(2)), m.group(3)   # wt_aa, pos, mut_aa

df_stiff   = pd.read_csv(STIFFLER_CSV)
S_stiff    = np.load(STRUCT_FEAT_PATH)    # (4996, 11)
E1v_stiff  = np.load(ESM1V_SCORES_PATH)  # (4996,)

# pos → struct features (all variants at same pos share the WT structural context)
pos_to_struct = {}
for i, row in df_stiff.iterrows():
    _, pos, _ = parse_mutation(row["mutant"])
    if pos not in pos_to_struct:
        pos_to_struct[pos] = S_stiff[i]

# (pos, mut_aa) → ESM-1v score
pos_mut_to_esm1v = {}
for i, row in df_stiff.iterrows():
    _, pos, mut_aa = parse_mutation(row["mutant"])
    pos_mut_to_esm1v[(pos, mut_aa)] = float(E1v_stiff[i])

print(f"  Position → struct lookup  : {len(pos_to_struct)} positions")
print(f"  (pos, mut) → ESM-1v lookup: {len(pos_mut_to_esm1v)} entries")


# ── Normalisation stats from full Stiffler dataset ────────────────────────────
X_stiff = np.load(STIFFLER_EMBED)         # (4996, 1280)
E1v_arr = E1v_stiff.reshape(-1, 1)

S_mean, S_std    = S_stiff.mean(0),     S_stiff.std(0) + 1e-8
E1v_mean, E1v_std = E1v_arr.mean(),    E1v_arr.std() + 1e-8

print(f"\nNormalisation stats (Stiffler full dataset):")
print(f"  Struct  mean range : [{S_mean.min():.3f}, {S_mean.max():.3f}]")
print(f"  ESM-1v  mean       : {E1v_mean:.3f}   std : {E1v_std:.3f}")


# ── ESM-2 embedding helper ────────────────────────────────────────────────────
_esm2_model  = None
_esm2_alpha  = None
_batch_conv  = None

def get_esm2():
    global _esm2_model, _esm2_alpha, _batch_conv
    if _esm2_model is None:
        import esm
        print("\n  Loading ESM-2 650M model for embedding …")
        _esm2_model, _esm2_alpha = esm.pretrained.esm2_t33_650M_UR50D()
        _batch_conv = _esm2_alpha.get_batch_converter()
        _esm2_model.eval().to(device)
    return _esm2_model, _esm2_alpha, _batch_conv

def compute_embeddings(sequences, batch_size=16):
    """Mean-pool ESM-2 representations over sequence length."""
    m, alpha, bc = get_esm2()
    all_embs = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        data       = [(f"seq_{j}", s) for j, s in enumerate(batch_seqs)]
        _, _, tokens = bc(data)
        tokens = tokens.to(device)
        with torch.no_grad():
            out = m(tokens, repr_layers=[33], return_contacts=False)
        reps = out["representations"][33]   # (B, L+2, 1280)
        # Mean pool over sequence positions (exclude BOS and EOS)
        for k, seq in enumerate(batch_seqs):
            all_embs.append(reps[k, 1 : len(seq) + 1].mean(0).cpu().numpy())
    return np.array(all_embs, dtype=np.float32)


# ── Evaluate each dataset ─────────────────────────────────────────────────────

def features_for_dataset(df_eval, dataset_name):
    """
    Returns (X_emb, S_feat, E1v_feat, y) for the evaluation dataset.
    X_emb : ESM-2 embeddings (computed or cached)
    S_feat: structural features via position lookup
    E1v   : ESM-1v scores via (pos, mut_aa) lookup
    y     : DMS_score
    """
    # ── ESM-2 embeddings (cache to Results/) ──────────────────────────────
    cache_path = RESULTS_DIR / f"{dataset_name}_esm2_embeddings.npy"
    if cache_path.exists():
        print(f"  Loading cached ESM-2 embeddings from {cache_path.name}")
        X_emb = np.load(cache_path)
    else:
        print(f"  Computing ESM-2 embeddings for {len(df_eval)} sequences …")
        from tqdm import tqdm
        seqs  = df_eval["mutated_sequence"].tolist()
        X_emb = compute_embeddings(seqs)
        np.save(cache_path, X_emb)
        print(f"  Cached → {cache_path}")

    # ── Structural features via position lookup ────────────────────────────
    S_feat   = np.zeros((len(df_eval), 11), dtype=np.float32)
    E1v_feat = np.zeros((len(df_eval), 1),  dtype=np.float32)
    missing_struct, missing_esm1v = 0, 0

    for i, row in df_eval.iterrows():
        _, pos, mut_aa = parse_mutation(row["mutant"])
        local_i = i - df_eval.index[0]   # row index within this df

        if pos in pos_to_struct:
            S_feat[local_i] = pos_to_struct[pos]
        else:
            missing_struct += 1

        if (pos, mut_aa) in pos_mut_to_esm1v:
            E1v_feat[local_i, 0] = pos_mut_to_esm1v[(pos, mut_aa)]
        else:
            missing_esm1v += 1

    if missing_struct:
        print(f"  WARNING: {missing_struct} variants have no struct feature (position not in Stiffler) → zeroed")
    if missing_esm1v:
        print(f"  WARNING: {missing_esm1v} variants have no ESM-1v score (not in Stiffler) → zeroed")

    # ── Normalise using Stiffler stats ─────────────────────────────────────
    S_norm   = (S_feat  - S_mean)   / S_std
    E1v_norm = (E1v_feat - E1v_mean) / E1v_std

    y = df_eval["DMS_score"].values.astype(float)
    return X_emb, S_norm, E1v_norm, y


def evaluate(X_emb, S_norm, E1v_norm, y):
    Xt = torch.FloatTensor(X_emb).to(device)
    St = torch.FloatTensor(S_norm).to(device)
    Et = torch.FloatTensor(E1v_norm).to(device)

    with torch.no_grad():
        preds = model(Xt, St, Et).cpu().numpy()

    rho,  _ = spearmanr(y, preds)
    r,    _ = pearsonr(y, preds)
    r2       = r2_score(y, preds)
    mse      = mean_squared_error(y, preds)
    return rho, r, r2, mse, preds


# ── Run evaluation ────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Cross-dataset generalization evaluation  (Fusion v3, no retraining)")
print("=" * 65)

results = []

for dataset_name, csv_path in EVAL_DATASETS:
    print(f"\n── {dataset_name} ──────────────────────────────────────────")
    df_eval = pd.read_csv(csv_path)
    print(f"  {len(df_eval)} variants")

    X_emb, S_norm, E1v_norm, y = features_for_dataset(df_eval, dataset_name)
    rho, r, r2, mse, preds = evaluate(X_emb, S_norm, E1v_norm, y)

    print(f"  Spearman ρ : {rho:.4f}")
    print(f"  Pearson  r : {r:.4f}")
    print(f"  R²         : {r2:.4f}")
    print(f"  MSE        : {mse:.4f}")

    results.append({
        "dataset":    dataset_name,
        "n_variants": len(df_eval),
        "spearman_rho": rho,
        "pearson_r":    r,
        "r2":           r2,
        "mse":          mse,
    })

    # Save per-dataset predictions
    pred_path = RESULTS_DIR / f"generalization_{dataset_name}_predictions.csv"
    pd.DataFrame({
        "mutant":           df_eval["mutant"].values,
        "true_fitness":     y,
        "predicted_fitness": preds,
        "residual":         y - preds,
    }).to_csv(pred_path, index=False)
    print(f"  Saved predictions → {pred_path.name}")


# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Summary")
print("=" * 65)
print(f"  {'Dataset':<20}  {'N':>5}  {'Spearman ρ':>10}  {'Pearson r':>9}  {'R²':>6}")
print(f"  {'─' * 60}")
print(f"  {'Stiffler_2015 (test)':<20}  {'749':>5}  {'0.8774':>10}  {'—':>9}  {'—':>6}  ← trained on this")
for r in results:
    print(f"  {r['dataset']:<20}  {r['n_variants']:>5}  {r['spearman_rho']:>10.4f}  "
          f"{r['pearson_r']:>9.4f}  {r['r2']:>6.4f}")
print(f"  {'─' * 60}")

# Save results CSV
results_df = pd.DataFrame(results)
out_csv = RESULTS_DIR / "generalization_results.csv"
results_df.to_csv(out_csv, index=False)
print(f"\nSaved summary → {out_csv}")
