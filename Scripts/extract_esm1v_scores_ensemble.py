"""
extract_esm1v_scores_ensemble.py
================================
Compute ESM-1v zero-shot fitness scores using all 5 ensemble members and
average them.  Averaging reduces variance and typically improves correlation
with experimental fitness by ~0.005–0.01 Spearman ρ over a single member.

Method (Meier et al., 2021 — ESM-1v paper):
  score(X→Y at pos p) = log P(Y | seq with pos p masked)
                       - log P(X | seq with pos p masked)

  Ensemble score = mean over members 1–5 of the above.

Each model is loaded, scored, and immediately released to keep peak GPU
memory at the single-model level (~2.5 GB fp32).

Output
------
Results/beta_lactamase_esm1v_ensemble_scores.npy   shape (4996,)   float32

The original Results/beta_lactamase_esm1v_scores.npy (member 1 only) is
not modified.
"""

import re
import torch
import esm
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR / "BLAT_ECOLX_Stiffler_2015.csv")
print(f"Loaded {len(df)} variants")

# ── Parse mutations ────────────────────────────────────────────────────────────
def parse_mutation(mutant_str):
    m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mutant_str.strip())
    if m is None:
        raise ValueError(f"Cannot parse mutation: {mutant_str!r}")
    wt_aa, pos, mut_aa = m.group(1), int(m.group(2)), m.group(3)
    return wt_aa, pos, mut_aa

mutations = [parse_mutation(s) for s in df["mutant"]]

# ── Reconstruct wild-type sequence ────────────────────────────────────────────
wt_aa_0, pos_0, _ = mutations[0]
mutated_seq_0 = df["mutated_sequence"].iloc[0]
wt_seq = list(mutated_seq_0)
wt_seq[pos_0 - 1] = wt_aa_0
wt_seq = "".join(wt_seq)

errs = 0
for (wt_aa, pos, mut_aa), seq in zip(mutations, df["mutated_sequence"]):
    if seq[pos - 1] != mut_aa:
        errs += 1
    if wt_seq[pos - 1] != wt_aa and wt_seq != seq:
        errs += 1
if errs == 0:
    print(f"WT sequence reconstructed OK  (len={len(wt_seq)})")
else:
    print(f"WARNING: {errs} sanity-check failures — proceed with caution")

# ── Group variants by mutation position ───────────────────────────────────────
pos_to_variants = defaultdict(list)
for i, (wt_aa, pos, mut_aa) in enumerate(mutations):
    pos_to_variants[pos].append((i, wt_aa, mut_aa))
print(f"Unique positions to score: {len(pos_to_variants)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── ESM-1v model loader functions (one per ensemble member) ───────────────────
ESM1V_LOADERS = [
    esm.pretrained.esm1v_t33_650M_UR90S_1,
    esm.pretrained.esm1v_t33_650M_UR90S_2,
    esm.pretrained.esm1v_t33_650M_UR90S_3,
    esm.pretrained.esm1v_t33_650M_UR90S_4,
    esm.pretrained.esm1v_t33_650M_UR90S_5,
]

all_scores = np.zeros((len(ESM1V_LOADERS), len(df)), dtype=np.float32)

# ── Score with each member ─────────────────────────────────────────────────────
for member_idx, loader_fn in enumerate(ESM1V_LOADERS, start=1):
    print(f"\n{'=' * 60}")
    print(f"ESM-1v member {member_idx} / {len(ESM1V_LOADERS)}")
    print(f"{'=' * 60}")

    model, alphabet = loader_fn()
    batch_converter  = alphabet.get_batch_converter()
    model.eval()
    model = model.to(device)

    mask_idx = alphabet.mask_idx

    # Tokenise WT once for this alphabet instance
    _, _, wt_tokens = batch_converter([("wt", wt_seq)])
    wt_tokens = wt_tokens[0]   # (L+2,)

    member_scores = np.zeros(len(df), dtype=np.float32)

    with torch.no_grad():
        for pos, variants_at_pos in tqdm(pos_to_variants.items(),
                                         desc=f"  Member {member_idx} — scoring positions"):
            tok_idx = pos   # BOS at 0; protein position p (1-indexed) → token index p

            masked = wt_tokens.clone()
            masked[tok_idx] = mask_idx
            masked = masked.unsqueeze(0).to(device)   # (1, L+2)

            out           = model(masked)["logits"]   # (1, L+2, vocab)
            logits_at_pos = out[0, tok_idx]           # (vocab,)
            log_probs     = torch.log_softmax(logits_at_pos, dim=-1)

            for variant_idx, wt_aa, mut_aa in variants_at_pos:
                wt_tok  = alphabet.get_idx(wt_aa)
                mut_tok = alphabet.get_idx(mut_aa)
                member_scores[variant_idx] = (log_probs[mut_tok] - log_probs[wt_tok]).item()

    all_scores[member_idx - 1] = member_scores
    print(f"  Member {member_idx} done — range [{member_scores.min():.3f}, {member_scores.max():.3f}]")

    # Release model and free GPU memory before loading the next member
    del model, alphabet, batch_converter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── Average across members ────────────────────────────────────────────────────
ensemble_scores = all_scores.mean(axis=0).astype(np.float32)

print(f"\n{'=' * 60}")
print(f"Ensemble average (members 1–5)")
print(f"{'=' * 60}")
print(f"Shape : {ensemble_scores.shape}   dtype : {ensemble_scores.dtype}")
print(f"Range : [{ensemble_scores.min():.3f}, {ensemble_scores.max():.3f}]   "
      f"mean : {ensemble_scores.mean():.3f}")

# Member-wise correlation with ensemble to sanity-check consistency
from scipy.stats import spearmanr
for i in range(len(ESM1V_LOADERS)):
    rho, _ = spearmanr(all_scores[i], ensemble_scores)
    print(f"  Member {i+1} vs ensemble  ρ = {rho:.4f}")

# ── Save ───────────────────────────────────────────────────────────────────────
out_path = RESULTS_DIR / "beta_lactamase_esm1v_ensemble_scores.npy"
np.save(out_path, ensemble_scores)
print(f"\nSaved ensemble scores → {out_path}")
