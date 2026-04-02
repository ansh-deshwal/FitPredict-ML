"""
extract_esm1v_scores.py
=======================
Compute ESM-1v zero-shot fitness scores (masked-marginal method) for every
single-point mutant in the Stiffler 2015 dataset.

Method (Meier et al., 2021 — ESM-1v paper):
  score(X→Y at pos p) = log P(Y | seq with pos p masked)
                       - log P(X | seq with pos p masked)

All variants share the same wild-type background, so we group mutations by
position: one forward pass per unique position → efficient (~263 passes for
4996 variants instead of 4996).

Output
------
Results/beta_lactamase_esm1v_scores.npy   shape (4996,)   float32
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
# Format: "H24C"  →  wt_aa='H', pos=24 (1-indexed), mut_aa='C'
def parse_mutation(mutant_str):
    m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mutant_str.strip())
    if m is None:
        raise ValueError(f"Cannot parse mutation: {mutant_str!r}")
    wt_aa, pos, mut_aa = m.group(1), int(m.group(2)), m.group(3)
    return wt_aa, pos, mut_aa

mutations = [parse_mutation(s) for s in df["mutant"]]

# ── Reconstruct wild-type sequence ────────────────────────────────────────────
# Take the first mutated_sequence and revert its single mutation.
wt_aa_0, pos_0, _ = mutations[0]
mutated_seq_0 = df["mutated_sequence"].iloc[0]
wt_seq = list(mutated_seq_0)
assert wt_seq[pos_0 - 1] != wt_aa_0 or True   # may already differ — just revert
wt_seq[pos_0 - 1] = wt_aa_0
wt_seq = "".join(wt_seq)

# Sanity check: every mutant should agree with wt_seq at all non-mutated positions
errs = 0
for i, ((wt_aa, pos, mut_aa), seq) in enumerate(zip(mutations, df["mutated_sequence"])):
    if seq[pos - 1] != mut_aa:
        errs += 1
    if wt_seq[pos - 1] != wt_aa and wt_seq != seq:
        errs += 1
if errs == 0:
    print(f"WT sequence reconstructed OK  (len={len(wt_seq)})")
else:
    print(f"WARNING: {errs} sanity-check failures — proceed with caution")

# ── Load ESM-1v model (ensemble member 1) ─────────────────────────────────────
print("Loading ESM-1v model 1 …")
model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
batch_converter  = alphabet.get_batch_converter()
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device)
print(f"Model loaded on {device}")

mask_idx = alphabet.mask_idx

# ── Group variants by mutation position ───────────────────────────────────────
# pos_to_variants: pos → [(variant_index, wt_aa, mut_aa), ...]
pos_to_variants = defaultdict(list)
for i, (wt_aa, pos, mut_aa) in enumerate(mutations):
    pos_to_variants[pos].append((i, wt_aa, mut_aa))

print(f"Unique positions to score: {len(pos_to_variants)}")

# ── Tokenise WT once so we can mask individual positions ──────────────────────
_, _, wt_tokens = batch_converter([("wt", wt_seq)])
wt_tokens = wt_tokens[0]   # (seq_len + 2,)   [BOS, aa1, aa2, …, EOS]

scores = np.zeros(len(df), dtype=np.float32)

# ── Score: one forward pass per unique position ───────────────────────────────
with torch.no_grad():
    for pos, variants_at_pos in tqdm(pos_to_variants.items(), desc="Scoring positions"):
        # Token index: BOS is at 0, so protein position p (1-indexed) → token index p
        tok_idx = pos   # i.e. pos + 0  (BOS offset = 1, but pos is 1-indexed → index = pos)

        masked = wt_tokens.clone()
        masked[tok_idx] = mask_idx
        masked = masked.unsqueeze(0).to(device)  # (1, L+2)

        out   = model(masked)["logits"]          # (1, L+2, vocab)
        logits_at_pos = out[0, tok_idx]          # (vocab,)
        log_probs     = torch.log_softmax(logits_at_pos, dim=-1)

        for variant_idx, wt_aa, mut_aa in variants_at_pos:
            wt_tok  = alphabet.get_idx(wt_aa)
            mut_tok = alphabet.get_idx(mut_aa)
            scores[variant_idx] = (log_probs[mut_tok] - log_probs[wt_tok]).item()

# ── Save ───────────────────────────────────────────────────────────────────────
out_path = RESULTS_DIR / "beta_lactamase_esm1v_scores.npy"
np.save(out_path, scores)
print(f"\nSaved ESM-1v scores → {out_path}")
print(f"Shape : {scores.shape}   dtype : {scores.dtype}")
print(f"Range : [{scores.min():.3f}, {scores.max():.3f}]   mean : {scores.mean():.3f}")
