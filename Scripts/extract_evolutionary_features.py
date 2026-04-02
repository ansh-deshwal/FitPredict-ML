'''
Authors: Anshita Sharma, Ansh Jain
Date: 2026

Stage 6 — Evolutionary feature extraction from Multiple Sequence Alignment (MSA).
Standalone script: extracts features, saves .npy and .csv. No fusion yet.

Libraries required:
    pandas, numpy, biopython

What this script does:
    1. Loads the MSA for β-lactamase (BLAT_ECOLX) from Data/BLAT_ECOLX_MSA.a2m
    2. Parses the alignment (A2M / FASTA format)
    3. Computes per-position evolutionary features:
         - Shannon entropy       (conservation: low = conserved, high = variable)
         - Gap fraction          (how often this position is deleted in homologs)
         - Amino acid frequencies (20-d profile — which AAs have been seen here)
    4. For each of the 4,996 mutations, extracts features at the mutated position
    5. Saves as .npy and .csv alongside existing feature files

Output feature vector per mutation (22 features total):
    [entropy, gap_frac, freq_A, freq_C, freq_D, ..., freq_Y]
    Shape: (4996, 22)

HOW TO GET THE MSA FILE:
----------------------------------------------------------------------------------
The MSA is distributed as part of the ProteinGym v1.3 bundle (5.2 GB zip).
Run these commands in your terminal from the project root:

    curl -o DMS_msa_files.zip \
      "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_msa_files.zip"
    unzip -j DMS_msa_files.zip "BLAT_ECOLX_1_b0.5.a2m" -d Data/
    mv Data/BLAT_ECOLX_1_b0.5.a2m Data/BLAT_ECOLX_MSA.a2m
    rm DMS_msa_files.zip

If you already have the zip or individual file from another ProteinGym source,
just place it at Data/BLAT_ECOLX_MSA.a2m and run this script normally.
----------------------------------------------------------------------------------
'''

import re
import pandas as pd
import numpy as np
from Bio import SeqIO
from pathlib import Path
import warnings

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent
CSV_PATH    = BASE_DIR / "Data"    / "BLAT_ECOLX_Stiffler_2015.csv"
MSA_PATH    = BASE_DIR / "Data"    / "BLAT_ECOLX_MSA.a2m"
OUTPUT_NPY  = BASE_DIR / "Results" / "beta_lactamase_evolutionary_features.npy"
OUTPUT_CSV  = BASE_DIR / "Results" / "beta_lactamase_evolutionary_features.csv"

# The 20 standard amino acids (alphabetical — consistent indexing)
AA_LIST   = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

# Integer encoding: AA → 0-19, gap ('-') → 20, unknown → 21
_LOOKUP = np.full(256, 21, dtype=np.int8)
_LOOKUP[ord("-")] = 20
for _i, _aa in enumerate(AA_LIST):
    _LOOKUP[ord(_aa)] = _i


# ── Step 1: Check MSA file ─────────────────────────────────────────────────────
def check_msa(msa_path: Path) -> None:
    if msa_path.exists():
        size_kb = msa_path.stat().st_size / 1024
        print(f"Found MSA file: {msa_path.name}  ({size_kb:.1f} KB)")
        if size_kb < 10:
            raise RuntimeError(
                f"MSA file is too small ({size_kb:.1f} KB) — likely incomplete.\n"
                "Re-download using the curl command at the top of this script."
            )
    else:
        raise FileNotFoundError(
            f"\nMSA file not found at: {msa_path}\n\n"
            "Download it with:\n\n"
            "  curl -o DMS_msa_files.zip \\\n"
            '    "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_msa_files.zip"\n'
            '  unzip -j DMS_msa_files.zip "BLAT_ECOLX_1_b0.5.a2m" -d Data/\n'
            "  mv Data/BLAT_ECOLX_1_b0.5.a2m Data/BLAT_ECOLX_MSA.a2m\n"
            "  rm DMS_msa_files.zip\n\n"
            "Note: the zip is 5.2 GB — only the one file is extracted."
        )


# ── Step 2: Parse A2M / FASTA alignment ───────────────────────────────────────
def parse_a2m(msa_path: Path) -> tuple[list[str], list[str]]:
    """
    Parse an A2M alignment file.

    A2M format rules:
      - Uppercase letters = match columns (aligned to reference)
      - Lowercase letters = insertions relative to reference → discard
      - '-' or '.'       = gap in a match column → keep as '-'

    Returns
    -------
    sequences : list[str]  match-state sequences, all identical length
    seq_ids   : list[str]  sequence identifiers
    """
    sequences, seq_ids = [], []

    ref_len = None

    for record in SeqIO.parse(str(msa_path), "fasta"):
        seq = "".join(
            c   if c.isupper() else
            "-" if c in ("-", ".") else
            ""   # lowercase insertion — discard
            for c in str(record.seq)
        )
        if len(seq) == 0:
            continue

        if ref_len is None:
            # First sequence defines the alignment length (reference match states)
            ref_len = len(seq)

        # Truncate sequences that extend beyond the reference's match states.
        # In A2M, char k = HMM match state k; the reference only has ref_len
        # match states so extra columns are irrelevant to BLAT_ECOLX positions.
        # Pad with '-' the rare cases that are shorter.
        if len(seq) >= ref_len:
            seq = seq[:ref_len]
        else:
            seq = seq + "-" * (ref_len - len(seq))

        sequences.append(seq)
        seq_ids.append(record.id)

    if len(sequences) == 0:
        raise ValueError("No sequences parsed from MSA — check file format.")

    aln_len = ref_len
    print(f"  Parsed {len(sequences):,} sequences")
    print(f"  Alignment length: {aln_len} columns (reference match states)")
    return sequences, seq_ids


# ── Step 3: Map reference positions → alignment columns ───────────────────────
def build_ref_to_col_map(sequences: list[str]) -> dict[int, int]:
    """
    The first sequence in the MSA is the reference (wild-type β-lactamase).
    Returns {residue_position (1-indexed) → alignment_column (0-indexed)}.
    Gap positions in the reference have no residue number and are skipped.
    """
    ref_seq    = sequences[0]
    ref_pos    = 0
    pos_to_col = {}

    for col_idx, char in enumerate(ref_seq):
        if char != "-":
            ref_pos += 1
            pos_to_col[ref_pos] = col_idx

    print(f"  Reference: {ref_pos} residues mapped to alignment columns")
    return pos_to_col


# ── Step 4: Vectorized per-column evolutionary statistics ─────────────────────
def compute_column_stats(sequences: list[str], alignment_len: int) -> dict:
    """
    Build a numpy integer matrix then compute all statistics in vectorized form.

    Encoding: AA → 0-19  |  gap ('-') → 20  |  unknown → 21

    Returns
    -------
    col_stats : dict  col_index → {gap_frac, entropy, aa_freq}
      Identical structure to the original, so the rest of main is unchanged.
    """
    n_seqs = len(sequences)

    # ── Build integer matrix (n_seqs × aln_len) ──────────────────────────────
    print(f"  Building alignment matrix ({n_seqs:,} × {alignment_len}) …")
    mat = np.empty((n_seqs, alignment_len), dtype=np.int8)
    for i, seq in enumerate(sequences):
        arr = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        mat[i] = _LOOKUP[arr]

    # ── Gap fraction per column ───────────────────────────────────────────────
    gap_fracs = (mat == 20).mean(axis=0)                     # (aln_len,)

    # ── AA counts per column: shape (aln_len, 20) ─────────────────────────────
    aa_counts = np.stack(
        [(mat == aa_idx).sum(axis=0) for aa_idx in range(20)],
        axis=1,
    ).astype(np.float64)                                     # (aln_len, 20)

    total_aa = aa_counts.sum(axis=1, keepdims=True)          # (aln_len, 1)

    # ── Frequencies with pseudocount ──────────────────────────────────────────
    aa_freqs = (aa_counts + 1e-6) / (total_aa + 20 * 1e-6)  # (aln_len, 20)
    # Fully gapped columns → zero frequency (no signal)
    aa_freqs = np.where(total_aa == 0, 0.0, aa_freqs)

    # ── Shannon entropy ───────────────────────────────────────────────────────
    safe = np.where(aa_freqs > 0, aa_freqs, 1.0)            # avoid log2(0)
    entropies = -(aa_freqs * np.log2(safe)).sum(axis=1)      # (aln_len,)
    entropies = np.where(total_aa[:, 0] == 0, 0.0, entropies)

    # ── Pack into dict (same interface as before) ─────────────────────────────
    col_stats = {}
    for col in range(alignment_len):
        col_stats[col] = {
            "gap_frac": float(gap_fracs[col]),
            "entropy" : float(entropies[col]),
            "aa_freq" : {aa: float(aa_freqs[col, i]) for i, aa in enumerate(AA_LIST)},
        }

    return col_stats


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Working directory : {BASE_DIR}\n")

    # 1. Check MSA file
    check_msa(MSA_PATH)

    # 2. Parse alignment
    print("\nParsing MSA...")
    sequences, _seq_ids = parse_a2m(MSA_PATH)
    alignment_len       = len(sequences[0])

    # 3. Reference position → column map
    print("\nBuilding reference position map...")
    pos_to_col = build_ref_to_col_map(sequences)

    # 4. Column statistics (vectorized)
    print("\nComputing per-column evolutionary statistics...")
    col_stats = compute_column_stats(sequences, alignment_len)
    print(f"  Done. {len(col_stats)} columns processed")

    # 5. Load mutations
    print(f"\nReading mutations from {CSV_PATH.name}...")
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print(f"  Mutations to process: {len(df)}")

    # Parse mutation positions with the same regex as extract_esm1v_scores.py
    def parse_pos(mutant_str: str) -> int:
        m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", mutant_str.strip())
        return int(m.group(2)) if m else 0

    positions = df["mutant"].apply(parse_pos)

    # 6. Assemble feature matrix
    print("\nExtracting evolutionary features...")
    rows              = []
    missing_positions = 0

    for pos in positions:
        if pos <= 0 or pos not in pos_to_col:
            rows.append([0.0] * 22)
            missing_positions += 1
            continue

        col   = pos_to_col[pos]
        stats = col_stats[col]

        entropy  = stats["entropy"]
        gap_frac = stats["gap_frac"]
        aa_freqs = [stats["aa_freq"][aa] for aa in AA_LIST]   # 20 values

        rows.append([entropy, gap_frac] + aa_freqs)

    if missing_positions > 0:
        warnings.warn(
            f"{missing_positions} positions could not be mapped to alignment "
            "(set to zeros). This is normal for a small number of edge positions."
        )

    features = np.array(rows, dtype=np.float32)

    # 7. Save outputs
    OUTPUT_NPY.parent.mkdir(exist_ok=True)
    np.save(OUTPUT_NPY, features)

    feature_names = ["entropy", "gap_frac"] + [f"freq_{aa}" for aa in AA_LIST]
    feature_df    = pd.DataFrame(features, columns=feature_names)
    feature_df.insert(0, "mutant", df["mutant"].values)
    feature_df.to_csv(OUTPUT_CSV, index=False)

    # 8. Summary
    print("\n" + "-" * 55)
    print("SUCCESS!")
    print(f"  MSA sequences    : {len(sequences):,}")
    print(f"  Alignment length : {alignment_len}")
    print(f"  Feature shape    : {features.shape}   ← should be (4996, 22)")
    print(f"  Missing positions: {missing_positions}")
    print(f"\n  Saved .npy → {OUTPUT_NPY.name}")
    print(f"  Saved .csv → {OUTPUT_CSV.name}")
    print("-" * 55)

    # 9. Sanity check
    print("\nSample output (first 5 mutations):")
    cols_to_show = ["mutant", "entropy", "gap_frac", "freq_A", "freq_L", "freq_V"]
    print(feature_df[cols_to_show].head(5).to_string(index=False))

    # 10. Quick stats
    high_conservation = (features[:, 0] < 1.0).sum()
    print(f"\nHighly conserved positions (entropy < 1.0 bit): "
          f"{high_conservation} / {len(df)} mutations")
    print("  → Low entropy = few AAs tolerated = mutations here likely harmful")


if __name__ == "__main__":
    main()
