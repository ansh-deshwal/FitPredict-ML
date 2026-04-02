'''
Author: Anshita Sharma
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

HOW TO GET THE MSA FILE (run ONE of these in terminal before running this script):
----------------------------------------------------------------------------------
Option 1 (Harvard server):
    curl -L -o Data/BLAT_ECOLX_MSA.a2m \
      "https://marks.hms.harvard.edu/proteingym/MSA_files/BLAT_ECOLX_1_b0.5.a2m"

Option 2 (GitHub raw):
    curl -L -o Data/BLAT_ECOLX_MSA.a2m \
      "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/MSA_files/BLAT_ECOLX_1_b0.5.a2m"

Option 3 (manual):
    Go to https://github.com/OATML-Markslab/ProteinGym/tree/main/MSA_files
    Find BLAT_ECOLX_1_b0.5.a2m → click Raw → Cmd+S → save as Data/BLAT_ECOLX_MSA.a2m
----------------------------------------------------------------------------------
'''

import pandas as pd
import numpy as np
from Bio import SeqIO
from pathlib import Path
from collections import Counter
import warnings

# ---------------------------------------------------------------------------
# Paths — mirrors the layout used in extract_structure_features.py
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).parent.parent
CSV_PATH    = BASE_DIR / "Data"    / "BLAT_ECOLX_Stiffler_2015.csv"
MSA_PATH    = BASE_DIR / "Data"    / "BLAT_ECOLX_MSA.a2m"
OUTPUT_NPY  = BASE_DIR / "Results" / "beta_lactamase_evolutionary_features.npy"
OUTPUT_CSV  = BASE_DIR / "Results" / "beta_lactamase_evolutionary_features.csv"

# The 20 standard amino acids (alphabetical — consistent indexing)
AA_LIST   = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


# ---------------------------------------------------------------------------
# Step 1: Check MSA file exists
# ---------------------------------------------------------------------------
def check_msa(msa_path: Path) -> None:
    if msa_path.exists():
        size_kb = msa_path.stat().st_size / 1024
        print(f"Found MSA file: {msa_path.name}  ({size_kb:.1f} KB)")
        if size_kb < 10:
            raise RuntimeError(
                f"MSA file is too small ({size_kb:.1f} KB) — likely incomplete.\n"
                "Re-download using one of the curl commands at the top of this script."
            )
    else:
        raise FileNotFoundError(
            f"\nMSA file not found at: {msa_path}\n\n"
            "Download it first using one of these commands in your terminal:\n\n"
            "  curl -L -o Data/BLAT_ECOLX_MSA.a2m \\\n"
            '    "https://marks.hms.harvard.edu/proteingym/MSA_files/BLAT_ECOLX_1_b0.5.a2m"\n\n'
            "  OR\n\n"
            "  curl -L -o Data/BLAT_ECOLX_MSA.a2m \\\n"
            '    "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/MSA_files/BLAT_ECOLX_1_b0.5.a2m"\n'
        )


# ---------------------------------------------------------------------------
# Step 2: Parse A2M / FASTA alignment
# ---------------------------------------------------------------------------
def parse_a2m(msa_path: Path) -> tuple:
    """
    Parse an A2M alignment file.

    A2M format rules:
      - Uppercase letters = match columns (aligned to reference)
      - Lowercase letters = insertions (not in reference) → discard
      - '-' or '.'       = gaps in match columns

    Returns
    -------
    sequences : list[str]  uppercase match-state sequences
    seq_ids   : list[str]  sequence identifiers
    """
    sequences, seq_ids = [], []

    for record in SeqIO.parse(str(msa_path), "fasta"):
        # Keep only match-state characters: uppercase letters and '-'
        seq = "".join(
            c for c in str(record.seq)
            if c.isupper() or c == "-"
        )
        if len(seq) == 0:
            continue
        sequences.append(seq.upper())
        seq_ids.append(record.id)

    if len(sequences) == 0:
        raise ValueError("No sequences parsed from MSA — check file format.")

    aln_len = len(sequences[0])
    print(f"  Parsed {len(sequences):,} sequences")
    print(f"  Alignment length: {aln_len} columns")
    return sequences, seq_ids


# ---------------------------------------------------------------------------
# Step 3: Map reference sequence positions → alignment columns
# ---------------------------------------------------------------------------
def build_ref_to_col_map(sequences: list) -> dict:
    """
    The first sequence in the MSA is the reference (wild-type β-lactamase).
    Build {residue_position (1-indexed) → alignment_column (0-indexed)}.
    Gaps in the reference ('-') have no residue position and are skipped.
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


# ---------------------------------------------------------------------------
# Step 4: Compute per-column evolutionary statistics
# ---------------------------------------------------------------------------
def compute_column_stats(sequences: list, alignment_len: int) -> dict:
    """
    For each alignment column compute:
      - gap_fraction    : fraction of sequences with '-' at this column
      - shannon_entropy : H = -sum(p * log2(p)) over non-gap AA frequencies
                          Low  = conserved position (few distinct AAs seen)
                          High = variable position  (many distinct AAs seen)
      - aa_frequencies  : {AA: frequency} using only non-gap characters
                          + pseudocount of 1e-6 to avoid log(0)
    """
    n_seqs    = len(sequences)
    col_stats = {}

    for col in range(alignment_len):
        chars = [seq[col] for seq in sequences if col < len(seq)]

        # Gap fraction
        gap_count = sum(1 for c in chars if c == "-")
        gap_frac  = gap_count / n_seqs

        # Non-gap amino acids only
        aa_chars = [c for c in chars if c in AA_TO_IDX]

        if len(aa_chars) == 0:
            # Fully gapped column — no signal
            col_stats[col] = {
                "gap_frac": 1.0,
                "entropy" : 0.0,
                "aa_freq" : {aa: 0.0 for aa in AA_LIST},
            }
            continue

        # AA frequencies with pseudocount
        counts  = Counter(aa_chars)
        total   = len(aa_chars)
        aa_freq = {
            aa: (counts.get(aa, 0) + 1e-6) / (total + 20 * 1e-6)
            for aa in AA_LIST
        }

        # Shannon entropy
        entropy = -sum(p * np.log2(p) for p in aa_freq.values() if p > 0)

        col_stats[col] = {
            "gap_frac": gap_frac,
            "entropy" : entropy,
            "aa_freq" : aa_freq,
        }

    return col_stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Working directory : {BASE_DIR}\n")

    # 1. Check MSA file exists
    check_msa(MSA_PATH)

    # 2. Parse alignment
    print("\nParsing MSA...")
    sequences, seq_ids = parse_a2m(MSA_PATH)
    alignment_len      = len(sequences[0])

    # 3. Reference position → column map
    print("\nBuilding reference position map...")
    pos_to_col = build_ref_to_col_map(sequences)

    # 4. Column statistics
    print("\nComputing per-column evolutionary statistics...")
    col_stats = compute_column_stats(sequences, alignment_len)
    print(f"  Done. {len(col_stats)} columns processed")

    # 5. Load mutations
    print(f"\nReading mutations from {CSV_PATH.name}...")
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df        = pd.read_csv(CSV_PATH)
    positions = df["mutant"].str[1:-1].apply(
        lambda x: int(x) if str(x).isdigit() else 0
    )
    print(f"  Mutations to process: {len(df)}")

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

        # Final vector: [entropy, gap_frac, freq_A, ..., freq_Y] → 22 values
        rows.append([entropy, gap_frac] + aa_freqs)

    if missing_positions > 0:
        warnings.warn(
            f"{missing_positions} positions could not be mapped to alignment "
            "(set to zeros). This is normal for a small number of edge positions."
        )

    features = np.array(rows, dtype=float)

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

    # 9. Sanity check — print a few rows
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
