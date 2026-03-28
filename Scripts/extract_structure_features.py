'''
Author: Arjun Sharma
Date: 15/02/2026 (Updated: 11/03/2026)
Libraries: Pandas, NumPy, Biopython, pathlib, urllib

Extracts 11 structural features per mutation from PDB 1M40 (beta-lactamase TEM-1):
  1. Download - Fetches the structure (PDB: 1M40) from RCSB
  2. Parse - Extracts Cα coordinates from chain A
  3. Load mutations - Reads mutant identifiers from CSV (e.g., "M182T")
  4. Burial - Counts Cα neighbours within 10 Å; binary buried flag (>20 neighbours)
  5. DSSP - Secondary structure one-hot (H/E/C), rASA, sin/cos of φ/ψ dihedral angles
  6. Contact map - Binary Cα–Cα contacts within 8 Å; per-residue contact count
  7. Output - (N × 11) matrix: [burial_score, is_buried, ss_H, ss_E, ss_C,
                                rASA, sin_phi, cos_phi, sin_psi, cos_psi, contact_count]

DSSP must be on PATH (sudo apt-get install dssp  or  conda install -c salilab dssp).
Falls back to burial-only features (DSSP columns zeroed) if mkdssp is unavailable.
'''
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser, DSSP
from pathlib import Path
import urllib.request
import warnings
import shutil

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
CSV_PATH = BASE_DIR / "Data/BLAT_ECOLX_Stiffler_2015.csv"
PDB_ID = "1M40"
PDB_PATH = BASE_DIR / f"Data/{PDB_ID}.pdb"
OUTPUT_PATH = BASE_DIR / "Results/beta_lactamase_structure_features.npy"
OUTPUT_CSV_PATH = BASE_DIR / "Results/beta_lactamase_structure_features.csv"

# Constants
BURIAL_THRESHOLD_ANGSTROM = 10.0   # radius for neighbour count
BURIAL_NEIGHBOUR_CUTOFF   = 20     # neighbours above this → buried
CONTACT_THRESHOLD_ANGSTROM = 8.0   # Cα–Cα distance for contact map

SS_CLASSES = ['H', 'E', 'C']       # helix / sheet / coil (other mapped → C)

# Helper functions
def download_pdb(pdb_code: str, pdb_filename: Path) -> None:
    if not pdb_filename.exists():
        print(f"Downloading {pdb_code} from RCSB...")
        urllib.request.urlretrieve(
            f"https://files.rcsb.org/download/{pdb_code}.pdb",
            pdb_filename
        )
        print("Download complete.")
    else:
        print(f"Found existing PDB file: {pdb_filename}")


def get_burial_score(res_id: int, ca_coords: dict, ca_coords_array: np.ndarray,
                     threshold: float = BURIAL_THRESHOLD_ANGSTROM) -> int:
    """Count Cα neighbours within *threshold* Å (excluding self)."""
    if res_id not in ca_coords:
        return 0
    target = ca_coords[res_id]
    distances = np.linalg.norm(ca_coords_array - target, axis=1)
    return int(np.sum(distances < threshold)) - 1


def build_contact_map(ca_coords: dict,
                      threshold: float = CONTACT_THRESHOLD_ANGSTROM) -> dict:
    """
    Build a binary Cα–Cα contact map.

    Returns
    -------
    contact_counts : dict  {res_id: int}
        Number of residues in contact with each residue.
    contact_matrix : np.ndarray  (N × N bool)
        Full pairwise contact matrix (residues in insertion order).
    residue_ids : list
        Ordered list of residue IDs matching matrix rows/cols.
    """
    residue_ids = list(ca_coords.keys())
    coords_array = np.array([ca_coords[r] for r in residue_ids])  # (N, 3)

    # Vectorised pairwise distances
    diff = coords_array[:, np.newaxis, :] - coords_array[np.newaxis, :, :]  # (N,N,3)
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))                        # (N, N)

    contact_matrix = (dist_matrix < threshold) & (dist_matrix > 0)           # exclude self
    contact_counts  = {rid: int(contact_matrix[i].sum())
                       for i, rid in enumerate(residue_ids)}

    return contact_counts, contact_matrix, residue_ids


def run_dssp(structure, pdb_path: Path):
    """
    Run DSSP via Biopython.  Returns a dict keyed by residue sequence number:
        {res_id: {'ss': str, 'rasa': float, 'phi': float, 'psi': float}}

    Falls back to None if DSSP binary is not found.
    """
    try:
        model  = structure[0]
        dssp_bin = shutil.which('mkdssp') or shutil.which('dssp') or '/usr/bin/mkdssp'
        dssp = DSSP(model, str(pdb_path), dssp=dssp_bin)

        dssp_data = {}
        for key in dssp.property_keys:
            res_data = dssp[key]
            # key = (chain_id, (' ', seq_num, ' '))
            seq_num = key[1][1]

            raw_ss = res_data[2]              # single-letter SS code
            # Biopython uses '-' for irregular; map everything except H/E → C
            ss = raw_ss if raw_ss in ('H', 'E') else 'C'

            rasa = res_data[3]                # relative ASA (0–1)
            phi  = res_data[4]                # degrees, may be 360.0 if undefined
            psi  = res_data[5]

            dssp_data[seq_num] = {
                'ss'  : ss,
                'rasa': float(rasa) if rasa is not None else 0.5,
                'phi' : float(phi)  if phi  is not None else 0.0,
                'psi' : float(psi)  if psi  is not None else 0.0,
            }
        print(f"  DSSP computed for {len(dssp_data)} residues.")
        return dssp_data

    except Exception as exc:
        warnings.warn(
            f"DSSP failed ({exc}). Secondary-structure / ASA features will be set to 0. "
            "Install DSSP (mkdssp) and ensure it is on PATH to enable these features."
        )
        return None


def encode_ss_onehot(ss: str) -> list:
    """Return [is_H, is_E, is_C] one-hot vector."""
    return [int(ss == cls) for cls in SS_CLASSES]


def normalise_angle(angle_deg: float):
    """
    Convert a backbone dihedral angle (degrees) to (sin, cos) pair.
    Undefined DSSP angles are returned as 360°; map these to (0, 0).
    """
    if abs(angle_deg) >= 360.0:
        return 0.0, 0.0
    rad = np.deg2rad(angle_deg)
    return float(np.sin(rad)), float(np.cos(rad))


# Main
def main():
    print(f"Working directory: {BASE_DIR}")

    # 1. Download & parse structure
    download_pdb(PDB_ID, PDB_PATH)

    print("Parsing 3D structure...")
    structure = PDBParser(QUIET=True).get_structure(PDB_ID, PDB_PATH)
    chain     = structure[0]['A']

    ca_coords       = {res.id[1]: res['CA'].coord
                       for res in chain if 'CA' in res}
    ca_coords_array = np.array(list(ca_coords.values()))

    # 2. Contact map (all residues at once — O(N²) but fast for single chains)
    print("Building contact map...")
    contact_counts, contact_matrix, residue_ids = build_contact_map(ca_coords)
    print(f"  Contact map shape: {contact_matrix.shape}")

    # 3. DSSP
    print("Running DSSP...")
    dssp_data = run_dssp(structure, PDB_PATH)
    dssp_available = dssp_data is not None

    # 4. Load mutations
    print(f"Reading data from {CSV_PATH.name}...")
    if not CSV_PATH.exists():
        print("ERROR: CSV file not found!")
        return

    df        = pd.read_csv(CSV_PATH)
    positions = df['mutant'].str[1:-1].apply(lambda x: int(x) if x.isdigit() else 0)
    print(f"Extracting structural features for {len(df)} mutations...")

    # 5. Assemble feature matrix
    rows = []
    for pos in positions:
        if pos <= 0:
            # Unknown / unparseable position → zeros
            rows.append([0] * 11)
            continue

        # --- Burial (original features) ---
        burial_score = get_burial_score(pos, ca_coords, ca_coords_array)
        is_buried    = int(burial_score > BURIAL_NEIGHBOUR_CUTOFF)

        # --- Contact map ---
        contact_count = contact_counts.get(pos, 0)

        # --- DSSP ---
        if dssp_available and pos in dssp_data:
            d          = dssp_data[pos]
            ss_onehot  = encode_ss_onehot(d['ss'])                 # [H, E, C]
            rasa       = d['rasa']
            sin_phi, cos_phi = normalise_angle(d['phi'])
            sin_psi, cos_psi = normalise_angle(d['psi'])
        else:
            ss_onehot        = [0, 0, 0]
            rasa             = 0.0
            sin_phi, cos_phi = 0.0, 0.0
            sin_psi, cos_psi = 0.0, 0.0

        row = [
            burial_score,   # 0  – Cα neighbour count within 10 Å
            is_buried,      # 1  – binary burial flag (>20 neighbours)
            ss_onehot[0],   # 2  – is helix (H)
            ss_onehot[1],   # 3  – is sheet (E)
            ss_onehot[2],   # 4  – is coil/other (C)
            rasa,           # 5  – relative accessible surface area [0,1]
            sin_phi,        # 6  – sin(φ) backbone dihedral
            cos_phi,        # 7  – cos(φ) backbone dihedral
            sin_psi,        # 8  – sin(ψ) backbone dihedral
            cos_psi,        # 9  – cos(ψ) backbone dihedral
            contact_count,  # 10 – number of Cα contacts within 8 Å
        ]
        rows.append(row)

    features = np.array(rows, dtype=float)

    # 6. Save
    np.save(OUTPUT_PATH, features)

    feature_names = [
        'burial_score', 'is_buried',
        'ss_helix', 'ss_sheet', 'ss_coil',
        'rASA',
        'sin_phi', 'cos_phi', 'sin_psi', 'cos_psi',
        'contact_count',
    ]
    feature_df = pd.DataFrame(features, columns=feature_names)
    feature_df.insert(0, 'mutant', df['mutant'].values)
    feature_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print("-" * 50)
    print("SUCCESS!")
    print(f"Saved structural features (.npy) to : {OUTPUT_PATH.name}")
    print(f"Saved structural features (.csv) to : {OUTPUT_CSV_PATH.name}")
    print(f"Feature matrix shape               : {features.shape}")
    print(f"Features ({len(feature_names)})    : {feature_names}")
    print("-" * 50)


if __name__ == "__main__":
    main()