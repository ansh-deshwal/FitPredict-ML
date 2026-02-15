'''
Author: Arjun Sharma
Date: 15/02/2026
Code Executing in this file does structure extraction from the embeddings.
Libraries required to make the code work:
Pandas, NumPy, Biopython, pathlib, urllib
What the code does and How it works:
1) What it does:
    This code extracts structural features from a protein 3D structure to characterize mutation positions as buried (inside) or 
    surface-exposed.
2) How it works:
1. Download - Fetches the beta-lactamase protein structure (PDB: 1M40) from RCSB database
2. Parse - Extracts 3D coordinates of all alpha-carbon atoms from the protein structure
3. Load mutations - Reads CSV file containing mutation data (e.g., "M182T")
4. Calculate burial - For each mutation position, counts neighboring atoms within 10Å radius
5. Generate features - Creates two features: burial score (neighbor count) and binary flag (buried if >20 neighbors)
6. Save - Exports feature matrix as numpy array for downstream analysis

The output correlates structural context with mutations to help predict their effects on protein function.
'''
import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
from pathlib import Path
import urllib.request

BASE_DIR = Path(r"C:\Users\hp694\OneDrive\Documents\Study\Engineering\sem6\DL\project\Model\Deep_Unfold")
CSV_PATH = BASE_DIR / "BLAT_ECOLX_Stiffler_2015.csv"
PDB_ID = "1M40"
PDB_PATH = BASE_DIR / f"{PDB_ID}.pdb"
OUTPUT_PATH = BASE_DIR / "beta_lactamase_structure_features.npy"

def download_pdb(pdb_code, pdb_filename):
    if not pdb_filename.exists():
        print(f"Downloading {pdb_code} from RCSB...")
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_code}.pdb", pdb_filename)
        print("Download complete.")
    else:
        print(f"Found existing PDB file: {pdb_filename}")

def get_residue_depth(chain, res_id, ca_coords, ca_coords_array, threshold=10.0):
    if res_id not in ca_coords:
        return 0
    
    target_coord = ca_coords[res_id]
    distances = np.linalg.norm(ca_coords_array - target_coord, axis=1)
    return np.sum(distances < threshold) - 1

def main():
    print(f"Working directory: {BASE_DIR}")
    
    download_pdb(PDB_ID, PDB_PATH)
    
    print("Parsing 3D structure...")
    structure = PDBParser(QUIET=True).get_structure(PDB_ID, PDB_PATH)
    chain = structure[0]['A']
    
    ca_coords = {res.id[1]: res['CA'].coord for res in chain if 'CA' in res}
    ca_coords_array = np.array(list(ca_coords.values()))
    
    print(f"Reading data from {CSV_PATH.name}...")
    if not CSV_PATH.exists():
        print("ERROR: CSV file not found!")
        return
    
    df = pd.read_csv(CSV_PATH)
    print(f"Extracting structural features for {len(df)} mutations...")
    
    positions = df['mutant'].str[1:-1].apply(lambda x: int(x) if x.isdigit() else 0)
    burial_scores = positions.apply(lambda pos: get_residue_depth(chain, pos, ca_coords, ca_coords_array) if pos > 0 else 0)
    
    features = np.column_stack([burial_scores, (burial_scores > 20).astype(int)])
    
    np.save(OUTPUT_PATH, features)
    
    print("-" * 30)
    print("SUCCESS!")
    print(f"Saved structural features to: {OUTPUT_PATH.name}")
    print(f"Feature matrix shape: {features.shape}")
    print("-" * 30)

if __name__ == "__main__":
    main()