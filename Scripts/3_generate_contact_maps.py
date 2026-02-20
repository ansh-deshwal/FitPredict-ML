"""
Step 3: Generate Contact Maps from PDB Structure

This script:
1. Loads protein structure (1M40.pdb)
2. Generates contact maps for all mutations
3. Saves contact maps as numpy arrays
"""

import sys
import numpy as np
import torch
from pathlib import Path
from Bio.PDB import PDBParser
from tqdm import tqdm

# Add Scripts directory to path
SCRIPTS_DIR = Path(__file__).parent
sys.path.append(str(SCRIPTS_DIR))

import config
from utils import extract_mutation_positions

# =====================================================================
# CONTACT MAP GENERATION
# =====================================================================

def generate_contact_map_for_position(pdb_file, ca_coords, mutation_pos, 
                                     threshold=8.0, map_size=256):
    """
    Generate contact map centered around mutation position
    
    Args:
        pdb_file: Path to PDB file
        ca_coords: Dictionary of {residue_id: CA_coordinate}
        mutation_pos: Mutation position
        threshold: Distance threshold in Angstroms
        map_size: Output map size (will be padded to this)
        
    Returns:
        contact_map: (map_size, map_size) binary matrix
    """
    n_residues = len(ca_coords)
    
    # Create distance matrix
    contact_map = np.zeros((n_residues, n_residues), dtype=np.float32)
    
    # Get coordinate array for vectorized computation
    positions = sorted(ca_coords.keys())
    coords = np.array([ca_coords[pos] for pos in positions])
    
    # Compute pairwise distances
    for i in range(n_residues):
        for j in range(i+1, n_residues):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < threshold:
                contact_map[i, j] = 1.0
                contact_map[j, i] = 1.0
    
    # Pad to map_size x map_size
    padded = np.zeros((map_size, map_size), dtype=np.float32)
    padded[:n_residues, :n_residues] = contact_map
    
    return padded


def extract_ca_coordinates(pdb_file):
    """
    Extract alpha-carbon coordinates from PDB file
    
    Returns:
        ca_coords: Dictionary {residue_id: numpy.array([x, y, z])}
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    # Get first model, chain A
    model = structure[0]
    if 'A' not in model:
        # Use first available chain
        chain_id = list(model.child_dict.keys())[0]
        chain = model[chain_id]
        print(f"Warning: Chain A not found, using chain {chain_id}")
    else:
        chain = model['A']
    
    # Extract CA coordinates
    ca_coords = {}
    for residue in chain:
        if 'CA' in residue:
            res_id = residue.id[1]  # Residue number
            ca_coords[res_id] = residue['CA'].coord
    
    return ca_coords


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("="*70)
    print("STEP 3: GENERATE CONTACT MAPS")
    print("="*70)
    
    # Check if PDB file exists
    if not config.PDB_FILE.exists():
        print(f"\n✗ ERROR: PDB file not found: {config.PDB_FILE}")
        print("\nPlease download 1M40.pdb from:")
        print("https://files.rcsb.org/download/1M40.pdb")
        print(f"\nAnd place it in: {config.DATA_DIR}")
        return
    
    # Load mutation data
    import pandas as pd
    df = pd.read_csv(config.CSV_FILE)
    positions = extract_mutation_positions(df)
    
    print(f"\nProcessing {len(positions)} mutations...")
    
    # Extract CA coordinates
    print("\nExtracting CA coordinates from PDB...")
    ca_coords = extract_ca_coordinates(config.PDB_FILE)
    print(f"  Extracted {len(ca_coords)} residues")
    
    # Generate contact maps
    print("\nGenerating contact maps...")
    contact_maps = []
    
    threshold = config.CONTACT_MAP_CONFIG['distance_threshold']
    map_size = config.CONTACT_MAP_CONFIG['map_size']
    
    for i, pos in enumerate(tqdm(positions, desc="Generating maps")):
        contact_map = generate_contact_map_for_position(
            config.PDB_FILE,
            ca_coords,
            pos,
            threshold=threshold,
            map_size=map_size
        )
        contact_maps.append(contact_map)
    
    # Stack into array
    contact_maps = np.array(contact_maps, dtype=np.float32)
    print(f"\nContact maps shape: {contact_maps.shape}")
    
    # Save
    output_file = config.DATA_DIR / "contact_maps_full.npy"
    np.save(output_file, contact_maps)
    
    print(f"\n✓ Saved contact maps to: {output_file}")
    
    # Statistics
    print(f"\nContact Map Statistics:")
    print(f"  Shape: {contact_maps.shape}")
    print(f"  Distance threshold: {threshold} Ã…")
    print(f"  Average contacts per map: {contact_maps.mean(axis=(1,2)).mean():.1f}")
    print(f"  Min contacts: {contact_maps.sum(axis=(1,2)).min():.0f}")
    print(f"  Max contacts: {contact_maps.sum(axis=(1,2)).max():.0f}")
    
    print("\n" + "="*70)
    print("✓ STEP 3 COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()