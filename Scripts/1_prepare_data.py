"""Step 1: Data preparation"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

SCRIPTS_DIR = Path(__file__).parent
sys.path.append(str(SCRIPTS_DIR))

import config
from utils import extract_mutation_positions, set_seed

def main():
    print("="*70)
    print("STEP 1: DATA PREPARATION")
    print("="*70)
    
    set_seed(config.TRAINING_CONFIG['random_seed'])
    
    print(f"\nLoading: {config.CSV_FILE}")
    df = pd.read_csv(config.CSV_FILE)
    print(f"  {len(df)} mutations")
    
    print(f"\nLoading: {config.ESM2_EMBEDDINGS_FILE}")
    try:
        X = np.load(config.ESM2_EMBEDDINGS_FILE)
        print(f"  Shape: {X.shape}")
    except:
        X = np.load(config.ESM2_EMBEDDINGS_FILE, mmap_mode='r')
        X = np.array(X)
        print(f"  Shape: {X.shape}")
    
    y = df["DMS_score"].values
    positions = extract_mutation_positions(df)
    
    print(f"\nData stats:")
    print(f"  Fitness: min={y.min():.2f}, max={y.max():.2f}")
    print(f"  Positions: min={positions.min()}, max={positions.max()}")
    
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=config.TRAINING_CONFIG['test_size'],
        random_state=config.TRAINING_CONFIG['random_seed'],
        shuffle=True
    )
    
    print(f"\nSplit: {len(train_idx)} train, {len(test_idx)} test")
    
    train_file = config.DATA_DIR / "train_data.npz"
    test_file = config.DATA_DIR / "test_data.npz"
    
    np.savez(train_file,
             indices=train_idx,
             esm2_embeddings=X[train_idx],
             labels=y[train_idx],
             positions=positions[train_idx])
    
    np.savez(test_file,
             indices=test_idx,
             esm2_embeddings=X[test_idx],
             labels=y[test_idx],
             positions=positions[test_idx])
    
    print(f"\n✓ Saved: {train_file}")
    print(f"✓ Saved: {test_file}")
    print("\n" + "="*70)
    print("✓ STEP 1 COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()