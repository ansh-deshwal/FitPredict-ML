import torch
import esm
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 1. Load your clean data with proper path handling
script_dir = Path(__file__).parent.parent / "Data"
df = pd.read_csv(script_dir / "BLAT_ECOLX_Stiffler_2015.csv")
sequences = df["mutated_sequence"].tolist()
labels = df["mutant"].tolist()

print(f"Loaded {len(sequences)} sequences from dataset")

# 2. Load the ESM-2 Model (650M parameters)
print("Loading ESM-2 model...")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # Disable dropout for deterministic results

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model loaded on {device}")

# 3. Process in batches to avoid running out of memory
# On an RTX 2050 (4GB VRAM), keep batch_size small (e.g., 2 or 4)
batch_size = 4
embeddings = []

print(f"Extracting embeddings for {len(sequences)} sequences...")

with torch.no_grad():
    for i in tqdm(range(0, len(sequences), batch_size), desc="Processing batches"):
        # Prepare batch
        batch_seqs = sequences[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]
        
        # Format for ESM: List of (id, sequence) tuples
        data = list(zip(batch_labels, batch_seqs))
        
        # Tokenize
        batch_labels_conv, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)
        
        # Extract features
        # repr_layers=[33] extracts features from the last layer
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_embeddings = results["representations"][33]
        
        # Mean Pooling: Average over the sequence length to get one vector per protein
        # We skip the first token (CLS) and last token (EOS)
        for j, seq_len in enumerate([len(s) for s in batch_seqs]):
            seq_emb = token_embeddings[j, 1 : seq_len + 1].mean(0).cpu().numpy()
            embeddings.append(seq_emb)

# 4. Save results
embeddings_array = np.vstack(embeddings)
print(f"Final embedding shape: {embeddings_array.shape}")  # Should be (N, 1280)

# Save as a numpy file (fast to load later)
output_path = script_dir / "beta_lactamase_esm2_embeddings.npy"
np.save(output_path, embeddings_array)
print(f"Saved embeddings to '{output_path}'")

# Optional: Save a summary
print(f"\nSummary:")
print(f"  - Number of sequences: {len(sequences)}")
print(f"  - Embedding dimension: {embeddings_array.shape[1]}")
print(f"  - Total size: {embeddings_array.nbytes / (1024**2):.2f} MB")
