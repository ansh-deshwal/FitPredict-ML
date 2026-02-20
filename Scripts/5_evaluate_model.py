"""
Step 5: Comprehensive Model Evaluation

This script:
1. Loads trained model
2. Performs ablation studies
3. Generates comprehensive visualizations
4. Creates results tables
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add Scripts directory to path
SCRIPTS_DIR = Path(__file__).parent
sys.path.append(str(SCRIPTS_DIR))

import config
from models import (
    ProteinAutoencoder,
    MultiModalFusionNetwork,
    ContactMapCNN,
    BiLSTMSequentialEncoder
)
from utils import (
    to_tensor,
    load_model,
    compute_metrics,
    print_metrics,
    set_seed
)

# =====================================================================
# ABLATION STUDY
# =====================================================================

def ablation_study(train_data, test_data, train_contact_maps, test_contact_maps, device):
    """
    Perform ablation study to measure component contributions
    """
    print("\n" + "="*70)
    print("ABLATION STUDY")
    print("="*70)
    
    results = {}
    
    # Load pre-trained autoencoder
    autoencoder = ProteinAutoencoder(
        config.AUTOENCODER_CONFIG['input_dim'],
        config.AUTOENCODER_CONFIG['latent_dim'],
        config.AUTOENCODER_CONFIG['dropout']
    )
    autoencoder = load_model(autoencoder, config.MODELS_DIR, "autoencoder")
    autoencoder.to(device)
    autoencoder.eval()
    
    # Test 1: Sequence only (frozen ESM-2 + MLP)
    print("\n1. Testing: Frozen ESM-2 + MLP (baseline)")
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(train_data['esm2_embeddings'], train_data['labels'])
    preds = ridge.predict(test_data['esm2_embeddings'])
    results['frozen_esm2'] = compute_metrics(test_data['labels'], preds)
    print_metrics(results['frozen_esm2'], "Frozen ESM-2 + Ridge")
    
    # Test 2: Sequence + Autoencoder
    print("\n2. Testing: Autoencoder compression")
    with torch.no_grad():
        X_test_tensor = to_tensor(test_data['esm2_embeddings'], device)
        compressed, _ = autoencoder(X_test_tensor)
        compressed_np = compressed.cpu().numpy()
    
    ridge2 = Ridge(alpha=1.0)
    ridge2.fit(
        autoencoder.encode(to_tensor(train_data['esm2_embeddings'], device)).cpu().detach().numpy(),
        train_data['labels']
    )
    preds2 = ridge2.predict(compressed_np)
    results['autoencoder_only'] = compute_metrics(test_data['labels'], preds2)
    print_metrics(results['autoencoder_only'], "Autoencoder + Ridge")
    
    # Test 3: Full fusion model
    print("\n3. Testing: Full Multi-Modal Fusion")
    fusion_model = MultiModalFusionNetwork(autoencoder, config).to(device)
    fusion_model = load_model(fusion_model, config.MODELS_DIR, "fusion_model_best")
    fusion_model.eval()
    
    X_test = to_tensor(test_data['esm2_embeddings'], device)
    pos_test = torch.LongTensor(test_data['positions']).to(device)
    cmap_test = to_tensor(test_contact_maps, device).unsqueeze(1)
    
    with torch.no_grad():
        preds3 = fusion_model(X_test, cmap_test, pos_test).cpu().numpy().flatten()
    
    results['full_fusion'] = compute_metrics(test_data['labels'], preds3)
    print_metrics(results['full_fusion'], "Full Fusion Model")
    
    # Create comparison table
    create_ablation_table(results)
    
    return results


def create_ablation_table(results):
    """Create and save ablation study table"""
    
    df = pd.DataFrame({
        'Model': [
            'Frozen ESM-2 + Ridge',
            'Autoencoder + Ridge',
            'Full Fusion (Ours)'
        ],
        'Spearman ρ': [
            results['frozen_esm2']['spearman_rho'],
            results['autoencoder_only']['spearman_rho'],
            results['full_fusion']['spearman_rho']
        ],
        'Pearson r': [
            results['frozen_esm2']['pearson_r'],
            results['autoencoder_only']['pearson_r'],
            results['full_fusion']['pearson_r']
        ],
        'R²': [
            results['frozen_esm2']['r2'],
            results['autoencoder_only']['r2'],
            results['full_fusion']['r2']
        ],
        'MSE': [
            results['frozen_esm2']['mse'],
            results['autoencoder_only']['mse'],
            results['full_fusion']['mse']
        ]
    })
    
    # Calculate improvements
    baseline_rho = results['frozen_esm2']['spearman_rho']
    df['Δ Spearman (%)'] = [
        0,
        100 * (results['autoencoder_only']['spearman_rho'] - baseline_rho) / baseline_rho,
        100 * (results['full_fusion']['spearman_rho'] - baseline_rho) / baseline_rho
    ]
    
    # Save table
    df.to_csv(config.METRICS_DIR / "ablation_study.csv", index=False)
    
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)
    print(df.to_string(index=False))
    print("\n✓ Saved to:", config.METRICS_DIR / "ablation_study.csv")
    
    # Plot
    plot_ablation_results(df)


def plot_ablation_results(df):
    """Create bar plot of ablation results"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df))
    width = 0.35
    
    rho_bars = ax.bar(x - width/2, df['Spearman ρ'], width, 
                      label='Spearman ρ', color='steelblue')
    pearson_bars = ax.bar(x + width/2, df['Pearson r'], width,
                          label='Pearson r', color='coral')
    
    ax.set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Correlation', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study: Component Contributions', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [rho_bars, pearson_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "ablation_study.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved ablation plot to: {config.FIGURES_DIR / 'ablation_study.png'}")


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("="*70)
    print("STEP 5: COMPREHENSIVE EVALUATION")
    print("="*70)
    
    # Set random seed
    set_seed(config.TRAINING_CONFIG['random_seed'])
    
    # Load data
    print("\nLoading data...")
    train_data = np.load(config.DATA_DIR / "train_data.npy", allow_pickle=True).item()
    test_data = np.load(config.DATA_DIR / "test_data.npy", allow_pickle=True).item()
    
    contact_maps_full = np.load(config.DATA_DIR / "contact_maps_full.npy")
    train_contact_maps = contact_maps_full[train_data['indices']]
    test_contact_maps = contact_maps_full[test_data['indices']]
    
    # Ablation study
    ablation_results = ablation_study(
        train_data,
        test_data,
        train_contact_maps,
        test_contact_maps,
        config.DEVICE
    )
    
    print("\n" + "="*70)
    print("✓ STEP 5 COMPLETE")
    print("="*70)
    
    print(f"\nAll results saved to:")
    print(f"  Metrics: {config.METRICS_DIR}")
    print(f"  Figures: {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()