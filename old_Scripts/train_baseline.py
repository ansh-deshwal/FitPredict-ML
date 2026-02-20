import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

# 1. Load Data and Embeddings
# Get the folder where this script is located
script_dir = Path(__file__).parent

print("Loading data...")
df = pd.read_csv(script_dir / "BLAT_ECOLX_Stiffler_2015.csv")
X = np.load(script_dir / "beta_lactamase_esm2_embeddings.npy")
y = df["DMS_score"].values

print(f"Dataset shape: {X.shape}")
print(f"Number of samples: {len(y)}")

# 2. Split Data (Train 80% / Test 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# 3. Train a Simple Regressor
print("\nTraining Ridge Regression baseline...")
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# 4. Evaluate
print("\nEvaluating model...")
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Calculate metrics
train_mse = mean_squared_error(y_train, train_preds)
test_mse = mean_squared_error(y_test, test_preds)

train_r2 = r2_score(y_train, train_preds)
test_r2 = r2_score(y_test, test_preds)

train_rho, _ = spearmanr(y_train, train_preds)
test_rho, _ = spearmanr(y_test, test_preds)

train_pearson, _ = pearsonr(y_train, train_preds)
test_pearson, _ = pearsonr(y_test, test_preds)

# Print results
print(f"\n{'='*50}")
print(f"{'BASELINE RESULTS':^50}")
print(f"{'='*50}")
print(f"\nTraining Set:")
print(f"  MSE:                {train_mse:.4f}")
print(f"  R² Score:           {train_r2:.4f}")
print(f"  Spearman ρ:         {train_rho:.4f}")
print(f"  Pearson r:          {train_pearson:.4f}")

print(f"\nTest Set:")
print(f"  MSE:                {test_mse:.4f}")
print(f"  R² Score:           {test_r2:.4f}")
print(f"  Spearman ρ:         {test_rho:.4f}")
print(f"  Pearson r:          {test_pearson:.4f}")
print(f"\n{'='*50}")

# 5. Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Training plot
axes[0].scatter(y_train, train_preds, alpha=0.4, s=15, edgecolors='none')
axes[0].plot([y_train.min(), y_train.max()], 
             [y_train.min(), y_train.max()], 
             'r--', lw=2, label='Perfect prediction')
axes[0].set_title(f"Training Set (Spearman ρ = {train_rho:.3f})", fontsize=12, fontweight='bold')
axes[0].set_xlabel("True Fitness", fontsize=11)
axes[0].set_ylabel("Predicted Fitness", fontsize=11)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Test plot
axes[1].scatter(y_test, test_preds, alpha=0.4, s=15, edgecolors='none', color='orange')
axes[1].plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect prediction')
axes[1].set_title(f"Test Set (Spearman ρ = {test_rho:.3f})", fontsize=12, fontweight='bold')
axes[1].set_xlabel("True Fitness", fontsize=11)
axes[1].set_ylabel("Predicted Fitness", fontsize=11)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

# Save plot
output_path = script_dir / "baseline_plot.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved plot to '{output_path}'")

# Optional: Save predictions for further analysis
results_df = pd.DataFrame({
    'true_fitness': y_test,
    'predicted_fitness': test_preds,
    'residual': y_test - test_preds
})
results_path = script_dir / "baseline_predictions.csv"
results_df.to_csv(results_path, index=False)
print(f"Saved predictions to '{results_path}'")

plt.show()
