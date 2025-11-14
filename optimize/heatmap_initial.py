from env.modules import *
from src.decorrelated_layer_initial import base_preds_val
from data.data_split import y_val

##-----------for HeatMap of Correlation between Base Models' Predictions -----------##
# Make DataFrame for easy plotting
base_preds_df = pd.DataFrame(base_preds_val, columns=[f"Base_{i}" for i in range(base_preds_val.shape[1])])

# Compute the correlation matrix
corr_matrix = base_preds_df.corr()

# Show correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("HeatMap: Correlation between base model predictions (validation set)")
plt.show()

# Print the average off-diagonal correlation                                        # k=1 >> starts one row above the diagonal (where i < j), ignoring self-comparisons.
off_diag_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]   # triu_indices_from >> Return the indices for the upper-triangle of arr. 
print(f"Mean off-diagonal correlation between base models: {np.mean(off_diag_corr):.3f}")

##-----------for HeatMap of Correlation between Base Models' Residuals -----------##
# Calculate residuals (errors) for each base model on test set
residuals = base_preds_val - y_val.values.reshape(-1, 1)

# Make a DataFrame for the residuals
resid_df = pd.DataFrame(residuals, columns=[f"Base_{i}" for i in range(base_preds_val.shape[1])])

# Correlation matrix for residuals
resid_corr_matrix = resid_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(resid_corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation between base model residuals (Validation set)")
plt.show()

# Print average off-diagonal correlation for residuals
off_diag_corr_resid = resid_corr_matrix.values[np.triu_indices_from(resid_corr_matrix.values, k=1)]
print(f"Mean off-diagonal correlation between base model residuals: {np.mean(off_diag_corr_resid):.3f}")