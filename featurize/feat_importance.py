from env.modules import *
from data.data_split import *
from src.decorrelated_layer_initial import base_models
from utils.repro_set_up import rng

# Store mean absolute SHAP for each model and feature
all_mean_abs_shap = np.zeros(X_train.shape[1])

for i, (model, feat_idx) in enumerate(base_models):
    # Only tree-based models are SHAP-compatible 
    if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
        explainer = shap.TreeExplainer(model)
        # Compute SHAP values only for the features this model used
        shap_values = explainer.shap_values(X_train.iloc[:, feat_idx])
        # Place mean |SHAP| values in their correct feature locations
        mean_abs_shap = np.zeros(X_train.shape[1])
        mean_abs_shap[feat_idx] = np.abs(shap_values).mean(axis=0)
        all_mean_abs_shap += mean_abs_shap

# Average across all base models
all_mean_abs_shap /= len(base_models)

# Get global feature ranking
feature_ranking = np.argsort(-all_mean_abs_shap)
sorted_feature_names = X_train.columns[feature_ranking]

print("Top 40 features by average SHAP value across ensemble: printed 20")
for i in range(20):
    print(f"{i+1:2d}. {sorted_feature_names[i]} ({all_mean_abs_shap[feature_ranking[i]]:.4f})")

# Bar plot of mean SHAP importances
plt.figure(figsize=(18, 5))
plt.bar(range(40), all_mean_abs_shap[feature_ranking[:40]])
plt.xticks(range(40), sorted_feature_names[:40], rotation=90)
plt.ylabel("Mean(|SHAP value|)")
plt.title("Top 20 Features by SHAP Importance (Averaged Over Ensemble)")
plt.tight_layout()
plt.show()


print(f"Max SHAP importance: {all_mean_abs_shap.max():.4f}")
print(f"Min SHAP importance: {all_mean_abs_shap.min():.4f}")
print(f"Mean SHAP importance: {all_mean_abs_shap.mean():.4f}")
print(f"Median SHAP importance: {np.median(all_mean_abs_shap):.4f}")

# Optionally, see the 90th, 75th, 50th, 25th percentile
percentiles = np.percentile(all_mean_abs_shap, [90, 75, 50, 25])
print(f"90th percentile: {percentiles[0]:.4f}")
print(f"75th percentile: {percentiles[1]:.4f}")
print(f"50th percentile (median): {percentiles[2]:.4f}")
print(f"25th percentile: {percentiles[3]:.4f}")