from env.modules import *
from data.data_split import *
from src.decorrelated_layer_initial import base_models
from featurize.feat_importance import sorted_feature_names
from utils.repro_set_up import rng

feature_counts = list(range(30, 136, 1))  # 30, 40, ..., 90
results = []

for top_n in feature_counts:
    selected_features = list(sorted_feature_names[:top_n])
    X_train_sub = X_train[selected_features]
    X_val_sub = X_val[selected_features]
    
    # Reset RNG for reproducibility
    rng = np.random.RandomState(42)
    
    N_BASE = 10
    base_models = []
    base_preds_train = np.zeros((X_train_sub.shape[0], N_BASE))
    base_preds_val = np.zeros((X_val_sub.shape[0], N_BASE))
    
    for i in range(N_BASE):
        idx = rng.choice(range(X_train_sub.shape[0]), size=X_train_sub.shape[0], replace=True)
        n_feat = int(X_train_sub.shape[1] * 0.6)
        feat_idx = rng.choice(range(X_train_sub.shape[1]), size=n_feat, replace=False)
        if i % 2 == 0:
            model = RandomForestRegressor(n_estimators=200, random_state=100+i)
        else:
            model = GradientBoostingRegressor(n_estimators=200, random_state=200+i)
        model.fit(X_train_sub.iloc[idx, feat_idx], y_train.iloc[idx])
        base_models.append((model, feat_idx))
        base_preds_train[:, i] = model.predict(X_train_sub.iloc[:, feat_idx])
        base_preds_val[:, i] = model.predict(X_val_sub.iloc[:, feat_idx])
    
    # Second-level stacking
    stacker = Ridge(alpha=1.0)
    stacker.fit(base_preds_train, y_train)
    final_pred = stacker.predict(base_preds_val)
    
    # Evaluate and store
    mae = mean_absolute_error(y_val, final_pred)
    r2 = r2_score(y_val, final_pred)
    results.append((top_n, mae, r2))
    print(f"Features: {top_n:2d} | MAE: {mae:.2f} °C | R²: {r2:.3f}")

# Convert results to DataFrame for plotting
results_df = pd.DataFrame(results, columns=["NumFeatures", "MAE", "R2"])

# Top 3 by highest R²
top_r2 = results_df.sort_values("R2", ascending=False).head(3)
print("\nTop 3 feature counts by highest R²:")
for _, row in top_r2.iterrows():
    print(f"Num features: {int(row['NumFeatures'])} | R²: {row['R2']:.3f} | MAE: {row['MAE']:.2f} °C")

# Top 3 by lowest MAE
top_mae = results_df.sort_values("MAE", ascending=True).head(3)
print("\nTop 3 feature counts by lowest MAE:")
for _, row in top_mae.iterrows():
    print(f"Num features: {int(row['NumFeatures'])} | MAE: {row['MAE']:.2f} °C | R²: {row['R2']:.3f}")

plt.figure(figsize=(7,4))
plt.plot(results_df["NumFeatures"], results_df["R2"], marker='o', label="R² (Validation)")
plt.xlabel("Number of Top Features")
plt.ylabel("R² (Validation)")
plt.title("Validation R² vs. Number of Top Features")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4))
plt.plot(results_df["NumFeatures"], results_df["MAE"], marker='o', color="orange", label="MAE (Validation)")
plt.xlabel("Number of Top Features")
plt.ylabel("MAE (Validation)")
plt.title("Validation MAE vs. Number of Top Features")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()