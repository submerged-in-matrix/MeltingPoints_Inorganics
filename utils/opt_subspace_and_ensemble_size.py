from env.modules import *
from data.data_split import X_train, y_train, X_val, y_val
from data.data_opt_feats import *


n_base_list = [6, 8, 10, 12, 14, 16, 18, 20]
subspace_fractions = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
results_ensemble = []

for N_BASE in n_base_list:
    for frac in subspace_fractions:
        rng_local = np.random.RandomState(42)
        base_models_local = []
        base_preds_train_local = np.zeros((X_train_sel.shape[0], N_BASE))
        base_preds_val_local = np.zeros((X_val_sel.shape[0], N_BASE))

        for i in range(N_BASE):
            idx_local = rng_local.choice(range(X_train_sel.shape[0]), size=X_train_sel.shape[0], replace=True)
            n_feat_local = int(X_train_sel.shape[1] * frac)
            feat_idx_local = rng_local.choice(range(X_train_sel.shape[1]), size=n_feat_local, replace=False)
            if i % 2 == 0:
                model_local = RandomForestRegressor(n_estimators=200, random_state=100+i)
            else:
                model_local = GradientBoostingRegressor(n_estimators=200, random_state=200+i)
            model_local.fit(X_train_sel.iloc[idx_local, feat_idx_local], y_train.iloc[idx_local])
            base_models_local.append((model_local, feat_idx_local))
            base_preds_train_local[:, i] = model_local.predict(X_train_sel.iloc[:, feat_idx_local])
            base_preds_val_local[:, i] = model_local.predict(X_val_sel.iloc[:, feat_idx_local])

        stacker_local = Ridge(alpha=1.0)
        stacker_local.fit(base_preds_train_local, y_train)
        final_pred_local = stacker_local.predict(base_preds_val_local)
        mae_local = mean_absolute_error(y_val, final_pred_local)
        r2_local = r2_score(y_val, final_pred_local)
        results_ensemble.append((N_BASE, frac, mae_local, r2_local))

# Convert results to DataFrame
tune_df_ensemble = pd.DataFrame(results_ensemble, columns=["N_BASE", "SubspaceFrac", "MAE", "R2"])

# --- Report best hyperparameters by highest R² ---
best_r2_row = tune_df_ensemble.loc[tune_df_ensemble["R2"].idxmax()]
print("\nBest hyperparameters by R²:")
print(f"N_BASE = {int(best_r2_row['N_BASE'])}, SubspaceFrac = {best_r2_row['SubspaceFrac']:.2f}, "
      f"MAE = {best_r2_row['MAE']:.2f}, R² = {best_r2_row['R2']:.3f}")

# --- Report best hyperparameters by lowest MAE ---
best_mae_row = tune_df_ensemble.loc[tune_df_ensemble["MAE"].idxmin()]
print("\nBest hyperparameters by MAE:")
print(f"N_BASE = {int(best_mae_row['N_BASE'])}, SubspaceFrac = {best_mae_row['SubspaceFrac']:.2f}, "
      f"MAE = {best_mae_row['MAE']:.2f}, R² = {best_mae_row['R2']:.3f}")