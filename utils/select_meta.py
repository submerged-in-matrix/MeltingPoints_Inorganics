from env.modules import *
from data.data_split import y_train, y_val
from src.decorrelated_layer_opt import *

meta_learners = [
    ("Ridge", Ridge(alpha=1.0)),
    ("Lasso", Lasso(alpha=0.01, max_iter=10000)),
    ("RF", RandomForestRegressor(n_estimators=200, random_state=42)),
    ("GB", GradientBoostingRegressor(n_estimators=200, random_state=42)),
    ("LGBM", LGBMRegressor(n_estimators=200, force_col_wise='true', random_state=42)),
    ("MLP", MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))
    
]

print("\nMeta-learner optimization (stacking level):")
results_meta = []
for name, meta_model in meta_learners:
    meta_model.fit(base_preds_train_final, y_train)
    pred = meta_model.predict(base_preds_val_final)
    mae = mean_absolute_error(y_val, pred)
    r2 = r2_score(y_val, pred)
    results_meta.append((name, mae, r2))
    print(f"{name:6} | MAE: {mae:.2f} | R²: {r2:.3f}")

# Report the best meta-learner by R² and MAE
results_meta_df = pd.DataFrame(results_meta, columns=["MetaLearner", "MAE", "R2"])
print("\nBest meta-learner by R²:")
print(results_meta_df.sort_values("R2", ascending=False).iloc[0])
print("\nBest meta-learner by MAE:")
print(results_meta_df.sort_values("MAE", ascending=True).iloc[0])