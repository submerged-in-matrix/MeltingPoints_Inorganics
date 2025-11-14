from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.ensemble import GradientBoostingRegressor as gb
from sklearn.neural_network import MLPRegressor as mlp
from lightgbm import LGBMRegressor as lgbm
from sklearn.neighbors import KNeighborsRegressor as knn
from env.modules import *
from data.data_split import y_train, y_val
from data.data_opt_feats import X_train_sel, X_val_sel


# Parameter dictionaries
rf_params = {'n_estimators': 1000, 'max_depth': 100, 'max_features': 15, 'bootstrap': False}
gb_params = {'n_estimators': 400, 'learning_rate': 0.0686, 'max_depth': 6, 'max_features': 0.637, 'min_samples_split': 5}
mlp_params = {'hidden_layer_sizes': (100,), 'max_iter': 1000}
lgbm_params = {'n_estimators': 200}
knn_params = {'n_neighbors': 50, 'weights': 'uniform'}

model_params = {
    'rf': rf_params,
    'gb': gb_params,
    'mlp': mlp_params,
    'lgbm': lgbm_params,
    'knn': knn_params
}

model_classes = {
    'rf': rf,
    'gb': gb,
    'mlp': mlp,
    'lgbm': lgbm,
    'knn': knn
}
block_variants = [
    ['rf']*5 + ['gb']*5,
    ['rf']*3 + ['gb']*4 + ['mlp']*3,
    ['rf']*3 + ['gb']*3 + ['mlp']*2 + ['lgbm']*2,
    ['rf']*3 + ['gb']*2 + ['mlp']*2 + ['lgbm']*2 + ['knn']*1
]
custom_variants = [
    ['gb','rf','gb','rf','gb','rf','gb','rf','gb','rf'],
    ['gb','rf','mlp','rf','gb','mlp','gb','rf','mlp','gb'],
    ['gb','rf','mlp','lgbm','rf','gb','lgbm','mlp','gb','rf'],
    ['gb','rf','mlp','lgbm','rf','gb','lgbm','mlp','gb','knn']]

subspace_frac = 0.7
results_8variants = []
rng_8variants = np.random.RandomState(42)

all_variants = block_variants + custom_variants
variant_labels = [f"Block {i+1}" for i in range(4)] + [f"Custom {i+1}" for i in range(4)]

for v, (order, label) in enumerate(zip(all_variants, variant_labels), 1):
    N_BASE = len(order)
    base_models_v = []
    base_preds_train_v = np.zeros((X_train_sel.shape[0], N_BASE))
    base_preds_val_v = np.zeros((X_val_sel.shape[0], N_BASE))
    
    for i, mname in enumerate(order):
        ModelClass = model_classes[mname]
        kwargs = model_params[mname].copy()
        if 'random_state' in ModelClass().get_params():
            kwargs['random_state'] = 1000 + v*100 + i
        idx_v = rng_8variants.choice(range(X_train_sel.shape[0]), size=X_train_sel.shape[0], replace=True)
        n_feat_v = int(X_train_sel.shape[1] * subspace_frac)
        feat_idx_v = rng_8variants.choice(range(X_train_sel.shape[1]), size=n_feat_v, replace=False)
        model_v = ModelClass(**kwargs)
        model_v.fit(X_train_sel.iloc[idx_v, feat_idx_v], y_train.iloc[idx_v])
        base_models_v.append((model_v, feat_idx_v))
        base_preds_train_v[:, i] = model_v.predict(X_train_sel.iloc[:, feat_idx_v])
        base_preds_val_v[:, i] = model_v.predict(X_val_sel.iloc[:, feat_idx_v])
    
    # Stacking with GB as meta-learner (GB selected on best R2, Ridge is also a good choice, selected on MAE)
    meta_gb_v = gb(n_estimators=200, random_state=999)
    meta_gb_v.fit(base_preds_train_v, y_train)
    final_pred_v = meta_gb_v.predict(base_preds_val_v)
    mae_v = mean_absolute_error(y_val, final_pred_v)
    r2_v = r2_score(y_val, final_pred_v)
    results_8variants.append((label, order, mae_v, r2_v))
    print(f"{label}: {order}")
    print(f"  MAE: {mae_v:.2f} | R²: {r2_v:.3f}\n")