from env.modules import *
from data.data_split import y_train
from data.data_opt_feats import X_train_sel, X_val_sel

# Re-create  RandomState for reproducibility
rng_ensemble = np.random.RandomState(42)

N_BASE_ensemble = 10
subspace_frac_ensemble = 0.70

base_models_final = []
base_preds_train_final = np.zeros((X_train_sel.shape[0], N_BASE_ensemble))
base_preds_val_final = np.zeros((X_val_sel.shape[0], N_BASE_ensemble))

for i in range(N_BASE_ensemble):
    idx = rng_ensemble.choice(range(X_train_sel.shape[0]), size=X_train_sel.shape[0], replace=True)
    n_feat = int(X_train_sel.shape[1] * subspace_frac_ensemble)
    feat_idx = rng_ensemble.choice(range(X_train_sel.shape[1]), size=n_feat, replace=False)
    if i % 2 == 0:
        model = RandomForestRegressor(
            n_estimators=1000,
            max_depth=100,
            max_features=15,
            bootstrap=False,
            random_state=100+i
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.06855932545815964,
            max_depth=6,
            max_features=0.6370300932851438,
            min_samples_split=5,
            random_state=200+i
        )
    model.fit(X_train_sel.iloc[idx, feat_idx], y_train.iloc[idx])
    base_models_final.append((model, feat_idx))
    base_preds_train_final[:, i] = model.predict(X_train_sel.iloc[:, feat_idx])
    base_preds_val_final[:, i] = model.predict(X_val_sel.iloc[:, feat_idx])