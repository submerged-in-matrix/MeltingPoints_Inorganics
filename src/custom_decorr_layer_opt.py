# --- Prepare all tuned models ---
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.ensemble import GradientBoostingRegressor as gb
from sklearn.neural_network import MLPRegressor as mlp
from env.modules import *
from data.data_scaling import *
from data.data_split import y_train, y_val

# --- Step 3: Build the ensemble in the specified order ---
custom_order = ['gb','rf','mlp','rf','gb','mlp','gb','rf','mlp','gb']
model_map = {
    'rf': rf(
            bootstrap=False,
             max_depth=100,
             max_features=15,
             n_estimators=1000,
             random_state=42
             ),
    
    'gb': gb(learning_rate=0.0686,  # 0.06855932545815964,
             max_depth=6,
             max_features=0.637, # 0.6370300932851438
             min_samples_split=5,
             n_estimators=400
             ),
    
    'mlp': mlp(
            activation='relu',
            alpha=0.01,
            batch_size=64,
            learning_rate='constant',
            learning_rate_init=0.00232, # 0.0023184882208720794
            max_iter=1000,
            solver='adam',
            hidden_layer_sizes=(200, 200, 200, 200),
            random_state=44
            )}

N_BASE_new = len(custom_order)
base_preds_train = np.zeros((X_train_scaled.shape[0], N_BASE_new))
base_preds_val = np.zeros((X_val_scaled.shape[0], N_BASE_new))

# Refit each model on bootstrapped subset for ensemble diversity (optionally use the same tuned objects  less randomness is desired)
rng_custom = np.random.RandomState(42)
base_models_final = []

for i, mname in enumerate(custom_order):
    model = model_map[mname]
    idx = rng_custom.choice(range(X_train_scaled.shape[0]), size=X_train_scaled.shape[0], replace=True)
    n_feat = int(X_train_scaled.shape[1] * 0.7)
    feat_idx = rng_custom.choice(range(X_train_scaled.shape[1]), size=n_feat, replace=False)
    
    # Clone and refit to bootstrapped data for diversity
    from sklearn.base import clone
    model_cloned = clone(model)
    if hasattr(model_cloned, 'random_state'):
        model_cloned.set_params(random_state=42 + i)      # For reproducibilioty
    model_cloned.fit(X_train_scaled.iloc[idx, feat_idx], y_train.iloc[idx])
    base_models_final.append((model_cloned, feat_idx))
    base_preds_train[:, i] = model_cloned.predict(X_train_scaled.iloc[:, feat_idx])
    base_preds_val[:, i] = model_cloned.predict(X_val_scaled.iloc[:, feat_idx])