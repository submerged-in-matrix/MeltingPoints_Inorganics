# --- Prepare all tuned models ---
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.ensemble import GradientBoostingRegressor as gb
from sklearn.neural_network import MLPRegressor as mlp
from env.modules import *
from data.data_scaling import *
from data.data_split import y_train, y_unseen

custom_order_unseen = ['gb','rf','mlp','rf','gb','mlp','gb','rf','mlp','gb']
model_map_unseen = {
    'rf': rf(
        bootstrap=False,
        max_depth=100,
        max_features=15,
        n_estimators=1000,
        random_state=42
    ),
    'gb': gb(
        learning_rate=0.0686,
        max_depth=6,
        max_features=0.637,
        min_samples_split=5,
        n_estimators=400,
        random_state=42
    ),
    'mlp': mlp(
        activation='relu',
        alpha=0.01,
        batch_size=64,
        learning_rate='constant',
        learning_rate_init=0.00232,
        max_iter=1000,
        solver='adam',
        hidden_layer_sizes=(200, 200, 200, 200, 200),
        random_state=44
    )
}

N_BASE_unseen = len(custom_order_unseen)
base_preds_train_unseen = np.zeros((X_train_scaled.shape[0], N_BASE_unseen))
base_preds_unseen = np.zeros((X_test_final.shape[0], N_BASE_unseen))

rng_custom_unseen = np.random.RandomState(42)
base_models_final_unseen = []

for i, mname in enumerate(custom_order_unseen):
    model = model_map_unseen[mname]
    idx_unseen = rng_custom_unseen.choice(range(X_train_scaled.shape[0]), size=X_train_scaled.shape[0], replace=True)
    n_feat_unseen = int(X_train_scaled.shape[1] * 0.7)
    feat_idx_unseen = rng_custom_unseen.choice(range(X_train_scaled.shape[1]), size=n_feat_unseen, replace=False)

    model_cloned_unseen = clone(model)
    if hasattr(model_cloned_unseen, 'random_state'):
        model_cloned_unseen.set_params(random_state=42 + i)

    model_cloned_unseen.fit(X_train_scaled.iloc[idx_unseen, feat_idx_unseen], y_train.iloc[idx_unseen])
    base_models_final_unseen.append((model_cloned_unseen, feat_idx_unseen))
    base_preds_train_unseen[:, i] = model_cloned_unseen.predict(X_train_scaled.iloc[:, feat_idx_unseen])
    base_preds_unseen[:, i] = model_cloned_unseen.predict(X_test_final.iloc[:, feat_idx_unseen])

# --- Stack with GB and evaluate on unseen data ---
gb_stack_unseen = gb(
    n_estimators=326,
    random_state=123,
    learning_rate=0.0543,
    max_depth=30,
    max_features=0.4423,
    min_samples_split=6
)

gb_stack_unseen.fit(base_preds_train_unseen, y_train)
final_pred_unseen = gb_stack_unseen.predict(base_preds_unseen)


mae_unseen = mean_absolute_error(y_unseen, final_pred_unseen)
r2_unseen = r2_score(y_unseen, final_pred_unseen)

print("\nTwo-level ensemble (GB, RF, MLP custom order; GB stacker) on Unseen Test Set:")
print(f"Unseen MAE: {mae_unseen:.2f} °C")
print(f"Unseen R²: {r2_unseen:.2f}")