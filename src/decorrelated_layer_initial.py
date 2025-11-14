from env.modules import *
from utils.repro_set_up import rng
from data.data_split import *

N_BASE = 10
base_models = []
base_preds_train = np.zeros((X_train.shape[0], N_BASE))
base_preds_val = np.zeros((X_val.shape[0], N_BASE))

for i in range(N_BASE):
    idx = rng.choice(range(X_train.shape[0]), size=X_train.shape[0], replace=True)
    # Select random subset of features
    n_feat = int(X_train.shape[1] * 0.6)
    feat_idx = rng.choice(range(X_train.shape[1]), size=n_feat, replace=False)
    if i % 2 == 0:
        model = RandomForestRegressor(n_estimators=200, max_features=35, random_state=100+i) 
    else:
        model = GradientBoostingRegressor(n_estimators=200, max_features=35, random_state=200+i)
    model.fit(X_train.iloc[idx, feat_idx], y_train.iloc[idx])
    base_models.append((model, feat_idx))
    base_preds_train[:, i] = model.predict(X_train.iloc[:, feat_idx])
    base_preds_val[:, i] = model.predict(X_val.iloc[:, feat_idx])