from env.modules import *
from data.data_scaling import *
from data.data_split import y_train

cv_meta = KFold(n_splits=3, shuffle=True, random_state=42)
mlp_bayes = BayesSearchCV(
    estimator=MLPRegressor(
        hidden_layer_sizes=(200, 200, 200, 200),  # Fixed here
        early_stopping=True,
        random_state=42
    ),
    search_spaces={
        "alpha": Real(1e-5, 1e-2, prior='log-uniform'),
        "learning_rate_init": Real(1e-4, 1e-2, prior='log-uniform'),
        "learning_rate": Categorical(["constant", "invscaling", "adaptive"]),
        "batch_size": Categorical(["auto", 32, 64, 128]),
        "activation": Categorical(["relu", "tanh"]),
        "solver": Categorical(["adam", "lbfgs"]),
        "max_iter": Integer(1000, 3000)
    },
    n_iter=40,
    scoring='neg_root_mean_squared_error',
    cv=cv_meta,
    random_state=42,
    n_jobs=-1,
    verbose=2
)
mlp_bayes.fit(X_train_scaled, y_train)
print("Best MLP parameters:", mlp_bayes.best_params_)
print("Best MLP RMSE (CV):", -mlp_bayes.best_score_)