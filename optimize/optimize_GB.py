from env.modules import *
from data.data_opt_feats import X_train_sel
from data.data_split import y_train

rng_meta = np.random.RandomState(42)
cv_meta = KFold(n_splits=5, shuffle=True, random_state=42)

# Tuning Gradient Boosting (same style)
gb_param_space = {
    'n_estimators': (100, 400),
    'max_depth': (3, 30),
    'max_features': (0.2, 1.0),
    'min_samples_split': (2, 12),
    'learning_rate': (0.01, 0.2),
    # Add more GBM params if needed
}

gb_bayes_meta = BayesSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    search_spaces=gb_param_space,
    n_iter=40,
    scoring='neg_mean_absolute_error',
    cv=cv_meta,
    random_state=42,
    n_jobs=-1,
    verbose=2
)
gb_bayes_meta.fit(X_train_sel, y_train)

print("Best Gradient Boosting params:")
print(gb_bayes_meta.best_params_)
print("Best MAE:", -gb_bayes_meta.best_score_)