from env.modules import *
from data.data_opt_feats import X_train_sel
from data.data_split import y_train

# For reproducibility and to match your ensemble structure
rng_meta = np.random.RandomState(42)
cv_meta = KFold(n_splits=5, shuffle=True, random_state=42)

# Tuning Random Forest (as one example)
rf_param_space = {
    'n_estimators': (400, 1000),
    'max_depth': (40, 100),
    'max_features': (13, 68),
    #'min_samples_split': (2, 10),
    #'min_samples_leaf': (1, 10),
    'bootstrap': (True, False),
}

rf_bayes_meta = BayesSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    search_spaces=rf_param_space,
    n_iter=40,  
    scoring='neg_mean_absolute_error',
    cv=cv_meta,
    random_state=42,
    n_jobs=-1,
    verbose=False,
)
rf_bayes_meta.fit(X_train_sel, y_train)

print("Best Random Forest params:")
print(rf_bayes_meta.best_params_)
print("Best MAE:", -rf_bayes_meta.best_score_)