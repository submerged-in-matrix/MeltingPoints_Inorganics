from sklearn.ensemble import GradientBoostingRegressor as gb
from env.modules import *
from data.data_split import  y_train,  y_val
from src.custom_decorr_layer_opt import base_preds_train, base_preds_val

# Define parameter search space (same as first layer)
gb_param_space = {
    'n_estimators': Integer(100, 400),
    'max_depth': Integer(3, 30),
    'max_features': Real(0.2, 1.0, prior='uniform'),
    'min_samples_split': Integer(2, 12),
    'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
}

# Set up cross-validation
cv = KFold(n_splits=3, shuffle=True, random_state=123)

# Define base model
gb_model = gb(random_state=123)

# Set up BayesSearchCV
opt = BayesSearchCV(
    estimator=gb_model,
    search_spaces=gb_param_space,
    n_iter=40,  
    scoring=make_scorer(mean_absolute_error, greater_is_better=False),
    cv=cv,
    n_jobs=-1,
    random_state=123,
    verbose=0
)

# Fit optimizer on stacking inputs
opt.fit(base_preds_train, y_train)

# Best model
best_gb_stack = opt.best_estimator_

# Predict and evaluate
final_pred = best_gb_stack.predict(base_preds_val)
mae = mean_absolute_error(y_val, final_pred)
r2 = r2_score(y_val, final_pred)

print(f"Optimized Stacking Model MAE: {mae:.3f}")
print(f"Optimized Stacking Model R²: {r2:.3f}")