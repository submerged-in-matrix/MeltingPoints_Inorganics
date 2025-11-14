from env.modules import *
from sklearn.ensemble import GradientBoostingRegressor as gb
from src.custom_decorr_layer_opt import base_preds_train, base_preds_val
from data.data_split import y_train, y_val

# Stack with GB and report ---
gb_stack = gb(n_estimators=326,
              random_state=123,
              learning_rate=0.0543, # 0.0542644365316447
              max_depth=30,
              max_features=0.4423, # 0.4422849984886695
              min_samples_split=6)

gb_stack.fit(base_preds_train, y_train)
final_pred = gb_stack.predict(base_preds_val)
from sklearn.metrics import mean_absolute_error, r2_score
mae = mean_absolute_error(y_val, final_pred)
r2 = r2_score(y_val, final_pred)

print("\nTwo-level ensemble (GB, RF, MLP custom order; GB stacker):")
print(f"Validation MAE: {mae:.2f} °C")
print(f"Validation R²: {r2:.3f}")