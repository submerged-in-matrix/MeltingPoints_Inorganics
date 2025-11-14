from env.modules import *

def regression_report(y_true, y_pred, label="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{label} MAE: {mae:.2f} °C")
    print(f"{label} R²: {r2:.3f}")
    return mae, r2