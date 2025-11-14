from env.modules import *
from utils.evaluation_helper import regression_report
from src.stacking_layer_initial import final_pred
from data.data_split import y_val

print("\nTwo-level stacked ensemble performance:")
regression_report(y_val, final_pred, "Stacked Ensemble")

plt.figure(figsize=(6,6))
plt.scatter(y_val, final_pred, label='Stacked Ensemble', alpha=0.8)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
plt.xlabel("Experimental melting point (°C)")
plt.ylabel("Predicted melting point (°C)")
plt.legend()
plt.title("Melting Point initial Prediction (Experimental vs Predicted)")
plt.tight_layout()
plt.show()