import matplotlib.pyplot as plt
from data.data_split import y_unseen
from src.two_lvl_ensemble   import * 

plt.figure(figsize=(8, 6))

# Plot true vs predicted for each sample
for i in range(len(y_unseen)):
    # draw a line between true and predicted values
    plt.plot([i, i], [y_unseen.iloc[i], final_pred_unseen[i]], color='gray', linestyle='-', alpha=0.5)

# scatter true values
plt.scatter(range(len(y_unseen)), y_unseen, color='blue', label='True', alpha=0.7)

# scatter predicted values
plt.scatter(range(len(y_unseen)), final_pred_unseen, color='orange', label='Predicted', alpha=0.7)

plt.xlabel('Compound Index')
plt.ylabel('Melting Point (°C)')
plt.title('True vs Predicted Melting Points (Unseen Test Set)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()