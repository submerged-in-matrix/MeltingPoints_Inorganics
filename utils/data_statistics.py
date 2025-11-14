from env.modules import *
from featurize.featurize_custom import X, y

# Compute variance for each feature
variances = X.var()

# Sort by variance, highest first
variances_sorted = variances.sort_values(ascending=False)

# Basic statistics
mean_mp = y.mean()
std_mp = y.std()
var_mp = y.var()
min_mp = y.min()
max_mp = y.max()

print(f"Melting point statistics: mean = {mean_mp:.2f} °C, std = {std_mp:.2f} °C, var = {var_mp:.2f}, min = {min_mp:.2f}, max = {max_mp:.2f}")

# Histogram
plt.figure(figsize=(7,4))
plt.hist(y, bins=30, color="skyblue", edgecolor="k", alpha=0.8)
plt.xlabel("Melting Point (°C)")
plt.ylabel("Count")
plt.title("Distribution of Melting Points in Dataset")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# Plot
plt.figure(figsize=(14, 5))
plt.bar(range(len(variances_sorted)), variances_sorted.values)
plt.xticks(range(len(variances_sorted)), variances_sorted.index, rotation=90, fontsize=8)
plt.xlabel("Feature")
plt.ylabel("Variance")
plt.title("Feature Variance Across the Dataset")
plt.tight_layout()
plt.show()