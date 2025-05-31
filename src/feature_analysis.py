import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load trained RF model
model = joblib.load("../models/random_forest_cw_balanced_fe.joblib")

feature_names = [f"ar_{i+1}" for i in range(6)] + [
    "mean", "std", "skew", "kurtosis", "peak_to_peak", "avg_slope", "max_abs_slope"
]

importances = model.feature_importances_

sorted_idx = np.argsort(importances)[::-1]
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_importances = importances[sorted_idx]

# Plot
plt.figure(figsize=(10, 5))
plt.barh(sorted_features[::-1], sorted_importances[::-1], color="forestgreen")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance_rf.png")
plt.show()
