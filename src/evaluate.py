import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# Choose the model you want to load
model_path = "../models/random_forest_cw_balanced_fe.joblib"
model = joblib.load(model_path)

data = np.load("../data/test_data_fe.npz")
X_test = data["X_test"]
y_test = data["y_test"]

y_pred = model.predict(X_test)

print("[RESULTS] Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n[RESULTS] Classification Report:")
print(classification_report(y_test, y_pred))
