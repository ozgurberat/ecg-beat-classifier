import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from data_preparation import build_dataset_from_filtered_df
from feature_extraction import extract_combined_features

# Load and prepare data
X_raw, y = build_dataset_from_filtered_df("../data/filtered_df.pkl")
X = extract_combined_features(X_raw, ar_order=6)

# Encode labels
y = np.array([1 if label == 'V' else 0 for label in y])  # 1 = V, 0 = N

# Split data (70% train, 15% val, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Define models and parameter grids
model_configs = {
    "logistic_regression": {
        "model": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "params": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs"]
        }
    },
    "random_forest": {
        "model": RandomForestClassifier(class_weight="balanced"),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        }
    },
    "svm": {
        "model": SVC(class_weight="balanced", probability=True),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["linear"]  # Removed 'rbf' to avoid slow training
        }
    }
}

# Prepare directory
os.makedirs("models", exist_ok=True)

# Train and evaluate
for name, config in model_configs.items():
    print(f"Training {name}...")

    grid = GridSearchCV(config["model"], config["params"], cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    joblib.dump(best_model, f"../models/{name}_cw_balanced_fe.joblib")

    y_pred = best_model.predict(X_val)
    print(f"[{name.upper()}] Best Parameters: {grid.best_params_}")
    print(f"[{name.upper()}] Validation Results:")
    print(classification_report(y_val, y_pred))
    print("=" * 60)

# Save test set
np.savez("../data/test_data_fe.npz", X_test=X_test, y_test=y_test)
