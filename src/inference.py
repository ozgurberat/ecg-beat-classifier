import numpy as np
import joblib
from feature_extraction import extract_combined_features

# Load trained RF model
MODEL_PATH = "../models/random_forest_cw_balanced_fe.joblib"
model = joblib.load(MODEL_PATH)


def predict_segment(ecg_segment, ar_order=6):
    """
    Predicts whether a single ECG segment is a Normal (N) or Ventricular (V) beat.

    Args:
        ecg_segment (np.ndarray): 1D array (already centered around R-peak, 300 ms)
        ar_order (int): Order for AR feature extraction.

    Returns:
        label (str): 'N' or 'V'
        probability (float): Probability for class 'V' (if available, else None)
    """
    if ecg_segment.ndim != 1:
        raise ValueError("Expected a 1D ECG segment array")

    segment_batch = np.expand_dims(ecg_segment, axis=0)  # Shape (1, segment_length)
    features = extract_combined_features(segment_batch, ar_order=ar_order)

    prediction = model.predict(features)[0]
    label = 'V' if prediction == 1 else 'N'

    # Try probability output (only supported by some models)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0][1]  # Probability of class 'V'
    else:
        proba = None

    return label, proba
