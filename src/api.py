from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib

from feature_extraction import extract_combined_features

MODEL_PATH = "../models/random_forest_cw_balanced_fe.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="ECG PVC Classifier API",
              description="Classify ECG segments as Normal (N) or Ventricular (V) Beats",
              version="1.0")


# Input schema
class SegmentInput(BaseModel):
    segment: list[float]  # raw 1D ECG segment (assumed 300ms centered around R-peak)


# Output schema
class PredictionOutput(BaseModel):
    label: str  # "N" or "V"
    probability_of_V: float


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: SegmentInput):
    segment = np.array(input_data.segment)

    if len(segment.shape) != 1:
        raise HTTPException(status_code=400, detail="Input segment must be a 1D array")

    try:
        features = extract_combined_features([segment])  # shape (1, n_features)
        prob = model.predict_proba(features)[0][1]  # probability of class '1' (PVC)
        label = "V" if prob >= 0.5 else "N"

        return PredictionOutput(label=label, probability_of_V=round(prob, 4))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
