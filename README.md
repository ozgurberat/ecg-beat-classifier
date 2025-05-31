# ECG PVC Classification

This project detects Premature Ventricular Contractions (PVCs) in ECG signals by classifying 300 ms segments centered around R-peaks as either:

- N: Normal beat

- V: Ventricular (PVC) beat

It combines classical signal processing with machine learning, providing:

- Feature extraction from ECG segments

- Multiple ML model training and selection

- Feature importance analysis

- Real-time inference API using FastAPI

## Features Extracted

From each 1D ECG segment (typically 108 samples, ~300 ms):

1. AR Coefficients (order=6)

    - Computed using Yule-Walker equations

2. Statistical Features

    - Mean

    - Standard deviation

    - Skewness

    - Kurtosis

    - Peak-to-peak amplitude

    - Average slope

    - Max absolute slope

These are combined into a single vector of 13 features per segment.

## Model Training

Implemented in ```train_model.py```

- Models Trained:

    - Logistic Regression

    - Random Forest (Best)

    - SVM

- GridSearchCV for hyperparameter tuning

- Balanced class weights

- Evaluation using validation accuracy, precision, recall

- Final model: Random Forest, saved as random_forest_cw_balanced_fe.joblib

## Feature Importance

Script: ```feature_analysis.py```

- Uses .feature_importances_ from Random Forest

- Plots ranked bar chart of all 13 features

## Inference

1. Local Prediction (for testing/debugging)
```
python3 inference.py
```
2. REST API

Start the API server:
```
cd src
uvicorn api:app --reload
```
This launches a local FastAPI app at ```http://127.0.0.1:8000```, allowing you to send ECG segments using test_client.py.

3. Send a Sample ECG Segment
```
python3 test_client.py
```
Expected Output:
```
Prediction: {'label': 'N', 'probability_of_V': 0.01}
```

## API Endpoint

POST ```/predict```
```
{
  "segment": [0.1, 0.03, -0.07, ...]  # ECG segment, 1D float list
}
```
Response:
```
{
  "label": "N",                  # or "V"
  "probability_of_V": 0.0134     # Float, 0-1
}
```

## Dataset

This project uses a subset of the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/) via PhysioNet.

Preprocessed data is stored as:
```data/filtered_df.pkl```
