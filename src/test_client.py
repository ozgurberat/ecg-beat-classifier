import requests
import numpy as np
import pandas as pd

# Load filtered_df and select one patient
df = pd.read_pickle("../data/filtered_df.pkl")
row = df.iloc[0]
signal = row['signal']
ann_indices = row['ann_indices']
ann_symbols = row['ann_symbols']
fs = row['sampling_rate']

# Pick one R-peak and extract the segment
target_ix = next(ix for ix, label in zip(ann_indices, ann_symbols) if label == 'N')
segment_len = int((300 / 1000) * fs)
half_len = segment_len // 2
start = target_ix - half_len
end = target_ix + half_len
segment = signal[start:end]

if len(segment) != segment_len:
    raise ValueError("Segment length mismatch!")

url = "http://127.0.0.1:8000/predict"
response = requests.post(url, json={"segment": segment.tolist()})

if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print("Error:", response.status_code, response.text)
