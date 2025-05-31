import numpy as np
import pandas as pd


def extract_segments_from_patient(signal, ann_indices, ann_symbols, sampling_rate, segment_ms=300,
                                  target_labels=('N', 'V')):
    """
    Extracts segments of `segment_ms` milliseconds centered around each target label.
    Returns a list of (segment, label) tuples.
    """
    segment_len = int((segment_ms / 1000) * sampling_rate)
    half_len = segment_len // 2

    segments = []
    labels = []

    for ix, label in zip(ann_indices, ann_symbols):
        if label not in target_labels:
            continue

        start = ix - half_len
        end = ix + half_len

        if start < 0 or end > len(signal):
            continue

        segment = signal[start:end]
        segments.append(segment)
        labels.append(label)

    return segments, labels


def build_dataset_from_filtered_df(filtered_df_path):
    df = pd.read_pickle(filtered_df_path)

    all_segments = []
    all_labels = []

    for _, row in df.iterrows():
        segments, labels = extract_segments_from_patient(
            signal=row['signal'],
            ann_indices=row['ann_indices'],
            ann_symbols=row['ann_symbols'],
            sampling_rate=row['sampling_rate']
        )
        all_segments.extend(segments)
        all_labels.extend(labels)

    return np.array(all_segments), np.array(all_labels)


if __name__ == "__main__":
    X, y = build_dataset_from_filtered_df("../data/filtered_df.pkl")
    print("X shape:", X.shape)
    print("y distribution:", {label: sum(y == label) for label in set(y)})

