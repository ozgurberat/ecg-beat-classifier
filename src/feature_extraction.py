import numpy as np
from statsmodels.regression.linear_model import yule_walker
from scipy.stats import skew, kurtosis


def extract_ar_features(segments, order=6, normalize=True):
    """
    Extracts autoregressive (AR) coefficients from each ECG segment using Yule-Walker method.

    Args:
        segments (np.ndarray): Array of 1D ECG segments (shape: [n_samples, segment_length])
        order (int): Order of the AR model.
        normalize (bool): Whether to z-score normalize each segment before extraction.

    Returns:
        np.ndarray: Feature matrix of shape (n_samples, order)
    """
    ar_features = []

    for seg in segments:
        if normalize:
            seg = (seg - np.mean(seg)) / np.std(seg)

        rho, _ = yule_walker(seg, order=order)
        ar_features.append(rho)

    return np.array(ar_features)


def extract_statistical_features(segments):
    stats_features = []
    for seg in segments:
        features = [
            np.mean(seg),
            np.std(seg),
            skew(seg),
            kurtosis(seg),
            np.max(seg) - np.min(seg),             # peak-to-peak
            np.mean(np.diff(seg)),                 # avg slope
            np.max(np.abs(np.diff(seg)))           # max absolute slope
        ]
        stats_features.append(features)
    return np.array(stats_features)


def extract_combined_features(segments, ar_order=6):
    """
    Combines AR + statistical features into a single feature vector per segment.
    Returns an (n_samples, n_features) matrix.
    """
    ar = extract_ar_features(segments, order=ar_order)
    stats = extract_statistical_features(segments)
    return np.hstack([ar, stats])
