from __future__ import annotations

import numpy as np

from .base import FeatureExtractor


class IdentityFeatures(FeatureExtractor):
    """Passthrough feature extractor.

    Flattens multi-channel time series into a flat feature vector per sample.
    Supports both batch mode (N, C, L) -> (N, C * L) and
    streaming/inference mode (C, L) -> (C * L,).
    """

    def transform(self, X: np.ndarray) -> np.ndarray:
        is_single = X.ndim == 2
        X_in = X[np.newaxis, ...] if is_single else X

        num_examples = X_in.shape[0]
        features = X_in.reshape(num_examples, -1)

        return features[0] if is_single else features
