from __future__ import annotations

import numpy as np


class FeatureExtractor:
    """Base interface for all feature extraction steps in the ML pipeline.

    Ensures a standardized scikit-learn-like fit/transform contract.
    """

    def fit(self, X: np.ndarray) -> FeatureExtractor:
        """Fits the feature extractor on a batch of training series X of shape (N, C, L).

        Default behavior is a no-op (stateless feature extractors).
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms a batch of time series of shape (N, C, L) into features (N, F).

        Returns:
            np.ndarray: Feature matrix of shape (N, F).
        """
        raise NotImplementedError
