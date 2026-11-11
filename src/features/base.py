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

class FeatureUnion(FeatureExtractor):
    """Combines multiple feature extractors by column-wise concatenation."""

    def __init__(self, extractors: list[FeatureExtractor], scale_features: bool = True):
        self.extractors = extractors
        self.scale_features = scale_features
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> FeatureUnion:
        for extractor in self.extractors:
            extractor.fit(X)

        if self.scale_features:
            feats = [ext.transform(X) for ext in self.extractors]
            concat = np.concatenate(feats, axis=1)
            # Use float64 accumulator for numerical stability on large datasets
            self.mean_ = np.mean(concat, axis=0, dtype=np.float64).astype(np.float32)
            self.std_ = np.std(concat, axis=0, dtype=np.float64).astype(np.float32)
            self.std_[self.std_ < 1e-8] = 1.0

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        feats = [ext.transform(X) for ext in self.extractors]
        is_single = X.ndim == 2

        concat = np.concatenate(feats, axis=0 if is_single else 1)

        if self.scale_features and self.mean_ is not None:
            concat = (concat - self.mean_) / self.std_

        return concat.astype(np.float32, copy=False)
