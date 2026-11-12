from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


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

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fits the extractor and transforms X in a single pass."""
        return self.fit(X).transform(X)


class FeatureUnion(FeatureExtractor):
    """Combines multiple feature extractors and standardizes the concatenated output."""

    def __init__(self, extractors: list[FeatureExtractor], scale_features: bool = True):
        self.extractors = extractors
        self.scale_features = scale_features
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self._scaler: StandardScaler | None = None

    def _extract_raw(self, X: np.ndarray) -> np.ndarray:
        feats = [ext.transform(X) for ext in self.extractors]
        axis = 0 if X.ndim == 2 else 1
        return np.concatenate(feats, axis=axis)

    def _freeze_scaler(self) -> None:
        """Freeze float32 statistics for low-latency inference and free temporary scaler."""
        if self._scaler is not None and self._scaler.mean_ is not None:
            self.mean_ = self._scaler.mean_.astype(np.float32)
            std = np.sqrt(self._scaler.var_).astype(np.float32)
            std[std < 1e-8] = 1.0
            self.std_ = std
        self._scaler = None

    def fit_extractors(self, X: np.ndarray) -> FeatureUnion:
        """Fit only the underlying sub-extractors (used in mini-batch initialization)."""
        for extractor in self.extractors:
            extractor.fit(X)
        self._scaler = StandardScaler() if self.scale_features else None
        self.mean_, self.std_ = None, None
        return self

    def partial_fit_scaler(self, X_batch: np.ndarray) -> FeatureUnion:
        """Incrementally update exact global feature mean and std on a raw batch X_batch."""
        if self.scale_features:
            if self._scaler is None:
                self._scaler = StandardScaler()
            self._scaler.partial_fit(self._extract_raw(X_batch))
        return self

    def finalize_scaler(self) -> FeatureUnion:
        """Finalize incremental standardization statistics."""
        self._freeze_scaler()
        return self

    def fit(self, X: np.ndarray) -> FeatureUnion:
        self.fit_transform(X)
        return self

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit extractors, compute exact full-batch standardization, and return scaled features once."""
        self.fit_extractors(X)
        concat = self._extract_raw(X)
        if self.scale_features and self._scaler is not None:
            self._scaler.fit(concat)
            self._freeze_scaler()
            concat = (concat - self.mean_) / self.std_
        return concat.astype(np.float32, copy=False)

    def transform(self, X: np.ndarray) -> np.ndarray:
        concat = self._extract_raw(X)
        if self.scale_features and self.mean_ is not None and self.std_ is not None:
            concat = (concat - self.mean_) / self.std_
        return concat.astype(np.float32, copy=False)
