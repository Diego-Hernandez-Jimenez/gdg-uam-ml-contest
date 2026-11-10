from __future__ import annotations

import numpy as np

from src.classifiers.base import ABSTAIN, Classifier
from src.features.base import FeatureExtractor
from src.preprocessing.base import Pipeline as PreprocessPipeline
from src.preprocessing.base import Step


class TimeSeriesPipeline:
    """Composable end-to-end pipeline: Preprocessing -> Noise Gate -> Features -> Classifier.

    Designed to be fitted in batch mode (N, C, L) and serialized directly with joblib
    for low-latency single-sample inference (C, L).
    """

    def __init__(
        self,
        preprocessing_steps: list[Step],
        feature_extractor: FeatureExtractor,
        classifier: Classifier,
        noise_threshold: float | None = None,
    ):
        self.preprocessing = PreprocessPipeline(preprocessing_steps)
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.noise_threshold = noise_threshold

    @staticmethod
    def _is_noisy(series: np.ndarray, threshold: float) -> bool | np.ndarray:
        """Evaluate high-frequency residual noise std(diff(x)) per sample."""
        diffs = np.diff(series, axis=-1)
        max_channel_noise = np.max(np.std(diffs, axis=-1), axis=-1)
        return max_channel_noise > threshold

    def fit(self, X: np.ndarray, y: np.ndarray) -> TimeSeriesPipeline:
        """Fit pipeline on a batch of series X of shape (N, C, L)."""
        X_clean = self.preprocessing.fit(X).transform(X)
        X_feats = self.feature_extractor.fit(X_clean).transform(X_clean)
        self.classifier.fit(X_feats, y)
        return self

    def predict(self, X: np.ndarray) -> int | np.ndarray:
        """Predict on a single sample (C, L) or a batch (N, C, L)."""
        is_single = X.ndim == 2
        X_clean = self.preprocessing.transform(X)

        # Single-sample fast path (competition inference)
        if is_single:
            if self.noise_threshold is not None and self._is_noisy(
                X_clean, self.noise_threshold
            ):
                return int(ABSTAIN)
            feats = self.feature_extractor.transform(X_clean)
            return int(self.classifier.predict(feats))

        # Batch path (training / cross-validation evaluation)
        feats = self.feature_extractor.transform(X_clean)
        preds = np.asarray(self.classifier.predict(feats), dtype=int)
        if self.noise_threshold is not None:
            noisy_mask = self._is_noisy(X_clean, self.noise_threshold)
            preds[noisy_mask] = ABSTAIN
        return preds
