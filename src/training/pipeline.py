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

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int | None = None,
        n_epochs: int = 3,
    ) -> TimeSeriesPipeline:
        """Fit pipeline in full-batch mode or memory-safe mini-batch mode."""
        n_samples = X.shape[0]

        # Standard full-batch in-memory path (single feature extraction pass via fit_transform)
        if batch_size is None or n_samples <= batch_size:
            X_clean = self.preprocessing.fit(X).transform(X)
            X_feats = self.feature_extractor.fit_transform(X_clean)
            self.classifier.fit(X_feats, y)
            return self

        # Memory-safe mini-batch path
        rng = np.random.default_rng(42)
        classes = np.unique(y)

        # 1. Fit preprocessing and sub-extractors on a representative subset
        subset_idx = rng.choice(n_samples, size=min(batch_size, n_samples), replace=False)
        X_sub_clean = self.preprocessing.fit(X[subset_idx]).transform(X[subset_idx])
        if hasattr(self.feature_extractor, "fit_extractors"):
            self.feature_extractor.fit_extractors(X_sub_clean)
        else:
            self.feature_extractor.fit(X_sub_clean)
        del X_sub_clean

        # Pass 1: Compute exact global feature mean & std across all mini-batches
        if getattr(self.feature_extractor, "scale_features", False):
            for start in range(0, n_samples, batch_size):
                X_b = self.preprocessing.transform(X[start : min(start + batch_size, n_samples)])
                self.feature_extractor.partial_fit_scaler(X_b)
            self.feature_extractor.finalize_scaler()

        # 2. If classifier supports incremental learning (SGD or IncrementalRidgeClassifier)
        if hasattr(self.classifier, "partial_fit_model"):
            if hasattr(self.classifier, "set_class_weights_from_labels"):
                self.classifier.set_class_weights_from_labels(y)

            effective_epochs = 1 if hasattr(self.classifier, "finalize_fit") else n_epochs
            for _ in range(effective_epochs):
                perm = rng.permutation(n_samples)
                for start in range(0, n_samples, batch_size):
                    idx = perm[start : min(start + batch_size, n_samples)]
                    X_b = self.preprocessing.transform(X[idx])
                    feats_b = self.feature_extractor.transform(X_b)
                    self.classifier.partial_fit_model(feats_b, y[idx], classes=classes)

            if hasattr(self.classifier, "finalize_fit"):
                self.classifier.finalize_fit()
            return self

        # 3. Fallback for RidgeClassifier: extract globally standardized features in chunks
        first_feats = self.feature_extractor.transform(self.preprocessing.transform(X[:1]))
        X_feats_all = np.empty((n_samples, first_feats.shape[1]), dtype=np.float32)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_b = self.preprocessing.transform(X[start:end])
            X_feats_all[start:end] = self.feature_extractor.transform(X_b)

        self.classifier.fit(X_feats_all, y)
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
