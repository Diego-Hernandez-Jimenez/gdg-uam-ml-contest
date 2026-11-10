from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

from .base import ABSTAIN, Classifier


class DummyAbstainClassifier(Classifier):
    """Trivial classifier that always predicts ABSTAIN (-1)."""

    def predict(self, X: np.ndarray) -> int | np.ndarray:
        if X.ndim == 1:
            return int(ABSTAIN)
        return np.full(X.shape[0], ABSTAIN, dtype=int)


class RidgeClassifier(Classifier):
    """Ridge classifier wrapper with built-in StandardScaler and abstention logic.
    
    Standardizes features internally (zero mean, unit variance) to ensure
    optimal conditioning for L2 regularization without data leakage.
    """

    def __init__(
        self,
        alphas: np.ndarray | tuple[float, ...] = np.logspace(-3, 3, 10),  # noqa: B008
        confidence_threshold: float | None = None,
        normalize_features: bool = True,
    ):
        """Initialize the RidgeClassifier.
        
        Args:
            alphas: Array of alpha values to evaluate in RidgeClassifierCV.
            confidence_threshold: Minimum margin required between top-1 and top-2
                logits. Below this threshold, ABSTAIN (-1) is returned.
            normalize_features: If True, fits and applies a StandardScaler
                strictly on the training features.
        """
        self.alphas = alphas
        self.confidence_threshold = confidence_threshold
        self.normalize_features = normalize_features
        self.scaler_: StandardScaler | None = None
        self.model_: RidgeClassifierCV | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeClassifier:
        """Fit scaler and linear classifier strictly on training data (No Leakage)."""
        X_in = np.asarray(X, dtype=np.float32)

        # 1. Fit and transform scaler strictly on the training set
        if self.normalize_features:
            self.scaler_ = StandardScaler(copy=False)
            X_train = self.scaler_.fit_transform(X_in)
        else:
            self.scaler_ = None
            X_train = X_in

        # 2. Fit RidgeClassifierCV on the standardized training features
        self.model_ = RidgeClassifierCV(alphas=self.alphas)
        self.model_.fit(X_train, y)
        return self

    def _should_abstain(self, scores: np.ndarray) -> np.ndarray:
        """Determine whether to abstain based on top-1 vs top-2 logit margin."""
        if self.confidence_threshold is None:
            return np.zeros(scores.shape[0], dtype=bool)

        if scores.ndim == 1 or scores.shape[1] == 1:
            margin = np.abs(scores.ravel())
        else:
            part = np.partition(scores, -2, axis=1)
            margin = part[:, -1] - part[:, -2]

        return margin < self.confidence_threshold

    def predict(self, X: np.ndarray) -> int | np.ndarray:
        """Predict class or ABSTAIN using frozen training statistics."""
        if self.model_ is None:
            raise ValueError("RidgeClassifier must be fitted before calling predict.")

        is_single = X.ndim == 1
        X_in = X[np.newaxis, :] if is_single else X
        X_f32 = np.asarray(X_in, dtype=np.float32)

        # 1. Transform test/streaming features using frozen training mean and scale
        if self.normalize_features and self.scaler_ is not None:
            # Low-latency vectorized standardization using fitted statistics
            X_scaled = (X_f32 - self.scaler_.mean_) / self.scaler_.scale_
        else:
            X_scaled = X_f32

        # 2. Single forward pass: compute decision function once
        scores = self.model_.decision_function(X_scaled)

        # 3. Derive predicted class index
        if scores.ndim == 1 or scores.shape[1] == 1:
            scores_1d = scores.ravel()
            indices = (scores_1d > 0).astype(int)
            abstain_mask = self._should_abstain(scores_1d)
        else:
            indices = np.argmax(scores, axis=1)
            abstain_mask = self._should_abstain(scores)

        preds = self.model_.classes_[indices].astype(int)

        # 4. Apply abstention mask
        if self.confidence_threshold is not None:
            preds[abstain_mask] = ABSTAIN

        return int(preds[0]) if is_single else preds
