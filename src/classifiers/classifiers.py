from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeClassifierCV, SGDClassifier
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
        normalize_features: bool = False,
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


class SGDLinearClassifier(Classifier):
    """Memory-efficient linear classifier using Stochastic Gradient Descent.
    
    Supports both standard in-memory fit() and out-of-core mini-batch training
    to prevent Out-Of-Memory (OOM) crashes on large datasets (e.g. 100k x 10k).
    """

    def __init__(
        self,
        loss: str = "log_loss",
        alpha: float = 1e-4,
        max_iter: int = 1000,
        confidence_threshold: float | None = None,
        normalize_features: bool = False,
        random_state: int = 42,
    ):
        """Initialize the SGDLinearClassifier.
        
        Args:
            loss: 'log_loss' (Logistic Regression) or 'hinge' (Linear SVM).
            alpha: L2 regularization strength (higher = more regularization).
            max_iter: Maximum epochs for standard in-memory fit().
            confidence_threshold: Minimum margin between top-1 and top-2 scores
                (or probabilities if log_loss) before returning ABSTAIN (-1).
            normalize_features: If True, standardizes features using StandardScaler.
            random_state: Seed for reproducible shuffling.
        """
        self.loss = loss
        self.alpha = alpha
        self.max_iter = max_iter
        self.confidence_threshold = confidence_threshold
        self.normalize_features = normalize_features
        self.random_state = random_state
        self.scaler_: StandardScaler | None = None
        self.model_: SGDClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> SGDLinearClassifier:
        """Standard fit on a feature matrix X of shape (N, F)."""
        X_in = np.asarray(X, dtype=np.float32)

        if self.normalize_features:
            self.scaler_ = StandardScaler(copy=False)
            X_train = self.scaler_.fit_transform(X_in)
        else:
            self.scaler_ = None
            X_train = X_in

        self.model_ = SGDClassifier(
            loss=self.loss,
            penalty="l2",
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model_.fit(X_train, y)
        return self

    def partial_fit_scaler(self, X_batch: np.ndarray) -> None:
        """Incrementally update feature mean and variance on a mini-batch."""
        if not self.normalize_features:
            return
        if self.scaler_ is None:
            self.scaler_ = StandardScaler()
        self.scaler_.partial_fit(np.asarray(X_batch, dtype=np.float32))

    def partial_fit_model(
        self, X_batch: np.ndarray, y_batch: np.ndarray, classes: np.ndarray
    ) -> None:
        """Incrementally update SGD weights on a standardized mini-batch."""
        X_in = np.asarray(X_batch, dtype=np.float32)
        if self.normalize_features and self.scaler_ is not None:
            X_in = (X_in - self.scaler_.mean_) / self.scaler_.scale_

        if self.model_ is None:
            self.model_ = SGDClassifier(
                loss=self.loss,
                penalty="l2",
                alpha=self.alpha,
                random_state=self.random_state,
                n_jobs=-1,
            )
        self.model_.partial_fit(X_in, y_batch, classes=classes)

    def _should_abstain(self, scores: np.ndarray) -> np.ndarray:
        """Determine whether to abstain based on top-1 vs top-2 margin."""
        if self.confidence_threshold is None:
            return np.zeros(scores.shape[0], dtype=bool)

        if scores.ndim == 1 or scores.shape[1] == 1:
            margin = np.abs(scores.ravel())
        else:
            part = np.partition(scores, -2, axis=1)
            margin = part[:, -1] - part[:, -2]

        return margin < self.confidence_threshold

    def predict(self, X: np.ndarray) -> int | np.ndarray:
        """Low-latency prediction identical in contract to RidgeClassifier."""
        if self.model_ is None:
            raise ValueError("SGDLinearClassifier must be fitted before calling predict.")

        is_single = X.ndim == 1
        X_in = X[np.newaxis, :] if is_single else X
        X_f32 = np.asarray(X_in, dtype=np.float32)

        if self.normalize_features and self.scaler_ is not None:
            X_scaled = (X_f32 - self.scaler_.mean_) / self.scaler_.scale_
        else:
            X_scaled = X_f32

        if self.confidence_threshold is None:
            preds = self.model_.predict(X_scaled).astype(int)
            return int(preds[0]) if is_single else preds

        scores = self.model_.decision_function(X_scaled)
        if scores.ndim == 1 or scores.shape[1] == 1:
            scores_1d = scores.ravel()
            indices = (scores_1d > 0).astype(int)
            abstain_mask = self._should_abstain(scores_1d)
        else:
            indices = np.argmax(scores, axis=1)
            abstain_mask = self._should_abstain(scores)

        preds = self.model_.classes_[indices].astype(int)
        preds[abstain_mask] = ABSTAIN

        return int(preds[0]) if is_single else preds
