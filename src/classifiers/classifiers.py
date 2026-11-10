from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeClassifierCV

from .base import ABSTAIN, Classifier


class DummyAbstainClassifier(Classifier):
    """Trivial classifier that always predicts ABSTAIN (-1).
    
    Useful as a baseline, fallback, or for pipeline smoke testing.
    """

    def predict(self, X: np.ndarray) -> int | np.ndarray:
        if X.ndim == 1:
            return int(ABSTAIN)
        return np.full(X.shape[0], ABSTAIN, dtype=int)


class RidgeClassifier(Classifier):
    """Ridge classifier wrapper with built-in abstention logic.
    
    Uses scikit-learn's RidgeClassifierCV under the hood, which is the 
    standard linear head for Rocket-family feature extractors.
    """

    def __init__(
        self,
        alphas: np.ndarray | tuple[float, ...] = np.logspace(-3, 3, 10),
        confidence_threshold: float | None = None,
    ):
        """Initialize the RidgeClassifier.
        
        Args:
            alphas: Array of alpha values to try during cross-validation.
            confidence_threshold: Minimum decision margin required to emit a 
                class prediction. If the margin falls below this threshold, 
                the classifier returns ABSTAIN (-1). If None, never abstains.
        """
        self.alphas = alphas
        self.confidence_threshold = confidence_threshold
        self.model_: RidgeClassifierCV | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeClassifier:
        """Fit the underlying RidgeClassifierCV model on features (N, F)."""
        self.model_ = RidgeClassifierCV(alphas=self.alphas)
        self.model_.fit(X, y)
        return self

    def _should_abstain(self, scores: np.ndarray) -> np.ndarray:
        """Determine whether to abstain for each sample based on raw decision scores.
        
        Can be overridden or extended with custom abstention heuristics 
        (e.g., max score threshold, top-1 vs top-2 margin, or entropy).
        
        Args:
            scores: Output from decision_function of shape (N,) for binary 
                    or (N, n_classes) for multiclass.
                    
        Returns:
            Boolean mask of shape (N,) where True indicates ABSTAIN.
        """
        if self.confidence_threshold is None:
            return np.zeros(scores.shape[0], dtype=bool)

        # Binary classification case: scores has shape (N,)
        if scores.ndim == 1:
            margin = np.abs(scores)
        # Multiclass case: compute margin between top-1 and top-2 logits
        else:
            part = np.partition(scores, -2, axis=1)
            margin = part[:, -1] - part[:, -2]

        return margin < self.confidence_threshold

    def predict(self, X: np.ndarray) -> int | np.ndarray:
        """Predict class labels or ABSTAIN (-1) with low-latency overhead."""
        if self.model_ is None:
            raise ValueError("RidgeClassifier must be fitted before calling predict.")

        is_single = X.ndim == 1
        X_in = X[np.newaxis, :] if is_single else X

        # Fast path when no abstention logic is active
        if self.confidence_threshold is None:
            preds = self.model_.predict(X_in).astype(int)
            return int(preds[0]) if is_single else preds

        # Compute raw decision scores once and derive both predictions and abstention
        scores = self.model_.decision_function(X_in)
        if scores.ndim == 1:
            indices = (scores > 0).astype(int)
        else:
            indices = np.argmax(scores, axis=1)

        preds = self.model_.classes_[indices].astype(int)
        abstain_mask = self._should_abstain(scores)
        preds[abstain_mask] = ABSTAIN

        return int(preds[0]) if is_single else preds
