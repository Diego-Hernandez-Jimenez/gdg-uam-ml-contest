from __future__ import annotations

import numpy as np

try:
    from competition_api import ABSTAIN
except ImportError:
    ABSTAIN = -1


class Classifier:
    """Base interface for all classifiers in the ML pipeline.

    Follows a scikit-learn-like contract while supporting both batch
    evaluation (N, F) and low-latency single-sample inference (F,).
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> Classifier:
        """Fit the classifier on feature matrix X of shape (N, F) and labels y of shape (N,)."""
        return self

    def predict(self, X: np.ndarray) -> int | np.ndarray:
        """Predict class label(s) in 0..7 or ABSTAIN (-1).

        Args:
            X: Feature vector of shape (F,) for single-sample inference,
               or feature matrix of shape (N, F) for batch evaluation.

        Returns:
            int if input is 1D, or np.ndarray of ints if input is 2D.
        """
        raise NotImplementedError
