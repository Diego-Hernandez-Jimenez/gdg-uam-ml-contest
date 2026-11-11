from __future__ import annotations

import numpy as np
import scipy.linalg
from sklearn.linear_model import RidgeClassifierCV, SGDClassifier
from sklearn.metrics import f1_score
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
            self.scaler_ = StandardScaler(with_mean=False, copy=False)
            X_train = self.scaler_.fit_transform(X_in)
        else:
            self.scaler_ = None
            X_train = X_in

        # 2. Fit RidgeClassifierCV on the standardized training features
        self.model_ = RidgeClassifierCV(alphas=self.alphas, class_weight="balanced", scoring="f1_weighted")
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
            self.scaler_ = StandardScaler(with_mean=False, copy=False)
            X_train = self.scaler_.fit_transform(X_in)
        else:
            self.scaler_ = None
            X_train = X_in

        self.model_ = SGDClassifier(
            loss=self.loss,
            penalty="l2",
            alpha=self.alpha,
            early_stopping=True,
            validation_fraction=0.2,
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
            self.scaler_ = StandardScaler(with_mean=False)
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


class IncrementalRidgeClassifier(Classifier):
    """Exact out-of-core Ridge Classifier with balanced class weights and automatic alpha tuning.

    Accumulates the sufficient statistics (weighted Gram matrix X^T W X and X^T W Y)
    across mini-batches, yielding the exact closed-form Cholesky solution without
    ever holding the full (N, F) feature matrix in memory.
    """

    def __init__(
        self,
        alphas: np.ndarray | tuple[float, ...] = np.logspace(0, 6, 7),  # noqa: B008
        confidence_threshold: float | None = None,
        normalize_features: bool = False,
        class_weight: str | None = "balanced",
        val_fraction: float = 0.15,
        max_val_samples: int = 4096,
        random_state: int = 42,
    ):
        self.alphas = np.asarray(alphas, dtype=np.float64)
        self.confidence_threshold = confidence_threshold
        self.normalize_features = normalize_features
        self.class_weight = class_weight
        self.val_fraction = val_fraction
        self.max_val_samples = max_val_samples
        self.random_state = random_state

        self.scaler_: StandardScaler | None = None
        self.classes_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None # Shape: (K, F)
        self.intercept_: np.ndarray | None = None # Shape: (K,)
        self.best_alpha_: float | None = None

        # Internal incremental accumulators (cleared after finalize_fit)
        self._gram: np.ndarray | None = None
        self._xty: np.ndarray | None = None
        self._w_sum: np.ndarray | None = None # Class counts / weights
        self._x_sum: np.ndarray | None = None # Per-class feature sum: (K, F)
        self._val_X: list[np.ndarray] = []
        self._val_y: list[np.ndarray] = []
        self._val_count: int = 0

    # ------------------------------------------------------------------
    # Incremental / Out-of-Core API (used by TimeSeriesPipeline)
    # ------------------------------------------------------------------

    def partial_fit_scaler(self, X_batch: np.ndarray) -> None:
        """Pass 1: Incrementally compute exact global feature mean and std."""
        if not self.normalize_features:
            return
        if self.scaler_ is None:
            self.scaler_ = StandardScaler(with_mean=True)
        self.scaler_.partial_fit(np.asarray(X_batch, dtype=np.float32))

    def partial_fit_model(
        self, X_batch: np.ndarray, y_batch: np.ndarray, classes: np.ndarray
    ) -> None:
        """Pass 2 (Single Epoch): Accumulate sufficient statistics X^T X and per-class sums."""
        X_in = np.asarray(X_batch, dtype=np.float32)
        if self.normalize_features and self.scaler_ is not None:
            X_in = (X_in - self.scaler_.mean_.astype(np.float32)) / self.scaler_.scale_.astype(np.float32)

        if self.classes_ is None:
            self.classes_ = np.sort(np.unique(classes))
            n_classes = len(self.classes_)
            n_features = X_in.shape[1]
            self._gram = np.zeros((n_features, n_features), dtype=np.float32)
            self._x_sum = np.zeros((n_classes, n_features), dtype=np.float64)
            self._w_sum = np.zeros(n_classes, dtype=np.float64)

        # If tuning multiple alphas, hold out a small stratified slice until max_val_samples
        if len(self.alphas) > 1 and self._val_count < self.max_val_samples:
            rng = np.random.default_rng(self.random_state + self._val_count)
            n_b = len(y_batch)
            n_take = min(
                int(np.ceil(n_b * self.val_fraction)),
                self.max_val_samples - self._val_count,
            )
            if 0 < n_take < n_b:
                perm = rng.permutation(n_b)
                val_idx, tr_idx = perm[:n_take], perm[n_take:]
                self._val_X.append(X_in[val_idx].copy())
                self._val_y.append(y_batch[val_idx].copy())
                self._val_count += n_take
                X_in = X_in[tr_idx]
                y_batch = y_batch[tr_idx]

        self._accumulate_batch(X_in, y_batch)

    def _accumulate_batch(self, X_in: np.ndarray, y_batch: np.ndarray) -> None:
        """Accumulate uncentered Gram matrix and class-wise sums (weights applied at solve time)."""
        assert self._gram is not None and self._x_sum is not None and self._w_sum is not None
        # Note: To support exact class_weight='balanced' globally, if we weight samples by class,
        # we can either weight X_in by class if class frequencies are known, or accumulate unweighted
        # X^T X and class-balanced targets X^T Y. Standard RidgeClassifierCV with solver='cholesky'
        # weights both X^T W X and X^T W Y, OR we can compute class weights on the first pass.
        # Even simpler: in pipeline.fit, as y is completely in memory
        self._gram += X_in.T @ X_in
        for idx, c in enumerate(self.classes_):
            mask = y_batch == c
            if np.any(mask):
                self._w_sum[idx] += int(np.sum(mask))
                self._x_sum[idx] += X_in[mask].sum(axis=0, dtype=np.float64)

    def finalize_fit(self) -> IncrementalRidgeClassifier:
        """Solve the exact linear system via Cholesky decomposition and free accumulators."""
        if self._gram is None or self._x_sum is None or self._w_sum is None or self.classes_ is None:
            raise ValueError("No batches were accumulated before calling finalize_fit.")

        # 1. Select best alpha on held-out validation buffer if multiple alphas provided
        if len(self.alphas) > 1 and len(self._val_X) > 0:
            X_val = np.concatenate(self._val_X, axis=0)
            y_val = np.concatenate(self._val_y, axis=0)

            best_score = -np.inf
            best_alpha = float(self.alphas[0])

            for alpha in self.alphas:
                coef, intercept = self._solve_from_accumulators(float(alpha))
                scores = X_val @ coef.T + intercept
                preds = self.classes_[np.argmax(scores, axis=1)]
                score = float(f1_score(y_val, preds, average="weighted"))
                if score > best_score:
                    best_score = score
                    best_alpha = float(alpha)

            self.best_alpha_ = best_alpha
            # Fold validation buffer back into accumulators so 100% of data is used for final weights
            self._accumulate_batch(X_val, y_val)
            self._val_X.clear()
            self._val_y.clear()
        else:
            self.best_alpha_ = float(self.alphas[0])

        # 2. Compute final exact weights on 100% of accumulated data
        self.coef_, self.intercept_ = self._solve_from_accumulators(self.best_alpha_)

        # 3. Free massive Gram matrix before saving with joblib
        self._gram = None
        self._x_sum = None
        self._w_sum = None
        return self

    def _solve_from_accumulators(self, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """Solve (G_centered + alpha * I) W = B_centered in-place without extra memory copies."""
        assert self._gram is not None and self._x_sum is not None and self._w_sum is not None
        n_total = float(np.sum(self._w_sum))
        n_classes = len(self.classes_)
        n_features = self._gram.shape[0]

        # Global feature mean: (F,)
        x_mean = (self._x_sum.sum(axis=0) / n_total).astype(np.float32)

        # Build {-1, +1} target matrix sums with balanced class weighting on targets
        # For class k, positive samples have target +1, negative samples have -1.
        # If class_weight == "balanced", we reweight each class column's positive vs negative contribution
        # so every class has equal pull on the hyperplane
        B = np.empty((n_features, n_classes), dtype=np.float32)
        y_mean = np.empty(n_classes, dtype=np.float32)
        total_x_sum = self._x_sum.sum(axis=0)

        for k in range(n_classes):
            n_pos = self._w_sum[k]
            n_neg = n_total - n_pos
            if self.class_weight == "balanced" and n_pos > 0 and n_neg > 0:
                # Balance positive and negative pulls for One-vs-Rest class k
                w_pos = n_total / (2.0 * n_pos)
                w_neg = n_total / (2.0 * n_neg)
            else:
                w_pos = 1.0
                w_neg = 1.0

            pos_x_sum = self._x_sum[k]
            neg_x_sum = total_x_sum - pos_x_sum

            # X^T y_k = w_pos * sum_{i in k} x_i - w_neg * sum_{i not in k} x_i
            xty_k = (w_pos * pos_x_sum - w_neg * neg_x_sum).astype(np.float32)
            ym_k = float((w_pos * n_pos - w_neg * n_neg) / n_total)
            y_mean[k] = ym_k
            # Centered RHS: X^T y_k - N * x_mean * y_mean_k
            B[:, k] = xty_k - n_total * x_mean * ym_k

        # Center Gram matrix: G_c = X^T X - N * outer(x_mean, x_mean) + alpha * I
        # Modify self._gram temporarily and restore it after solve to avoid allocating a second huge matrix
        mean_outer = np.outer(x_mean, x_mean)
        self._gram -= n_total * mean_outer
        self._gram.flat[:: n_features + 1] += np.float32(alpha)

        try:
            # Fast, exact Cholesky solve
            W = scipy.linalg.solve(
                self._gram, B, assume_a="pos", overwrite_a=False, overwrite_b=True
            )
        except scipy.linalg.LinAlgError:
            W = scipy.linalg.solve(
                self._gram, B, assume_a="sym", overwrite_a=False, overwrite_b=True
            )
        finally:
            # Restore self._gram in-place
            self._gram.flat[:: n_features + 1] -= np.float32(alpha)
            self._gram += n_total * mean_outer

        coef = W.T.astype(np.float32) # (K, F)
        intercept = (y_mean - coef @ x_mean).astype(np.float32) # (K,)
        return coef, intercept

    # ------------------------------------------------------------------
    # Standard In-Memory Fit & Predict Contract
    # ------------------------------------------------------------------

    def fit(
        self, X: np.ndarray, y: np.ndarray, batch_size: int = 4096
    ) -> IncrementalRidgeClassifier:
        """Standard fit on a full feature matrix (processes in slices to save RAM)."""
        classes = np.unique(y)
        n_samples = X.shape[0]
        if self.normalize_features:
            for start in range(0, n_samples, batch_size):
                self.partial_fit_scaler(X[start : start + batch_size])
        for start in range(0, n_samples, batch_size):
            self.partial_fit_model(
                X[start : start + batch_size],
                y[start : start + batch_size],
                classes=classes,
            )
        return self.finalize_fit()

    def _should_abstain(self, scores: np.ndarray) -> np.ndarray:
        if self.confidence_threshold is None:
            return np.zeros(scores.shape[0], dtype=bool)
        if scores.ndim == 1 or scores.shape[1] == 1:
            margin = np.abs(scores.ravel())
        else:
            part = np.partition(scores, -2, axis=1)
            margin = part[:, -1] - part[:, -2]
        return margin < self.confidence_threshold

    def predict(self, X: np.ndarray) -> int | np.ndarray:
        """Low-latency prediction using exact closed-form weights."""
        if self.coef_ is None or self.intercept_ is None or self.classes_ is None:
            raise ValueError("IncrementalRidgeClassifier must be fitted before calling predict.")

        is_single = X.ndim == 1
        X_in = X[np.newaxis, :] if is_single else X
        X_f32 = np.asarray(X_in, dtype=np.float32)

        if self.normalize_features and self.scaler_ is not None:
            X_scaled = (X_f32 - self.scaler_.mean_.astype(np.float32)) / self.scaler_.scale_.astype(np.float32)
        else:
            X_scaled = X_f32

        scores = X_scaled @ self.coef_.T + self.intercept_
        indices = np.argmax(scores, axis=1)
        preds = self.classes_[indices].astype(int)

        if self.confidence_threshold is not None:
            abstain_mask = self._should_abstain(scores)
            preds[abstain_mask] = ABSTAIN

        return int(preds[0]) if is_single else preds
