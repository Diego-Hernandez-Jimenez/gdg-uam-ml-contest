from __future__ import annotations

import warnings

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
        self.model_ = RidgeClassifierCV(alphas=self.alphas, class_weight="balanced", scoring="f1_macro")
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
    """Exact out-of-core Ridge Classifier with balanced class weights and trace-scaled alpha tuning.

    Matches scikit-learn's RidgeClassifier(class_weight='balanced') mathematically by
    accumulating the exact sample-weighted Gram matrix X^T D_w X and class sums across batches.
    """

    def __init__(
        self,
        alphas: np.ndarray | tuple[float, ...] = (0.01, 0.05, 0.2, 1.0, 5.0, 20.0, 100.0),
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
        self.class_weights_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.best_alpha_: float | None = None

        # Incremental accumulators (freed after finalize_fit)
        self._gram: np.ndarray | None = None
        self._w_sum: np.ndarray | None = None
        self._wx_sum: np.ndarray | None = None
        self._val_X: list[np.ndarray] = []
        self._val_y: list[np.ndarray] = []
        self._val_count: int = 0

    def set_class_weights_from_labels(self, y_all: np.ndarray) -> None:
        """Pre-compute exact global balanced class weights before streaming batches."""
        classes, counts = np.unique(y_all, return_counts=True)
        self.classes_ = np.sort(classes)
        if self.class_weight == "balanced":
            n_total = float(len(y_all))
            n_classes = float(len(classes))
            self.class_weights_ = (n_total / (n_classes * counts)).astype(np.float64)
        else:
            self.class_weights_ = np.ones(len(classes), dtype=np.float64)

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
        """Pass 2 (Single Epoch): Accumulate weighted Gram matrix X^T D_w X and weighted sums."""
        X_in = np.asarray(X_batch, dtype=np.float32)
        if self.normalize_features and self.scaler_ is not None:
            scale = np.where(self.scaler_.scale_ < 1e-6, 1.0, self.scaler_.scale_).astype(np.float32)
            X_in = (X_in - self.scaler_.mean_.astype(np.float32)) / scale

        if self.classes_ is None:
            self.classes_ = np.sort(np.unique(classes))
        if self.class_weights_ is None:
            # Fallback if set_class_weights_from_labels was not called beforehand
            self.set_class_weights_from_labels(y_batch)

        n_classes = len(self.classes_)
        n_features = X_in.shape[1]

        if self._gram is None:
            self._gram = np.zeros((n_features, n_features), dtype=np.float32)
            self._wx_sum = np.zeros((n_classes, n_features), dtype=np.float64)
            self._w_sum = np.zeros(n_classes, dtype=np.float64)

        # Hold out validation slice for tuning alpha
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

        self._accumulate_weighted_batch(X_in, y_batch)

    def _accumulate_weighted_batch(self, X_in: np.ndarray, y_batch: np.ndarray) -> None:
        """Accumulate exact sample-weighted X^T D_w X and class weighted sums."""
        assert (
            self._gram is not None
            and self._wx_sum is not None
            and self._w_sum is not None
            and self.classes_ is not None
            and self.class_weights_ is not None
        )
        # Map each sample's label to its class index [0..K-1]
        class_idx = np.searchsorted(self.classes_, y_batch)
        sample_w = self.class_weights_[class_idx]  # (B,)

        # Weight rows by sqrt(w_i) so X_w.T @ X_w == X_in.T @ diag(w) @ X_in
        sqrt_w = np.sqrt(sample_w, dtype=np.float32)[:, np.newaxis]
        X_w = X_in * sqrt_w
        self._gram += X_w.T @ X_w

        for k in range(len(self.classes_)):
            mask = class_idx == k
            if np.any(mask):
                w_k = self.class_weights_[k]
                count_k = int(np.sum(mask))
                self._w_sum[k] += w_k * count_k
                self._wx_sum[k] += w_k * X_in[mask].sum(axis=0, dtype=np.float64)

    def finalize_fit(self) -> IncrementalRidgeClassifier:
        """Solve exact weighted system via Cholesky and free memory."""
        if (
            self._gram is None
            or self._wx_sum is None
            or self._w_sum is None
            or self.classes_ is None
        ):
            raise ValueError("No batches were accumulated before calling finalize_fit.")

        # 1. Select optimal trace-scaled alpha on held-out validation buffer
        if len(self.alphas) > 1 and len(self._val_X) > 0:
            X_val = np.concatenate(self._val_X, axis=0)
            y_val = np.concatenate(self._val_y, axis=0)
            self._val_X.clear()
            self._val_y.clear()

            w_total_tr = float(np.sum(self._w_sum))
            x_mean_tr = (self._wx_sum.sum(axis=0) / w_total_tr).astype(np.float32)
            B_tr, y_mean_tr = self._build_rhs(w_total_tr, x_mean_tr)

            # Center Gram matrix in-place once for the validation grid search
            self._gram -= w_total_tr * np.outer(x_mean_tr, x_mean_tr)
            mean_diag = max(float(np.trace(self._gram)) / self._gram.shape[0], 1.0)

            best_score = -np.inf
            best_alpha = float(self.alphas[0])

            for rel_alpha in self.alphas:
                eff_alpha = float(rel_alpha) * mean_diag
                coef, intercept = self._solve_centered(
                    eff_alpha, B_tr, x_mean_tr, y_mean_tr, overwrite_gram=False
                )
                scores = X_val @ coef.T + intercept
                preds = self.classes_[np.argmax(scores, axis=1)]
                # Macro F1 aligns directly with per-class balanced accuracy / geomean!
                score = float(f1_score(y_val, preds, average="macro"))
                if score > best_score:
                    best_score = score
                    best_alpha = float(rel_alpha)

            # Restore uncentered Gram matrix and fold validation data back in
            self._gram += w_total_tr * np.outer(x_mean_tr, x_mean_tr)
            self.best_alpha_ = best_alpha
            self._accumulate_weighted_batch(X_val, y_val)
            del X_val, y_val
        else:
            self.best_alpha_ = float(self.alphas[0])

        # 2. Final solve on 100% of accumulated data
        w_total = float(np.sum(self._w_sum))
        x_mean = (self._wx_sum.sum(axis=0) / w_total).astype(np.float32)
        B, y_mean = self._build_rhs(w_total, x_mean)

        self._gram -= w_total * np.outer(x_mean, x_mean)
        mean_diag_full = max(float(np.trace(self._gram)) / self._gram.shape[0], 1.0)
        eff_best_alpha = self.best_alpha_ * mean_diag_full

        self.coef_, self.intercept_ = self._solve_centered(
            eff_best_alpha, B, x_mean, y_mean, overwrite_gram=True
        )

        # 3. Free accumulators
        self._gram = None
        self._wx_sum = None
        self._w_sum = None
        return self

    def _build_rhs(
        self, w_total: float, x_mean: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct centered RHS matrix B = X^T D_w (Y - y_mean) for Y in {-1, +1}."""
        assert self._wx_sum is not None and self._w_sum is not None and self.classes_ is not None
        n_classes = len(self.classes_)
        n_features = x_mean.shape[0]

        B = np.empty((n_features, n_classes), dtype=np.float32)
        y_mean = np.empty(n_classes, dtype=np.float32)
        total_wx_sum = self._wx_sum.sum(axis=0)

        for k in range(n_classes):
            # In {-1, +1} encoding: y_ik = +1 if i in class k else -1
            # Therefore: sum_i w_i x_i y_ik = 2 * sum_{i in k} w_i x_i - sum_all w_i x_i
            xty_k = (2.0 * self._wx_sum[k] - total_wx_sum).astype(np.float32)
            ym_k = float((2.0 * self._w_sum[k] - w_total) / w_total)
            y_mean[k] = ym_k
            B[:, k] = xty_k - w_total * x_mean * ym_k

        return B, y_mean

    def _solve_centered(
        self,
        eff_alpha: float,
        B: np.ndarray,
        x_mean: np.ndarray,
        y_mean: np.ndarray,
        overwrite_gram: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve (G_centered + eff_alpha * I) W = B."""
        assert self._gram is not None
        n_features = self._gram.shape[0]
        self._gram.flat[:: n_features + 1] += np.float32(eff_alpha)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", scipy.linalg.LinAlgWarning)
            try:
                W = scipy.linalg.solve(
                    self._gram,
                    B,
                    assume_a="pos",
                    overwrite_a=overwrite_gram,
                    overwrite_b=False,
                )
            except scipy.linalg.LinAlgError:
                W = scipy.linalg.solve(
                    self._gram,
                    B,
                    assume_a="sym",
                    overwrite_a=overwrite_gram,
                    overwrite_b=False,
                )
            finally:
                if not overwrite_gram:
                    self._gram.flat[:: n_features + 1] -= np.float32(eff_alpha)

        coef = W.T.astype(np.float32)
        intercept = (y_mean - coef @ x_mean).astype(np.float32)
        return coef, intercept

    def fit(
        self, X: np.ndarray, y: np.ndarray, batch_size: int = 4096
    ) -> IncrementalRidgeClassifier:
        """Standard fit on a full feature matrix (processes in slices to save RAM)."""
        self.set_class_weights_from_labels(y)
        classes = self.classes_
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
        if self.coef_ is None or self.intercept_ is None or self.classes_ is None:
            raise ValueError("IncrementalRidgeClassifier must be fitted before calling predict.")

        is_single = X.ndim == 1
        X_in = X[np.newaxis, :] if is_single else X
        X_f32 = np.asarray(X_in, dtype=np.float32)

        if self.normalize_features and self.scaler_ is not None:
            scale = np.where(self.scaler_.scale_ < 1e-6, 1.0, self.scaler_.scale_).astype(np.float32)
            X_scaled = (X_f32 - self.scaler_.mean_.astype(np.float32)) / scale
        else:
            X_scaled = X_f32

        scores = X_scaled @ self.coef_.T + self.intercept_
        indices = np.argmax(scores, axis=1)
        preds = self.classes_[indices].astype(int)

        if self.confidence_threshold is not None:
            abstain_mask = self._should_abstain(scores)
            preds[abstain_mask] = ABSTAIN

        return int(preds[0]) if is_single else preds
