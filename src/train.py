from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.classifiers.base import ABSTAIN
from src.classifiers.classifiers import DummyAbstainClassifier, RidgeClassifier
from src.features.identity import IdentityFeatures
from src.features.minirocket import MiniRocket
from src.features.multirocket import MultiRocket
from src.performance import (
    accuracy_all,
    accuracy_per_class,
    confusion_matrix,
    geomean_score,
)
from src.pipeline import TimeSeriesPipeline
from src.preprocessing.steps import (
    HampelFilter,
    Identity,
    NaNInterpolation,
    RobustScale,
)

# ==============================================================================
# 1. MODEL PRESETS (Clean & explicit factory functions, no complex registry)
# ==============================================================================


def build_model(
    model_name: str,
    num_features: int = 10_000,
    confidence_threshold: float | None = None,
    noise_threshold: float | None = None,
) -> TimeSeriesPipeline:
    """Build a TimeSeriesPipeline instance by preset name and hyperparameters."""
    presets: dict[str, Callable[[], TimeSeriesPipeline]] = {
        "dummy": lambda: TimeSeriesPipeline(
            preprocessing_steps=[Identity()],
            feature_extractor=IdentityFeatures(),
            classifier=DummyAbstainClassifier(),
        ),
        "minirocket_raw": lambda: TimeSeriesPipeline(
            preprocessing_steps=[NaNInterpolation()],
            feature_extractor=MiniRocket(num_features=num_features),
            classifier=RidgeClassifier(confidence_threshold=confidence_threshold),
            noise_threshold=noise_threshold,
        ),
        "minirocket_robust": lambda: TimeSeriesPipeline(
            preprocessing_steps=[
                NaNInterpolation(),
                HampelFilter(window=5, n_sigmas=3.0),
                RobustScale(),
            ],
            feature_extractor=MiniRocket(num_features=num_features),
            classifier=RidgeClassifier(confidence_threshold=confidence_threshold),
            noise_threshold=noise_threshold,
        ),
        "multirocket_raw": lambda: TimeSeriesPipeline(
            preprocessing_steps=[NaNInterpolation()],
            feature_extractor=MultiRocket(num_features=num_features),
            classifier=RidgeClassifier(confidence_threshold=confidence_threshold),
            noise_threshold=noise_threshold,
        ),
        "multirocket_robust": lambda: TimeSeriesPipeline(
            preprocessing_steps=[
                NaNInterpolation(),
                HampelFilter(window=5, n_sigmas=3.0),
                RobustScale(),
            ],
            feature_extractor=MultiRocket(num_features=num_features),
            classifier=RidgeClassifier(confidence_threshold=confidence_threshold),
            noise_threshold=noise_threshold,
        ),
    }
    if model_name not in presets:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Available: {list(presets.keys())}"
        )
    return presets[model_name]()


# ==============================================================================
# 2. METRICS COMPUTATION (Global accuracy, per-class accuracy, abstention stats)
# ==============================================================================


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 8
) -> dict:
    """Compute global metrics, official score, and per-class metrics."""
    total_samples = len(y_true)
    abstained_mask = y_pred == -1
    num_abstained = int(np.sum(abstained_mask))
    num_covered = total_samples - num_abstained

    # Accuracy on non-abstained (covered) samples
    if num_covered > 0:
        accuracy_on_covered = float(
            np.mean(y_true[~abstained_mask] == y_pred[~abstained_mask])
        )
    else:
        accuracy_on_covered = 0.0

    # Per-class metrics
    per_class_acc = accuracy_per_class(y_true, y_pred, n_classes=num_classes)
    per_class = {}
    for c in range(num_classes):
        class_mask = y_true == c
        class_total = int(np.sum(class_mask))
        if class_total == 0:
            continue

        class_preds = y_pred[class_mask]
        class_abstained = int(np.sum(class_preds == -1))

        per_class[str(c)] = {
            "total_samples": class_total,
            "abstained_samples": class_abstained,
            "abstention_rate": float(class_abstained / class_total),
            "accuracy_all": float(per_class_acc[c])
            if per_class_acc[c] is not None
            else 0.0,
        }

    return {
        "total_samples": total_samples,
        "abstained_samples": num_abstained,
        "abstention_rate": float(num_abstained / total_samples)
        if total_samples > 0
        else 0.0,
        "coverage": float(num_covered / total_samples) if total_samples > 0 else 0.0,
        "global_accuracy_all": accuracy_all(y_true, y_pred),
        "accuracy_on_covered": accuracy_on_covered,
        "official_geomean_score": geomean_score(y_true, y_pred, n_classes=num_classes),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, n_classes=num_classes
        ).tolist(),
        "per_class": per_class,
    }


# ==============================================================================
# 3. TRAINING & EVALUATION WORKFLOW (With or without Cross-Validation)
# ==============================================================================


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    cv_folds: int = 0,
    num_features: int = 10_000,
    confidence_threshold: float | None = None,
    noise_threshold: float | None = None,
    artifacts_dir: str | Path = "artifacts",
) -> tuple[Path, Path]:
    """Run optional CV evaluation, fit final model on full data, and export artifacts."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_path = Path(artifacts_dir)
    metrics_dir = artifacts_path / "metrics"
    artifacts_path.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cross-Validation (if cv_folds > 1) using Out-Of-Fold predictions
    cv_metrics = None
    if cv_folds and cv_folds > 1:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        oof_preds = np.full(len(y), ABSTAIN, dtype=int)

        for train_idx, val_idx in skf.split(X, y):
            fold_model = build_model(
                model_name=model_name,
                num_features=num_features,
                confidence_threshold=confidence_threshold,
                noise_threshold=noise_threshold,
            )
            fold_model.fit(X[train_idx], y[train_idx])
            oof_preds[val_idx] = fold_model.predict(X[val_idx])

        cv_metrics = compute_metrics(y, oof_preds)

    # 2. Fit final model on 100% of the dataset
    final_model = build_model(
        model_name=model_name,
        num_features=num_features,
        confidence_threshold=confidence_threshold,
        noise_threshold=noise_threshold,
    )
    final_model.fit(X, y)
    train_preds = final_model.predict(X)
    train_metrics = compute_metrics(y, train_preds)

    # 3. Save model artifacts (.joblib)
    params_path = artifacts_path / f"params_{model_name}_{timestamp}.joblib"
    latest_model_path = artifacts_path / f"latest_{model_name}.joblib"
    latest_default_path = artifacts_path / "latest.joblib"

    joblib.dump(final_model, params_path)
    joblib.dump(final_model, latest_model_path)
    joblib.dump(final_model, latest_default_path)

    # 4. Save metrics artifact (.json)
    metrics_payload = {
        "model_name": model_name,
        "timestamp": timestamp,
        "config": {
            "cv_folds": cv_folds,
            "num_features": num_features,
            "confidence_threshold": confidence_threshold,
            "noise_threshold": noise_threshold,
        },
        "train_metrics": train_metrics,
        "cv_oof_metrics": cv_metrics,
    }
    metrics_path = metrics_dir / f"metrics_{model_name}_{timestamp}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    return params_path, metrics_path


# ==============================================================================
# 4. CLI ENTRY POINT
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train Time-Series Classification Pipeline."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training .npz file (keys: X, y).",
    )
    parser.add_argument(
        "--model-name", type=str, default="minirocket_robust", help="Model preset name."
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of StratifiedKFold splits (0 or 1 to disable).",
    )
    parser.add_argument(
        "--num-features", type=int, default=10_000, help="Number of Rocket features."
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Classifier decision margin threshold.",
    )
    parser.add_argument(
        "--noise-threshold",
        type=float,
        default=None,
        help="Residual noise std(diff) threshold.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Output directory for artifacts.",
    )
    args = parser.parse_args()

    with np.load(args.data) as loader:
        # Support both (X, y) and (series, labels) key conventions
        x_key = (
            "X"
            if "X" in loader.files
            else ("series" if "series" in loader.files else loader.files[0])
        )
        y_key = (
            "y"
            if "y" in loader.files
            else ("labels" if "labels" in loader.files else loader.files[1])
        )
        X = loader[x_key]
        y = loader[y_key]

    params_path, metrics_path = train_and_evaluate(
        X=X,
        y=y,
        model_name=args.model_name,
        cv_folds=args.cv_folds,
        num_features=args.num_features,
        confidence_threshold=args.confidence_threshold,
        noise_threshold=args.noise_threshold,
        artifacts_dir=args.artifacts_dir,
    )
    print(f"Saved model parameters to: {params_path}")
    print(f"Saved evaluation metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
