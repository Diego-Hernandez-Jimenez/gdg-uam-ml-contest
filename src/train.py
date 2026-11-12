from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.classifiers.base import ABSTAIN
from src.training.config import TrainConfig, build_pipeline
from src.training.performance import (
    accuracy_all,
    accuracy_per_class,
    confusion_matrix,
    geomean_score,
)

# ==============================================================================
# 1. METRICS COMPUTATION (Global accuracy, per-class accuracy, abstention stats)
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
# 2. TRAINING & EVALUATION WORKFLOW (With or without Cross-Validation)
# ==============================================================================


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    config: TrainConfig,
) -> tuple[Path, Path]:
    """Run optional CV evaluation, fit final model on full data, and export artifacts."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_path = Path(config.artifacts_dir)
    params_dir = artifacts_path / "parameters"
    metrics_dir = artifacts_path / "metrics"
    artifacts_path.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    params_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cross-Validation (if cv_folds > 1) using Out-Of-Fold predictions
    cv_metrics = None
    if config.training.cv_folds and config.training.cv_folds > 1:
        skf = StratifiedKFold(
            n_splits=config.training.cv_folds, shuffle=True, random_state=42
        )
        oof_preds = np.full(len(y), ABSTAIN, dtype=int)
        for train_idx, val_idx in skf.split(X, y):
            fold_model = build_pipeline(config)
            fold_model.fit(
                X[train_idx],
                y[train_idx],
                batch_size=config.training.batch_size,
                n_epochs=config.training.epochs,
            )
            oof_preds[val_idx] = fold_model.predict(X[val_idx])
        cv_metrics = compute_metrics(y, oof_preds)

    # 2. Fit final model on 100% of the dataset
    final_model = build_pipeline(config)
    final_model.fit(
        X, y, batch_size=config.training.batch_size, n_epochs=config.training.epochs
    )

    train_preds = final_model.predict(X)
    train_metrics = compute_metrics(y, train_preds)

    # 3. Save model artifacts (.joblib)
    params_path = params_dir / f"{config.name}_{timestamp}.joblib"
    latest_model_path = artifacts_path / f"latest_{config.name}.joblib"
    latest_default_path = artifacts_path / "latest.joblib"

    joblib.dump(final_model, params_path)
    joblib.dump(final_model, latest_model_path)
    joblib.dump(final_model, latest_default_path)

    # 4. Save metrics artifact (.json)
    metrics_payload = {
        "model_name": config.name,
        "timestamp": timestamp,
        "config": config.to_dict(),
        "train_metrics": train_metrics,
        "cv_oof_metrics": cv_metrics,
    }
    metrics_path = metrics_dir / f"metrics_{config.name}_{timestamp}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    return params_path, metrics_path


# ==============================================================================
# 3. CLI ENTRY POINT
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
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)

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

    params_path, metrics_path = train_and_evaluate(X=X, y=y, config=config)
    print(f"Saved model parameters to: {params_path}")
    print(f"Saved evaluation metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
