from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.classifiers.classifiers import (
    DummyAbstainClassifier,
    RidgeClassifier,
    SGDLinearClassifier,
)
from src.features.identity import IdentityFeatures
from src.features.minirocket import MiniRocket
from src.features.multirocket import MultiRocket
from src.preprocessing.steps import (
    HampelFilter,
    Identity,
    NaNInterpolation,
    RobustScale,
)
from src.training.pipeline import TimeSeriesPipeline

PREPROCESSING_MAP: dict[str, type] = {
    "identity": Identity,
    "nan_interpolation": NaNInterpolation,
    "hampel": HampelFilter,
    "robust_scale": RobustScale,
}

FEATURE_MAP: dict[str, type] = {
    "identity": IdentityFeatures,
    "minirocket": MiniRocket,
    "multirocket": MultiRocket,
}

CLASSIFIER_MAP: dict[str, type] = {
    "dummy": DummyAbstainClassifier,
    "ridge": RidgeClassifier,
    "sgd": SGDLinearClassifier,
}


@dataclass
class TrainConfig:
    """Configuration contract for training pipeline and workflow."""

    name: str
    preprocessing: list[str | dict[str, Any]] = field(
        default_factory=lambda: ["nan_interpolation"]
    )
    feature_extractor: str = "minirocket"
    num_features: int = 10_000
    classifier: str = "ridge"
    confidence_threshold: float | None = None
    normalize_features: bool = False
    noise_threshold: float | None = None
    cv_folds: int | None = None
    batch_size: int | None = None
    epochs: int = 3
    artifacts_dir: str = "artifacts"

    def __post_init__(self) -> None:
        # Validate SGD-exclusive training parameters
        if self.classifier != "sgd" and (
            self.batch_size is not None or self.epochs != 3
        ):
            raise ValueError(
                "batch_size and epochs are only valid with classifier 'sgd'."
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        """Load configuration from a YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to dictionary."""
        return asdict(self)


def build_pipeline(config: TrainConfig) -> TimeSeriesPipeline:
    """Instantiate a TimeSeriesPipeline from a TrainConfig contract."""
    steps = []
    for step in config.preprocessing:
        if isinstance(step, str):
            steps.append(PREPROCESSING_MAP[step]())
        elif isinstance(step, dict):
            name, params = next(iter(step.items()))
            steps.append(PREPROCESSING_MAP[name](**(params or {})))

    fe_cls = FEATURE_MAP[config.feature_extractor]
    feature_extractor = (
        fe_cls()
        if config.feature_extractor == "identity"
        else fe_cls(num_features=config.num_features)
    )

    clf_cls = CLASSIFIER_MAP[config.classifier]
    if config.classifier == "dummy":
        classifier = clf_cls()
    elif config.classifier == "ridge":
        classifier = clf_cls(
            confidence_threshold=config.confidence_threshold,
            normalize_features=config.normalize_features,
        )
    else:
        classifier = clf_cls(confidence_threshold=config.confidence_threshold)

    return TimeSeriesPipeline(
        preprocessing_steps=steps,
        feature_extractor=feature_extractor,
        classifier=classifier,
        noise_threshold=config.noise_threshold,
    )
