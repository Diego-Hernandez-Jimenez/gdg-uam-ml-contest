from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.classifiers.classifiers import (
    DummyAbstainClassifier,
    IncrementalRidgeClassifier,
    RidgeClassifier,
    SGDLinearClassifier,
)
from src.features.base import FeatureExtractor, FeatureUnion
from src.features.baseline_features import BaselineFeaturesExtractor
from src.features.identity import IdentityFeatures
from src.features.minirocket import MiniRocket
from src.features.multirocket import MultiRocket
from src.features.summary_stats import SummaryStatsExtractor
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
    "summary_stats": SummaryStatsExtractor,
    "baseline": BaselineFeaturesExtractor,
}

CLASSIFIER_MAP: dict[str, type] = {
    "dummy": DummyAbstainClassifier,
    "ridge": RidgeClassifier,
    "incremental_ridge": IncrementalRidgeClassifier,
    "sgd": SGDLinearClassifier,
}


@dataclass
class TrainConfig:
    """Configuration contract for training pipeline and workflow."""

    name: str
    preprocessing: list[str | dict[str, Any]] = field(
        default_factory=lambda: ["nan_interpolation"]
    )
    feature_extractor: str | list[str | dict[str, Any]] = "minirocket"
    num_features: int = 1000
    classifier: str = "ridge"
    confidence_threshold: float | None = None
    normalize_features: bool = False
    noise_threshold: float | None = None
    cv_folds: int | None = None
    batch_size: int | None = None
    epochs: int = 3
    artifacts_dir: str = "artifacts"

    def __post_init__(self) -> None:
        if self.classifier not in ("sgd", "incremental_ridge", "ridge") and (
            self.batch_size is not None or self.epochs != 3
        ):
            raise ValueError(
                "batch_size and epochs are only valid with 'sgd', 'ridge', or 'incremental_ridge'."
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


def _instantiate_single_extractor(
    spec: str | dict[str, Any], default_num_features: int
) -> FeatureExtractor:
    """Helper to instantiate a single FeatureExtractor from string or dict spec."""
    if isinstance(spec, str):
        name, params = spec, {}
    elif isinstance(spec, dict):
        name, params = next(iter(spec.items()))
        params = params or {}
    else:
        raise TypeError(f"Invalid feature extractor specification: {spec}")

    fe_cls = FEATURE_MAP[name]
    if name in ("minirocket", "multirocket") and "num_features" not in params:
        params["num_features"] = default_num_features

    return fe_cls(**params)


def build_pipeline(config: TrainConfig) -> TimeSeriesPipeline:
    """Instantiate a TimeSeriesPipeline from a TrainConfig contract."""
    # 1. Build Preprocessing Steps
    steps = []
    for step in config.preprocessing:
        if isinstance(step, str):
            steps.append(PREPROCESSING_MAP[step]())
        elif isinstance(step, dict):
            name, params = next(iter(step.items()))
            steps.append(PREPROCESSING_MAP[name](**(params or {})))

    # 2. Build Feature Extractor (Single or FeatureUnion if length > 1)
    fe_specs = (
        [config.feature_extractor]
        if isinstance(config.feature_extractor, str)
        else config.feature_extractor
    )
    extractors = [
        _instantiate_single_extractor(spec, config.num_features)
        for spec in fe_specs
    ]

    if len(extractors) > 1:
        feature_extractor: FeatureExtractor = FeatureUnion(
            extractors=extractors,
            scale_features=config.normalize_features,
        )
    else:
        feature_extractor = extractors[0]

    # 3. Build Classifier
    clf_cls = CLASSIFIER_MAP[config.classifier]
    if config.classifier == "dummy":
        classifier = clf_cls()
    elif config.classifier == "ridge":
        classifier = clf_cls(
            confidence_threshold=config.confidence_threshold,
            # Avoid double scaling if FeatureUnion already scaled the features
            normalize_features=(
                config.normalize_features and len(extractors) == 1
            ),
        )
    else:
        classifier = clf_cls(
            confidence_threshold=config.confidence_threshold,
            normalize_features=(
                config.normalize_features and len(extractors) == 1
            ),
        )

    return TimeSeriesPipeline(
        preprocessing_steps=steps,
        feature_extractor=feature_extractor,
        classifier=classifier,
        noise_threshold=config.noise_threshold,
    )
