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
class PreprocessingConfig:
    """Preprocessing stage: ordered list of cleaning/filtering steps."""

    steps: list[str | dict[str, Any]] = field(
        default_factory=lambda: ["nan_interpolation"]
    )


@dataclass
class FeatureExtractorConfig:
    """Feature extraction stage: components to compute and standardize."""

    components: str | list[str | dict[str, Any]] = "minirocket"
    normalize_features: bool = True


@dataclass
class ClassifierConfig:
    """Classifier stage: model choice and its hyperparameters."""

    model: str = "ridge"
    confidence_threshold: float | None = None
    noise_threshold: float | None = None


@dataclass
class TrainingConfig:
    """Training loop controls: batching, epochs and cross-validation."""

    cv_folds: int | None = None
    batch_size: int | None = None
    epochs: int = 3


@dataclass
class TrainConfig:
    """Configuration contract for training pipeline and workflow."""

    name: str
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    feature_extractor: FeatureExtractorConfig = field(
        default_factory=FeatureExtractorConfig
    )
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    artifacts_dir: str = "artifacts"

    def __post_init__(self) -> None:
        if self.classifier.model not in ("sgd", "incremental_ridge", "ridge") and (
            self.training.batch_size is not None or self.training.epochs != 3
        ):
            raise ValueError(
                "batch_size and epochs are only valid with 'sgd', 'ridge', or 'incremental_ridge'."
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        """Load configuration from a YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls(
            name=data["name"],
            preprocessing=PreprocessingConfig(**(data.get("preprocessing") or {})),
            feature_extractor=FeatureExtractorConfig(
                **(data.get("feature_extractor") or {})
            ),
            classifier=ClassifierConfig(**(data.get("classifier") or {})),
            training=TrainingConfig(**(data.get("training") or {})),
            artifacts_dir=data.get("artifacts_dir", "artifacts"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to dictionary."""
        return asdict(self)


def _instantiate_single_extractor(spec: str | dict[str, Any]) -> FeatureExtractor:
    """Helper to instantiate a single FeatureExtractor from string or dict spec."""
    if isinstance(spec, str):
        name, params = spec, {}
    elif isinstance(spec, dict):
        name, params = next(iter(spec.items()))
        params = params or {}
    else:
        raise TypeError(f"Invalid feature extractor specification: {spec}")

    return FEATURE_MAP[name](**params)


def build_pipeline(config: TrainConfig) -> TimeSeriesPipeline:
    """Instantiate a TimeSeriesPipeline from a TrainConfig contract."""
    # 1. Build Preprocessing Steps
    steps = []
    for step in config.preprocessing.steps:
        if isinstance(step, str):
            steps.append(PREPROCESSING_MAP[step]())
        elif isinstance(step, dict):
            name, params = next(iter(step.items()))
            steps.append(PREPROCESSING_MAP[name](**(params or {})))

    # 2. Build Feature Extractor (wrap in FeatureUnion if multiple or if normalization is enabled)
    fe_specs = (
        [config.feature_extractor.components]
        if isinstance(config.feature_extractor.components, str)
        else config.feature_extractor.components
    )
    extractors = [_instantiate_single_extractor(spec) for spec in fe_specs]

    if len(extractors) > 1 or config.feature_extractor.normalize_features:
        feature_extractor: FeatureExtractor = FeatureUnion(
            extractors=extractors,
            scale_features=config.feature_extractor.normalize_features,
        )
    else:
        feature_extractor = extractors[0]

    # 3. Build Classifier
    clf_cls = CLASSIFIER_MAP[config.classifier.model]
    classifier = (
        clf_cls()
        if config.classifier.model == "dummy"
        else clf_cls(confidence_threshold=config.classifier.confidence_threshold)
    )

    return TimeSeriesPipeline(
        preprocessing_steps=steps,
        feature_extractor=feature_extractor,
        classifier=classifier,
        noise_threshold=config.classifier.noise_threshold,
    )
