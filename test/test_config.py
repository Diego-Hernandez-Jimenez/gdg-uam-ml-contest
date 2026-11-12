import pytest

from src.training.config import (
    ClassifierConfig,
    TrainConfig,
    TrainingConfig,
    build_pipeline,
)
from src.training.pipeline import TimeSeriesPipeline

# Shared minimal nested YAML; reused across tests to avoid duplicating the schema.
MINIMAL_YAML = """
name: minirocket_test
preprocessing:
  steps:
    - nan_interpolation
    - hampel:
        window: 5
        n_sigmas: 3.0
    - robust_scale
feature_extractor:
  components:
    - minirocket:
        num_features: 200
classifier:
  model: ridge
  confidence_threshold: 0.1
training:
  cv_folds: 2
  batch_size: null
"""


@pytest.fixture
def minimal_yaml_path(tmp_path):
    """Write MINIMAL_YAML to a temp file and return its path."""
    path = tmp_path / "config.yaml"
    path.write_text(MINIMAL_YAML, encoding="utf-8")
    return path


def test_defaults_build_valid_pipeline():
    """A bare TrainConfig (only name) must fall back to sane defaults and build a pipeline."""
    config = TrainConfig(name="defaults")
    assert config.preprocessing.steps == ["nan_interpolation"]
    assert config.feature_extractor.components == "minirocket"
    assert config.classifier.model == "ridge"
    assert isinstance(build_pipeline(config), TimeSeriesPipeline)


def test_from_yaml_maps_nested_schema(minimal_yaml_path):
    """from_yaml must populate each nested block from its own YAML section."""
    config = TrainConfig.from_yaml(minimal_yaml_path)
    assert config.name == "minirocket_test"
    assert config.preprocessing.steps[0] == "nan_interpolation"
    assert config.preprocessing.steps[1] == {"hampel": {"window": 5, "n_sigmas": 3.0}}
    assert config.feature_extractor.components == [
        {"minirocket": {"num_features": 200}}
    ]
    assert config.classifier.model == "ridge"
    assert config.classifier.confidence_threshold == 0.1
    assert config.training.cv_folds == 2
    assert isinstance(build_pipeline(config), TimeSeriesPipeline)


def test_batch_size_epochs_invalid_for_non_incremental_classifier():
    """batch_size/epochs are only meaningful for sgd, ridge, or incremental_ridge."""
    with pytest.raises(ValueError, match="only valid with"):
        TrainConfig(
            name="invalid",
            classifier=ClassifierConfig(model="dummy"),
            training=TrainingConfig(batch_size=32),
        )
