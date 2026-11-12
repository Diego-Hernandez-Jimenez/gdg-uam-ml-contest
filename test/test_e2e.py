import os
import tempfile

import joblib
import numpy as np
import pytest

from src.classifiers.base import ABSTAIN
from src.classifiers.classifiers import DummyAbstainClassifier, RidgeClassifier
from src.features.identity import IdentityFeatures
from src.features.minirocket import MiniRocket
from src.preprocessing.steps import (
    HampelFilter,
    Identity,
    NaNInterpolation,
    RobustScale,
)
from src.training.config import TrainConfig, build_pipeline
from src.training.performance import confusion_matrix, geomean_score
from src.training.pipeline import TimeSeriesPipeline


@pytest.fixture(scope="module")
def synthetic_8class_dataset():
    """Generates a synthetic 8-class time-series dataset of shape (N=12, C=4, L=256).
    
    Includes simulated anomalies (NaNs and severe Spikes) to test pipeline robust steps.
    """
    np.random.seed(42)
    N, C, L = 12, 4, 256
    
    X = np.random.randn(N, C, L).astype(np.float32)
    # Inject anomalies
    X[0, 0, 10:20] = np.nan
    X[1, 1, 150] = 100.0  # Huge spike outlier
    
    # 8 classes (0..7)
    y = np.random.randint(0, 8, size=N)
    return X, y


# ==============================================================================
# INTEGRATION TESTS
# ==============================================================================

def test_full_pipeline_e2e_fit_predict_save_load(synthetic_8class_dataset):
    """Verifies E2E robust pipeline training, single-sample prediction, and joblib serialization."""
    X, y = synthetic_8class_dataset
    
    # 1. Instantiate full active components
    pipeline = TimeSeriesPipeline(
        preprocessing_steps=[
            NaNInterpolation(),
            HampelFilter(window=5, n_sigmas=3.0),
            RobustScale()
        ],
        feature_extractor=MiniRocket(num_features=500),  # small feature space for test speed
        classifier=RidgeClassifier(confidence_threshold=0.1),
        noise_threshold=3.5
    )
    
    # 2. Fit Phase: Batch training (N, C, L)
    pipeline.fit(X, y)
    
    # 3. Predict Phase (Batch): (N, C, L) -> (N,)
    batch_preds = pipeline.predict(X)
    assert batch_preds.shape == (X.shape[0],)
    assert set(batch_preds).issubset(set(range(8)).union({ABSTAIN}))
    
    # 4. Predict Phase (Streaming/Inference): (C, L) -> int class
    single_sample = X[0]  # Shape: (4, 256)
    single_pred = pipeline.predict(single_sample)
    
    assert isinstance(single_pred, (int, np.integer))
    assert single_pred in set(range(8)).union({ABSTAIN})
    
    # 5. Serialization and Loading test (simulating the trainer saving & model.py loading)
    with tempfile.TemporaryDirectory() as tmpdir:
        weights_path = os.path.join(tmpdir, "weights.joblib")
        joblib.dump(pipeline, weights_path)
        
        # Load from disk
        loaded_pipeline = joblib.load(weights_path)
        
        # Verify predictions match exactly
        loaded_pred = loaded_pipeline.predict(single_sample)
        assert loaded_pred == single_pred


def test_passthrough_identity_pipeline_e2e(synthetic_8class_dataset):
    """Verifies that composition with pass-through (Identity) steps functions perfectly."""
    X, y = synthetic_8class_dataset
    X_clean = np.nan_to_num(X, nan=0.0)  # Avoid scikit-learn NaNs crash on raw passthrough
    
    pipeline = TimeSeriesPipeline(
        preprocessing_steps=[Identity()],
        feature_extractor=IdentityFeatures(),
        classifier=RidgeClassifier(confidence_threshold=None)
    )
    
    pipeline.fit(X_clean, y)
    
    # Test streaming prediction
    single_sample = X_clean[0]
    pred = pipeline.predict(single_sample)
    
    assert isinstance(pred, (int, np.integer))
    assert 0 <= pred <= 7


def test_dummy_abstain_pipeline_e2e(synthetic_8class_dataset):
    """Verifies composition when using the Dummy fallback classifier (always abstains)."""
    X, y = synthetic_8class_dataset
    
    pipeline = TimeSeriesPipeline(
        preprocessing_steps=[Identity()],
        feature_extractor=IdentityFeatures(),
        classifier=DummyAbstainClassifier()
    )
    
    pipeline.fit(X, y)
    
    # Every streaming prediction must fall back to ABSTAIN
    for i in range(len(X)):
        pred = pipeline.predict(X[i])
        assert pred == ABSTAIN


def test_performance_metrics_integration(synthetic_8class_dataset):
    """Validates metrics computation from pipeline outputs using src/performance.py."""
    X, y = synthetic_8class_dataset
    
    pipeline = TimeSeriesPipeline(
        preprocessing_steps=[NaNInterpolation(), RobustScale()],
        feature_extractor=IdentityFeatures(),
        classifier=RidgeClassifier(confidence_threshold=0.1)
    )
    pipeline.fit(X, y)
    
    preds = pipeline.predict(X)
    
    # Compute metrics
    score = geomean_score(y, preds, n_classes=8)
    cm = confusion_matrix(y, preds, n_classes=8)
    
    assert isinstance(score, float)
    assert score >= 0.0
    # 8 classes, 9 columns (8 classes + 1 abstain column)
    assert cm.shape == (8, 9)


def test_yaml_config_and_sgd_validation(synthetic_8class_dataset, tmp_path):
    """Verifies YAML config loading and pipeline fit/predict with SGD mini-batch parameters."""
    X, y = synthetic_8class_dataset
    yaml_content = """
name: minirocket_sgd_test
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
  model: sgd
  confidence_threshold: 0.1
  noise_threshold: 3.5
training:
  cv_folds: 2
  batch_size: 6
  epochs: 2
"""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")
    cfg = TrainConfig.from_yaml(cfg_file)
    pipeline = build_pipeline(cfg)
    pipeline.fit(X, y, batch_size=cfg.training.batch_size, n_epochs=cfg.training.epochs)
    preds = pipeline.predict(X)
    assert preds.shape == (X.shape[0],)
