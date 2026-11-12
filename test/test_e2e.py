import os
import tempfile

import joblib
import numpy as np
import pytest

from src.classifiers.base import ABSTAIN
from src.classifiers.classifiers import (
    DummyAbstainClassifier,
    IncrementalRidgeClassifier,
    RidgeClassifier,
)
from src.features.base import FeatureUnion
from src.features.identity import IdentityFeatures
from src.features.minirocket import MiniRocket
from src.features.summary_stats import SummaryStatsExtractor
from src.preprocessing.steps import (
    HampelFilter,
    Identity,
    NaNInterpolation,
    RobustScale,
)
from src.training.config import (
    ClassifierConfig,
    FeatureExtractorConfig,
    PreprocessingConfig,
    TrainConfig,
    TrainingConfig,
    build_pipeline,
)
from src.training.performance import confusion_matrix, geomean_score
from src.training.pipeline import TimeSeriesPipeline


@pytest.fixture(scope="module")
def synthetic_8class_dataset():
    """Generates a synthetic 8-class time-series dataset of shape (N=16, C=4, L=256).

    Guarantees all 8 classes (0..7) are present (2 samples per class) and includes
    simulated anomalies (NaNs and severe Spikes) to test robust preprocessing.
    """
    rng = np.random.default_rng(42)
    N, C, L = 16, 4, 256

    X = rng.standard_normal((N, C, L), dtype=np.float32)
    # Inject anomalies
    X[0, 0, 10:20] = np.nan
    X[1, 1, 150] = 100.0  # Huge spike outlier

    # Guarantee all 8 classes (0..7) are represented
    y = np.repeat(np.arange(8, dtype=int), N // 8)
    rng.shuffle(y)
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
            RobustScale(),
        ],
        feature_extractor=MiniRocket(
            num_features=500
        ),  # small feature space for test speed
        classifier=RidgeClassifier(confidence_threshold=0.1),
        noise_threshold=3.5,
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
    assert single_pred == batch_preds[0]

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
    X_clean = np.nan_to_num(
        X, nan=0.0
    )  # Avoid scikit-learn NaNs crash on raw passthrough

    pipeline = TimeSeriesPipeline(
        preprocessing_steps=[Identity()],
        feature_extractor=IdentityFeatures(),
        classifier=RidgeClassifier(confidence_threshold=None),
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
        classifier=DummyAbstainClassifier(),
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
        classifier=RidgeClassifier(confidence_threshold=0.1),
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
  normalize_features: true
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


# ==============================================================================
# RIDGE vs INCREMENTAL_RIDGE & NORMALIZATION MATRIX TESTS
# ==============================================================================


@pytest.mark.parametrize("clf_cls", [RidgeClassifier, IncrementalRidgeClassifier])
@pytest.mark.parametrize("scale_features", [False, True])
@pytest.mark.parametrize("batch_size", [None, 5])
def test_ridge_variants_normalization_and_streaming_parity(
    synthetic_8class_dataset, clf_cls, scale_features, batch_size
):
    """Verifies Ridge and IncrementalRidge across full-batch and mini-batch modes,
    with and without FeatureUnion standardization, ensuring exact parity between
    batch evaluation (N, 4, 256) and single-sample streaming inference (4, 256).
    """
    X, y = synthetic_8class_dataset

    feature_extractor = FeatureUnion(
        extractors=[SummaryStatsExtractor(), IdentityFeatures()],
        scale_features=scale_features,
    )
    pipeline = TimeSeriesPipeline(
        preprocessing_steps=[NaNInterpolation(), RobustScale()],
        feature_extractor=feature_extractor,
        classifier=clf_cls(confidence_threshold=0.05),
        noise_threshold=4.0,
    )

    # 1. Fit (Full-batch if batch_size is None, Mini-batch if batch_size=5)
    pipeline.fit(X, y, batch_size=batch_size)

    # Verify standardization state matches configuration
    if scale_features:
        assert feature_extractor.mean_ is not None
        assert feature_extractor.std_ is not None
        assert feature_extractor._scaler is None  # Temporary sklearn object freed
    else:
        assert feature_extractor.mean_ is None
        assert feature_extractor.std_ is None

    # 2. Batch prediction: (N, 4, 256) -> (N,)
    batch_preds = pipeline.predict(X)
    assert batch_preds.shape == (X.shape[0],)
    assert set(batch_preds).issubset(set(range(8)).union({ABSTAIN}))

    # 3. Sample-by-sample streaming prediction: (4, 256) -> int
    # Must match batch predictions 100% for every sample
    streaming_preds = np.array([pipeline.predict(X[i]) for i in range(len(X))])
    assert np.array_equal(batch_preds, streaming_preds)


def test_exact_standardization_equivalence_full_vs_minibatch(synthetic_8class_dataset):
    """Verifies that FeatureUnion computes the exact same global mean_ and std_
    whether trained in full-batch mode or incrementally across mini-batches.
    """
    X, y = synthetic_8class_dataset
    # Use deterministic preprocessing and deterministic extractors
    X_clean = NaNInterpolation().fit(X).transform(X)

    fe_full = FeatureUnion(
        extractors=[SummaryStatsExtractor(), IdentityFeatures()],
        scale_features=True,
    )
    fe_incremental = FeatureUnion(
        extractors=[SummaryStatsExtractor(), IdentityFeatures()],
        scale_features=True,
    )

    pipe_full = TimeSeriesPipeline(
        preprocessing_steps=[Identity()],
        feature_extractor=fe_full,
        classifier=IncrementalRidgeClassifier(),
    )
    pipe_inc = TimeSeriesPipeline(
        preprocessing_steps=[Identity()],
        feature_extractor=fe_incremental,
        classifier=IncrementalRidgeClassifier(),
    )

    pipe_full.fit(X_clean, y, batch_size=None)
    pipe_inc.fit(X_clean, y, batch_size=5)

    # Global mean and std must match to floating-point precision
    np.testing.assert_allclose(
        fe_full.mean_, fe_incremental.mean_, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(fe_full.std_, fe_incremental.std_, rtol=1e-5, atol=1e-6)

    # Predictions on single samples must be identical
    for i in range(len(X_clean)):
        assert pipe_full.predict(X_clean[i]) == pipe_inc.predict(X_clean[i])


@pytest.mark.parametrize("model_name", ["ridge", "incremental_ridge"])
@pytest.mark.parametrize("normalize", [False, True])
def test_build_pipeline_contract_single_extractor_normalization(
    synthetic_8class_dataset, model_name, normalize
):
    """Verifies that build_pipeline properly handles a single extractor with or without
    normalization (without needing YAML files) and predicts sample-by-sample.
    """
    X, y = synthetic_8class_dataset

    cfg = TrainConfig(
        name=f"test_{model_name}_norm_{normalize}",
        preprocessing=PreprocessingConfig(steps=["nan_interpolation", "robust_scale"]),
        feature_extractor=FeatureExtractorConfig(
            components="summary_stats",
            normalize_features=normalize,
        ),
        classifier=ClassifierConfig(
            model=model_name,
            confidence_threshold=0.05,
        ),
        training=TrainingConfig(batch_size=6),
    )

    pipeline = build_pipeline(cfg)

    # When normalize=True, even a single extractor must be wrapped in FeatureUnion
    if normalize:
        assert isinstance(pipeline.feature_extractor, FeatureUnion)
        assert pipeline.feature_extractor.scale_features is True
    else:
        assert isinstance(pipeline.feature_extractor, SummaryStatsExtractor)

    pipeline.fit(X, y, batch_size=cfg.training.batch_size)

    # Verify streaming inference contract on (C=4, L=256)
    pred = pipeline.predict(X[0])
    assert isinstance(pred, (int, np.integer))
    assert pred in set(range(8)).union({ABSTAIN})
