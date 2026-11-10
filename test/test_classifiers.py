import numpy as np
import pytest
from sklearn.datasets import make_classification

# Absolute imports from the project structure
from src.classifiers.base import ABSTAIN
from src.classifiers.classifiers import DummyAbstainClassifier, RidgeClassifier


@pytest.fixture(scope="module")
def toy_multiclass_dataset():
    """Generates a simple 8-class classification dataset for testing.

    Contains exactly 8 classes (0..7) and 10 features per sample.
    """
    X, y = make_classification(
        n_samples=40, n_features=10, n_informative=8, n_classes=8, random_state=42
    )
    return X.astype(np.float32), y


@pytest.fixture
def fitted_ridge(toy_multiclass_dataset):
    """Provides a fitted RidgeClassifier on the toy 8-class dataset without abstention."""
    X, y = toy_multiclass_dataset
    clf = RidgeClassifier()
    clf.fit(X, y)
    return clf


# ==============================================================================
# TESTS FOR DUMMY ABSTAIN CLASSIFIER (IDENTITY)
# ==============================================================================


def test_dummy_abstain_classifier_batch(toy_multiclass_dataset):
    """Validates DummyAbstainClassifier returns ABSTAIN array for 2D batch inputs."""
    X, _ = toy_multiclass_dataset
    clf = DummyAbstainClassifier()
    preds = clf.predict(X)

    assert preds.shape == (X.shape[0],)
    assert np.all(preds == ABSTAIN)


def test_dummy_abstain_classifier_streaming():
    """Validates DummyAbstainClassifier returns single int ABSTAIN for 1D stream input."""
    clf = DummyAbstainClassifier()
    single_sample = np.array(
        [0.5, -1.2, 3.1, 0.0, 1.1, -0.4, 0.9, 1.2, 0.1, -0.2], dtype=np.float32
    )
    pred = clf.predict(single_sample)

    assert isinstance(pred, int)
    assert pred == ABSTAIN


# ==============================================================================
# TESTS FOR RIDGE CLASSIFIER (STANDARD & STREAMING) ON 8 CLASSES
# ==============================================================================


def test_ridge_classifier_standard_batch(toy_multiclass_dataset, fitted_ridge):
    """Validates standard RidgeClassifier prediction output in batch mode (8 classes)."""
    X, _ = toy_multiclass_dataset
    preds = fitted_ridge.predict(X)

    assert preds.shape == (X.shape[0],)
    assert set(preds).issubset(
        set(range(8))
    )  # Class IDs must fall strictly within 0..7


def test_ridge_classifier_standard_streaming(fitted_ridge):
    """Validates standard RidgeClassifier prediction output in single-sample mode (1D)."""
    single_sample = np.array(
        [0.1, -0.5, 1.2, -0.8, 0.3, 0.4, -0.2, 1.1, -1.0, 0.5], dtype=np.float32
    )
    pred = fitted_ridge.predict(single_sample)

    assert isinstance(pred, int)
    assert pred in set(range(8))  # Returns a single int class in 0..7


def test_ridge_not_fitted_raises():
    """Ensures prediction on an unfitted RidgeClassifier instance raises ValueError."""
    clf = RidgeClassifier()
    with pytest.raises(
        ValueError, match="RidgeClassifier must be fitted before calling predict."
    ):
        clf.predict(np.array([1.0, 2.0]))


# ==============================================================================
# TESTS FOR RIDGE CLASSIFIER WITH ABSTENTION ON 8 CLASSES
# ==============================================================================


def test_ridge_multiclass_abstention_boundary(toy_multiclass_dataset):
    """Ensures RidgeClassifier correctly calculates top-1 vs top-2 margin in 8-class setup."""
    X, y = toy_multiclass_dataset

    # Large confidence threshold forces abstention on uncertain/boundary samples
    clf = RidgeClassifier(confidence_threshold=1.5)
    clf.fit(X, y)

    # 1. Test Batch mode
    preds_batch = clf.predict(X)
    assert (
        ABSTAIN in preds_batch
    )  # Some samples must fallback to ABSTAIN due to low margins

    # 2. Test Streaming mode on a highly uncertain sample
    # A vector of zeros gives flat/very low margin scores across classes, triggering abstention
    boundary_sample = np.zeros(10, dtype=np.float32)
    pred_stream = clf.predict(boundary_sample)

    assert isinstance(pred_stream, int)
    assert pred_stream == ABSTAIN
