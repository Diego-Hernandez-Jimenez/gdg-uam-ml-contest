import numpy as np
import pytest

from src.classifiers.base import ABSTAIN
from src.classifiers.classifiers import RidgeClassifier
from src.features.minirocket import MiniRocket

# Absolute imports from the project structure
from src.pipeline import TimeSeriesPipeline
from src.preprocessing.steps import HampelFilter, NaNInterpolation, RobustScale


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

    # 5. Serialization and Loading test (simu
