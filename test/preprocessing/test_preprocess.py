import os

import numpy as np
import pytest

# Absolute imports from the src package structure
from src.preprocessing.base import Pipeline
from src.preprocessing.steps import (
    HampelFilter,
    Identity,
    NaNInterpolation,
    RobustScale,
)


@pytest.fixture(scope="session", autouse=True)
def setup_synthetic_npz_file():
    """Generates a synthetic .npz file with exactly (4, 256) shape.
    
    Includes realistic anomalies matching your exact specification:
    - Channel 0: Sinusoidal wave with a burst of consecutive NaNs (length 10).
    - Channel 1: Cosine wave with a severe amplitude spike at index 128 (outlier).
    - Channel 2: High amplitude wave with a large DC bias (offset = 100).
    - Channel 3: Baseline random normal noise.
    """
    np.random.seed(42)
    t = np.linspace(0, 10, 256)  # Exactly 256 time steps
    
    # Channel 0: NaNs burst
    ch0 = np.sin(t)
    ch0[50:60] = np.nan
    
    # Channel 1: Outlier spike
    ch1 = np.cos(t)
    ch1[128] = 50.0  
    
    # Channel 2: High offset and scale
    ch2 = 10.0 * np.sin(t) + 100.0
    
    # Channel 3: Random Gaussian noise
    ch3 = np.random.randn(256)
    
    # Stack to create shape (4, 256)
    synthetic_data = np.vstack([ch0, ch1, ch2, ch3])
    
    # Save the synthetic data inside the test directory
    filepath = os.path.join(os.path.dirname(__file__), "synthetic_test_data.npz")
    np.savez(filepath, data=synthetic_data)
    
    yield filepath
    
    # Teardown: clean up the test file after tests finish
    if os.path.exists(filepath):
        os.remove(filepath)


@pytest.fixture
def test_data(setup_synthetic_npz_file):
    """Loads the synthetic npz dataset of shape (4, 256) for individual tests."""
    with np.load(setup_synthetic_npz_file) as loader:
        data = loader["data"]
        # Double check exact dimensional constraint
        assert data.shape == (4, 256), f"Expected shape (4, 256), got {data.shape}"
        return data


def test_identity_step(test_data):
    """Verifies that Identity class returns the array completely unchanged with correct dimensions."""
    step = Identity()
    transformed = step.transform(test_data)
    
    assert transformed.shape == (4, 256)
    np.testing.assert_array_equal(transformed, test_data)


def test_nan_interpolation(test_data):
    """Validates linear interpolation of NaN values on the (4, 256) series."""
    step = NaNInterpolation()
    transformed = step.transform(test_data)
    
    assert transformed.shape == (4, 256)
    # Assert no NaNs remain in the output array
    assert not np.isnan(transformed).any()
    
    # Assert that non-NaN original values remain intact
    valid_mask = ~np.isnan(test_data)
    np.testing.assert_array_equal(transformed[valid_mask], test_data[valid_mask])


def test_hampel_filter(test_data):
    """Validates that Hampel filter correctly suppresses isolated spikes.
    
    Needs a NaN-free input, so we run NaNInterpolation first.
    """
    clean_nans = NaNInterpolation().transform(test_data)
    
    step = HampelFilter(window=5, n_sigmas=3.0)
    transformed = step.transform(clean_nans)
    
    assert transformed.shape == (4, 256)
    
    # Assert that the huge spike in Channel 1 (index 128) was suppressed
    original_spike = clean_nans[1, 128]
    replaced_val = transformed[1, 128]
    
    assert original_spike == 50.0
    assert replaced_val < 10.0  # Successfully suppressed close to local median
    
    # Assert other non-outlier points remain untouched
    np.testing.assert_allclose(transformed[1, :50], clean_nans[1, :50], rtol=1e-5)


def test_robust_scale(test_data):
    """Validates centering and scaling around median and IQR on (4, 256) series."""
    clean_nans = NaNInterpolation().transform(test_data)
    
    step = RobustScale()
    transformed = step.transform(clean_nans)
    
    assert transformed.shape == (4, 256)
    
    # Assert median is centered at 0 per channel (each channel has length 256)
    medians = np.median(transformed, axis=1)
    np.testing.assert_allclose(medians, 0.0, atol=1e-7)


def test_pipeline_composition(test_data):
    """Validates the compositionality of the pipeline running all steps on (4, 256) series."""
    pipeline = Pipeline([
        NaNInterpolation(),
        HampelFilter(window=5, n_sigmas=3.0),
        RobustScale()
    ])
    
    transformed = pipeline.transform(test_data)
    
    # Verify shape is preserved
    assert transformed.shape == (4, 256)
    # Verify the output is NaN-free
    assert not np.isnan(transformed).any()
    # Verify centering (RobustScale worked)
    np.testing.assert_allclose(np.median(transformed, axis=1), 0.0, atol=1e-7)
    # Verify spike suppression (Hampel worked, scaled value is in bounds)
    assert transformed[1, 128] < 3.0
