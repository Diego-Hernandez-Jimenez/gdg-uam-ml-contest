import numpy as np
import pytest

# Absolute imports from the project root structure
from src.features.identity import IdentityFeatures
from src.features.minirocket import MiniRocket


@pytest.fixture
def batch_synthetic_data():
    """Generates a minimal batch dataset with shape (N=3, C=4, L=256)."""
    np.random.seed(42)
    return np.random.randn(3, 4, 256).astype(np.float32)


@pytest.fixture
def streaming_synthetic_sample():
    """Generates a single streaming test sample with shape (C=4, L=256)."""
    np.random.seed(42)
    return np.random.randn(4, 256).astype(np.float32)


# ==============================================================================
# TESTS FOR IDENTITY FEATURES
# ==============================================================================

def test_identity_features_batch(batch_synthetic_data):
    """Validates IdentityFeatures in batch mode (N, C, L) -> (N, C * L)."""
    extractor = IdentityFeatures()
    features = extractor.transform(batch_synthetic_data)
    
    # Expected output: (3, 4 * 256) = (3, 1024)
    assert features.shape == (3, 1024)
    np.testing.assert_array_equal(features[0], batch_synthetic_data[0].flatten())


def test_identity_features_streaming(streaming_synthetic_sample):
    """Validates IdentityFeatures in streaming mode (C, L) -> (C * L,)."""
    extractor = IdentityFeatures()
    features = extractor.transform(streaming_synthetic_sample)
    
    # Expected output: (4 * 256,) = (1024,)
    assert features.shape == (1024,)
    np.testing.assert_array_equal(features, streaming_synthetic_sample.flatten())


# ==============================================================================
# TESTS FOR MINIROCKET FEATURES
# ==============================================================================

def test_minirocket_features_batch(batch_synthetic_data):
    """Validates MiniRocket in batch mode (N, C, L) -> (N, F) using dynamic assertion."""
    extractor = MiniRocket(num_features=1000, max_dilations_per_kernel=32)
    
    # Fit and transform
    extractor.fit(batch_synthetic_data)
    features = extractor.transform(batch_synthetic_data)
    
    # Extract the dynamic number of features successfully built during fit
    expected_features = extractor.parameters_[4].shape[0]
    
    assert features.shape == (3, expected_features)
    assert not np.isnan(features).any()
    assert (features >= 0.0).all() and (features <= 1.0).all()


def test_minirocket_features_streaming(batch_synthetic_data, streaming_synthetic_sample):
    """Validates MiniRocket in streaming mode (C, L) -> (F,) using dynamic assertion."""
    extractor = MiniRocket(num_features=1000, max_dilations_per_kernel=32)
    extractor.fit(batch_synthetic_data)
    
    # Transform streaming sample
    features = extractor.transform(streaming_synthetic_sample)
    
    expected_features = extractor.parameters_[4].shape[0]
    
    # Assert shape is 1D (F,) matching the single sample inference contract
    assert features.shape == (expected_features,)
    assert not np.isnan(features).any()
    assert (features >= 0.0).all() and (features <= 1.0).all()


def test_minirocket_not_fitted_raises(streaming_synthetic_sample):
    """Asserts that transform on an unfitted MiniRocket raises a ValueError."""
    extractor = MiniRocket(num_features=1000)
    with pytest.raises(ValueError, match="MiniRocket must be fitted before transforming."):
        extractor.transform(streaming_synthetic_sample)
