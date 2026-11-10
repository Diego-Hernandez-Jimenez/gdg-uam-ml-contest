import numpy as np
import pytest

# Absolute imports from the project root structure
from src.features.identity import IdentityFeatures
from src.features.minirocket import MiniRocket
from src.features.multirocket import MultiRocket


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
# PARAMETRIZED TESTS FOR ROCKET-FAMILY EXTRACTORS (MINIROCKET & MULTIROCKET)
# ==============================================================================

@pytest.mark.parametrize("extractor_class", [MiniRocket, MultiRocket])
def test_rocket_features_batch(batch_synthetic_data, extractor_class):
    """Validates both Rocket extractors in batch mode using dynamic assertions."""
    extractor = extractor_class(num_features=500, max_dilations_per_kernel=32)
    
    # Fit and transform
    extractor.fit(batch_synthetic_data)
    features = extractor.transform(batch_synthetic_data)
    
    # Calculate dynamic expected features
    base_features = extractor.parameters_[4].shape[0]
    if extractor_class is MultiRocket:
        diff_features = extractor.parameters1_[4].shape[0]
        expected_features = (base_features + diff_features) * 4  # 4 pooling features per kernel
    else:
        expected_features = base_features  # 1 PPV feature per kernel
        
    assert features.shape == (3, expected_features)
    assert not np.isnan(features).any()
    # Check that pooled features (PPV, longest stretch, mean) contain valid, stable ranges
    assert not (features == 0).all()


@pytest.mark.parametrize("extractor_class", [MiniRocket, MultiRocket])
def test_rocket_features_streaming(batch_synthetic_data, streaming_synthetic_sample, extractor_class):
    """Validates both Rocket extractors in single-sample streaming mode (C, L) -> (F,)."""
    extractor = extractor_class(num_features=500, max_dilations_per_kernel=32)
    extractor.fit(batch_synthetic_data)
    
    # Transform streaming sample
    features = extractor.transform(streaming_synthetic_sample)
    
    # Calculate dynamic expected features
    base_features = extractor.parameters_[4].shape[0]
    if extractor_class is MultiRocket:
        diff_features = extractor.parameters1_[4].shape[0]
        expected_features = (base_features + diff_features) * 4
    else:
        expected_features = base_features
        
    # Assert shape is 1D (F,) matching the single sample inference contract
    assert features.shape == (expected_features,)
    assert not np.isnan(features).any()


@pytest.mark.parametrize("extractor_class", [MiniRocket, MultiRocket])
def test_rocket_not_fitted_raises(streaming_synthetic_sample, extractor_class):
    """Asserts that transform on an unfitted Rocket extractor raises a ValueError."""
    extractor = extractor_class(num_features=500)
    with pytest.raises(ValueError, match=f"{extractor_class.__name__} must be fitted before transforming."):
        extractor.transform(streaming_synthetic_sample)
