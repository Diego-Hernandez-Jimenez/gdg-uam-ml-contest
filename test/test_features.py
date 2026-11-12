import numpy as np
import pytest

# Absolute imports from the project root structure
from src.features.base import FeatureUnion
from src.features.identity import IdentityFeatures
from src.features.minirocket import MiniRocket
from src.features.multirocket import MultiRocket
from src.features.summary_stats import SummaryStatsExtractor


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
# TESTS FOR SUMMARY STATS EXTRACTOR
# ==============================================================================


def test_summary_stats_batch_and_streaming(batch_synthetic_data):
    """Validates SummaryStatsExtractor shapes and consistency between batch and streaming."""
    extractor = SummaryStatsExtractor()

    # 1. Batch mode: (N=3, C=4, L=256) -> (3, 6 * 4) = (3, 24)
    batch_feats = extractor.fit(batch_synthetic_data).transform(batch_synthetic_data)
    assert batch_feats.shape == (3, 24)
    assert not np.isnan(batch_feats).any()
    assert (batch_feats >= 0).all()  # All 6 stats/ratios are strictly non-negative

    # 2. Streaming mode on the first sample of the batch: (C=4, L=256) -> (24,)
    single_feats = extractor.transform(batch_synthetic_data[0])
    assert single_feats.shape == (24,)
    np.testing.assert_allclose(batch_feats[0], single_feats, rtol=1e-5)


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
        expected_features = (
            base_features + diff_features
        ) * 4  # 4 pooling features per kernel
    else:
        expected_features = base_features  # 1 PPV feature per kernel

    assert features.shape == (3, expected_features)
    assert not np.isnan(features).any()
    # Check that pooled features (PPV, longest stretch, mean) contain valid, stable ranges
    assert not (features == 0).all()


@pytest.mark.parametrize("extractor_class", [MiniRocket, MultiRocket])
def test_rocket_features_streaming(
    batch_synthetic_data, streaming_synthetic_sample, extractor_class
):
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
    with pytest.raises(
        ValueError,
        match=f"{extractor_class.__name__} must be fitted before transforming.",
    ):
        extractor.transform(streaming_synthetic_sample)


# ==============================================================================
# TESTS FOR FEATURE UNION (CONCATENATION & NORMALIZATION)
# ==============================================================================


def test_feature_union_unscaled(batch_synthetic_data):
    """Validates FeatureUnion concatenation in batch and streaming modes without scaling."""
    rocket = MiniRocket(num_features=500, max_dilations_per_kernel=32)
    stats = SummaryStatsExtractor()
    union = FeatureUnion(extractors=[rocket, stats], scale_features=False)

    union.fit(batch_synthetic_data)
    batch_out = union.transform(batch_synthetic_data)

    expected_rocket_dim = rocket.parameters_[4].shape[0]
    expected_total_dim = expected_rocket_dim + 24

    # 1. Validate concatenated batch shape
    assert batch_out.shape == (3, expected_total_dim)

    # 2. Verify exact match of the summary stats slice at the end of the feature vector
    direct_stats = stats.transform(batch_synthetic_data)
    np.testing.assert_allclose(batch_out[:, -24:], direct_stats, rtol=1e-5)

    # 3. Validate streaming mode (1D output) and consistency with batch[0]
    single_out = union.transform(batch_synthetic_data[0])
    assert single_out.shape == (expected_total_dim,)
    np.testing.assert_allclose(batch_out[0], single_out, rtol=1e-5)


def test_feature_union_scaled(batch_synthetic_data):
    """Validates that FeatureUnion properly standardizes concatenated features."""
    rocket = MiniRocket(num_features=500, max_dilations_per_kernel=32)
    stats = SummaryStatsExtractor()
    union = FeatureUnion(extractors=[rocket, stats], scale_features=True)

    union.fit(batch_synthetic_data)
    batch_out = union.transform(batch_synthetic_data)

    assert union.mean_ is not None
    assert union.std_ is not None
    assert not np.isnan(batch_out).any()

    # Non-constant features across the training batch must have zero mean and unit variance
    non_const_mask = np.std(batch_out, axis=0) > 1e-6
    assert non_const_mask.any()
    np.testing.assert_allclose(
        np.mean(batch_out[:, non_const_mask], axis=0), 0.0, atol=1e-4
    )
    np.testing.assert_allclose(
        np.std(batch_out[:, non_const_mask], axis=0), 1.0, atol=1e-4
    )

    # Streaming mode must apply the exact same frozen mean_ and std_
    single_out = union.transform(batch_synthetic_data[0])
    np.testing.assert_allclose(batch_out[0], single_out, rtol=1e-5)
