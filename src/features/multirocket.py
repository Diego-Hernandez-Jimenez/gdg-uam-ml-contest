"""MultiRocket feature extractor (naive multivariate extension).

Reference:
    Chang Wei Tan, Angus Dempster, Christoph Bergmeir, Geoffrey I Webb.
    "MultiRocket: Multiple pooling operators and transformations for fast and
    effective time series classification". https://arxiv.org/abs/2102.00457

Builds on MiniRocket's 84 fixed +/-1 kernels, but instead of a single PPV feature
per (kernel, dilation, bias) combination, extracts 4 pooling features from each
convolution output, and runs the whole pipeline twice: once on the raw series and
once on its first-order difference.

This module only extracts features (`fit` + `transform`); the classifier that
trains on top of them is a separate component, not included here.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from .base import FeatureExtractor

KERNEL_LENGTH = 9   # fixed kernel length used throughout the paper
NUM_KERNELS = 84     # C(9,3): number of distinct +/-1 weight kernels of length 9
HALF_KERNEL = KERNEL_LENGTH // 2
N_FEATURES_PER_KERNEL = 4   # PPV, longest stretch, mean, mean index (see _write_pooling_features)

# Quick win: this lookup table was previously duplicated verbatim inside both
# _fit_biases and transform; it never changes, so it is now built once here and
# reused by both (numba treats module-level arrays as compile-time constants).
# Equivalent to: np.array(list(itertools.combinations(range(9), 3)))
# fmt: off
_INDICES = np.array((
    0, 1, 2, 0, 1, 3, 0, 1, 4, 0, 1, 5, 0, 1, 6, 0, 1, 7, 0, 1, 8,
    0, 2, 3, 0, 2, 4, 0, 2, 5, 0, 2, 6, 0, 2, 7, 0, 2, 8, 0, 3, 4,
    0, 3, 5, 0, 3, 6, 0, 3, 7, 0, 3, 8, 0, 4, 5, 0, 4, 6, 0, 4, 7,
    0, 4, 8, 0, 5, 6, 0, 5, 7, 0, 5, 8, 0, 6, 7, 0, 6, 8, 0, 7, 8,
    1, 2, 3, 1, 2, 4, 1, 2, 5, 1, 2, 6, 1, 2, 7, 1, 2, 8, 1, 3, 4,
    1, 3, 5, 1, 3, 6, 1, 3, 7, 1, 3, 8, 1, 4, 5, 1, 4, 6, 1, 4, 7,
    1, 4, 8, 1, 5, 6, 1, 5, 7, 1, 5, 8, 1, 6, 7, 1, 6, 8, 1, 7, 8,
    2, 3, 4, 2, 3, 5, 2, 3, 6, 2, 3, 7, 2, 3, 8, 2, 4, 5, 2, 4, 6,
    2, 4, 7, 2, 4, 8, 2, 5, 6, 2, 5, 7, 2, 5, 8, 2, 6, 7, 2, 6, 8,
    2, 7, 8, 3, 4, 5, 3, 4, 6, 3, 4, 7, 3, 4, 8, 3, 5, 6, 3, 5, 7,
    3, 5, 8, 3, 6, 7, 3, 6, 8, 3, 7, 8, 4, 5, 6, 4, 5, 7, 4, 5, 8,
    4, 6, 7, 4, 6, 8, 4, 7, 8, 5, 6, 7, 5, 6, 8, 5, 7, 8, 6, 7, 8
), dtype=np.int32).reshape(NUM_KERNELS, 3)
# fmt: on

@njit("float32[:](float64[:,:,:],int32[:],int32[:],int32[:],int32[:],float32[:])",
      fastmath=True, parallel=False, cache=True)
def _fit_biases(X: np.ndarray, num_channels_per_combination: np.ndarray,
                channel_indices: np.ndarray, dilations: np.ndarray,
                num_features_per_dilation: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Sample one bias per (kernel, dilation, quantile) combination from random training examples."""
    num_examples, _num_channels, input_length = X.shape

    num_features = NUM_KERNELS * np.sum(num_features_per_dilation)
    biases = np.zeros(num_features, dtype=np.float32)

    feature_index_start = 0
    combination_index = 0
    num_channels_start = 0

    for dilation_index in range(len(dilations)):
        dilation = dilations[dilation_index]
        padding = ((KERNEL_LENGTH - 1) * dilation) // 2
        num_features_this_dilation = num_features_per_dilation[dilation_index]

        for kernel_index in range(NUM_KERNELS):
            feature_index_end = feature_index_start + num_features_this_dilation
            num_channels_this_combination = num_channels_per_combination[combination_index]
            num_channels_end = num_channels_start + num_channels_this_combination
            channels_this_combination = channel_indices[num_channels_start:num_channels_end]

            _X = X[np.random.randint(num_examples)][channels_this_combination]

            A = -_X          # A = alpha * X = -X
            G = _X + _X + _X  # G = gamma * X = 3X

            # Quick win: X is float64 but internal arithmetic is deliberately kept
            # in float32 (matching the dtype of `biases`); casting once here
            # replaces a zero-alloc + full overwrite with a single array op.
            C_alpha = A.astype(np.float32)

            C_gamma = np.zeros((KERNEL_LENGTH, num_channels_this_combination, input_length), dtype=np.float32)
            C_gamma[HALF_KERNEL] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(HALF_KERNEL):
                C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                C_gamma[gamma_index, :, -end:] = G[:, :end]
                end += dilation

            for gamma_index in range(HALF_KERNEL + 1, KERNEL_LENGTH):
                C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
                C_gamma[gamma_index, :, :-start] = G[:, start:]
                start += dilation

            index_0, index_1, index_2 = _INDICES[kernel_index]
            C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]
            C = np.sum(C, axis=0)

            biases[feature_index_start:feature_index_end] = \
                np.quantile(C, quantiles[feature_index_start:feature_index_end])

            feature_index_start = feature_index_end
            combination_index += 1
            num_channels_start = num_channels_end

    return biases


def _fit_dilations(input_length: int, num_features: int,
                    max_dilations_per_kernel: int) -> tuple[np.ndarray, np.ndarray]:
    """Pick log-spaced dilations and how many quantile-features to assign to each."""
    num_features_per_kernel = num_features // NUM_KERNELS
    true_max_dilations_per_kernel = min(num_features_per_kernel, max_dilations_per_kernel)
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = np.log2((input_length - 1) / (KERNEL_LENGTH - 1))
    dilations, num_features_per_dilation = np.unique(
        np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(np.int32),
        return_counts=True,
    )
    num_features_per_dilation = (num_features_per_dilation * multiplier).astype(np.int32)

    remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)

    return dilations, num_features_per_dilation


def _quantiles(n: int) -> np.ndarray:
    """Low-discrepancy (golden-ratio) sequence used to assign quantiles to kernel/dilation combos."""
    return np.array([(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype=np.float32)


def fit(X: np.ndarray, num_features: int = 10_000, max_dilations_per_kernel: int = 32) -> tuple:
    """Fit MultiRocket kernels on X: (num_examples, num_channels, input_length).

    Call this twice to build a full MultiRocket feature set: once on the raw
    series, once on their first-order difference (`np.diff(X, 1)`), then pass
    both parameter tuples into `transform`. Returns the parameters tuple
    (num_channels_per_combination, channel_indices, dilations,
    num_features_per_dilation, biases).
    """
    _, num_channels, input_length = X.shape

    dilations, num_features_per_dilation = _fit_dilations(input_length, num_features, max_dilations_per_kernel)
    num_features_per_kernel = np.sum(num_features_per_dilation)
    quantiles = _quantiles(NUM_KERNELS * num_features_per_kernel)

    num_combinations = NUM_KERNELS * len(dilations)
    max_num_channels = min(num_channels, KERNEL_LENGTH)
    max_exponent = np.log2(max_num_channels + 1)
    num_channels_per_combination = (2 ** np.random.uniform(0, max_exponent, num_combinations)).astype(np.int32)

    channel_indices = np.zeros(num_channels_per_combination.sum(), dtype=np.int32)

    num_channels_start = 0
    for combination_index in range(num_combinations):
        num_channels_this_combination = num_channels_per_combination[combination_index]
        num_channels_end = num_channels_start + num_channels_this_combination
        channel_indices[num_channels_start:num_channels_end] = np.random.choice(
            num_channels, num_channels_this_combination, replace=False
        )
        num_channels_start = num_channels_end

    biases = _fit_biases(X, num_channels_per_combination, channel_indices,
                          dilations, num_features_per_dilation, quantiles)

    return num_channels_per_combination, channel_indices, dilations, num_features_per_dilation, biases


# Quick win: the original inlined this ~25-line block identically 4 times (once
# per combination of {base series, first-order difference} x {unpadded, padded
# trim}), differing only in which array view and output offset were used. Factored
# into a single helper that every call site invokes, cutting ~150 duplicated lines
# down to this one function + 4 short call sites, with no change in arithmetic.
@njit(cache=True, fastmath=True)
def _write_pooling_features(feature_row: np.ndarray, base_index: int, stride: int,
                             c_view: np.ndarray, bias: float) -> None:
    """Compute MultiRocket's 4 pooling features for one (kernel, bias) pair over
    `c_view` and write them into `feature_row` at base_index, +stride, +2*stride,
    +3*stride: PPV (proportion of values above bias), the longest stretch of
    consecutive values at-or-below bias, the mean of the values above bias (offset
    by bias, matching the published implementation), and their mean index.
    """
    n = c_view.shape[0]
    ppv = 0
    last_val = 0
    max_stretch = 0.0
    mean_index = 0
    mean = 0.0

    for j in range(n):
        if c_view[j] > bias:
            ppv += 1
            mean_index += j
            mean += c_view[j] + bias
        elif c_view[j] < bias:
            stretch = j - last_val
            max_stretch = max(max_stretch, stretch)
            last_val = j
    stretch = n - 1 - last_val
    max_stretch = max(max_stretch, stretch)

    feature_row[base_index] = ppv / n
    feature_row[base_index + stride] = max_stretch
    feature_row[base_index + 2 * stride] = mean / ppv if ppv > 0 else 0
    feature_row[base_index + 3 * stride] = mean_index / ppv if ppv > 0 else -1


@njit(
    "float32[:,:](float64[:,:,:],float64[:,:,:],"
    "Tuple((int32[:],int32[:],int32[:],int32[:],float32[:])),"
    "Tuple((int32[:],int32[:],int32[:],int32[:],float32[:])),int32)",
    fastmath=True, parallel=True, cache=True)
def transform(X: np.ndarray, X1: np.ndarray, parameters, parameters1,
              n_features_per_kernel: int = 4) -> np.ndarray:
    """Apply fitted MultiRocket kernels to the base series `X` and its
    first-order difference `X1` (`X1 = np.diff(X, 1)`, computed by the caller).

    `parameters`/`parameters1` are the tuples returned by `fit(X, ...)` and
    `fit(X1, ...)` respectively. Returns features, shape
    (num_examples, (num_features + num_features1) * n_features_per_kernel).

    Note: the first-order-difference pass reuses the SAME channel/combination
    assignment (`num_channels_per_combination`, `channel_indices`) fitted for the
    base series in `parameters` -- only dilations/biases differ (`parameters1`).
    This matches the published implementation.
    """
    num_examples, num_channels, input_length = X.shape

    num_channels_per_combination, channel_indices, dilations, num_features_per_dilation, biases = parameters
    _, _, dilations1, num_features_per_dilation1, biases1 = parameters1

    num_dilations = len(dilations)
    num_dilations1 = len(dilations1)

    num_features = NUM_KERNELS * np.sum(num_features_per_dilation)
    num_features1 = NUM_KERNELS * np.sum(num_features_per_dilation1)

    features = np.zeros((num_examples, (num_features + num_features1) * n_features_per_kernel), dtype=np.float32)
    # Quick win: avoid a pointless float division + cast (always exact: shape[1]
    # is a multiple of n_features_per_kernel=4, hence even).
    n_features_per_transform = features.shape[1] // 2

    for example_index in prange(num_examples):
        feature_row = features[example_index]

        # --- base series -------------------------------------------------
        _X = X[example_index]
        A = -_X
        G = _X + _X + _X

        feature_index_start = 0
        combination_index = 0
        num_channels_start = 0

        for dilation_index in range(num_dilations):
            _padding0 = dilation_index % 2
            dilation = dilations[dilation_index]
            padding = ((KERNEL_LENGTH - 1) * dilation) // 2
            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = A.astype(np.float32)
            C_gamma = np.zeros((KERNEL_LENGTH, num_channels, input_length), dtype=np.float32)
            C_gamma[HALF_KERNEL] = G

            start = dilation
            end = input_length - padding
            for gamma_index in range(HALF_KERNEL):
                C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                C_gamma[gamma_index, :, -end:] = G[:, :end]
                end += dilation
            for gamma_index in range(HALF_KERNEL + 1, KERNEL_LENGTH):
                C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
                C_gamma[gamma_index, :, :-start] = G[:, start:]
                start += dilation

            for kernel_index in range(NUM_KERNELS):
                feature_index_end = feature_index_start + num_features_this_dilation
                num_channels_this_combination = num_channels_per_combination[combination_index]
                num_channels_end = num_channels_start + num_channels_this_combination
                channels_this_combination = channel_indices[num_channels_start:num_channels_end]

                _padding1 = (_padding0 + kernel_index) % 2
                index_0, index_1, index_2 = _INDICES[kernel_index]
                C = C_alpha[channels_this_combination] + \
                    C_gamma[index_0][channels_this_combination] + \
                    C_gamma[index_1][channels_this_combination] + \
                    C_gamma[index_2][channels_this_combination]
                C = np.sum(C, axis=0)
                c_view = C if _padding1 == 0 else C[padding:-padding]

                for feature_count in range(num_features_this_dilation):
                    feature_index = feature_index_start + feature_count
                    _write_pooling_features(feature_row, feature_index, num_features,
                                             c_view, biases[feature_index])

                feature_index_start = feature_index_end
                combination_index += 1
                num_channels_start = num_channels_end

        # --- first-order difference (reuses channel assignment, see docstring) ---
        _X1 = X1[example_index]
        A1 = -_X1
        G1 = _X1 + _X1 + _X1

        feature_index_start = 0
        combination_index = 0
        num_channels_start = 0

        for dilation_index in range(num_dilations1):
            _padding0 = dilation_index % 2
            dilation = dilations1[dilation_index]
            padding = ((KERNEL_LENGTH - 1) * dilation) // 2
            num_features_this_dilation = num_features_per_dilation1[dilation_index]

            C_alpha = A1.astype(np.float32)
            C_gamma = np.zeros((KERNEL_LENGTH, num_channels, input_length - 1), dtype=np.float32)
            C_gamma[HALF_KERNEL] = G1

            start = dilation
            end = input_length - padding
            for gamma_index in range(HALF_KERNEL):
                C_alpha[:, -end:] = C_alpha[:, -end:] + A1[:, :end]
                C_gamma[gamma_index, :, -end:] = G1[:, :end]
                end += dilation
            for gamma_index in range(HALF_KERNEL + 1, KERNEL_LENGTH):
                C_alpha[:, :-start] = C_alpha[:, :-start] + A1[:, start:]
                C_gamma[gamma_index, :, :-start] = G1[:, start:]
                start += dilation

            for kernel_index in range(NUM_KERNELS):
                feature_index_end = feature_index_start + num_features_this_dilation
                num_channels_this_combination = num_channels_per_combination[combination_index]
                num_channels_end = num_channels_start + num_channels_this_combination
                channels_this_combination = channel_indices[num_channels_start:num_channels_end]

                _padding1 = (_padding0 + kernel_index) % 2
                index_0, index_1, index_2 = _INDICES[kernel_index]
                C = C_alpha[channels_this_combination] + \
                    C_gamma[index_0][channels_this_combination] + \
                    C_gamma[index_1][channels_this_combination] + \
                    C_gamma[index_2][channels_this_combination]
                C = np.sum(C, axis=0)
                c_view = C if _padding1 == 0 else C[padding:-padding]

                for feature_count in range(num_features_this_dilation):
                    feature_index = feature_index_start + feature_count
                    _write_pooling_features(feature_row, feature_index + n_features_per_transform,
                                             num_features, c_view, biases1[feature_index])

                feature_index_start = feature_index_end
                combination_index += 1
                num_channels_start = num_channels_end

    return features


class MultiRocket(FeatureExtractor):
    """MultiRocket feature extractor wrapped in the standard FeatureExtractor contract."""

    def __init__(self, num_features: int = 10_000, max_dilations_per_kernel: int = 32):
        self.num_features = num_features
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.parameters_ = None
        self.parameters1_ = None

    def fit(self, X: np.ndarray) -> MultiRocket:
        # MultiRocket works on float64 input for both fit and transform
        X_f64 = X.astype(np.float64, copy=False)
        X1_f64 = np.diff(X_f64, axis=-1)
        
        # Fit on base series
        self.parameters_ = fit(X_f64, self.num_features, self.max_dilations_per_kernel)
        # Fit on first-order difference
        self.parameters1_ = fit(X1_f64, self.num_features, self.max_dilations_per_kernel)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.parameters_ is None or self.parameters1_ is None:
            raise ValueError("MultiRocket must be fitted before transforming.")
            
        is_single = X.ndim == 2
        X_in = X[np.newaxis, ...] if is_single else X
        
        # Numba's compiled transform signature requires contiguous float64 arrays
        X_f64 = np.ascontiguousarray(X_in, dtype=np.float64)
        X1_f64 = np.ascontiguousarray(np.diff(X_f64, axis=-1), dtype=np.float64)
        
        features = transform(X_f64, X1_f64, self.parameters_, self.parameters1_, 4)
        return features[0] if is_single else features
