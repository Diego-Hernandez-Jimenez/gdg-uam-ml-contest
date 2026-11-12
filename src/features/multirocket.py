"""MultiRocket feature extractor (optimized zero-allocation float32 implementation)."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .base import FeatureExtractor

KERNEL_LENGTH = 9
NUM_KERNELS = 84
HALF_KERNEL = KERNEL_LENGTH // 2
N_FEATURES_PER_KERNEL = 4

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


@njit(
    "float32[:](float32[:,:,:],int32[:],int32[:],int32[:],int32[:],float32[:])",
    fastmath=True,
    parallel=False,
    cache=True,
)
def _fit_biases(
    X: np.ndarray,
    num_channels_per_combination: np.ndarray,
    channel_indices: np.ndarray,
    dilations: np.ndarray,
    num_features_per_dilation: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
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
            num_channels_this_combination = num_channels_per_combination[
                combination_index
            ]
            num_channels_end = num_channels_start + num_channels_this_combination
            channels_this_combination = channel_indices[
                num_channels_start:num_channels_end
            ]

            _X = X[np.random.randint(num_examples)][channels_this_combination]

            A = -_X
            G = _X + _X + _X

            C_alpha = A.copy()
            C_gamma = np.zeros(
                (KERNEL_LENGTH, num_channels_this_combination, input_length),
                dtype=np.float32,
            )
            C_gamma[HALF_KERNEL] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(HALF_KERNEL):
                C_alpha[:, -end:] += A[:, :end]
                C_gamma[gamma_index, :, -end:] = G[:, :end]
                end += dilation

            for gamma_index in range(HALF_KERNEL + 1, KERNEL_LENGTH):
                C_alpha[:, :-start] += A[:, start:]
                C_gamma[gamma_index, :, :-start] = G[:, start:]
                start += dilation

            index_0, index_1, index_2 = _INDICES[kernel_index]
            C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]
            C = np.sum(C, axis=0)

            biases[feature_index_start:feature_index_end] = np.quantile(
                C, quantiles[feature_index_start:feature_index_end]
            )

            feature_index_start = feature_index_end
            combination_index += 1
            num_channels_start = num_channels_end

    return biases


def _fit_dilations(
    input_length: int, num_features: int, max_dilations_per_kernel: int
) -> tuple[np.ndarray, np.ndarray]:
    num_features_per_kernel = num_features // NUM_KERNELS
    true_max_dilations_per_kernel = min(
        num_features_per_kernel, max_dilations_per_kernel
    )
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = np.log2((input_length - 1) / (KERNEL_LENGTH - 1))
    dilations, num_features_per_dilation = np.unique(
        np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(
            np.int32
        ),
        return_counts=True,
    )
    num_features_per_dilation = (num_features_per_dilation * multiplier).astype(
        np.int32
    )

    remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)

    return dilations, num_features_per_dilation


def _quantiles(n: int) -> np.ndarray:
    return np.array(
        [(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype=np.float32
    )


def fit(
    X: np.ndarray, num_features: int = 10_000, max_dilations_per_kernel: int = 32
) -> tuple:
    X_f32 = np.ascontiguousarray(X, dtype=np.float32)
    _, num_channels, input_length = X_f32.shape

    dilations, num_features_per_dilation = _fit_dilations(
        input_length, num_features, max_dilations_per_kernel
    )
    num_features_per_kernel = np.sum(num_features_per_dilation)
    quantiles = _quantiles(NUM_KERNELS * num_features_per_kernel)

    num_combinations = NUM_KERNELS * len(dilations)
    max_num_channels = min(num_channels, KERNEL_LENGTH)
    max_exponent = np.log2(max_num_channels + 1)
    num_channels_per_combination = (
        2 ** np.random.uniform(0, max_exponent, num_combinations)
    ).astype(np.int32)

    channel_indices = np.zeros(num_channels_per_combination.sum(), dtype=np.int32)

    num_channels_start = 0
    for combination_index in range(num_combinations):
        num_channels_this_combination = num_channels_per_combination[combination_index]
        num_channels_end = num_channels_start + num_channels_this_combination
        channel_indices[num_channels_start:num_channels_end] = np.random.choice(
            num_channels, num_channels_this_combination, replace=False
        )
        num_channels_start = num_channels_end

    biases = _fit_biases(
        X_f32,
        num_channels_per_combination,
        channel_indices,
        dilations,
        num_features_per_dilation,
        quantiles,
    )

    return (
        num_channels_per_combination,
        channel_indices,
        dilations,
        num_features_per_dilation,
        biases,
    )


@njit(cache=True, fastmath=True)
def _write_pooling_features(
    feature_row: np.ndarray,
    base_index: int,
    stride: int,
    C: np.ndarray,
    start_t: int,
    end_t: int,
    bias: np.float32,
) -> None:
    """Zero-allocation pooling over C[start_t:end_t] using pure integer/float32 registers."""
    n = end_t - start_t
    ppv = 0
    last_val = 0
    max_stretch = 0
    mean_index = 0
    mean = np.float32(0.0)

    for j in range(n):
        val = C[start_t + j]
        if val > bias:
            ppv += 1
            mean_index += j
            mean += val + bias
        elif val < bias:
            stretch = j - last_val
            max_stretch = max(max_stretch, stretch)
            last_val = j

    stretch = n - 1 - last_val
    max_stretch = max(max_stretch, stretch)

    feature_row[base_index] = np.float32(ppv) / np.float32(n)
    feature_row[base_index + stride] = np.float32(max_stretch)
    if ppv > 0:
        inv_ppv = np.float32(1.0) / np.float32(ppv)
        feature_row[base_index + 2 * stride] = mean * inv_ppv
        feature_row[base_index + 3 * stride] = np.float32(mean_index) * inv_ppv
    else:
        feature_row[base_index + 2 * stride] = np.float32(0.0)
        feature_row[base_index + 3 * stride] = np.float32(-1.0)


@njit(cache=True, fastmath=True)
def _transform_single_representation(
    _X: np.ndarray,
    num_channels_per_combination: np.ndarray,
    channel_indices: np.ndarray,
    dilations: np.ndarray,
    num_features_per_dilation: np.ndarray,
    biases: np.ndarray,
    feature_row: np.ndarray,
    row_offset: int,
    num_features_this_rep: int,
    C_alpha: np.ndarray,
    C_gamma: np.ndarray,
    C: np.ndarray,
) -> None:
    """Process one 2D series representation (base or diff) using pre-allocated buffers."""
    num_channels, input_length = _X.shape  # noqa: RUF059
    num_dilations = len(dilations)

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

        C_alpha[:, :input_length] = A
        C_gamma[:, :, :input_length] = 0.0
        C_gamma[HALF_KERNEL, :, :input_length] = G

        start = dilation
        end = input_length - padding
        for gamma_index in range(HALF_KERNEL):
            C_alpha[:, input_length - end : input_length] += A[:, :end]
            C_gamma[gamma_index, :, input_length - end : input_length] = G[:, :end]
            end += dilation
        for gamma_index in range(HALF_KERNEL + 1, KERNEL_LENGTH):
            C_alpha[:, : input_length - start] += A[:, start:]
            C_gamma[gamma_index, :, : input_length - start] = G[:, start:]
            start += dilation

        for kernel_index in range(NUM_KERNELS):
            feature_index_end = feature_index_start + num_features_this_dilation
            num_channels_this_combination = num_channels_per_combination[
                combination_index
            ]
            num_channels_end = num_channels_start + num_channels_this_combination

            _padding1 = (_padding0 + kernel_index) % 2
            index_0, index_1, index_2 = _INDICES[kernel_index]

            # Accumulate channels directly into pre-allocated 1D buffer C (0 mallocs)
            ch0 = channel_indices[num_channels_start]
            for t in range(input_length):
                C[t] = (
                    C_alpha[ch0, t]
                    + C_gamma[index_0, ch0, t]
                    + C_gamma[index_1, ch0, t]
                    + C_gamma[index_2, ch0, t]
                )
            for ch_idx in range(num_channels_start + 1, num_channels_end):
                ch = channel_indices[ch_idx]
                for t in range(input_length):
                    C[t] += (
                        C_alpha[ch, t]
                        + C_gamma[index_0, ch, t]
                        + C_gamma[index_1, ch, t]
                        + C_gamma[index_2, ch, t]
                    )

            start_t = 0 if _padding1 == 0 else padding
            end_t = input_length if _padding1 == 0 else input_length - padding

            for feature_count in range(num_features_this_dilation):
                feature_index = feature_index_start + feature_count
                _write_pooling_features(
                    feature_row,
                    row_offset + feature_index,
                    num_features_this_rep,
                    C,
                    start_t,
                    end_t,
                    biases[feature_index],
                )

            feature_index_start = feature_index_end
            combination_index += 1
            num_channels_start = num_channels_end


def _transform_core(
    X: np.ndarray,
    X1: np.ndarray,
    parameters,
    parameters1,
    n_features_per_kernel: int = 4,
) -> np.ndarray:
    num_examples, num_channels, input_length = X.shape

    (
        num_channels_per_combination,
        channel_indices,
        dilations,
        num_features_per_dilation,
        biases,
    ) = parameters
    _, _, dilations1, num_features_per_dilation1, biases1 = parameters1

    num_features = NUM_KERNELS * np.sum(num_features_per_dilation)
    num_features1 = NUM_KERNELS * np.sum(num_features_per_dilation1)

    features = np.empty(
        (num_examples, (num_features + num_features1) * n_features_per_kernel),
        dtype=np.float32,
    )
    offset_diff = num_features * n_features_per_kernel

    for example_index in prange(num_examples):
        feature_row = features[example_index]

        # Pre-allocate scratch buffers once per sample (reused across base & diff)
        C_alpha = np.empty((num_channels, input_length), dtype=np.float32)
        C_gamma = np.empty(
            (KERNEL_LENGTH, num_channels, input_length), dtype=np.float32
        )
        C = np.empty(input_length, dtype=np.float32)

        # 1. Base series pass
        _transform_single_representation(
            X[example_index],
            num_channels_per_combination,
            channel_indices,
            dilations,
            num_features_per_dilation,
            biases,
            feature_row,
            0,
            num_features,
            C_alpha,
            C_gamma,
            C,
        )

        # 2. First-order difference pass (reuses channel assignments and scratch buffers)
        _transform_single_representation(
            X1[example_index],
            num_channels_per_combination,
            channel_indices,
            dilations1,
            num_features_per_dilation1,
            biases1,
            feature_row,
            offset_diff,
            num_features1,
            C_alpha,
            C_gamma,
            C,
        )

    return features


_SIG = (
    "float32[:,:](float32[:,:,:],float32[:,:,:],"
    "Tuple((int32[:],int32[:],int32[:],int32[:],float32[:])),"
    "Tuple((int32[:],int32[:],int32[:],int32[:],float32[:])),int32)"
)
transform = njit(_SIG, fastmath=True, parallel=True, cache=True)(_transform_core)
_transform_single = njit(_SIG, fastmath=True, parallel=False, cache=True)(
    _transform_core
)


class MultiRocket(FeatureExtractor):
    """MultiRocket feature extractor wrapped in the standard FeatureExtractor contract."""

    def __init__(self, num_features: int = 10_000, max_dilations_per_kernel: int = 32):
        self.num_features = num_features
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.parameters_ = None
        self.parameters1_ = None

    def fit(self, X: np.ndarray) -> MultiRocket:
        X_f32 = np.ascontiguousarray(X, dtype=np.float32)
        X1_f32 = np.ascontiguousarray(np.diff(X_f32, axis=-1), dtype=np.float32)

        self.parameters_ = fit(X_f32, self.num_features, self.max_dilations_per_kernel)
        self.parameters1_ = fit(
            X1_f32, self.num_features, self.max_dilations_per_kernel
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.parameters_ is None or self.parameters1_ is None:
            raise ValueError("MultiRocket must be fitted before transforming.")

        is_single = X.ndim == 2
        X_in = X[np.newaxis, ...] if is_single else X

        X_f32 = np.ascontiguousarray(X_in, dtype=np.float32)
        X1_f32 = np.ascontiguousarray(np.diff(X_f32, axis=-1), dtype=np.float32)

        if is_single:
            return _transform_single(
                X_f32, X1_f32, self.parameters_, self.parameters1_, 4
            )[0]
        return transform(X_f32, X1_f32, self.parameters_, self.parameters1_, 4)
