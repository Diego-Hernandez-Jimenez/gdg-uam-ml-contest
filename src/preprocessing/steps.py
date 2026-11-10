from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .base import Step


class Identity(Step):
    """Passthrough step that returns the input unchanged."""

    def transform(self, series: np.ndarray) -> np.ndarray:
        return series


class NaNInterpolation(Step):
    """Linear NaN interpolation per channel.

    Avoids artificial high-frequency artifacts that zero-filling would inject.
    If an entire channel contains only NaNs, it falls back to 0.0.

    Supports both batch mode (N, C, L) and streaming/inference mode (C, L).
    """

    def transform(self, series: np.ndarray) -> np.ndarray:
        # Avoid modifying the original input in place
        out = np.array(series, dtype=np.float64)

        # Unify dimensions internally to (N, C, L)
        is_single = out.ndim == 2
        X_3d = out[np.newaxis, ...] if is_single else out

        N, C, L = X_3d.shape
        t = np.arange(L)

        # Highly optimized memory contiguous loops for interpolation
        for i in range(N):
            for c in range(C):
                channel = X_3d[i, c]
                missing = np.isnan(channel)
                if missing.all():
                    channel[:] = 0.0
                elif missing.any():
                    channel[missing] = np.interp(
                        t[missing], t[~missing], channel[~missing]
                    )

        return out


class HampelFilter(Step):
    """Selective outlier filter using local median and MAD (Median Absolute Deviation).

    Suppresses impulsive spikes without smoothing original signal dynamics.
    Requires a gap-free series (should run AFTER NaNInterpolation).

    Fully vectorized across channels and samples for optimal latency.
    Supports both batch mode (N, C, L) and streaming/inference mode (C, L).
    """

    def __init__(self, window: int = 5, n_sigmas: float = 3.0):
        self.window = window
        self.n_sigmas = n_sigmas

    def transform(self, series: np.ndarray) -> np.ndarray:
        half = self.window // 2

        # Build dynamic padding based on array dimensions (pad only the last axis 'L')
        pad_width = [(0, 0)] * (series.ndim - 1) + [(half, half)]
        padded = np.pad(series, pad_width, mode="edge")

        # Shape: (..., L, window) - sliding window over the last dimension (-1)
        windows = sliding_window_view(padded, self.window, axis=-1)
        med = np.median(windows, axis=-1)

        # 1.4826 scales MAD to be consistent with standard deviation under normality
        mad = np.median(np.abs(windows - med[..., None]), axis=-1) * 1.4826 + 1e-9

        is_spike = np.abs(series - med) > self.n_sigmas * mad
        return np.where(is_spike, med, series)


class RobustScale(Step):
    """Local robust scaling per channel using Median and IQR (Interquartile Range).

    Removes DC bias and scales amplitude, keeping outliers from distorting the bounds.
    Should run AFTER HampelFilter.

    Supports both batch mode (N, C, L) and streaming/inference mode (C, L).
    """

    def __init__(self, epsilon: float = 1e-9):
        self.epsilon = epsilon

    def transform(self, series: np.ndarray) -> np.ndarray:
        median = np.median(series, axis=-1, keepdims=True)
        q75, q25 = np.percentile(series, [75, 25], axis=-1, keepdims=True)
        iqr = (q75 - q25) + self.epsilon
        return (series - median) / iqr
