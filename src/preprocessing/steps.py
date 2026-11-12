from __future__ import annotations

import numpy as np
from numba import njit
from numpy.lib.stride_tricks import sliding_window_view

from .base import Step


class Identity(Step):
    """Passthrough step that returns the input unchanged."""

    def transform(self, series: np.ndarray) -> np.ndarray:
        return series


class NaNInterpolation(Step):
    """Linear NaN interpolation per channel maintaining float32 precision."""

    def transform(self, series: np.ndarray) -> np.ndarray:
        if not np.isnan(series).any():
            return np.asarray(series, dtype=np.float32)

        out = np.array(series, dtype=np.float32, copy=True)
        is_single = out.ndim == 2
        X_3d = out[np.newaxis, ...] if is_single else out

        N, C, L = X_3d.shape
        t = np.arange(L, dtype=np.float32)

        for i in range(N):
            for c in range(C):
                channel = X_3d[i, c]
                missing = np.isnan(channel)
                if missing.all():
                    channel[:] = 0.0
                elif missing.any():
                    channel[missing] = np.interp(
                        t[missing], t[~missing], channel[~missing]
                    ).astype(np.float32)

        return out


@njit(fastmath=True, cache=True)
def _hampel_single_2d(series: np.ndarray, window: int, n_sigmas: float) -> np.ndarray:
    C, L = series.shape
    half = window // 2
    out = np.empty_like(series)
    buf = np.empty(window, dtype=series.dtype)
    dev = np.empty(window, dtype=series.dtype)

    for c in range(C):
        for t in range(L):
            for k in range(window):
                idx = t + k - half
                if idx < 0:
                    idx = 0
                elif idx >= L:
                    idx = L - 1
                buf[k] = series[c, idx]
            buf.sort()
            med = buf[half]
            for k in range(window):
                dev[k] = abs(buf[k] - med)
            dev.sort()
            mad = dev[half] * 1.4826 + 1e-9
            val = series[c, t]
            out[c, t] = med if abs(val - med) > n_sigmas * mad else val
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
        if series.ndim == 2:
            return _hampel_single_2d(np.ascontiguousarray(series), self.window, self.n_sigmas)

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
    """Local robust scaling per channel using Median and IQR in a single pass."""

    def __init__(self, epsilon: float = 1e-9):
        self.epsilon = np.float32(epsilon)

    def transform(self, series: np.ndarray) -> np.ndarray:
        # Single sort pass for 25%, 50% (median), and 75% quantiles
        q25, median, q75 = np.percentile(series, [25, 50, 75], axis=-1, keepdims=True)
        iqr = (q75 - q25) + self.epsilon
        return ((series - median) / iqr).astype(np.float32, copy=False)


