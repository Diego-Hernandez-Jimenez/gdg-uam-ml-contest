from __future__ import annotations

import numpy as np

from .base import FeatureExtractor


class SummaryStatsExtractor(FeatureExtractor):
    """Extracts summary statistical features and scale-invariant ratios per channel.

    For C channels, yields 6 * C features:
    - Root Mean Square (energy/amplitude)
    - Interquartile Range (robust dispersion)
    - Variance of temporal first differences (roughness / high-frequency modulation)
    - Cross-channel IQR ratio (scale-invariant relative channel dominance)
    - Normalized roughness (scale-free Hjorth mobility proxy: std(diff) / IQR)
    - Zero-crossing rate of first differences (scale- and offset-invariant inflection rate)
    """

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Works seamlessly for both (C, L) -> (6*C,) and (N, C, L) -> (N, 6*C)
        eps = 1e-8

        rms = np.sqrt(np.mean(X**2, axis=-1))
        q75, q25 = np.percentile(X, [75, 25], axis=-1)
        iqr = q75 - q25
        diff = np.diff(X, axis=-1)
        diff_var = np.var(diff, axis=-1)

        # 1. Cross-channel relative dispersion (invariant to global amplitude shifts)
        iqr_ratio = iqr / (np.sum(iqr, axis=-1, keepdims=True) + eps)

        # 2. Scale-invariant roughness / Hjorth mobility proxy
        norm_roughness = np.sqrt(diff_var) / np.maximum(iqr, eps)

        # 3. Zero-crossing rate of first differences (inflection point frequency)
        zcr_diff = np.mean(
            (diff[..., :-1] * diff[..., 1:]) < 0, axis=-1, dtype=np.float32
        )

        return np.concatenate(
            [rms, iqr, diff_var, iqr_ratio, norm_roughness, zcr_diff], axis=-1
        ).astype(np.float32)
