from __future__ import annotations

import numpy as np

from .base import FeatureExtractor


class SummaryStatsExtractor(FeatureExtractor):
    """Extracts summary statistical features (RMS, IQR, diff variance) per channel.

    For C channels, yields 3 * C features:
    - Root Mean Square (energy/amplitude)
    - Interquartile Range (robust dispersion)
    - Variance of temporal first differences (roughness / high-frequency modulation)
    """

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Works seamlessly for both (C, L) -> (3*C,) and (N, C, L) -> (N, 3*C)
        rms = np.sqrt(np.mean(X**2, axis=-1))
        q75, q25 = np.percentile(X, [75, 25], axis=-1)
        iqr = q75 - q25
        diff_var = np.var(np.diff(X, axis=-1), axis=-1)

        return np.concatenate([rms, iqr, diff_var], axis=-1).astype(np.float32)
