from __future__ import annotations

import numpy as np

from .base import FeatureExtractor

QUANTILES = (10, 25, 50, 75, 90)
ACF_LAGS = (1, 2, 4, 8, 16)
N_BANDS = 4


class BaselineFeaturesExtractor(FeatureExtractor):
    """Extracts the complete set of hand-crafted statistical, temporal, spectral,
    and cross-channel features from the competition baseline model.

    Vectorized to support both batch (N, C, L) and single-sample (C, L) inputs.
    For C=4 channels and default constants, yields 78 features per sample:
    - Level & scale on raw input: 2 * C = 8
    - Distribution quantiles on z: 5 * C = 20
    - Roughness of first diffs on z: 2 * C = 8
    - Autocorrelation at lags on z: 5 * C = 20
    - Log spectral band power on z: 4 * C = 16
    - Cross-channel correlations: C * (C - 1) // 2 = 6
    """

    def __init__(
        self,
        quantiles: tuple[int, ...] = QUANTILES,
        acf_lags: tuple[int, ...] = ACF_LAGS,
        n_bands: int = N_BANDS,
    ):
        self.quantiles = quantiles
        self.acf_lags = acf_lags
        self.n_bands = n_bands

    def transform(self, X: np.ndarray) -> np.ndarray:
        is_single = X.ndim == 2
        if is_single:
            X = X[np.newaxis, ...]  # Promote (C, L) -> (1, C, L)

        N, C, L = X.shape

        # 1. Level and scale features on input signal: (N, C, 2)
        mean = X.mean(axis=-1, keepdims=True)
        std = X.std(axis=-1, keepdims=True)
        level_scale = np.concatenate([mean, std], axis=-1)

        # Standardize per sample and channel for shape/spectral/correlation groups
        z = (X - mean) / np.where(std < 1e-8, 1.0, std)

        # 2. Distribution features (percentiles): (Q, N, C) -> (N, C, Q)
        dist = np.percentile(z, self.quantiles, axis=-1).transpose(1, 2, 0)

        # 3. Roughness features: (N, C, 2)
        steps = np.diff(z, axis=-1)
        roughness = np.stack([steps.std(axis=-1), np.abs(steps).mean(axis=-1)], axis=-1)

        # 4. Autocorrelation features: (N, C, len(acf_lags))
        acf = np.stack(
            [np.mean(z[:, :, lag:] * z[:, :, :-lag], axis=-1) for lag in self.acf_lags],
            axis=-1,
        )

        # 5. Spectral features: (N, C, n_bands)
        power = np.abs(np.fft.rfft(z, axis=-1))[:, :, 1:] ** 2
        bands = np.array_split(power, self.n_bands, axis=-1)
        spectral = np.log1p(np.stack([band.sum(axis=-1) for band in bands], axis=-1))

        # Concatenate all per-channel groups: (N, C, 2 + Q + 2 + Lags + Bands) -> flatten to (N, C * F_ch)
        per_channel = np.concatenate(
            [level_scale, dist, roughness, acf, spectral], axis=-1
        ).reshape(N, -1)

        # 6. Cross-channel correlation features: (N, C, C) -> upper triangle (N, C*(C-1)//2)
        corr = np.matmul(z, z.transpose(0, 2, 1)) / L
        triu_idx = np.triu_indices(C, k=1)
        cross_channel = corr[:, triu_idx[0], triu_idx[1]]

        # Final feature matrix: (N, 78)
        out = np.concatenate([per_channel, cross_channel], axis=1).astype(np.float32)

        return out[0] if is_single else out
