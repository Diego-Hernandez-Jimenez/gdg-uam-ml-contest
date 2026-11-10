from __future__ import annotations

import numpy as np


class Step:
    """Base interface for all preprocessing steps in the pipeline."""
    
    def fit(self, X: np.ndarray) -> Step:
        """Fit the preprocessing step on a training set X.
        
        Default behavior is a no-op (stateless step).
        """
        return self

    def transform(self, series: np.ndarray) -> np.ndarray:
        """Apply the transformation to a series of shape (C, L).
        
        Must be optimized for low-latency inference.
        """
        raise NotImplementedError


class Pipeline(Step):
    """Sequentially chains multiple preprocessing steps together.
    
    Enables highly flexible and composable pipelines.
    """
    
    def __init__(self, steps: list[Step]):
        self.steps = steps

    def fit(self, X: np.ndarray) -> Pipeline:
        """Fits all steps in the sequence."""
        for step in self.steps:
            step.fit(X)
        return self

    def transform(self, series: np.ndarray) -> np.ndarray:
        """Transforms the input sequentially through all steps."""
        out = series
        for step in self.steps:
            out = step.transform(out)
        return out
