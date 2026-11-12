import os
from pathlib import Path

import joblib

from competition_api import Sample

HERE = Path(__file__).parent


class Model:
    """Official Competition Model Class."""

    def __init__(self):
        # Allow selecting specific weights via env var, defaulting to artifacts/latest.joblib
        default_path = HERE / "artifacts" / "latest.joblib"
        weights_path = Path(os.environ.get("MODEL_WEIGHTS", default_path))
        self.pipeline = joblib.load(weights_path)

    def predict(self, x: Sample) -> int:
        # x.series is a read-only float32 array of shape (4, 256)
        # series = np.asarray(x.series, dtype=np.float64)
        return int(self.pipeline.predict(x.series))
