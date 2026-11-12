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
        return int(self.pipeline.predict(x.series))

model = Model()
print(f"ridge alpha: {model.pipeline.classifier.model_.alpha_}")