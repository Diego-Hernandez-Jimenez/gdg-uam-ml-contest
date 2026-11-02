from pathlib import Path

from competition_api import ABSTAIN, Sample

HERE = Path(__file__).parent  # load your files relative to this file


class Model:
    def __init__(self):
        # Load weights here. Untimed, but must finish within the setup timeout.
        self.weights = HERE / "weights.pt"

    def predict(self, x: Sample) -> int:
        # Return a class id in 0..7, or ABSTAIN (-1) to pass.
        return ABSTAIN