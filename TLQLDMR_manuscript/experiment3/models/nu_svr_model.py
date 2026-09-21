from dataclasses import dataclass

import numpy as np
from sklearn.svm import NuSVR


@dataclass
class NuSVRConfig:
    nu: float = 0.5
    C: float = 10.0
    gamma: float = 0.01
    kernel: str = "rbf"


class NuSVRIntervalModel:
    def __init__(self, config: NuSVRConfig | None = None):
        self.config = config or NuSVRConfig()
        self.model = NuSVR(
            nu=self.config.nu,
            C=self.config.C,
            gamma=self.config.gamma,
            kernel=self.config.kernel,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pred = self.model.predict(X)
        return pred, pred
