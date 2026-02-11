from dataclasses import dataclass

import numpy as np
from sklearn.svm import NuSVR


@dataclass
class NuSVRConfig:
    C: float = 10.0
    nu: float = 0.5
    gamma: str = "scale"
    kernel: str = "rbf"


class NuSVRModel:
    def __init__(self, config: NuSVRConfig | None = None):
        self.config = config or NuSVRConfig()
        self.model = NuSVR(
            C=self.config.C,
            nu=self.config.nu,
            gamma=self.config.gamma,
            kernel=self.config.kernel,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
