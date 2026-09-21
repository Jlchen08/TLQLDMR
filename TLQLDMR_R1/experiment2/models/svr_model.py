from dataclasses import dataclass

import numpy as np
from sklearn.svm import SVR


@dataclass
class SVRConfig:
    C: float = 10.0
    epsilon: float = 0.1
    gamma: str = "scale"
    kernel: str = "rbf"


class SVRModel:
    def __init__(self, config: SVRConfig | None = None):
        self.config = config or SVRConfig()
        self.model = SVR(
            C=self.config.C,
            epsilon=self.config.epsilon,
            gamma=self.config.gamma,
            kernel=self.config.kernel,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
