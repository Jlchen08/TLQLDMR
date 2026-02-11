from dataclasses import dataclass

import numpy as np
from sklearn.kernel_ridge import KernelRidge


@dataclass
class KernelRidgeConfig:
    alpha: float = 1.0
    gamma: float = 0.01
    kernel: str = "rbf"


class KernelRidgeModel:
    def __init__(self, config: KernelRidgeConfig | None = None):
        self.config = config or KernelRidgeConfig()
        self.model = KernelRidge(
            alpha=self.config.alpha,
            gamma=self.config.gamma,
            kernel=self.config.kernel,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
