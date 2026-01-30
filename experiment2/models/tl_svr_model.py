from dataclasses import dataclass

import numpy as np
from sklearn.svm import SVR


@dataclass
class TLSVRConfig:
    C: float = 10.0
    epsilon: float = 0.1
    gamma: str = "scale"
    kernel: str = "rbf"
    source_weight: float = 0.2


class TLSVRModel:
    def __init__(self, config: TLSVRConfig | None = None):
        self.config = config or TLSVRConfig()
        self.model = SVR(
            C=self.config.C,
            epsilon=self.config.epsilon,
            gamma=self.config.gamma,
            kernel=self.config.kernel,
        )

    def fit(self, X_S: np.ndarray, y_S: np.ndarray, X_T: np.ndarray, y_T: np.ndarray) -> None:
        X_all = np.concatenate([X_S, X_T], axis=0)
        y_all = np.concatenate([y_S, y_T], axis=0)
        weights = np.concatenate(
            [
                np.full(len(X_S), self.config.source_weight, dtype=np.float32),
                np.ones(len(X_T), dtype=np.float32),
            ],
            axis=0,
        )
        self.model.fit(X_all, y_all, sample_weight=weights)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
