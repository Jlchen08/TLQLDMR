from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import KNeighborsRegressor


@dataclass
class KNNConfig:
    n_neighbors: int = 10
    weights: str = "distance"
    p: int = 2


class KNNModel:
    def __init__(self, config: KNNConfig | None = None):
        self.config = config or KNNConfig()
        self.model = KNeighborsRegressor(
            n_neighbors=self.config.n_neighbors,
            weights=self.config.weights,
            p=self.config.p,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
