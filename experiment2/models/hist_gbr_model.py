from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass
class HistGBRConfig:
    max_depth: int | None = 6
    max_iter: int = 200
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 20
    l2_regularization: float = 0.0
    random_state: int = 42


class HistGBRModel:
    def __init__(self, config: HistGBRConfig | None = None):
        self.config = config or HistGBRConfig()
        self.model = HistGradientBoostingRegressor(
            max_depth=self.config.max_depth,
            max_iter=self.config.max_iter,
            learning_rate=self.config.learning_rate,
            max_leaf_nodes=self.config.max_leaf_nodes,
            min_samples_leaf=self.config.min_samples_leaf,
            l2_regularization=self.config.l2_regularization,
            random_state=self.config.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
