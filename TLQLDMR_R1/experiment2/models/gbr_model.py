from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


@dataclass
class GBRConfig:
    n_estimators: int = 120
    learning_rate: float = 0.05
    max_depth: int = 3
    subsample: float = 0.8
    min_samples_leaf: int = 5
    random_state: int = 42


class GBRModel:
    def __init__(self, config: GBRConfig | None = None):
        self.config = config or GBRConfig()
        self.model = GradientBoostingRegressor(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            subsample=self.config.subsample,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
