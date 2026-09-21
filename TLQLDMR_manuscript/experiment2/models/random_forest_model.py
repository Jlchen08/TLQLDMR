from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor


@dataclass
class RandomForestConfig:
    n_estimators: int = 300
    max_depth: int | None = 12
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str | float = "sqrt"
    random_state: int = 42


class RandomForestModel:
    def __init__(self, config: RandomForestConfig | None = None):
        self.config = config or RandomForestConfig()
        self.model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            random_state=self.config.random_state,
            n_jobs=1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
