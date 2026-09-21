from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


@dataclass
class QuantileGBRConfig:
    n_estimators: int = 200
    learning_rate: float = 0.05
    max_depth: int = 3
    min_samples_leaf: int = 20
    tau_low: float = 0.05
    tau_high: float = 0.95
    random_state: int = 42


class QuantileGBRIntervalModel:
    def __init__(self, config: QuantileGBRConfig | None = None):
        self.config = config or QuantileGBRConfig()
        self.model_low = GradientBoostingRegressor(
            loss="quantile",
            alpha=self.config.tau_low,
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
        )
        self.model_high = GradientBoostingRegressor(
            loss="quantile",
            alpha=self.config.tau_high,
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model_low.fit(X, y)
        self.model_high.fit(X, y)

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lower = self.model_low.predict(X)
        upper = self.model_high.predict(X)
        return lower, upper
