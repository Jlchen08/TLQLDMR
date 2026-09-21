from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor


@dataclass
class QuantileRFConfig:
    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_leaf: int = 5
    random_state: int = 42
    n_jobs: int = -1
    tau_low: float = 0.05
    tau_high: float = 0.95


class QuantileRFIntervalModel:
    def __init__(self, config: QuantileRFConfig | None = None):
        self.config = config or QuantileRFConfig()
        self.model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not hasattr(self.model, "estimators_"):
            raise RuntimeError("QuantileRFIntervalModel is not fitted.")
        preds = np.stack([tree.predict(X) for tree in self.model.estimators_], axis=0)
        lower = np.quantile(preds, self.config.tau_low, axis=0)
        upper = np.quantile(preds, self.config.tau_high, axis=0)
        return lower, upper
