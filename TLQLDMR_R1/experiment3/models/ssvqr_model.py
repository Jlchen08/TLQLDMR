from dataclasses import dataclass

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import QuantileRegressor


@dataclass
class SSVQRConfig:
    gamma: float = 0.02
    n_components: int = 300
    alpha: float = 0.001
    tau_low: float = 0.05
    tau_high: float = 0.95
    random_state: int = 42


class SparseSVQRIntervalModel:
    def __init__(self, config: SSVQRConfig | None = None):
        self.config = config or SSVQRConfig()
        self.rbf = RBFSampler(
            gamma=self.config.gamma,
            n_components=self.config.n_components,
            random_state=self.config.random_state,
        )
        self.model_low = QuantileRegressor(
            quantile=self.config.tau_low, alpha=self.config.alpha, solver="highs"
        )
        self.model_high = QuantileRegressor(
            quantile=self.config.tau_high, alpha=self.config.alpha, solver="highs"
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Z = self.rbf.fit_transform(X)
        self.model_low.fit(Z, y)
        self.model_high.fit(Z, y)

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Z = self.rbf.transform(X)
        lower = self.model_low.predict(Z)
        upper = self.model_high.predict(Z)
        return lower, upper
