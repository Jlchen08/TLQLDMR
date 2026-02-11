from dataclasses import dataclass

import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.linear_model import QuantileRegressor


@dataclass
class NFSSVQRConfig:
    gamma: float = 0.02
    n_components: int = 400
    alpha_fs: float = 0.001
    alpha_q: float = 0.001
    max_features: int = 120
    tau_low: float = 0.05
    tau_high: float = 0.95
    random_state: int = 42


class NFSSVQRIntervalModel:
    def __init__(self, config: NFSSVQRConfig | None = None):
        self.config = config or NFSSVQRConfig()
        self.rbf = RBFSampler(
            gamma=self.config.gamma,
            n_components=self.config.n_components,
            random_state=self.config.random_state,
        )
        self.selector = SelectFromModel(
            Lasso(alpha=self.config.alpha_fs, max_iter=5000, random_state=self.config.random_state),
            max_features=self.config.max_features,
        )
        self.model_low = QuantileRegressor(
            quantile=self.config.tau_low, alpha=self.config.alpha_q, solver="highs"
        )
        self.model_high = QuantileRegressor(
            quantile=self.config.tau_high, alpha=self.config.alpha_q, solver="highs"
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        Z = self.rbf.fit_transform(X)
        self.selector.fit(Z, y)
        Z_sel = self.selector.transform(Z)
        self.model_low.fit(Z_sel, y)
        self.model_high.fit(Z_sel, y)

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Z = self.rbf.transform(X)
        Z_sel = self.selector.transform(Z)
        lower = self.model_low.predict(Z_sel)
        upper = self.model_high.predict(Z_sel)
        return lower, upper
