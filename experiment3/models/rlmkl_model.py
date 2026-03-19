from dataclasses import dataclass, field

import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class RLMKLConfig:
    gammas: tuple[float, ...] = field(default_factory=lambda: (0.001, 0.003, 0.008))
    n_components: int = 120
    alpha: float = 0.0015
    tau_low: float = 0.05
    tau_high: float = 0.95
    random_state: int = 42


class RLMKLIntervalModel:
    """
    Low-rank multiple-kernel interval regressor.

    Each kernel is approximated by a Nyström map, then the feature blocks are
    combined by data-driven kernel weights before fitting two quantile
    regressors for the lower and upper bounds.
    """

    def __init__(self, config: RLMKLConfig | None = None):
        self.config = config or RLMKLConfig()
        self.mappers = [
            Nystroem(
                kernel="rbf",
                gamma=gamma,
                n_components=self.config.n_components,
                random_state=self.config.random_state + idx,
            )
            for idx, gamma in enumerate(self.config.gammas)
        ]
        self.block_weights_: np.ndarray | None = None
        self.scaler = StandardScaler(with_mean=False)
        self.model_low = QuantileRegressor(
            quantile=self.config.tau_low,
            alpha=self.config.alpha,
            solver="highs",
        )
        self.model_high = QuantileRegressor(
            quantile=self.config.tau_high,
            alpha=self.config.alpha,
            solver="highs",
        )

    def _fit_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        blocks = [mapper.fit_transform(X) for mapper in self.mappers]
        y_center = y - np.mean(y)
        scores = []
        for block in blocks:
            proj = block.T @ y_center
            score = float(np.linalg.norm(proj)) / (float(np.linalg.norm(block)) + 1e-8)
            scores.append(max(score, 1e-8))
        weights = np.asarray(scores, dtype=np.float64)
        weights /= np.sum(weights)
        self.block_weights_ = weights
        combined = np.hstack([np.sqrt(w) * block for w, block in zip(weights, blocks)])
        return self.scaler.fit_transform(combined)

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        if self.block_weights_ is None:
            raise RuntimeError("RLMKL model is not fitted.")
        blocks = [mapper.transform(X) for mapper in self.mappers]
        combined = np.hstack(
            [np.sqrt(w) * block for w, block in zip(self.block_weights_, blocks)]
        )
        return self.scaler.transform(combined)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if len(X) == 0:
            raise ValueError("RLMKL received empty training data.")
        Z = self._fit_features(X, y)
        self.model_low.fit(Z, y)
        self.model_high.fit(Z, y)

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Z = self._transform_features(np.asarray(X, dtype=np.float64))
        lower = self.model_low.predict(Z)
        upper = self.model_high.predict(Z)
        return lower, upper
