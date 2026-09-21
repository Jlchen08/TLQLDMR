from dataclasses import dataclass

import numpy as np
from sklearn.svm import SVR


@dataclass
class ARASVRConfig:
    C: float = 10.0
    epsilon: float = 0.05
    gamma: float = 0.001
    corr_threshold: float = 0.4
    min_features: int = 10


class ARASVRModel:
    """SVR with correlation-based associated feature selection."""

    def __init__(self, config: ARASVRConfig | None = None):
        self.config = config or ARASVRConfig()
        self.model = SVR(
            C=self.config.C,
            epsilon=self.config.epsilon,
            gamma=self.config.gamma,
            kernel="rbf",
        )
        self.selected_idx: np.ndarray | None = None

    @staticmethod
    def _corrcoef(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        y_centered = y - y.mean()
        y_std = y_centered.std()
        if y_std == 0:
            return np.zeros(X.shape[1], dtype=np.float64)
        X_centered = X - X.mean(axis=0)
        X_std = X_centered.std(axis=0)
        denom = X_std * y_std
        denom = np.where(denom == 0, 1.0, denom)
        corr = (X_centered * y_centered[:, None]).mean(axis=0) / denom
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        return corr

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).flatten()
        corr = np.abs(self._corrcoef(X, y))
        selected = np.where(corr >= self.config.corr_threshold)[0]
        if selected.size < int(self.config.min_features):
            topk = np.argsort(corr)[::-1][: int(self.config.min_features)]
            selected = np.unique(topk)
        if selected.size == 0:
            selected = np.arange(X.shape[1])
        self.selected_idx = selected
        self.model.fit(X[:, self.selected_idx], y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.selected_idx is None:
            raise RuntimeError("Model has not been fitted")
        X = np.asarray(X, dtype=np.float64)
        return self.model.predict(X[:, self.selected_idx])
