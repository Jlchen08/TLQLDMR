from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor


@dataclass
class BRFConfig:
    n_feature_groups: int = 4
    n_feature_nodes: int = 20
    n_enhance_nodes: int = 50
    activation: str = "tanh"
    n_estimators: int = 150
    max_depth: int | None = 6
    min_samples_leaf: int = 4
    max_features: float = 0.5
    random_state: int = 42


class BroadRandomForestModel:
    """Broad Random Forest (BLS feature expansion + RF)."""

    def __init__(self, config: BRFConfig | None = None):
        self.config = config or BRFConfig()
        self._rng = np.random.default_rng(self.config.random_state)
        self._feature_weights: list[tuple[np.ndarray, np.ndarray]] = []
        self._enhance_weights: tuple[np.ndarray, np.ndarray] | None = None
        self.model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            random_state=self.config.random_state,
            n_jobs=1,
        )

    def _activate(self, X: np.ndarray) -> np.ndarray:
        if self.config.activation == "tanh":
            return np.tanh(X)
        if self.config.activation == "sigmoid":
            X_clip = np.clip(X, -50, 50)
            return 1.0 / (1.0 + np.exp(-X_clip))
        if self.config.activation == "relu":
            return np.maximum(0.0, X)
        raise ValueError(f"Unsupported activation: {self.config.activation}")

    def _build_feature_nodes(self, X: np.ndarray) -> np.ndarray:
        n_features = X.shape[1]
        feature_nodes = []
        self._feature_weights = []
        for _ in range(int(self.config.n_feature_groups)):
            W = self._rng.standard_normal((n_features, self.config.n_feature_nodes))
            b = self._rng.standard_normal((self.config.n_feature_nodes,))
            self._feature_weights.append((W, b))
            Z = self._activate(X @ W + b)
            feature_nodes.append(Z)
        return np.concatenate(feature_nodes, axis=1)

    def _build_enhance_nodes(self, Z: np.ndarray) -> np.ndarray:
        W = self._rng.standard_normal((Z.shape[1], self.config.n_enhance_nodes))
        b = self._rng.standard_normal((self.config.n_enhance_nodes,))
        self._enhance_weights = (W, b)
        H = self._activate(Z @ W + b)
        return H

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if not self._feature_weights:
            raise RuntimeError("Model has not been fitted")
        feature_nodes = []
        for W, b in self._feature_weights:
            feature_nodes.append(self._activate(X @ W + b))
        Z = np.concatenate(feature_nodes, axis=1)
        if self._enhance_weights is None:
            raise RuntimeError("Model has not been fitted")
        W_e, b_e = self._enhance_weights
        H = self._activate(Z @ W_e + b_e)
        return np.concatenate([Z, H], axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).flatten()
        Z = self._build_feature_nodes(X)
        H = self._build_enhance_nodes(Z)
        F = np.concatenate([Z, H], axis=1)
        self.model.fit(F, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        F = self._transform(X)
        return self.model.predict(F)
