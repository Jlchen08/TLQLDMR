from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass
class ClusteredGBRConfig:
    n_clusters: int = 4
    max_depth: int | None = 6
    max_iter: int = 200
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 20
    l2_regularization: float = 0.0
    random_state: int = 42
    min_cluster_samples: int = 50


class ClusteredGBRModel:
    """Cluster-then-GBT regression (KMeans + per-cluster GBT)."""

    def __init__(self, config: ClusteredGBRConfig | None = None):
        self.config = config or ClusteredGBRConfig()
        self.kmeans: KMeans | None = None
        self.cluster_models: dict[int, HistGradientBoostingRegressor] = {}
        self.global_model: HistGradientBoostingRegressor | None = None

    def _build_regressor(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            max_depth=self.config.max_depth,
            max_iter=self.config.max_iter,
            learning_rate=self.config.learning_rate,
            max_leaf_nodes=self.config.max_leaf_nodes,
            min_samples_leaf=self.config.min_samples_leaf,
            l2_regularization=self.config.l2_regularization,
            random_state=self.config.random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).flatten()

        self.kmeans = KMeans(
            n_clusters=self.config.n_clusters,
            random_state=self.config.random_state,
            n_init=10,
        )
        labels = self.kmeans.fit_predict(X)

        self.global_model = self._build_regressor()
        self.global_model.fit(X, y)

        self.cluster_models = {}
        for cluster_id in range(self.config.n_clusters):
            mask = labels == cluster_id
            if mask.sum() < self.config.min_cluster_samples:
                continue
            model = self._build_regressor()
            model.fit(X[mask], y[mask])
            self.cluster_models[cluster_id] = model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.kmeans is None or self.global_model is None:
            raise RuntimeError("Model has not been fitted")
        X = np.asarray(X, dtype=np.float64)
        labels = self.kmeans.predict(X)
        preds = np.empty(X.shape[0], dtype=np.float64)
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            model = self.cluster_models.get(int(cluster_id), self.global_model)
            preds[mask] = model.predict(X[mask])
        return preds
