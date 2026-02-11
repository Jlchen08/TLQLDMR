from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge


@dataclass
class StackingWPFConfig:
    rf_n_estimators: int = 150
    rf_max_depth: int = 6
    rf_min_samples_leaf: int = 4
    rf_max_features: float = 0.5
    gbr_n_estimators: int = 120
    gbr_learning_rate: float = 0.05
    gbr_max_depth: int = 3
    gbr_subsample: float = 0.8
    gbr_min_samples_leaf: int = 5
    knn_n_neighbors: int = 10
    knn_weights: str = "distance"
    knn_p: int = 2
    meta_alpha: float = 1.0
    cv: int = 3
    random_state: int = 42


class StackingWPFModel:
    def __init__(self, config: StackingWPFConfig | None = None):
        self.config = config or StackingWPFConfig()
        rf = RandomForestRegressor(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            min_samples_leaf=self.config.rf_min_samples_leaf,
            max_features=self.config.rf_max_features,
            random_state=self.config.random_state,
            n_jobs=1,
        )
        gbr = GradientBoostingRegressor(
            n_estimators=self.config.gbr_n_estimators,
            learning_rate=self.config.gbr_learning_rate,
            max_depth=self.config.gbr_max_depth,
            subsample=self.config.gbr_subsample,
            min_samples_leaf=self.config.gbr_min_samples_leaf,
            random_state=self.config.random_state,
        )
        knn = KNeighborsRegressor(
            n_neighbors=self.config.knn_n_neighbors,
            weights=self.config.knn_weights,
            p=self.config.knn_p,
        )
        meta = Ridge(alpha=self.config.meta_alpha, random_state=self.config.random_state)

        self.model = StackingRegressor(
            estimators=[("rf", rf), ("gbr", gbr), ("knn", knn)],
            final_estimator=meta,
            cv=self.config.cv,
            passthrough=False,
            n_jobs=None,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
