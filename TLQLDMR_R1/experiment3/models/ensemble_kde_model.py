from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KernelDensity


@dataclass
class EnsembleKDEConfig:
    n_bins: int = 5
    kde_bandwidth: float = 0.15
    kde_samples: int = 4000
    rf_estimators: int = 200
    gbr_estimators: int = 150
    random_state: int = 42


class EnsembleKDEIntervalModel:
    def __init__(self, config: EnsembleKDEConfig | None = None):
        self.config = config or EnsembleKDEConfig()
        self.rf = RandomForestRegressor(
            n_estimators=self.config.rf_estimators,
            max_depth=None,
            min_samples_leaf=5,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        self.gbr = GradientBoostingRegressor(
            n_estimators=self.config.gbr_estimators,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=20,
            random_state=self.config.random_state,
        )
        self.bin_edges: np.ndarray | None = None
        self.residual_quantiles: list[tuple[float, float]] | None = None

    def _predict_point(self, X: np.ndarray) -> np.ndarray:
        pred_rf = self.rf.predict(X)
        pred_gbr = self.gbr.predict(X)
        return 0.5 * pred_rf + 0.5 * pred_gbr

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        self.rf.fit(X_train, y_train)
        self.gbr.fit(X_train, y_train)

        pred_cal = self._predict_point(X_cal)
        residuals = y_cal - pred_cal

        n_bins = max(2, int(self.config.n_bins))
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        self.bin_edges = np.quantile(pred_cal, quantiles)
        self.bin_edges[0] -= 1e-6
        self.bin_edges[-1] += 1e-6

        self.residual_quantiles = []
        for b in range(n_bins):
            mask = (pred_cal >= self.bin_edges[b]) & (pred_cal < self.bin_edges[b + 1])
            if np.sum(mask) < 20:
                res_bin = residuals
            else:
                res_bin = residuals[mask]
            kde = KernelDensity(bandwidth=self.config.kde_bandwidth, kernel="gaussian")
            kde.fit(res_bin.reshape(-1, 1))
            samples = kde.sample(self.config.kde_samples, random_state=self.config.random_state)
            q_low, q_high = np.quantile(samples, [0.05, 0.95])
            self.residual_quantiles.append((float(q_low), float(q_high)))

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.bin_edges is None or self.residual_quantiles is None:
            raise RuntimeError("EnsembleKDEIntervalModel is not fitted.")

        pred = self._predict_point(X)
        lower = np.empty_like(pred)
        upper = np.empty_like(pred)

        for i, val in enumerate(pred):
            bin_idx = np.searchsorted(self.bin_edges, val, side="right") - 1
            bin_idx = int(np.clip(bin_idx, 0, len(self.residual_quantiles) - 1))
            q_low, q_high = self.residual_quantiles[bin_idx]
            lower[i] = val + q_low
            upper[i] = val + q_high

        return lower, upper
