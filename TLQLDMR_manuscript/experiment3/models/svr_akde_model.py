from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.svm import SVR


@dataclass
class SVRAKDEConfig:
    C: float = 10.0
    gamma: float = 0.01
    epsilon: float = 0.1
    kernel: str = "rbf"
    n_bins: int = 5
    bandwidth: float = 0.2
    kde_samples: int = 4000
    alpha: float = 0.5
    tau_low: float = 0.05
    tau_high: float = 0.95
    random_state: int = 42


class SVRAKDEIntervalModel:
    def __init__(self, config: SVRAKDEConfig | None = None):
        self.config = config or SVRAKDEConfig()
        self.model = SVR(
            C=self.config.C,
            gamma=self.config.gamma,
            epsilon=self.config.epsilon,
            kernel=self.config.kernel,
        )
        self.bin_edges: np.ndarray | None = None
        self.residual_quantiles: list[tuple[float, float]] | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        self.model.fit(X_train, y_train)

        pred_cal = self.model.predict(X_cal)
        residuals = y_cal - pred_cal

        n_bins = max(2, int(self.config.n_bins))
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        self.bin_edges = np.quantile(pred_cal, quantiles)
        self.bin_edges[0] -= 1e-6
        self.bin_edges[-1] += 1e-6

        rng = np.random.default_rng(self.config.random_state)
        self.residual_quantiles = []
        for b in range(n_bins):
            mask = (pred_cal >= self.bin_edges[b]) & (pred_cal < self.bin_edges[b + 1])
            res_bin = residuals[mask] if np.sum(mask) >= 20 else residuals
            if len(res_bin) == 0:
                self.residual_quantiles.append((0.0, 0.0))
                continue

            kde = KernelDensity(bandwidth=self.config.bandwidth, kernel="gaussian")
            kde.fit(res_bin.reshape(-1, 1))
            log_f = kde.score_samples(res_bin.reshape(-1, 1))
            f_hat = np.exp(log_f)
            eps = 1e-12
            g = float(np.exp(np.mean(np.log(f_hat + eps))))
            lambda_i = (g / (f_hat + eps)) ** self.config.alpha

            idx = rng.integers(0, len(res_bin), size=self.config.kde_samples)
            base = res_bin[idx]
            lam = lambda_i[idx]
            noise = rng.normal(0.0, self.config.bandwidth * lam)
            samples = base + noise
            q_low, q_high = np.quantile(samples, [self.config.tau_low, self.config.tau_high])
            self.residual_quantiles.append((float(q_low), float(q_high)))

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.bin_edges is None or self.residual_quantiles is None:
            raise RuntimeError("SVRAKDEIntervalModel is not fitted.")

        pred = self.model.predict(X)
        lower = np.empty_like(pred)
        upper = np.empty_like(pred)

        for i, val in enumerate(pred):
            bin_idx = np.searchsorted(self.bin_edges, val, side="right") - 1
            bin_idx = int(np.clip(bin_idx, 0, len(self.residual_quantiles) - 1))
            q_low, q_high = self.residual_quantiles[bin_idx]
            lower[i] = val + q_low
            upper[i] = val + q_high

        return lower, upper
