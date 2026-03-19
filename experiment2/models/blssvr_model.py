from dataclasses import dataclass

import numpy as np
from sklearn.kernel_approximation import Nystroem


@dataclass
class BLSSVRConfig:
    C: float = 6.0
    gamma: float = 0.002
    n_components: int = 600
    loss_scale: float = 0.08
    max_iter: int = 12
    tol: float = 1e-4
    random_state: int = 42


class BLSSVRModel:
    """
    Bounded-loss LS-SVR style baseline.

    The model uses a low-rank Nyström expansion of the RBF kernel and fits
    an iteratively reweighted ridge regressor, where the sample weights are
    updated by a bounded influence function to reduce the impact of outliers.
    """

    def __init__(self, config: BLSSVRConfig | None = None):
        self.config = config or BLSSVRConfig()
        self.mapper = Nystroem(
            kernel="rbf",
            gamma=self.config.gamma,
            n_components=self.config.n_components,
            random_state=self.config.random_state,
        )
        self.coef_: np.ndarray | None = None
        self.bias_: float = 0.0

    def _solve_weighted_ridge(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        sqrt_w = np.sqrt(np.clip(sample_weights, 1e-6, None))
        Z_w = Z * sqrt_w[:, None]
        y_w = y * sqrt_w

        ones = sqrt_w[:, None]
        design = np.hstack([Z_w, ones])
        reg = np.eye(design.shape[1], dtype=np.float64) / max(self.config.C, 1e-8)
        reg[-1, -1] = 0.0

        lhs = design.T @ design + reg
        rhs = design.T @ y_w
        beta = np.linalg.solve(lhs, rhs)
        return beta[:-1], float(beta[-1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if len(X) == 0:
            raise ValueError("BLSSVR received empty training data.")

        Z = self.mapper.fit_transform(X)
        sample_weights = np.ones(len(y), dtype=np.float64)
        coef = np.zeros(Z.shape[1], dtype=np.float64)
        bias = float(np.mean(y))

        for _ in range(self.config.max_iter):
            prev_coef = coef.copy()
            prev_bias = bias

            coef, bias = self._solve_weighted_ridge(Z, y, sample_weights)
            pred = Z @ coef + bias
            resid = y - pred

            scale = max(
                self.config.loss_scale,
                1.4826 * float(np.median(np.abs(resid))) + 1e-6,
            )
            scaled = resid / scale
            sample_weights = 1.0 / (1.0 + scaled * scaled)

            delta = np.linalg.norm(coef - prev_coef) / (np.linalg.norm(prev_coef) + 1e-8)
            delta += abs(bias - prev_bias) / (abs(prev_bias) + 1e-8)
            if delta < self.config.tol:
                break

        self.coef_ = coef
        self.bias_ = bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("BLSSVR model is not fitted.")
        Z = self.mapper.transform(np.asarray(X, dtype=np.float64))
        return Z @ self.coef_ + self.bias_
