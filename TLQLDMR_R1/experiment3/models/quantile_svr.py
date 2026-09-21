from dataclasses import dataclass
from typing import Optional

import numpy as np
import cvxopt


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X_norm = np.sum(X * X, axis=1)[:, None]
    Y_norm = np.sum(Y * Y, axis=1)[None, :]
    dist = X_norm + Y_norm - 2.0 * X @ Y.T
    return np.exp(-gamma * dist)


def _linear_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float64) @ np.asarray(Y, dtype=np.float64).T


@dataclass
class QuantileSVRConfig:
    C: float = 10.0
    gamma: float = 0.01
    tau: float = 0.5
    kernel: str = "rbf"
    max_samples: Optional[int] = None


class QuantileSVR:
    def __init__(self, config: QuantileSVRConfig | None = None):
        self.config = config or QuantileSVRConfig()
        self.X_train: Optional[np.ndarray] = None
        self.beta: Optional[np.ndarray] = None
        self.b: float = 0.0

    def _kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        if self.config.kernel == "linear":
            return _linear_kernel(X, Y)
        return _rbf_kernel(X, Y, gamma=self.config.gamma)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)

        if self.config.max_samples is not None and len(X) > self.config.max_samples:
            X = X[: self.config.max_samples]
            y = y[: self.config.max_samples]

        n = len(X)
        if n == 0:
            raise ValueError("QuantileSVR received empty training data.")

        K = self._kernel(X, X)
        K = K + 1e-8 * np.eye(n)

        P_top = np.concatenate([K, -K], axis=1)
        P_bottom = np.concatenate([-K, K], axis=1)
        P = np.concatenate([P_top, P_bottom], axis=0)

        q = np.concatenate([-y, y], axis=0)

        C_tau = self.config.C * self.config.tau
        C_inv = self.config.C * (1.0 - self.config.tau)
        ub = np.concatenate([np.full(n, C_tau), np.full(n, C_inv)])

        G = np.vstack([np.eye(2 * n), -np.eye(2 * n)])
        h = np.concatenate([ub, np.zeros(2 * n)])

        A = np.concatenate([np.ones(n), -np.ones(n)])[None, :]
        b = np.zeros(1)

        cvxopt.solvers.options["show_progress"] = False
        sol = cvxopt.solvers.qp(
            cvxopt.matrix(P),
            cvxopt.matrix(q),
            cvxopt.matrix(G),
            cvxopt.matrix(h),
            cvxopt.matrix(A),
            cvxopt.matrix(b),
        )

        z = np.array(sol["x"]).reshape(-1)
        alpha = z[:n]
        alpha_star = z[n:]
        beta = alpha - alpha_star

        # Bias estimation
        support_mask = (
            ((alpha > 1e-6) & (alpha < C_tau - 1e-6))
            | ((alpha_star > 1e-6) & (alpha_star < C_inv - 1e-6))
        )
        if np.any(support_mask):
            idx = np.where(support_mask)[0]
            K_sv = K[idx]
            b_vals = y[idx] - K_sv @ beta
            b_est = float(np.mean(b_vals))
        else:
            b_est = float(np.mean(y - K @ beta))

        self.X_train = X
        self.beta = beta
        self.b = b_est

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.beta is None:
            raise RuntimeError("QuantileSVR model is not fitted.")
        K = self._kernel(np.asarray(X, dtype=np.float64), self.X_train)
        return K @ self.beta + self.b
