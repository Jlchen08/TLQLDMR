from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FLSVRConfig:
    C: float = 10.0
    epsilon: float = 0.05
    gamma: float = 0.001
    max_iter: int = 50
    tol: float = 1e-4
    alpha: float | None = None


class FLSVRModel:
    """Functional-iteration Lagrangian SVR (FLSVR)."""

    def __init__(self, config: FLSVRConfig | None = None):
        self.config = config or FLSVRConfig()
        self._train_aug: np.ndarray | None = None
        self._u: np.ndarray | None = None
        self._q_inv: np.ndarray | None = None

    @staticmethod
    def _rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
        X1_sq = np.sum(X1 * X1, axis=1, keepdims=True)
        X2_sq = np.sum(X2 * X2, axis=1, keepdims=True).T
        dist = X1_sq + X2_sq - 2.0 * X1 @ X2.T
        return np.exp(-gamma * dist)

    def _build_augmented(self, X: np.ndarray) -> np.ndarray:
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.concatenate([X, ones], axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).flatten()
        m = X.shape[0]
        if m == 0:
            raise ValueError("Empty training set for FLSVR")

        G = self._build_augmented(X)
        K = self._rbf_kernel(G, G, self.config.gamma)
        Q = np.eye(m, dtype=np.float64) / float(self.config.C) + K

        # Precompute inverse for iterative updates
        Q_inv = np.linalg.inv(Q)
        self._q_inv = Q_inv

        e = np.ones(m, dtype=np.float64)
        r1 = y - self.config.epsilon * e
        r2 = -y - self.config.epsilon * e

        if self.config.alpha is None:
            eigvals = np.linalg.eigvalsh(Q)
            lambda_min = max(float(eigvals[0]), 1e-8)
            upper = 2.0 / (3.0 * float(self.config.C))
            alpha = min(upper * 0.9, max(lambda_min * 1.1, 1e-6))
            if alpha <= 0 or alpha >= upper:
                alpha = upper * 0.5
        else:
            alpha = float(self.config.alpha)

        u = np.zeros(m, dtype=np.float64)

        for _ in range(int(self.config.max_iter)):
            u_plus = np.maximum(u, 0.0)
            term1 = Q @ u - r1 - alpha * u_plus
            term2 = -Q @ u - r2 - alpha * (u_plus - u)
            rhs = y + 0.5 * (np.abs(term1) - np.abs(term2) - alpha * u)
            u_new = Q_inv @ rhs
            if np.linalg.norm(u_new - u) < self.config.tol:
                u = u_new
                break
            u = u_new

        self._train_aug = G
        self._u = u

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._train_aug is None or self._u is None:
            raise RuntimeError("Model has not been fitted")
        X = np.asarray(X, dtype=np.float64)
        G_test = self._build_augmented(X)
        K_test = self._rbf_kernel(G_test, self._train_aug, self.config.gamma)
        return K_test @ self._u
