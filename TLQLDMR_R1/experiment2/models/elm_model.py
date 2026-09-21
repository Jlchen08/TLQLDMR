from dataclasses import dataclass

import numpy as np


@dataclass
class ELMConfig:
    n_hidden: int = 500
    activation: str = "sigmoid"
    reg: float = 1e-3
    seed: int = 42


class ELMModel:
    def __init__(self, config: ELMConfig | None = None):
        self.config = config or ELMConfig()
        self.W = None
        self.b = None
        self.beta = None

    def _activate(self, X: np.ndarray) -> np.ndarray:
        if self.config.activation == "sigmoid":
            X_clip = np.clip(X, -50, 50)
            return 1.0 / (1.0 + np.exp(-X_clip))
        if self.config.activation == "tanh":
            return np.tanh(X)
        if self.config.activation == "relu":
            return np.maximum(0.0, X)
        raise ValueError(f"Unsupported activation: {self.config.activation}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.config.seed)
        n_features = X.shape[1]
        self.W = rng.standard_normal((n_features, self.config.n_hidden)).astype(np.float32)
        self.b = rng.standard_normal((self.config.n_hidden,)).astype(np.float32)

        H = self._activate(X @ self.W + self.b)
        HtH = H.T @ H
        reg_I = self.config.reg * np.eye(HtH.shape[0], dtype=np.float32)
        self.beta = np.linalg.solve(HtH + reg_I, H.T @ y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        H = self._activate(X @ self.W + self.b)
        return H @ self.beta
