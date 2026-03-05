from dataclasses import dataclass

import numpy as np

from experiment3.models.quantile_svr import QuantileSVR, QuantileSVRConfig


@dataclass
class StandardSVQRConfig:
    C: float = 8.0
    gamma: float = 0.01
    tau_low: float = 0.05
    tau_high: float = 0.95
    max_samples: int | None = 800


class StandardSVQRIntervalModel:
    """
    Standard SVQR baseline with pinball loss (two independent quantile SVRs).
    """

    def __init__(self, config: StandardSVQRConfig | None = None):
        self.config = config or StandardSVQRConfig()
        self.model_low = QuantileSVR(
            QuantileSVRConfig(
                C=self.config.C,
                gamma=self.config.gamma,
                tau=self.config.tau_low,
                kernel="rbf",
                max_samples=self.config.max_samples,
            )
        )
        self.model_high = QuantileSVR(
            QuantileSVRConfig(
                C=self.config.C,
                gamma=self.config.gamma,
                tau=self.config.tau_high,
                kernel="rbf",
                max_samples=self.config.max_samples,
            )
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model_low.fit(X, y)
        self.model_high.fit(X, y)

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lower = self.model_low.predict(X)
        upper = self.model_high.predict(X)
        return lower, upper
