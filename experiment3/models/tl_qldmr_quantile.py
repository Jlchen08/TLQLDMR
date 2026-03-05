from dataclasses import dataclass

import numpy as np

from Train_TL_QLDMR import TL_QLDMR
from Predict_TL_QLDMR import Predictor


@dataclass
class TLQLDMRQuantileConfig:
    lambda1: float = 0.01
    lambda2: float = 0.01
    C_S: float = 1.0
    C_T: float = 10.0
    kernel_gamma: float = 0.02
    solver: str = "fast_nystrom"
    nystrom_n_components: int = 200
    nystrom_lr: float = 0.01
    nystrom_epochs: int = 50
    nystrom_batch_size: int = 256
    variance_mode: str = "asymmetric"
    asym_scale: float = 0.25


class TLQLDMRQuantileIntervalModel:
    def __init__(self, config: TLQLDMRQuantileConfig, tau_low: float = 0.05, tau_high: float = 0.95):
        self.config = config
        self.tau_low = tau_low
        self.tau_high = tau_high

        self.model_low = TL_QLDMR(
            lambda1=config.lambda1,
            lambda2=config.lambda2,
            C_S=config.C_S,
            C_T=config.C_T,
            tau=tau_low,
            kernel_gamma=config.kernel_gamma,
            solver=config.solver,
            nystrom_n_components=config.nystrom_n_components,
            nystrom_lr=config.nystrom_lr,
            nystrom_epochs=config.nystrom_epochs,
            nystrom_batch_size=config.nystrom_batch_size,
            variance_mode=config.variance_mode,
            asym_scale=config.asym_scale,
        )
        self.model_high = TL_QLDMR(
            lambda1=config.lambda1,
            lambda2=config.lambda2,
            C_S=config.C_S,
            C_T=config.C_T,
            tau=tau_high,
            kernel_gamma=config.kernel_gamma,
            solver=config.solver,
            nystrom_n_components=config.nystrom_n_components,
            nystrom_lr=config.nystrom_lr,
            nystrom_epochs=config.nystrom_epochs,
            nystrom_batch_size=config.nystrom_batch_size,
            variance_mode=config.variance_mode,
            asym_scale=config.asym_scale,
        )
        self.predictor_low: Predictor | None = None
        self.predictor_high: Predictor | None = None

    def fit(self, X_S: np.ndarray, y_S: np.ndarray, X_T: np.ndarray, y_T: np.ndarray) -> None:
        success_low = self.model_low.fit(X_S, y_S, X_T, y_T)
        success_high = self.model_high.fit(X_S, y_S, X_T, y_T)
        if not success_low or not success_high:
            raise RuntimeError("TL-QLDMR training failed for quantile interval.")
        self.predictor_low = Predictor(self.model_low)
        self.predictor_high = Predictor(self.model_high)

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.predictor_low is None or self.predictor_high is None:
            self.predictor_low = Predictor(self.model_low)
            self.predictor_high = Predictor(self.model_high)
        lower = self.predictor_low.predict(X)
        upper = self.predictor_high.predict(X)
        return lower, upper
