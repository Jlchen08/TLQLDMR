from dataclasses import dataclass

import numpy as np

from Train_TL_QLDMR import TL_QLDMR
from Predict_TL_QLDMR import Predictor


@dataclass
class TLQLDMRConfig:
    lambda1: float = 0.01
    lambda2: float = 0.01
    C_S: float = 1.0
    C_T: float = 10.0
    tau: float = 0.5
    kernel_gamma: float = 0.02
    solver: str = "fast_nystrom"
    nystrom_n_components: int = 200
    nystrom_lr: float = 0.01
    nystrom_epochs: int = 50
    nystrom_batch_size: int = 256


class TLQLDMRMedianModel:
    def __init__(self, config: TLQLDMRConfig | None = None):
        self.config = config or TLQLDMRConfig()
        self.model = TL_QLDMR(
            lambda1=self.config.lambda1,
            lambda2=self.config.lambda2,
            C_S=self.config.C_S,
            C_T=self.config.C_T,
            tau=self.config.tau,
            kernel_gamma=self.config.kernel_gamma,
            solver=self.config.solver,
            nystrom_n_components=self.config.nystrom_n_components,
            nystrom_lr=self.config.nystrom_lr,
            nystrom_epochs=self.config.nystrom_epochs,
            nystrom_batch_size=self.config.nystrom_batch_size,
        )
        self.predictor = None

    def fit(self, X_S: np.ndarray, y_S: np.ndarray, X_T: np.ndarray, y_T: np.ndarray) -> None:
        success = self.model.fit(X_S, y_S, X_T, y_T)
        if not success:
            raise RuntimeError("TL-QLDMR training failed")
        self.predictor = Predictor(self.model)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.predictor is None:
            self.predictor = Predictor(self.model)
        return self.predictor.predict(X)
