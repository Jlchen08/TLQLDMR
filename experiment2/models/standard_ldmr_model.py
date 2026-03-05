from dataclasses import dataclass

import numpy as np

from Train_TL_QLDMR import TL_QLDMR
from Predict_TL_QLDMR import Predictor


@dataclass
class StandardLDMRConfig:
    lambda1: float = 0.0005
    C: float = 120.0
    tau: float = 0.5
    kernel_gamma: float = 0.002
    solver: str = "fast_nystrom"
    nystrom_n_components: int = 1000
    nystrom_lr: float = 0.01
    nystrom_epochs: int = 64
    nystrom_batch_size: int = 256


class StandardLDMRModel:
    """
    Standard single-domain LDMR baseline.
    It uses only target-domain samples and symmetric variance regularization.
    """

    def __init__(self, config: StandardLDMRConfig | None = None):
        self.config = config or StandardLDMRConfig()
        self.model = TL_QLDMR(
            lambda1=self.config.lambda1,
            lambda2=0.0,
            C_S=0.0,
            C_T=self.config.C,
            tau=self.config.tau,
            kernel_gamma=self.config.kernel_gamma,
            solver=self.config.solver,
            nystrom_n_components=self.config.nystrom_n_components,
            nystrom_lr=self.config.nystrom_lr,
            nystrom_epochs=self.config.nystrom_epochs,
            nystrom_batch_size=self.config.nystrom_batch_size,
            variance_mode="symmetric",
            asym_scale=0.0,
        )
        self.predictor = None

    def fit(self, X_T: np.ndarray, y_T: np.ndarray) -> None:
        X_S_empty = np.empty((0, X_T.shape[1]), dtype=np.float32)
        y_S_empty = np.empty((0,), dtype=np.float32)
        success = self.model.fit(X_S_empty, y_S_empty, X_T, y_T)
        if not success:
            raise RuntimeError("Standard LDMR training failed")
        self.predictor = Predictor(self.model)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.predictor is None:
            self.predictor = Predictor(self.model)
        return self.predictor.predict(X)
