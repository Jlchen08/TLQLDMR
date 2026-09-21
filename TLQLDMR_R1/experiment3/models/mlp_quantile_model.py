from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn


def _pinball_loss(pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    diff = target - pred
    return torch.maximum(tau * diff, (tau - 1.0) * diff).mean()


class _MLPQuantile(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Iterable[int] = (64, 32)):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class MLPQuantileConfig:
    hidden_dims: tuple[int, ...] = (64, 32)
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    tau_low: float = 0.05
    tau_high: float = 0.95
    n_members: int = 3
    seed: int = 42


class MLPQuantileEnsembleIntervalModel:
    def __init__(self, input_dim: int, config: MLPQuantileConfig | None = None):
        self.input_dim = input_dim
        self.config = config or MLPQuantileConfig()
        self.members: list[_MLPQuantile] = []

    def _train_member(self, model: _MLPQuantile, X: np.ndarray, y: np.ndarray, seed: int) -> None:
        torch.manual_seed(seed)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        X_tensor = torch.from_numpy(X.astype(np.float32))
        y_tensor = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

        n = len(X_tensor)
        for epoch in range(self.config.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.config.batch_size):
                idx = perm[i : i + self.config.batch_size]
                batch_x = X_tensor[idx]
                batch_y = y_tensor[idx]
                pred = model(batch_x)
                pred_low = pred[:, 0:1]
                pred_high = pred[:, 1:2]
                loss = _pinball_loss(pred_low, batch_y, self.config.tau_low)
                loss += _pinball_loss(pred_high, batch_y, self.config.tau_high)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.members = []
        for i in range(self.config.n_members):
            model = _MLPQuantile(self.input_dim, self.config.hidden_dims)
            self._train_member(model, X, y, seed=self.config.seed + i)
            self.members.append(model)

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.members:
            raise RuntimeError("MLPQuantileEnsembleIntervalModel is not fitted.")

        X_tensor = torch.from_numpy(X.astype(np.float32))
        preds = []
        for model in self.members:
            model.eval()
            with torch.no_grad():
                out = model(X_tensor).cpu().numpy()
            preds.append(out)
        preds = np.stack(preds, axis=0)
        mean_pred = np.mean(preds, axis=0)
        lower = mean_pred[:, 0]
        upper = mean_pred[:, 1]
        return lower, upper
