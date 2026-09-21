from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn


def _pinball_loss(pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    diff = target - pred
    return torch.maximum(tau * diff, (tau - 1.0) * diff).mean()


class _TransformerQuantile(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.pos_emb
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.fc(pooled)


@dataclass
class TransformerQuantileConfig:
    d_model: int = 64
    n_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 128
    dropout: float = 0.1
    epochs: int = 15
    batch_size: int = 128
    lr: float = 1e-3
    tau_low: float = 0.05
    tau_high: float = 0.95
    n_members: int = 2
    seed: int = 42


class TransformerQuantileEnsembleIntervalModel:
    def __init__(self, input_dim: int, seq_len: int, config: TransformerQuantileConfig | None = None):
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.config = config or TransformerQuantileConfig()
        self.members: list[_TransformerQuantile] = []

    def _train_member(self, model: _TransformerQuantile, X: np.ndarray, y: np.ndarray, seed: int) -> None:
        torch.manual_seed(seed)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        X_tensor = torch.from_numpy(X.astype(np.float32))
        y_tensor = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

        n = len(X_tensor)
        for _ in range(self.config.epochs):
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
            model = _TransformerQuantile(
                input_dim=self.input_dim,
                seq_len=self.seq_len,
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                num_layers=self.config.num_layers,
                ff_dim=self.config.ff_dim,
                dropout=self.config.dropout,
            )
            self._train_member(model, X, y, seed=self.config.seed + i)
            self.members.append(model)

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.members:
            raise RuntimeError("TransformerQuantileEnsembleIntervalModel is not fitted.")

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
