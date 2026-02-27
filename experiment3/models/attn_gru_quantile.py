from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def _pinball_loss(pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    diff = target - pred
    return torch.maximum(tau * diff, (tau - 1.0) * diff).mean()


class _AttnGRUQuantile(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attn_query = nn.Parameter(torch.randn(hidden_dim))
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        scores = torch.matmul(output, self.attn_query)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(output * weights.unsqueeze(-1), dim=1)
        return self.fc(context)


@dataclass
class AttnGRUQuantileConfig:
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    tau_low: float = 0.05
    tau_high: float = 0.95
    seed: int = 42


class AttnGRUQuantileIntervalModel:
    def __init__(self, input_dim: int, config: AttnGRUQuantileConfig | None = None):
        self.input_dim = input_dim
        self.config = config or AttnGRUQuantileConfig()
        self.model = _AttnGRUQuantile(
            input_dim=self.input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        torch.manual_seed(self.config.seed)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        X_tensor = torch.from_numpy(X.astype(np.float32))
        y_tensor = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

        n = len(X_tensor)
        for _ in range(self.config.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.config.batch_size):
                idx = perm[i : i + self.config.batch_size]
                batch_x = X_tensor[idx]
                batch_y = y_tensor[idx]
                pred = self.model(batch_x)
                pred_low = pred[:, 0:1]
                pred_high = pred[:, 1:2]
                loss = _pinball_loss(pred_low, batch_y, self.config.tau_low)
                loss += _pinball_loss(pred_high, batch_y, self.config.tau_high)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        X_tensor = torch.from_numpy(X.astype(np.float32))
        with torch.no_grad():
            out = self.model(X_tensor).cpu().numpy()
        lower = out[:, 0]
        upper = out[:, 1]
        return lower, upper
