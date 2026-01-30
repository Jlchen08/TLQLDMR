from dataclasses import dataclass

import numpy as np


@dataclass
class TLLSTMConfig:
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    lr: float = 1e-3
    pretrain_epochs: int = 10
    finetune_epochs: int = 10
    batch_size: int = 128
    val_ratio: float = 0.1
    patience: int = 3
    device: str = "auto"


class TLLSTMRegressor:
    def __init__(self, input_size: int, config: TLLSTMConfig):
        import torch
        import torch.nn as nn

        self.torch = torch
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(config.hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.head(out)
        return out

    def parameters(self):
        return list(self.lstm.parameters()) + list(self.head.parameters())


class TLLSTMModel:
    def __init__(self, input_size: int, config: TLLSTMConfig | None = None):
        self.config = config or TLLSTMConfig()
        self.input_size = input_size
        self.device = self._get_device()
        self.net = TLLSTMRegressor(input_size, self.config)
        self.net.lstm.to(self.device)
        self.net.head.to(self.device)

    def _get_device(self) -> str:
        if self.config.device != "auto":
            return self.config.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _create_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        ds = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=self.config.batch_size, shuffle=shuffle)

    def _train_epochs(self, X: np.ndarray, y: np.ndarray, epochs: int) -> None:
        import torch
        import torch.nn as nn

        n_total = len(X)
        n_val = max(1, int(n_total * self.config.val_ratio))
        n_train = max(1, n_total - n_val)

        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:], y[n_train:]

        train_loader = self._create_loader(X_train, y_train, shuffle=True)
        val_loader = self._create_loader(X_val, y_val, shuffle=False)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr)
        criterion = nn.MSELoss()

        best_val = float("inf")
        patience_left = self.config.patience
        best_state = None

        for _ in range(epochs):
            self.net.lstm.train()
            self.net.head.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                preds = self.net.forward(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

            self.net.lstm.eval()
            self.net.head.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    preds = self.net.forward(xb)
                    val_loss += criterion(preds, yb).item() * len(xb)
            val_loss /= max(1, len(X_val))

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                patience_left = self.config.patience
                best_state = {
                    "lstm": self.net.lstm.state_dict(),
                    "head": self.net.head.state_dict(),
                }
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            self.net.lstm.load_state_dict(best_state["lstm"])
            self.net.head.load_state_dict(best_state["head"])

    def fit(self, X_S: np.ndarray, y_S: np.ndarray, X_T: np.ndarray, y_T: np.ndarray) -> None:
        if len(X_S) > 0:
            self._train_epochs(X_S, y_S, self.config.pretrain_epochs)
        self._train_epochs(X_T, y_T, self.config.finetune_epochs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        loader = self._create_loader(X, np.zeros(len(X), dtype=np.float32), shuffle=False)
        self.net.lstm.eval()
        self.net.head.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(self.device)
                out = self.net.forward(xb).squeeze(-1).cpu().numpy()
                preds.append(out)
        return np.concatenate(preds, axis=0)
