"""Direct multi-step LSTM forecaster — outputs `horizon` values from last hidden."""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 horizon: int, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, features) → (batch, horizon)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])
