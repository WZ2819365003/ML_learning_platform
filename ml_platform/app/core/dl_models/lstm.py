"""LSTM / GRU deep-learning trainer.

Treats each sample's feature vector as a single-step sequence
(batch, 1, features), allowing stacked bidirectional LSTM/GRU cells.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from app.core.dl_trainer import BaseDLTrainer


# ---------------------------------------------------------------------------
# nn.Module
# ---------------------------------------------------------------------------

class _LSTMModule(nn.Module):
    def __init__(
        self,
        input_dim:     int,
        output_dim:    int,
        hidden_size:   int,
        num_layers:    int,
        bidirectional: bool,
        dropout:       float,
        cell_type:     str,   # "lstm" | "gru"
        fc_hidden:     int,
    ):
        super().__init__()
        rnn_cls = nn.LSTM if cell_type == "lstm" else nn.GRU
        # dropout only applies between layers (not on last layer)
        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=rnn_dropout,
        )
        rnn_out_dim = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(rnn_out_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features)  →  treat as (batch, seq=1, features)
        x = x.unsqueeze(1)
        out, _ = self.rnn(x)
        # take the last time-step output
        last = out[:, -1, :]
        return self.fc(last)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class LSTMTrainer(BaseDLTrainer):
    def __init__(self):
        super().__init__()
        self.model_type = "lstm"

    def build_model(self, input_dim: int, output_dim: int, arch_config: dict) -> nn.Module:
        return _LSTMModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=int(arch_config.get("hidden_size",   128)),
            num_layers=int(arch_config.get("num_layers",    2)),
            bidirectional=bool(arch_config.get("bidirectional", False)),
            dropout=float(arch_config.get("dropout",      0.3)),
            cell_type=str(arch_config.get("cell_type",    "lstm")),
            fc_hidden=int(arch_config.get("fc_hidden",    64)),
        )
