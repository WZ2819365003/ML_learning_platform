"""CNN-1D deep-learning trainer.

Each feature is treated as one channel in a 1-D sequence.
Input shape fed to Conv1d: (batch, 1, num_features).
AdaptiveAvgPool1d(output_size=8) normalises variable sequence lengths
after stacking conv layers, avoiding manual size arithmetic.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from app.core.dl_trainer import BaseDLTrainer


# ---------------------------------------------------------------------------
# nn.Module
# ---------------------------------------------------------------------------

class _CNN1DModule(nn.Module):
    _POOL_OUT = 8   # fixed adaptive pool output length

    def __init__(
        self,
        input_dim:      int,
        output_dim:     int,
        num_filters:    int,
        kernel_size:    int,
        num_conv_layers: int,
        dropout:        float,
        fc_hidden:      int,
    ):
        super().__init__()
        conv_blocks: list[nn.Module] = []
        in_ch = 1
        for _ in range(num_conv_layers):
            conv_blocks += [
                nn.Conv1d(in_ch, num_filters,
                          kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ]
            in_ch = num_filters
        conv_blocks.append(nn.AdaptiveAvgPool1d(self._POOL_OUT))
        self.conv = nn.Sequential(*conv_blocks)

        flat_dim = num_filters * self._POOL_OUT
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features) → (batch, 1, features) for Conv1d
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class CNN1DTrainer(BaseDLTrainer):
    def __init__(self):
        super().__init__()
        self.model_type = "cnn1d"

    def build_model(self, input_dim: int, output_dim: int, arch_config: dict) -> nn.Module:
        return _CNN1DModule(
            input_dim=input_dim,
            output_dim=output_dim,
            num_filters=int(arch_config.get("num_filters",     64)),
            kernel_size=int(arch_config.get("kernel_size",     3)),
            num_conv_layers=int(arch_config.get("num_conv_layers", 2)),
            dropout=float(arch_config.get("dropout",        0.3)),
            fc_hidden=int(arch_config.get("fc_hidden",      128)),
        )
