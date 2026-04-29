"""Causal-conv TCN forecaster — direct multi-step output."""
from __future__ import annotations

import torch
import torch.nn as nn


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        self._pad = (kernel_size - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size, padding=self._pad, dilation=dilation)

    def forward(self, x):
        out = super().forward(x)
        return out[:, :, : -self._pad] if self._pad > 0 else out


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.drop(self.relu(self.conv1(x)))
        out = self.drop(self.relu(self.conv2(out)))
        return self.relu(out + self.skip(x))


class TCNForecaster(nn.Module):
    def __init__(self, input_size: int, channels: int, kernel_size: int,
                 num_layers: int, horizon: int, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = input_size
        for i in range(num_layers):
            layers.append(TCNBlock(prev, channels, kernel_size, dilation=2 ** i, dropout=dropout))
            prev = channels
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(channels, horizon)

    def forward(self, x):
        # x: (batch, lookback, features) → permute to (batch, features, lookback)
        h = self.tcn(x.transpose(1, 2))
        return self.head(h[:, :, -1])
