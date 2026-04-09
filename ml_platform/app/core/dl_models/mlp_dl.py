"""MLP deep-learning trainer — fully-connected network with Dropout + BatchNorm."""
from __future__ import annotations

import torch.nn as nn

from app.core.dl_trainer import BaseDLTrainer


# ---------------------------------------------------------------------------
# nn.Module
# ---------------------------------------------------------------------------

_ACT = {
    "relu":       nn.ReLU,
    "gelu":       nn.GELU,
    "tanh":       nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


class _MLPDLModule(nn.Module):
    def __init__(
        self,
        input_dim:    int,
        output_dim:   int,
        hidden_layers: list[int],
        dropout:      float,
        batch_norm:   bool,
        activation:   str,
    ):
        super().__init__()
        act_cls = _ACT.get(activation, nn.ReLU)
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class MLPDLTrainer(BaseDLTrainer):
    def __init__(self):
        super().__init__()
        self.model_type = "mlp_dl"

    def build_model(self, input_dim: int, output_dim: int, arch_config: dict) -> nn.Module:
        hidden_layers = arch_config.get("hidden_layers", [256, 128])
        if isinstance(hidden_layers, str):
            hidden_layers = [int(x.strip()) for x in hidden_layers.split(",") if x.strip()]
        dropout    = float(arch_config.get("dropout", 0.3))
        batch_norm = bool(arch_config.get("batch_norm", True))
        activation = str(arch_config.get("activation", "relu"))
        return _MLPDLModule(input_dim, output_dim, hidden_layers, dropout, batch_norm, activation)
