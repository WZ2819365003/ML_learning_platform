"""Lightweight Transformer deep-learning trainer for tabular data.

Architecture:
  Each input feature is projected to d_model dimensions, forming a sequence
  of (num_features) tokens.  A learnable CLS token is prepended; the encoder
  reads the full sequence and the CLS output feeds into the classification /
  regression head.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from app.core.dl_trainer import BaseDLTrainer


# ---------------------------------------------------------------------------
# nn.Module
# ---------------------------------------------------------------------------

class _TransformerModule(nn.Module):
    def __init__(
        self,
        input_dim:          int,
        output_dim:         int,
        d_model:            int,
        nhead:              int,
        num_encoder_layers: int,
        dim_feedforward:    int,
        dropout:            float,
    ):
        super().__init__()
        # Ensure d_model is divisible by nhead
        if d_model % nhead != 0:
            d_model = max(nhead, (d_model // nhead) * nhead)

        # Project each scalar feature to d_model
        self.feature_proj = nn.Linear(1, d_model)
        # Learnable positional embedding for each feature position + CLS
        self.pos_embed = nn.Embedding(input_dim + 1, d_model)
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,       # pre-LN for training stability
        )
        self.encoder  = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers,
                                              enable_nested_tensor=False)
        self.dropout  = nn.Dropout(p=dropout)
        self.head     = nn.Linear(d_model, output_dim)
        self._d_model = d_model
        self._input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features)
        batch = x.size(0)

        # (batch, features, 1) → (batch, features, d_model)
        tokens = self.feature_proj(x.unsqueeze(-1))

        # Add positional embedding (positions 1..features; 0 reserved for CLS)
        pos = torch.arange(1, self._input_dim + 1, device=x.device)
        tokens = tokens + self.pos_embed(pos)

        # Prepend CLS token
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)   # (batch, features+1, d_model)

        # CLS positional embedding
        cls_pos = torch.zeros(1, dtype=torch.long, device=x.device)
        tokens[:, :1, :] = tokens[:, :1, :] + self.pos_embed(cls_pos)

        out = self.encoder(tokens)          # (batch, features+1, d_model)
        cls_out = self.dropout(out[:, 0])   # CLS token
        return self.head(cls_out)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TransformerTrainer(BaseDLTrainer):
    def __init__(self):
        super().__init__()
        self.model_type = "transformer"

    def build_model(self, input_dim: int, output_dim: int, arch_config: dict) -> nn.Module:
        d_model = int(arch_config.get("d_model",            64))
        nhead   = int(arch_config.get("nhead",              4))
        # Guard: d_model must be divisible by nhead
        if d_model % nhead != 0:
            d_model = max(nhead, (d_model // nhead) * nhead)
        return _TransformerModule(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=int(arch_config.get("num_encoder_layers", 2)),
            dim_feedforward=int(arch_config.get("dim_feedforward",   256)),
            dropout=float(arch_config.get("dropout",             0.1)),
        )
