"""Deep MLP classifier for antibacterial activity prediction.

2048 → 1024 → 512 → 256 → 128 → 1 with BatchNorm + ReLU + dropout. This is the
single definition of the architecture (previously duplicated in _common.py and the
modeling notebook — keep them identical if the notebook is ever re-run).
"""
from __future__ import annotations

import torch.nn as nn


class AntibacterialMLP(nn.Module):
    def __init__(self, input_dim: int = 2048, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
