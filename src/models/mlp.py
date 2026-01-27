"""
MLP模型

用于统计特征输入

本项目统一双头输出：SOH + RUL
"""

import torch
import torch.nn as nn
from .base import BaseModel


class MLP(BaseModel):
    """多层感知机（双头）"""

    INPUT_TYPE = 'features'

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dims: list = [64, 32],
        dropout: float = 0.1,
        activation: str = 'relu',
    ):
        super().__init__()

        act_fn = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'silu': nn.SiLU,
            'leaky_relu': nn.LeakyReLU,
        }.get(activation, nn.ReLU)

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                act_fn(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        self.soh_head = nn.Linear(prev_dim, 1)
        self.rul_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prev_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        return {
            'soh': self.soh_head(feat),
            'rul': self.rul_head(feat),
        }


class ResidualMLP(BaseModel):
    """带残差连接的MLP（双头）"""

    INPUT_TYPE = 'features'

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 64,
        num_blocks: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout)
            for _ in range(num_blocks)
        ])

        self.soh_head = nn.Linear(hidden_dim, 1)
        self.rul_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return {
            'soh': self.soh_head(x),
            'rul': self.rul_head(x),
        }


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))
