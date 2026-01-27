"""
CNN模型（双头输出：SOH + RUL）

包含：
1. CNN1D: 用于单循环/固定长度时序输入 (batch, T, C)
2. CNN2D: 用于热力图输入 (batch, H, W, C)

注意：本项目统一只使用3通道：voltage/current/time
"""

import torch
import torch.nn as nn
from .base import BaseModel


class CNN1D(BaseModel):
    """1D卷积神经网络（双头）"""

    INPUT_TYPE = 'sequence'

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        prev_channels = in_channels

        for i in range(num_layers):
            out_channels = hidden_channels * (2 ** min(i, 2))
            layers.extend([
                nn.Conv1d(prev_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            ])
            prev_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.soh_head = nn.Linear(prev_channels, 1)
        self.rul_head = nn.Sequential(
            nn.Linear(prev_channels, prev_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prev_channels, 1),
        )

    def forward(self, x: torch.Tensor):
        # (batch, T, C) -> (batch, C, T)
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = self.global_pool(x).squeeze(-1)
        return {
            'soh': self.soh_head(x),
            'rul': self.rul_head(x),
        }


class CNN2D(BaseModel):
    """2D卷积神经网络（双头）"""

    INPUT_TYPE = 'image'

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 32,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        prev_channels = in_channels

        for i in range(num_layers):
            out_channels = hidden_channels * (2 ** min(i, 2))
            layers.extend([
                nn.Conv2d(prev_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout),
            ])
            prev_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.soh_head = nn.Linear(prev_channels, 1)
        self.rul_head = nn.Sequential(
            nn.Linear(prev_channels, prev_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prev_channels, 1),
        )

    def forward(self, x: torch.Tensor):
        # (batch, H, W, C) -> (batch, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = self.conv_layers(x)
        x = self.global_pool(x).view(x.size(0), -1)
        return {
            'soh': self.soh_head(x),
            'rul': self.rul_head(x),
        }


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet2D(BaseModel):
    """简化版ResNet（双头）"""

    INPUT_TYPE = 'image'

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 32,
        num_blocks: list = [2, 2, 2],
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)

        self.layer1 = self._make_layer(hidden_channels, hidden_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(hidden_channels, hidden_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(hidden_channels * 2, hidden_channels * 4, num_blocks[2], stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        feat_dim = hidden_channels * 4
        self.soh_head = nn.Linear(feat_dim, 1)
        self.rul_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 1),
        )

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int):
        layers = [ResNetBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x).view(x.size(0), -1)
        return {
            'soh': self.soh_head(x),
            'rul': self.rul_head(x),
        }
