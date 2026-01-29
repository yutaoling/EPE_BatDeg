"""Fourier Neural Operator (FNO) 模型（双头输出：SOH + RUL）

用于学习算子映射，适合捕捉物理系统的动态特性。

注意：本项目统一只使用3通道：v_delta/i_delta/q_norm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseModel


class SpectralConv1d(nn.Module):
    """1D谱卷积层"""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        return torch.einsum("bim,iom->bom", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            batch_size, self.out_channels, x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNOBlock1d(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral_conv = SpectralConv1d(width, width, modes)
        self.linear = nn.Conv1d(width, width, 1)
        self.norm = nn.BatchNorm1d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.spectral_conv(x)
        x2 = self.linear(x)
        x = x1 + x2
        x = self.norm(x)
        x = F.gelu(x)
        return x


class FNO(BaseModel):
    """1D FNO（双头）"""

    INPUT_TYPE = 'sequence'

    def __init__(
        self,
        in_channels: int = 3,
        width: int = 32,
        modes: int = 16,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.fc0 = nn.Linear(in_channels, width)

        self.layers = nn.ModuleList([
            FNOBlock1d(width, modes)
            for _ in range(num_layers)
        ])

        self.backbone = nn.Sequential(
            nn.Linear(width, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.soh_head = nn.Linear(64, 1)
        self.rul_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor):
        # (batch, T, C)
        x = self.fc0(x)  # (batch, T, width)
        x = x.transpose(1, 2)  # (batch, width, T)

        for layer in self.layers:
            x = layer(x)

        x = x.mean(dim=-1)  # (batch, width)
        feat = self.backbone(x)  # (batch, 64)

        return {
            'soh': self.soh_head(feat),
            'rul': self.rul_head(feat),
        }


class SpectralConv2d(nn.Module):
    """2D谱卷积层"""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batch_size, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2D(BaseModel):
    """2D FNO（双头）"""

    INPUT_TYPE = 'image'

    def __init__(
        self,
        in_channels: int = 3,
        width: int = 32,
        modes1: int = 12,
        modes2: int = 12,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.fc0 = nn.Linear(in_channels, width)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                SpectralConv2d(width, width, modes1, modes2),
                nn.Conv2d(width, width, 1),
                nn.BatchNorm2d(width),
            ]))

        self.backbone = nn.Sequential(
            nn.Linear(width, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.soh_head = nn.Linear(64, 1)
        self.rul_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor):
        # (batch, H, W, C)
        x = self.fc0(x)  # (batch, H, W, width)
        x = x.permute(0, 3, 1, 2)  # (batch, width, H, W)

        for spectral_conv, linear, norm in self.layers:
            x1 = spectral_conv(x)
            x2 = linear(x)
            x = x1 + x2
            x = norm(x)
            x = F.gelu(x)

        x = x.mean(dim=[-2, -1])  # (batch, width)
        feat = self.backbone(x)

        return {
            'soh': self.soh_head(feat),
            'rul': self.rul_head(feat),
        }
