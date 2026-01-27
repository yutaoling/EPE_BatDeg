"""
Transformer模型（双头输出：SOH + RUL）

包含：
1. TransformerModel: 标准Transformer编码器
2. TransformerWithCLS: 带CLS token的Transformer

注意：本项目统一只使用3通道：voltage/current/time
"""

import torch
import torch.nn as nn
import math
from .base import BaseModel


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """可学习的位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerModel(BaseModel):
    """Transformer编码器（双头）"""

    INPUT_TYPE = 'sequence'

    def __init__(
        self,
        in_channels: int = 3,
        d_model: int = 64,
        nhead: int = 3,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 500,
        pos_encoding: str = 'sinusoidal',
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.LayerNorm(d_model),
        )

        if pos_encoding == 'learned':
            self.pos_encoder = LearnedPositionalEncoding(d_model, max_len, dropout)
        else:
            self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.backbone = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.soh_head = nn.Linear(d_model, 1)
        self.rul_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)

    def forward(self, x: torch.Tensor):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        feat = self.backbone(x)
        return {
            'soh': self.soh_head(feat),
            'rul': self.rul_head(feat),
        }


class TransformerWithCLS(BaseModel):
    """带CLS token的Transformer（双头）"""

    INPUT_TYPE = 'sequence'

    def __init__(
        self,
        in_channels: int = 3,
        d_model: int = 64,
        nhead: int = 3,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 500,
    ):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model),
            nn.LayerNorm(d_model),
        )

        self.pos_encoder = LearnedPositionalEncoding(d_model, max_len + 1, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        self.backbone = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.soh_head = nn.Linear(d_model, 1)
        self.rul_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)

        x = self.input_proj(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        cls_output = x[:, 0]
        feat = self.backbone(cls_output)
        return {
            'soh': self.soh_head(feat),
            'rul': self.rul_head(feat),
        }
