"""
LSTM模型（双头输出：SOH + RUL）

包含：
1. LSTM: 单向/双向LSTM
2. BiLSTM: 双向LSTM封装
3. CNNLSTM: CNN + LSTM混合
4. AttentionLSTM: 注意力LSTM

注意：本项目统一只使用3通道：v_delta/i_delta/q_norm
"""

import torch
import torch.nn as nn
from .base import BaseModel


class LSTM(BaseModel):
    """LSTM模型（双头）"""

    INPUT_TYPE = 'sequence'

    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_output_size = hidden_size * (2 if bidirectional else 1)

        self.backbone = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.soh_head = nn.Linear(hidden_size, 1)
        self.rul_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor):
        _, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            out = torch.cat([h_forward, h_backward], dim=1)
        else:
            out = h_n[-1]

        feat = self.backbone(out)
        return {
            'soh': self.soh_head(feat),
            'rul': self.rul_head(feat),
        }


class BiLSTM(LSTM):
    """双向LSTM"""

    def __init__(self, **kwargs):
        kwargs['bidirectional'] = True
        super().__init__(**kwargs)


class CNNLSTM(BaseModel):
    """CNN + LSTM混合模型（双头）"""

    INPUT_TYPE = 'sequence'

    def __init__(
        self,
        in_channels: int = 3,
        cnn_channels: int = 32,
        lstm_hidden: int = 64,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        self.backbone = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.soh_head = nn.Linear(lstm_hidden, 1)
        self.rul_head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 1),
        )

    def forward(self, x: torch.Tensor):
        # (batch, T, C) -> CNN expects (batch, C, T)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)

        _, (h_n, _) = self.lstm(x)

        if self.lstm.bidirectional:
            out = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            out = h_n[-1]

        feat = self.backbone(out)
        return {
            'soh': self.soh_head(feat),
            'rul': self.rul_head(feat),
        }


class AttentionLSTM(BaseModel):
    """带注意力机制的LSTM（双头）"""

    INPUT_TYPE = 'sequence'

    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        self.backbone = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.soh_head = nn.Linear(hidden_size, 1)
        self.rul_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor):
        lstm_out, _ = self.lstm(x)

        attn_scores = self.attention(lstm_out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        context = (lstm_out * attn_weights).sum(dim=1)

        feat = self.backbone(context)
        return {
            'soh': self.soh_head(feat),
            'rul': self.rul_head(feat),
        }
