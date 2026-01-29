"""Vision Transformer (ViT) 模型（双头输出：SOH + RUL）

用于热力图输入

注意：本项目统一只使用3通道：v_delta/i_delta/q_norm
"""

import torch
import torch.nn as nn
from .base import BaseModel


class PatchEmbedding(nn.Module):
    """将图像分割成patch并嵌入"""

    def __init__(
        self,
        img_height: int,
        img_width: int,
        patch_height: int,
        patch_width: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()

        self.num_patches_h = img_height // patch_height
        self.num_patches_w = img_width // patch_width
        self.num_patches = self.num_patches_h * self.num_patches_w

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_height, patch_width),
            stride=(patch_height, patch_width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim({embed_dim}) must be divisible by num_heads({num_heads}). "
                f"Please change num_heads (e.g. 4/8) or embed_dim."
            )

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(BaseModel):
    """Vision Transformer（双头）"""

    INPUT_TYPE = 'image'

    def __init__(
        self,
        img_height: int = 100,
        img_width: int = 200,
        in_channels: int = 3,
        patch_height: int = 10,
        patch_width: int = 20,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_height,
            img_width,
            patch_height,
            patch_width,
            in_channels,
            embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.backbone = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.soh_head = nn.Linear(embed_dim, 1)
        self.rul_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor):
        # (batch, H, W, C) -> (batch, C, H, W)
        x = x.permute(0, 3, 1, 2)
        B = x.shape[0]

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_output = x[:, 0]

        feat = self.backbone(cls_output)
        return {
            'soh': self.soh_head(feat),
            'rul': self.rul_head(feat),
        }


class SimpleViT(BaseModel):
    """简化版ViT（双头）"""

    INPUT_TYPE = 'image'

    def __init__(
        self,
        img_height: int = 100,
        img_width: int = 200,
        in_channels: int = 3,
        patch_height: int = 10,
        patch_width: int = 20,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim({embed_dim}) must be divisible by num_heads({num_heads})."
            )

        self.patch_embed = PatchEmbedding(
            img_height,
            img_width,
            patch_height,
            patch_width,
            in_channels,
            embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)

        self.soh_head = nn.Linear(embed_dim, 1)
        self.rul_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2)
        B = x.shape[0]

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed

        x = self.transformer(x)
        x = self.norm(x)

        cls_out = x[:, 0]
        return {
            'soh': self.soh_head(cls_out),
            'rul': self.rul_head(cls_out),
        }
