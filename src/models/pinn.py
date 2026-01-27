"""
PINN模型（Physics-Informed Neural Network）

基于Wang 2024的思路：统计特征输入 + 物理约束损失。

本项目统一双头输出：SOH + RUL。
注意：PINN的物理约束目前仅对 SOH 头生效（RUL不做物理约束）。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Tuple

from .base import BaseModel


class PINN(BaseModel):
    """Physics-Informed Neural Network（双头）"""

    INPUT_TYPE = 'features'

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dims: list = [64, 32],
        dropout: float = 0.1,
        # 物理约束权重
        monotonicity_weight: float = 0.1,
        bound_weight: float = 0.1,
    ):
        super().__init__()

        self.monotonicity_weight = monotonicity_weight
        self.bound_weight = bound_weight

        # backbone
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hd),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hd
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

    def compute_physics_loss(self, soh_pred: torch.Tensor, meta: Dict = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """对 SOH 头计算物理约束损失"""
        loss_dict: Dict[str, float] = {}
        total_loss = torch.tensor(0.0, device=soh_pred.device)

        # 边界约束：SOH应在[0,1]
        if self.bound_weight > 0:
            lower = torch.relu(-soh_pred).mean()
            upper = torch.relu(soh_pred - 1).mean()
            bound_loss = lower + upper
            total_loss = total_loss + self.bound_weight * bound_loss
            loss_dict['bound_loss'] = float(bound_loss.detach().cpu())

        # 单调性约束：batch内相邻样本（若有顺序）SOH不应上升
        if self.monotonicity_weight > 0 and soh_pred.numel() > 1:
            diff = soh_pred[1:] - soh_pred[:-1]
            mono_loss = torch.relu(diff).mean()
            total_loss = total_loss + self.monotonicity_weight * mono_loss
            loss_dict['monotonicity_loss'] = float(mono_loss.detach().cpu())

        return total_loss, loss_dict

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        save_path: str = None,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """训练PINN（无早停/无scheduler）。

        说明：此处仍沿用 BaseModel.fit 的接口，但内部加上 physics_loss。
        label 使用 train_loader 中的 batch['label']。
        当上层采用双目标训练时，应使用统一的 BaseModel.fit（后续会统一）。
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        mse = nn.MSELoss()

        history = {'train_loss': [], 'val_loss': [], 'physics_loss': []}
        iterator = tqdm(range(epochs), desc='Training PINN') if verbose else range(epochs)

        for epoch in iterator:
            self.train()
            train_loss = 0.0
            physics_loss_total = 0.0

            for batch in train_loader:
                x = batch['feature'].to(self._device)
                y = batch['label'].to(self._device)

                optimizer.zero_grad()
                out = self(x)
                soh_pred = out['soh'].squeeze()

                data_loss = mse(soh_pred, y)
                physics_loss, _ = self.compute_physics_loss(soh_pred, batch.get('meta'))

                loss = data_loss + physics_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += float(data_loss.detach().cpu())
                physics_loss_total += float(physics_loss.detach().cpu())

            train_loss /= max(1, len(train_loader))
            physics_loss_total /= max(1, len(train_loader))
            history['train_loss'].append(train_loss)
            history['physics_loss'].append(physics_loss_total)

            if val_loader is not None:
                # PINN验证时只对label（通常是SOH）做loss
                val_total = 0.0
                self.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        x = batch['feature'].to(self._device)
                        y = batch['label'].to(self._device)
                        out = self(x)
                        soh_pred = out['soh'].squeeze()
                        val_total += float(mse(soh_pred, y).detach().cpu())
                val_loss = val_total / max(1, len(val_loader))
                history['val_loss'].append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}: train={train_loss:.6f}, physics={physics_loss_total:.6f}"
                if val_loader is not None:
                    msg += f", val={history['val_loss'][-1]:.6f}"
                tqdm.write(msg)

        if save_path:
            self.save(save_path)

        return history


class DegradationPINN(BaseModel):
    """基于降解方程的PINN（双头占位实现）

    注意：该模型原设计更偏 SOH，一旦统一训练框架（soh/rul双loss）后可再细化。
    当前先提供双头结构，保持接口一致。
    """

    INPUT_TYPE = 'features'

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dims: list = [64, 32],
        dropout: float = 0.1,
        physics_weight: float = 0.1,
    ):
        super().__init__()
        self.physics_weight = physics_weight

        # predictor backbone
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hd),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hd
        self.backbone = nn.Sequential(*layers)

        self.soh_head = nn.Linear(prev_dim, 1)
        self.rul_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(prev_dim, 1),
        )

        # degradation param head（a,b）
        self.degradation_net = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        return {
            'soh': self.soh_head(feat),
            'rul': self.rul_head(feat),
        }

    def predict_degradation_params(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self.degradation_net(feat)
        a = torch.sigmoid(params[:, 0]) * 0.5
        b = torch.sigmoid(params[:, 1]) + 0.5
        return a, b

    def compute_degradation_loss(
        self,
        pred_soh: torch.Tensor,
        cycle_number: torch.Tensor,
        feat: torch.Tensor,
    ) -> torch.Tensor:
        a, b = self.predict_degradation_params(feat)
        k = cycle_number.float()
        soh_physics = 1 - a * torch.pow(k / 1000, b)
        return torch.mean((pred_soh.squeeze() - soh_physics) ** 2)
