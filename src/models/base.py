"""
模型基类

定义统一的训练、评估、保存/加载接口。

本版本统一为双头输出训练：
- 模型 forward 返回 {'soh': ..., 'rul': ...}
- 数据 batch 提供 soh/rul/rul_mask/label/target_type 等字段
- loss 根据 target_type 选择或加权相加（both: 1.0*soh + 0.1*rul）

注意：已按要求移除 early stopping 与 scheduler。
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional, Any
from pathlib import Path


class BaseModel(nn.Module):
    """模型基类"""

    INPUT_TYPE: str = 'sequence'  # 'features', 'sequence', 'image'

    def __init__(self):
        super().__init__()
        self._device = 'cpu'

    @property
    def input_type(self) -> str:
        return self.INPUT_TYPE

    @property
    def device(self) -> str:
        return self._device

    def forward(self, x: torch.Tensor):
        raise NotImplementedError

    def to(self, device: str):
        self._device = device
        return super().to(device)

    def _param_device(self) -> torch.device:
        """始终以模型参数所在 device 为准，避免 self._device 与真实参数 device 不一致。"""
        return next(self.parameters()).device

    @staticmethod
    def _as_pred_dict(pred):
        """兼容：如果子类仍返回Tensor，这里包成dict（SOH=rul=同值）。"""
        if isinstance(pred, dict):
            return pred
        return {'soh': pred, 'rul': pred}

    @staticmethod
    def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return torch.mean((pred - target) ** 2)
        mask = mask.float()
        denom = mask.sum().clamp(min=1.0)
        return torch.sum(((pred - target) ** 2) * mask) / denom

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        save_path: str = None,
        verbose: bool = True,
        target_type: str = 'both',
        soh_loss_weight: float = 1.0,
        rul_loss_weight: float = 0.1,
    ) -> Dict[str, list]:
        """训练模型（无早停/无scheduler）。"""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        history = {
            'train_loss': [],
            'train_soh_loss': [],
            'train_rul_loss': [],
            'val_loss': [],
            'val_soh_loss': [],
            'val_rul_loss': [],
        }

        iterator = tqdm(range(epochs), desc='Training') if verbose else range(epochs)

        for epoch in iterator:
            self.train()
            total_loss = 0.0
            total_soh = 0.0
            total_rul = 0.0
            n_batches = 0

            for batch in train_loader:
                dev = self._param_device()

                x = batch['feature'].to(dev)
                soh = batch['soh'].to(dev)
                rul = batch['rul'].to(dev)
                rul_mask = batch.get('rul_mask')
                if rul_mask is not None:
                    rul_mask = rul_mask.to(dev)

                optimizer.zero_grad()

                pred = self._as_pred_dict(self(x))
                soh_pred = pred['soh'].squeeze()
                rul_pred = pred['rul'].squeeze()

                soh_loss = torch.mean((soh_pred - soh) ** 2)
                rul_loss = self._masked_mse(rul_pred, rul, rul_mask)

                if target_type == 'soh':
                    loss = soh_loss
                elif target_type == 'rul':
                    loss = rul_loss
                else:
                    loss = soh_loss_weight * soh_loss + rul_loss_weight * rul_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += float(loss.detach().cpu())
                total_soh += float(soh_loss.detach().cpu())
                total_rul += float(rul_loss.detach().cpu())
                n_batches += 1

            total_loss /= max(1, n_batches)
            total_soh /= max(1, n_batches)
            total_rul /= max(1, n_batches)

            history['train_loss'].append(total_loss)
            history['train_soh_loss'].append(total_soh)
            history['train_rul_loss'].append(total_rul)

            if val_loader is not None:
                val_metrics = self._evaluate_losses(
                    val_loader,
                    target_type=target_type,
                    soh_loss_weight=soh_loss_weight,
                    rul_loss_weight=rul_loss_weight,
                )
                history['val_loss'].append(val_metrics['loss'])
                history['val_soh_loss'].append(val_metrics['soh_loss'])
                history['val_rul_loss'].append(val_metrics['rul_loss'])

            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}: loss={total_loss:.6f}, soh={total_soh:.6f}, rul={total_rul:.6f}"
                if val_loader is not None and len(history['val_loss']) > 0:
                    msg += f", val={history['val_loss'][-1]:.6f}"
                tqdm.write(msg)

        if save_path:
            self.save(save_path)

        return history

    @torch.no_grad()
    def _evaluate_losses(
        self,
        loader: DataLoader,
        target_type: str,
        soh_loss_weight: float,
        rul_loss_weight: float,
    ) -> Dict[str, float]:
        self.eval()

        total_loss = 0.0
        total_soh = 0.0
        total_rul = 0.0
        n_batches = 0

        dev = self._param_device()

        for batch in loader:
            x = batch['feature'].to(dev)
            soh = batch['soh'].to(dev)
            rul = batch['rul'].to(dev)
            rul_mask = batch.get('rul_mask')
            if rul_mask is not None:
                rul_mask = rul_mask.to(dev)

            pred = self._as_pred_dict(self(x))
            soh_pred = pred['soh'].squeeze()
            rul_pred = pred['rul'].squeeze()

            soh_loss = torch.mean((soh_pred - soh) ** 2)
            rul_loss = self._masked_mse(rul_pred, rul, rul_mask)

            if target_type == 'soh':
                loss = soh_loss
            elif target_type == 'rul':
                loss = rul_loss
            else:
                loss = soh_loss_weight * soh_loss + rul_loss_weight * rul_loss

            total_loss += float(loss.detach().cpu())
            total_soh += float(soh_loss.detach().cpu())
            total_rul += float(rul_loss.detach().cpu())
            n_batches += 1

        total_loss /= max(1, n_batches)
        total_soh /= max(1, n_batches)
        total_rul /= max(1, n_batches)

        return {'loss': total_loss, 'soh_loss': total_soh, 'rul_loss': total_rul}

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Dict[str, np.ndarray]:
        """批量预测，返回 {'soh':..., 'rul':...}。"""
        self.eval()
        soh_preds = []
        rul_preds = []

        dev = self._param_device()

        for batch in loader:
            x = batch['feature'].to(dev)
            pred = self._as_pred_dict(self(x))
            soh_preds.append(pred['soh'].detach().cpu())
            rul_preds.append(pred['rul'].detach().cpu())
        return {
            'soh': torch.cat(soh_preds).squeeze(-1).numpy(),
            'rul': torch.cat(rul_preds).squeeze(-1).numpy(),
        }

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'model_state_dict': self.state_dict(),
                'model_class': self.__class__.__name__,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self._device, weights_only=True)
        self.load_state_dict(checkpoint['model_state_dict'])

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))

        mask = y_true != 0
        if mask.any():
            mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        else:
            mape = float('inf')

        # R2
        denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if denom > 0:
            r2 = 1.0 - float(np.sum((y_true - y_pred) ** 2)) / denom
        else:
            r2 = float('nan')

        return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}

    def evaluate_with_predictions(self, loader: DataLoader) -> Dict[str, Any]:
        """评估并返回样本级预测，用于多指标与可视化。"""
        preds = self.predict(loader)

        soh_true = np.concatenate([batch['soh'].numpy() for batch in loader])
        rul_true = np.concatenate([batch['rul'].numpy() for batch in loader])
        rul_mask = np.concatenate([batch.get('rul_mask', torch.ones_like(batch['rul'])).numpy() for batch in loader])

        out: Dict[str, Any] = {
            'soh_true': soh_true,
            'soh_pred': preds['soh'],
            'rul_true': rul_true,
            'rul_pred': preds['rul'],
            'rul_mask': rul_mask,
        }

        out['soh_metrics'] = self.compute_metrics(soh_true, preds['soh'])

        valid = rul_mask.flatten() > 0.5
        if valid.any():
            out['rul_metrics'] = self.compute_metrics(rul_true[valid], preds['rul'][valid])
        else:
            out['rul_metrics'] = {'RMSE': float('inf'), 'MAE': float('inf'), 'MAPE': float('inf'), 'R2': float('nan')}

        return out

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """默认评估 SOH 与 RUL 两套指标（RUL只评估有效样本）。"""
        out = self.evaluate_with_predictions(loader)
        soh_m = out['soh_metrics']
        rul_m = out['rul_metrics']

        return {
            'SOH_RMSE': soh_m['RMSE'],
            'SOH_MAE': soh_m['MAE'],
            'SOH_MAPE': soh_m['MAPE'],
            'SOH_R2': soh_m['R2'],
            'RUL_RMSE': rul_m['RMSE'],
            'RUL_MAE': rul_m['MAE'],
            'RUL_MAPE': rul_m['MAPE'],
            'RUL_R2': rul_m['R2'],
        }
