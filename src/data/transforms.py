"""
数据变换模块

包含：
1. Z-Score标准化
2. Log尺度变换
3. Min-Max归一化
4. 协议无关归一化（Protocol-Invariant Transform）
"""

import torch
import numpy as np
from typing import Optional, Union, Dict


class BaseTransform:
    """数据变换基类"""
    
    def __init__(self):
        self._fitted = False
    
    def fit(self, data: Union[torch.Tensor, np.ndarray]):
        """拟合变换参数"""
        raise NotImplementedError
    
    def transform(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """应用变换"""
        raise NotImplementedError
    
    def inverse_transform(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """逆变换"""
        raise NotImplementedError
    
    def fit_transform(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """拟合并变换"""
        self.fit(data)
        return self.transform(data)
    
    def __call__(self, data):
        return self.transform(data)


class ZScoreTransform(BaseTransform):
    """
    Z-Score标准化
    
    x_normalized = (x - mean) / std
    """
    
    def __init__(self, dim: int = 0, eps: float = 1e-8):
        """
        Args:
            dim: 计算统计量的维度
            eps: 防止除零的小量
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self._mean = None
        self._std = None
    
    def fit(self, data: Union[torch.Tensor, np.ndarray]):
        """计算均值和标准差"""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        
        self._mean = data.mean(dim=self.dim, keepdim=True)
        self._std = data.std(dim=self.dim, keepdim=True).clamp(min=self.eps)
        self._fitted = True
    
    def transform(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """应用Z-Score标准化"""
        assert self._fitted, "Transform not fitted! Call fit() first."
        
        is_numpy = isinstance(data, np.ndarray)
        if is_numpy:
            data = torch.from_numpy(data)
        
        result = (data - self._mean) / self._std
        
        return result.numpy() if is_numpy else result
    
    def inverse_transform(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """逆Z-Score变换"""
        assert self._fitted, "Transform not fitted! Call fit() first."
        
        is_numpy = isinstance(data, np.ndarray)
        if is_numpy:
            data = torch.from_numpy(data)
        
        result = data * self._std + self._mean
        
        return result.numpy() if is_numpy else result
    
    def to(self, device: str):
        """移动到指定设备"""
        if self._mean is not None:
            self._mean = self._mean.to(device)
            self._std = self._std.to(device)
        return self
    
    @property
    def mean(self):
        return self._mean
    
    @property
    def std(self):
        return self._std


class LogScaleTransform(BaseTransform):
    """
    Log尺度变换
    
    x_transformed = log(x + offset)
    
    常用于RUL等非负且分布偏斜的标签
    """
    
    def __init__(self, offset: float = 1.0, base: float = np.e):
        """
        Args:
            offset: 偏移量，确保log的输入为正
            base: 对数底数
        """
        super().__init__()
        self.offset = offset
        self.base = base
        self._fitted = True  # 无状态变换，不需要fit
    
    def fit(self, data: Union[torch.Tensor, np.ndarray]):
        """Log变换是无状态的，不需要fit"""
        pass
    
    def transform(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """应用Log变换"""
        is_numpy = isinstance(data, np.ndarray)
        if is_numpy:
            if self.base == np.e:
                return np.log(data + self.offset)
            else:
                return np.log(data + self.offset) / np.log(self.base)
        else:
            if self.base == np.e:
                return torch.log(data + self.offset)
            else:
                return torch.log(data + self.offset) / np.log(self.base)
    
    def inverse_transform(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """逆Log变换"""
        is_numpy = isinstance(data, np.ndarray)
        if is_numpy:
            if self.base == np.e:
                return np.exp(data) - self.offset
            else:
                return np.power(self.base, data) - self.offset
        else:
            if self.base == np.e:
                return torch.exp(data) - self.offset
            else:
                return torch.pow(self.base, data) - self.offset


class MinMaxTransform(BaseTransform):
    """
    Min-Max归一化
    
    x_normalized = (x - min) / (max - min)
    
    将数据缩放到[0, 1]范围
    """
    
    def __init__(self, dim: int = 0, eps: float = 1e-8):
        """
        Args:
            dim: 计算统计量的维度
            eps: 防止除零的小量
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        self._min = None
        self._max = None
    
    def fit(self, data: Union[torch.Tensor, np.ndarray]):
        """计算最小值和最大值"""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        
        self._min = data.min(dim=self.dim, keepdim=True).values
        self._max = data.max(dim=self.dim, keepdim=True).values
        self._fitted = True
    
    def transform(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """应用Min-Max归一化"""
        assert self._fitted, "Transform not fitted! Call fit() first."
        
        is_numpy = isinstance(data, np.ndarray)
        if is_numpy:
            data = torch.from_numpy(data)
        
        range_ = (self._max - self._min).clamp(min=self.eps)
        result = (data - self._min) / range_
        
        return result.numpy() if is_numpy else result
    
    def inverse_transform(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """逆Min-Max变换"""
        assert self._fitted, "Transform not fitted! Call fit() first."
        
        is_numpy = isinstance(data, np.ndarray)
        if is_numpy:
            data = torch.from_numpy(data)
        
        range_ = self._max - self._min
        result = data * range_ + self._min
        
        return result.numpy() if is_numpy else result
    
    def to(self, device: str):
        """移动到指定设备"""
        if self._min is not None:
            self._min = self._min.to(device)
            self._max = self._max.to(device)
        return self


class SequentialTransform(BaseTransform):
    """
    组合多个变换
    """
    
    def __init__(self, transforms: list):
        """
        Args:
            transforms: 变换列表
        """
        super().__init__()
        self.transforms = transforms
    
    def fit(self, data: Union[torch.Tensor, np.ndarray]):
        """依次拟合所有变换"""
        current_data = data
        for t in self.transforms:
            t.fit(current_data)
            current_data = t.transform(current_data)
        self._fitted = True
    
    def transform(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """依次应用所有变换"""
        result = data
        for t in self.transforms:
            result = t.transform(result)
        return result
    
    def inverse_transform(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """逆序应用所有逆变换"""
        result = data
        for t in reversed(self.transforms):
            result = t.inverse_transform(result)
        return result
    
    def to(self, device: str):
        """移动到指定设备"""
        for t in self.transforms:
            if hasattr(t, 'to'):
                t.to(device)
        return self


# ============== 协议无关变换 (Protocol-Invariant Transforms) ==============

class ProtocolInvariantTransform(BaseTransform):
    """
    协议无关的数据归一化
    
    针对时序数据 (sequence/image)，将不同充电协议的数据归一化到统一范围，
    消除绝对数值差异，保留相对变化模式。
    
    归一化策略：
    - 电压: 归一化到 [0, 1]，基于单样本的 min-max
    - 电流: 归一化到 [-1, 1]，基于单样本的 max(abs)
    - 时间: 归一化到 [0, 1]，基于单样本的 max
    
    适用于：
    - sequence 数据: shape (T, C) 或 (batch, T, C)
    - image 数据: shape (H, W, C) 或 (batch, H, W, C)
    
    """
    
    # 默认通道顺序
    CHANNEL_NAMES = ['v_delta', 'i_delta', 'q_norm']
    
    def __init__(
        self, 
        channels: list = None,
        voltage_range: tuple = (0, 1),
        current_range: tuple = (-1, 1),
        time_range: tuple = (0, 1),
        missing_value: float = 0.0,
        missing_threshold: float = 1e-6,
        eps: float = 1e-8,
    ):
        """
        Args:
            channels: 通道名称列表，默认 ['voltage', 'current', 'time']
            voltage_range: 电压归一化目标范围
            current_range: 电流归一化目标范围
            time_range: 时间归一化目标范围
            missing_value: 缺失通道的填充值
            missing_threshold: 判断数据缺失的阈值（标准差小于此值视为缺失）
            eps: 防止除零的小量
        """
        super().__init__()
        self.channels = channels or self.CHANNEL_NAMES
        # 为保持兼容：保留原参数名，但现在 time_range 对应 q_norm
        self.voltage_range = voltage_range
        self.current_range = current_range
        self.time_range = time_range
        self.missing_value = missing_value
        self.missing_threshold = missing_threshold
        self.eps = eps
        self._fitted = True  # 无状态变换，不需要全局 fit
        
        # 构建通道索引映射
        self._channel_idx = {name: i for i, name in enumerate(self.channels)}
    
    def fit(self, data: Union[torch.Tensor, np.ndarray]):
        """协议归一化是按样本的，不需要全局 fit"""
        pass
    
    def transform(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        应用协议无关归一化
        
        Args:
            data: 时序数据，shape 为 (T, C), (H, W, C) 或带 batch 维度
        
        Returns:
            归一化后的数据，保持原始 shape
        """
        is_numpy = isinstance(data, np.ndarray)
        if not is_numpy:
            data = data.numpy()
        
        # 复制数据避免修改原始数据
        result = data.copy()
        
        # 判断数据维度
        # (T, C) -> sequence 单样本
        # (H, W, C) -> image 单样本
        # (B, T, C) -> sequence batch
        # (B, H, W, C) -> image batch
        
        if result.ndim == 2:
            # (T, C) - 单个 sequence
            result = self._normalize_single(result)
        elif result.ndim == 3:
            # (H, W, C) - 单个 image 或 (B, T, C) - batch sequence
            # 通过最后一维判断
            if result.shape[-1] == len(self.channels):
                # 可能是 image (H, W, C) 或 batch sequence (B, T, C)
                # 这里假设是 image，对整个 image 做归一化
                result = self._normalize_image(result)
            else:
                # batch sequence
                for i in range(result.shape[0]):
                    result[i] = self._normalize_single(result[i])
        elif result.ndim == 4:
            # (B, H, W, C) - batch image
            for i in range(result.shape[0]):
                result[i] = self._normalize_image(result[i])
        
        if not is_numpy:
            result = torch.from_numpy(result)
        
        return result
    
    def _is_missing(self, data: np.ndarray) -> bool:
        """检测数据是否缺失（全0、常量或标准差极小）"""
        if np.all(data == 0):
            return True
        if np.std(data) < self.missing_threshold:
            return True
        return False
    
    def _normalize_single(self, data: np.ndarray) -> np.ndarray:
        """归一化单个 sequence (T, C)"""
        result = data.copy()
        
        for ch_name, ch_idx in self._channel_idx.items():
            if ch_idx >= result.shape[-1]:
                continue
            
            channel_data = result[:, ch_idx]
            
            if ch_name == 'v_delta':
                result[:, ch_idx] = self._minmax_normalize(
                    channel_data, self.voltage_range
                )
            elif ch_name == 'i_delta':
                result[:, ch_idx] = self._symmetric_normalize(
                    channel_data, self.current_range
                )
            elif ch_name == 'q_norm':
                result[:, ch_idx] = self._time_normalize(
                    channel_data, self.time_range
                )
        
        return result
    
    def _normalize_image(self, data: np.ndarray) -> np.ndarray:
        """归一化单个 image (H, W, C)"""
        result = data.copy()
        
        for ch_name, ch_idx in self._channel_idx.items():
            if ch_idx >= result.shape[-1]:
                continue
            
            channel_data = result[:, :, ch_idx]
            
            if ch_name == 'v_delta':
                result[:, :, ch_idx] = self._minmax_normalize(
                    channel_data, self.voltage_range
                )
            elif ch_name == 'i_delta':
                result[:, :, ch_idx] = self._symmetric_normalize(
                    channel_data, self.current_range
                )
            elif ch_name == 'q_norm':
                result[:, :, ch_idx] = self._time_normalize(
                    channel_data, self.time_range
                )
        
        return result
    
    def _minmax_normalize(self, data: np.ndarray, target_range: tuple) -> np.ndarray:
        """Min-Max 归一化到目标范围"""
        d_min, d_max = data.min(), data.max()
        d_range = d_max - d_min
        
        if d_range < self.eps:
            # 所有值相同，返回目标范围的中点
            return np.full_like(data, (target_range[0] + target_range[1]) / 2)
        
        # 先归一化到 [0, 1]
        normalized = (data - d_min) / d_range
        # 再映射到目标范围
        t_min, t_max = target_range
        return normalized * (t_max - t_min) + t_min
    
    def _symmetric_normalize(self, data: np.ndarray, target_range: tuple) -> np.ndarray:
        """对称归一化（适用于电流，保持正负关系）"""
        abs_max = np.abs(data).max()
        
        if abs_max < self.eps:
            return np.zeros_like(data)
        
        # 归一化到 [-1, 1]
        normalized = data / abs_max
        # 映射到目标范围
        t_min, t_max = target_range
        t_mid = (t_min + t_max) / 2
        t_half = (t_max - t_min) / 2
        return normalized * t_half + t_mid
    
    def _time_normalize(self, data: np.ndarray, target_range: tuple) -> np.ndarray:
        """时间归一化（从0开始）"""
        t_min = data.min()
        t_max = data.max()
        t_range = t_max - t_min
        
        if t_range < self.eps:
            return np.full_like(data, target_range[0])
        
        # 从最小值开始归一化
        normalized = (data - t_min) / t_range
        # 映射到目标范围
        r_min, r_max = target_range
        return normalized * (r_max - r_min) + r_min
    
    def inverse_transform(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        协议归一化是不可逆的（信息丢失）
        返回原数据的警告
        """
        import warnings
        warnings.warn(
            "ProtocolInvariantTransform is not invertible. "
            "Returning data as-is."
        )
        return data


class ProtocolAwareTransform(BaseTransform):
    """
    协议感知的归一化
    
    根据数据集/协议的先验知识进行归一化，使用预定义的参考值。
    这允许不同数据集使用相同的归一化标准，便于跨域比较。
    
    预定义的协议参考值：
    - LFP (MATR, HUST, XJTU): V_cutoff=3.6V, typical V_range=[2.0, 3.6]
    - LCO (CALCE): V_cutoff=4.2V, typical V_range=[2.75, 4.2]
    - NMC (RWTH, TJU): V_cutoff=4.2V, typical V_range=[2.5, 4.2]
    """
    
    # 预定义的协议参数
    PROTOCOL_PARAMS = {
        'LFP': {
            'v_min': 2.0, 'v_max': 3.6,
            'i_max': 3.0,  # 典型最大充电电流 (A)
        },
        'LCO': {
            'v_min': 2.75, 'v_max': 4.2,
            'i_max': 2.0,
        },
        'NMC': {
            'v_min': 2.5, 'v_max': 4.2,
            'i_max': 3.5,
        },
        'NCA': {
            'v_min': 2.5, 'v_max': 4.2,
            'i_max': 3.5,
        },
        'default': {
            'v_min': 2.0, 'v_max': 4.2,
            'i_max': 5.0,
        },
    }
    
    # 数据集到协议的映射
    DATASET_PROTOCOL = {
        'MATR': 'LFP',
        'HUST': 'LFP',
        'XJTU': 'LFP',
        'CALCE': 'LCO',
        'RWTH': 'NMC',
        'TJU': 'NCA',
        'NASA': 'default',
    }
    
    def __init__(
        self,
        protocol: str = None,
        dataset: str = None,
        channels: list = None,
        missing_value: float = 0.0,
        missing_threshold: float = 1e-6,
        eps: float = 1e-8,
    ):
        """
        Args:
            protocol: 协议类型 ('LFP', 'LCO', 'NMC', 'NCA', 'default')
            dataset: 数据集名称，自动映射到协议
            channels: 通道名称列表
            missing_value: 缺失通道的填充值
            missing_threshold: 判断数据缺失的阈值
            eps: 防止除零的小量
        """
        super().__init__()
        
        # 确定协议
        if protocol is not None:
            self.protocol = protocol
        elif dataset is not None:
            self.protocol = self.DATASET_PROTOCOL.get(dataset, 'default')
        else:
            self.protocol = 'default'
        
        self.params = self.PROTOCOL_PARAMS[self.protocol]
        self.channels = channels or ['voltage', 'current', 'time']
        self.missing_value = missing_value
        self.missing_threshold = missing_threshold
        self.eps = eps
        self._fitted = True
        
        self._channel_idx = {name: i for i, name in enumerate(self.channels)}
    
    def _is_missing(self, data: np.ndarray) -> bool:
        """检测数据是否缺失"""
        if np.all(data == 0):
            return True
        if np.std(data) < self.missing_threshold:
            return True
        return False
    
    def fit(self, data: Union[torch.Tensor, np.ndarray]):
        """使用预定义参数，不需要 fit"""
        pass
    
    def transform(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """应用协议感知归一化"""
        is_numpy = isinstance(data, np.ndarray)
        if not is_numpy:
            data = data.numpy()
        
        result = data.copy()
        
        # 获取协议参数
        v_min, v_max = self.params['v_min'], self.params['v_max']
        i_max = self.params['i_max']
        
        # 归一化各通道
        if 'voltage' in self._channel_idx:
            idx = self._channel_idx['voltage']
            if result.ndim == 2:
                result[:, idx] = (result[:, idx] - v_min) / (v_max - v_min + self.eps)
            elif result.ndim == 3:
                result[:, :, idx] = (result[:, :, idx] - v_min) / (v_max - v_min + self.eps)
        
        if 'current' in self._channel_idx:
            idx = self._channel_idx['current']
            if result.ndim == 2:
                result[:, idx] = result[:, idx] / (i_max + self.eps)
            elif result.ndim == 3:
                result[:, :, idx] = result[:, :, idx] / (i_max + self.eps)
        
        if 'time' in self._channel_idx:
            idx = self._channel_idx['time']
            # 时间归一化到 [0, 1]（基于单样本）
            if result.ndim == 2:
                t_data = result[:, idx]
                t_range = t_data.max() - t_data.min()
                if t_range > self.eps:
                    result[:, idx] = (t_data - t_data.min()) / t_range
            elif result.ndim == 3:
                t_data = result[:, :, idx]
                t_range = t_data.max() - t_data.min()
                if t_range > self.eps:
                    result[:, :, idx] = (t_data - t_data.min()) / t_range
        
        # Clip 到 [0, 1] 范围（电压）或 [-1, 1]（电流）
        result = np.clip(result, -1, 1)
        
        if not is_numpy:
            result = torch.from_numpy(result)
        
        return result
    
    def inverse_transform(self, data: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """逆变换（仅电压和电流可精确恢复）"""
        is_numpy = isinstance(data, np.ndarray)
        if not is_numpy:
            data = data.numpy()
        
        result = data.copy()
        
        v_min, v_max = self.params['v_min'], self.params['v_max']
        i_max = self.params['i_max']
        
        if 'voltage' in self._channel_idx:
            idx = self._channel_idx['voltage']
            if result.ndim == 2:
                result[:, idx] = result[:, idx] * (v_max - v_min) + v_min
            elif result.ndim == 3:
                result[:, :, idx] = result[:, :, idx] * (v_max - v_min) + v_min
        
        if 'current' in self._channel_idx:
            idx = self._channel_idx['current']
            if result.ndim == 2:
                result[:, idx] = result[:, idx] * i_max
            elif result.ndim == 3:
                result[:, :, idx] = result[:, :, idx] * i_max
        
        # 时间无法精确恢复
        
        if not is_numpy:
            result = torch.from_numpy(result)
        
        return result
