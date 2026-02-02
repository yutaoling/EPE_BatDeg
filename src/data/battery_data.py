"""
电池数据类 - 简化版
借鉴BatteryML的设计，但去掉不必要的复杂性
"""

import pickle
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import sys

# 为了兼容旧版本pickle文件，注册模块别名
# 旧版本使用 'data.battery_data' 模块路径
if 'data' not in sys.modules or not hasattr(sys.modules.get('data'), 'battery_data'):
    class _DataModule:
        pass
    _data_mod = _DataModule()
    sys.modules['data'] = _data_mod


@dataclass
class CycleData:
    """
    单个循环的数据
    
    只保留充电区间的数据（根据项目需求）
    """
    cycle_number: int
    
    # 充电曲线数据（必需）
    voltage: np.ndarray              # 电压序列 (V)
    current: np.ndarray              # 电流序列 (A)
    time: np.ndarray                 # 时间序列 (s)
    
    # 循环级别的标签
    capacity: float = None           # 本循环容量 (Ah)
    soh: float = None                # SOH值 (0-1)
    
    # 扩展字段
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """数据类型转换"""
        self.voltage = np.asarray(self.voltage, dtype=np.float32)
        self.current = np.asarray(self.current, dtype=np.float32)
        self.time = np.asarray(self.time, dtype=np.float32)
    
    def __len__(self):
        return len(self.voltage)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'cycle_number': self.cycle_number,
            'voltage': self.voltage,
            'current': self.current,
            'time': self.time,
            'capacity': self.capacity,
            'soh': self.soh,
            'extra': self.extra,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CycleData':
        """从字典创建"""
        return cls(**data)


@dataclass
class BatteryData:
    """
    单个电池的完整数据
    
    包含元信息和所有循环数据
    """
    # 必需字段
    cell_id: str                     # 电池ID（唯一标识）
    dataset: str                     # 来源数据集名称
    nominal_capacity: float          # 标称容量 (Ah)
    cycles: List[CycleData]          # 所有循环数据
    
    # 电池元信息（可选）
    chemistry: str = None            # 化学体系 (LFP, NCM, NCA等)
    form_factor: str = None          # 电池形态 (18650, 21700, pouch等)
    
    # 充电协议信息（可选）
    charge_cutoff_voltage: float = 4.2   # 充电截止电压 (V)
    charge_cutoff_current: float = None  # 充电截止电流 (A)
    
    # 生命周期信息
    eol_cycle: int = None            # EOL循环数（80% SOH时）
    eol_threshold: float = 0.8       # EOL阈值
    
    # 扩展字段
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self):
        return len(self.cycles)
    
    def __getitem__(self, idx) -> CycleData:
        return self.cycles[idx]
    
    # ============== 便捷属性 ==============
    
    def get_soh_array(self) -> np.ndarray:
        """获取所有循环的SOH数组"""
        return np.array([c.soh for c in self.cycles if c.soh is not None])
    
    def get_capacity_array(self) -> np.ndarray:
        """获取所有循环的容量数组"""
        return np.array([c.capacity for c in self.cycles if c.capacity is not None])
    
    def get_cycle_numbers(self) -> np.ndarray:
        """获取循环编号数组"""
        return np.array([c.cycle_number for c in self.cycles])
    
    @property
    def total_cycles(self) -> int:
        """总循环数"""
        return len(self.cycles)
    
    # ============== SOH/RUL计算 ==============
    
    def compute_soh(self, reference_capacity: float = None):
        """
        计算每个循环的SOH
        
        Args:
            reference_capacity: 参考容量，默认使用第一次循环的容量
        """
        # 如果没有指定参考容量：使用“前10个有效循环容量的均值”作为参考容量
        if reference_capacity is None:
            caps = [c.capacity for c in self.cycles if c.capacity is not None]
            if len(caps) > 0:
                n_base = min(10, len(caps))
                reference_capacity = float(np.mean(caps[:n_base]))
            else:
                reference_capacity = self.nominal_capacity
        
        ref_cap = reference_capacity
        # 为所有循环计算SOH（即使容量为None也尝试计算）
        for cycle in self.cycles:
            if cycle.capacity is not None:
                cycle.soh = cycle.capacity / ref_cap
            # 如果容量为None，SOH保持为None（不强制计算）
    
    def compute_eol(self, threshold: float = 0.8) -> int:
        """
        计算EOL循环数
        
        Args:
            threshold: SOH阈值，默认0.8
        
        Returns:
            EOL循环数，如果未达到则返回None
        """
        self.eol_threshold = threshold
        
        # 直接遍历cycles，找到第一个SOH低于阈值的循环
        # 这样可以正确处理SOH为None的情况
        for cycle in self.cycles:
            if cycle.soh is not None and cycle.soh < threshold:
                self.eol_cycle = int(cycle.cycle_number)
                return self.eol_cycle
        
        # 未找到低于阈值的循环
        self.eol_cycle = None
        return self.eol_cycle
    
    def get_rul_at_cycle(self, cycle_idx: int) -> int:
        """
        获取指定循环的RUL
        
        Args:
            cycle_idx: 循环索引
        
        Returns:
            RUL值（剩余循环数）
        """
        if self.eol_cycle is None:
            self.compute_eol()
        
        if self.eol_cycle is None:
            return None
        
        current_cycle = self.cycles[cycle_idx].cycle_number
        return max(0, self.eol_cycle - current_cycle)
    
    # ============== 数据筛选 ==============
    
    def filter_cycles(self, min_cycle: int = None, max_cycle: int = None) -> 'BatteryData':
        """
        筛选循环范围
        
        Returns:
            新的BatteryData对象
        """
        filtered_cycles = []
        for cycle in self.cycles:
            if min_cycle is not None and cycle.cycle_number < min_cycle:
                continue
            if max_cycle is not None and cycle.cycle_number > max_cycle:
                continue
            filtered_cycles.append(cycle)
        
        # 创建新对象
        new_data = BatteryData(
            cell_id=self.cell_id,
            dataset=self.dataset,
            nominal_capacity=self.nominal_capacity,
            cycles=filtered_cycles,
            chemistry=self.chemistry,
            form_factor=self.form_factor,
            charge_cutoff_voltage=self.charge_cutoff_voltage,
            charge_cutoff_current=self.charge_cutoff_current,
            eol_cycle=self.eol_cycle,
            eol_threshold=self.eol_threshold,
            extra=self.extra.copy(),
        )
        return new_data
    
    # ============== 保存/加载 ==============
    
    def save(self, path: str):
        """保存到文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path: str) -> 'BatteryData':
        """从文件加载 - 支持旧版本pickle格式"""
        # 确保兼容性模块已注册
        if 'data' not in sys.modules or not hasattr(sys.modules.get('data'), 'battery_data'):
            class _DataModule:
                pass
            _data_mod = _DataModule()
            _data_mod.battery_data = sys.modules[__name__]
            sys.modules['data'] = _data_mod
            sys.modules['data.battery_data'] = sys.modules[__name__]
        
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    # ============== 可视化辅助 ==============
    
    def summary(self) -> str:
        """返回电池数据摘要"""
        soh_array = self.get_soh_array()
        summary_str = f"""
========== Battery Summary ==========
Cell ID:          {self.cell_id}
Dataset:          {self.dataset}
Chemistry:        {self.chemistry or 'Unknown'}
Form Factor:      {self.form_factor or 'Unknown'}
Nominal Capacity: {self.nominal_capacity:.3f} Ah
Total Cycles:     {self.total_cycles}
SOH Range:        {soh_array.min():.3f} - {soh_array.max():.3f}
EOL Cycle:        {self.eol_cycle or 'Not reached'}
=====================================
"""
        return summary_str
    
    def __repr__(self):
        return f"BatteryData(cell_id='{self.cell_id}', dataset='{self.dataset}', cycles={len(self.cycles)})"
