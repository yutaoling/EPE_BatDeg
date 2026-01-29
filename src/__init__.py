# BatterySOH - 锂电池健康状态预测框架
# Copyright (c) 2025

# 基础数据类（不依赖torch）
from .data.battery_data import BatteryData, CycleData


# 延迟导入需要torch的模块
def _lazy_import():
    """延迟导入需要torch的模块"""
    from .data import BatteryDataset, create_dataloaders
    from .models import BaseModel
    return BatteryDataset, BaseModel, create_dataloaders

__version__ = '0.1.0'
__all__ = [
    'BatteryData', 'CycleData', 
    'data', 'models', 'train', 'experiments'
]
