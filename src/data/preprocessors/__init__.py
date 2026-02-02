"""
数据预处理模块

将不同数据集的原始数据转换为统一的BatteryData格式

支持的数据集：
- MATR (MIT-Stanford-Toyota): MATLAB v7.3 HDF5格式
- CALCE (马里兰大学): Excel/CSV + ZIP格式
- HUST (华中科技大学): Pickle + ZIP格式
- RWTH (亚琛工业大学): CSV + 多层ZIP格式
- XJTU (西安交通大学): MATLAB v7格式
- TJU (天津大学): CSV + ZIP格式
- NASA (NASA Ames): 加速寿命测试CSV + ZIP格式
"""

from pathlib import Path
from typing import List, Dict
import warnings

from ..battery_data import BatteryData

# 导入各数据集预处理器
from .matr import preprocess_matr
from .calce import preprocess_calce
from .hust import preprocess_hust
from .rwth import preprocess_rwth
from .xjtu import preprocess_xjtu
from .tju import preprocess_tju
from .nasa import preprocess_nasa

# 导入通用工具函数
from .base import (
    find_charging_segment,
    calc_capacity,
    get_discharge_capacity,
    get_charge_capacity,
    load_processed_batteries,
    load_all_processed,
)


# 预处理器注册表
PREPROCESSORS = {
    'MATR': preprocess_matr,
    'CALCE': preprocess_calce,
    'HUST': preprocess_hust,
    'RWTH': preprocess_rwth,
    'XJTU': preprocess_xjtu,
    'TJU': preprocess_tju,
    'NASA': preprocess_nasa,
}


def preprocess_dataset(
    dataset_name: str,
    raw_dir: str,
    output_dir: str,
    **kwargs,
) -> List[BatteryData]:
    """
    统一的预处理接口
    
    Args:
        dataset_name: 数据集名称 ('MATR', 'CALCE', 'HUST', 'RWTH', 'XJTU', 'TJU', 'NASA')
        raw_dir: 原始数据目录
        output_dir: 输出目录
        **kwargs: 传递给具体预处理器的参数
    
    Returns:
        BatteryData列表
    
    Example:
        >>> batteries = preprocess_dataset('MATR', 'data/raw/MATR', 'data/processed/MATR')
    """
    if dataset_name not in PREPROCESSORS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(PREPROCESSORS.keys())}")
    
    return PREPROCESSORS[dataset_name](raw_dir, output_dir, **kwargs)


def preprocess_all(
    raw_base_dir: str,
    output_base_dir: str,
    datasets: List[str] = None,
    **kwargs,
) -> Dict[str, List[BatteryData]]:
    """
    批量预处理所有数据集
    
    Args:
        raw_base_dir: 原始数据根目录
        output_base_dir: 输出根目录
        datasets: 要处理的数据集列表，默认处理所有
        **kwargs: 传递给具体预处理器的参数
    
    Returns:
        {dataset_name: [BatteryData, ...], ...}
    
    Example:
        >>> all_batteries = preprocess_all('data/raw', 'data/processed')
    """
    if datasets is None:
        datasets = list(PREPROCESSORS.keys())
    
    raw_base_dir = Path(raw_base_dir)
    output_base_dir = Path(output_base_dir)
    
    results = {}
    
    for dataset in datasets:
        raw_dir = raw_base_dir / dataset
        output_dir = output_base_dir / dataset
        
        if not raw_dir.exists():
            warnings.warn(f"Raw directory not found: {raw_dir}")
            continue
        
        print(f"\n{'='*50}")
        print(f"Processing {dataset}")
        print(f"{'='*50}")
        
        try:
            batteries = preprocess_dataset(dataset, str(raw_dir), str(output_dir), **kwargs)
            results[dataset] = batteries
        except Exception as e:
            warnings.warn(f"Error processing {dataset}: {e}")
            results[dataset] = []
    
    # 打印总结
    print(f"\n{'='*50}")
    print("Summary")
    print(f"{'='*50}")
    total = 0
    for dataset, batteries in results.items():
        print(f"{dataset}: {len(batteries)} batteries")
        total += len(batteries)
    print(f"Total: {total} batteries")
    
    return results


__all__ = [
    # 数据集预处理器
    'preprocess_matr',
    'preprocess_calce',
    'preprocess_hust',
    'preprocess_rwth',
    'preprocess_xjtu',
    'preprocess_tju',
    'preprocess_nasa',
    # 统一接口
    'preprocess_dataset',
    'preprocess_all',
    # 工具函数
    'find_charging_segment',
    'calc_capacity',
    'get_discharge_capacity',
    'get_charge_capacity',
    'load_processed_batteries',
    'load_all_processed',
    # 预处理器注册表
    'PREPROCESSORS',
]
