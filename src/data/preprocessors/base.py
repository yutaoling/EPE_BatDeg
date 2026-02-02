"""
通用预处理工具函数
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import warnings

from ..battery_data import BatteryData, CycleData


def find_charging_segment(
    voltage: np.ndarray,
    current: np.ndarray,
    time: np.ndarray,
    current_threshold: float = 0.01,
) -> dict:
    """
    从完整循环数据中提取充电段
    
    Args:
        voltage, current, time: 完整循环数据（仅电压、电流、时间）
        current_threshold: 电流阈值，大于此值视为充电
    
    Returns:
        充电段数据字典（仅包含voltage, current, time）
    """
    charging_mask = current > current_threshold
    
    if not charging_mask.any():
        return {
            'voltage': voltage,
            'current': current,
            'time': time,
        }
    
    diff = np.diff(charging_mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    
    if len(starts) == 0 and charging_mask[0]:
        starts = np.array([0])
    if len(ends) == 0 and charging_mask[-1]:
        ends = np.array([len(voltage)])
    
    if len(starts) == 0 or len(ends) == 0:
        return {
            'voltage': voltage,
            'current': current,
            'time': time,
        }
    
    if starts[0] > ends[0]:
        ends = ends[1:]
    if len(starts) > len(ends):
        starts = starts[:len(ends)]
    
    if len(starts) == 0 or len(ends) == 0:
        return {
            'voltage': voltage,
            'current': current,
            'time': time,
        }
    
    lengths = ends - starts
    longest_idx = np.argmax(lengths)
    start, end = starts[longest_idx], ends[longest_idx]
    
    result = {
        'voltage': voltage[start:end],
        'current': current[start:end],
        'time': time[start:end] - time[start],
    }
    
    return result


def calc_capacity(current: np.ndarray, time: np.ndarray, is_charge: bool = True) -> np.ndarray:
    """
    计算累积容量
    
    Args:
        current: 电流数组 (A)
        time: 时间数组 (s)
        is_charge: True为充电容量，False为放电容量
    
    Returns:
        累积容量数组 (Ah)
    """
    Q = np.zeros_like(current, dtype=np.float64)
    for i in range(1, len(current)):
        dt = time[i] - time[i-1]
        if is_charge and current[i] > 0:
            Q[i] = Q[i-1] + current[i] * dt / 3600
        elif not is_charge and current[i] < 0:
            Q[i] = Q[i-1] - current[i] * dt / 3600
        else:
            Q[i] = Q[i-1]
    return Q


def get_discharge_capacity(current: np.ndarray, time: np.ndarray) -> float:
    """获取放电容量"""
    Q = calc_capacity(current, time, is_charge=False)
    return float(np.max(Q)) if len(Q) > 0 else 0.0


def get_charge_capacity(current: np.ndarray, time: np.ndarray) -> float:
    """获取充电容量"""
    Q = calc_capacity(current, time, is_charge=True)
    return float(np.max(Q)) if len(Q) > 0 else 0.0


def load_processed_batteries(processed_dir: str) -> List[BatteryData]:
    """
    加载已处理的电池数据
    
    Args:
        processed_dir: 处理后数据目录
    
    Returns:
        BatteryData列表
    """
    processed_dir = Path(processed_dir)
    batteries = []
    
    pkl_files = list(processed_dir.glob('*.pkl'))
    
    for pkl_file in tqdm(pkl_files, desc=f'Loading from {processed_dir.name}'):
        try:
            battery = BatteryData.load(pkl_file)
            batteries.append(battery)
        except Exception as e:
            warnings.warn(f"Error loading {pkl_file}: {e}")
    
    return batteries


def load_all_processed(processed_base_dir: str) -> Dict[str, List[BatteryData]]:
    """
    加载所有已处理的数据集
    
    Args:
        processed_base_dir: 处理后数据根目录
    
    Returns:
        {dataset_name: [BatteryData, ...], ...}
    """
    processed_base_dir = Path(processed_base_dir)
    results = {}
    
    for dataset_dir in processed_base_dir.iterdir():
        if dataset_dir.is_dir():
            batteries = load_processed_batteries(str(dataset_dir))
            if len(batteries) > 0:
                results[dataset_dir.name] = batteries
    
    return results
