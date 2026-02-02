"""
XJTU数据集预处理 (西安交通大学)

数据格式：
- 多个Batch目录，每个包含.mat文件（MATLAB v7格式，非HDF5）
- 使用scipy.io.loadmat读取
- data: 结构体数组 (1, num_cycles)，包含每个循环的数据
  - voltage_V: 电压序列
  - current_A: 电流序列
  - relative_time_min: 相对时间（分钟）
  - temperature_C: 温度序列
  - capacity_Ah: 容量序列
  - description: 循环描述
- summary: 包含循环寿命和统计信息
  - charge_capacity_Ah: 每循环充电容量
  - discharge_capacity_Ah: 每循环放电容量
  - cycle_life: 总循环数
"""

import scipy.io as sio
import numpy as np
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any
from tqdm import tqdm

from ..battery_data import BatteryData, CycleData
from .base import find_charging_segment, calc_capacity


# XJTU数据集配置
XJTU_BATCHES = {
    'Batch-1': {'prefix': '2C', 'charge_rate': '2C', 'nominal_capacity': 2.0},
    'Batch-2': {'prefix': '3C', 'charge_rate': '3C', 'nominal_capacity': 2.0},
    'Batch-3': {'prefix': 'R2.5', 'charge_rate': 'R2.5', 'nominal_capacity': 2.0},
    'Batch-4': {'prefix': 'R3', 'charge_rate': 'R3', 'nominal_capacity': 2.0},
    'Batch-5': {'prefix': 'RW', 'charge_rate': 'RW', 'nominal_capacity': 2.0},
    'Batch-6': {'prefix': 'Sim_satellite', 'charge_rate': 'satellite', 'nominal_capacity': 2.0},
}


def preprocess_xjtu(
    raw_dir: str,
    output_dir: str,
    nominal_capacity: float = 2.0,
    verbose: bool = True,
) -> List[BatteryData]:
    """
    预处理XJTU数据集
    
    Args:
        raw_dir: 原始数据目录
        output_dir: 输出目录
        nominal_capacity: 标称容量 (Ah)，默认2.0Ah
        verbose: 是否显示进度
    
    Returns:
        BatteryData列表
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    batteries = []
    
    # 处理每个Batch
    for batch_name, batch_config in XJTU_BATCHES.items():
        batch_dir = raw_dir / batch_name
        
        if not batch_dir.exists():
            if verbose:
                print(f"[INFO] Batch directory not found: {batch_dir}, skipping...")
            continue
        
        # 查找该batch的所有.mat文件
        mat_files = list(batch_dir.glob('*.mat'))
        
        if len(mat_files) == 0:
            continue
        
        if verbose:
            print(f"\n[INFO] Processing {batch_name} ({len(mat_files)} files)...")
            mat_files = tqdm(mat_files, desc=f'Processing {batch_name}')
        
        for mat_file in mat_files:
            try:
                battery = _process_xjtu_cell(
                    mat_file, 
                    batch_name, 
                    batch_config,
                    nominal_capacity,
                    output_dir
                )
                if battery is not None and len(battery.cycles) > 10:
                    batteries.append(battery)
                    if verbose:
                        tqdm.write(f'Saved: {battery.cell_id} ({len(battery.cycles)} cycles)')
            except Exception as e:
                warnings.warn(f"Error processing {mat_file.name}: {e}")
    
    print(f"XJTU: Processed {len(batteries)} batteries")
    return batteries


def _process_xjtu_cell(
    mat_file: Path,
    batch_name: str,
    batch_config: Dict[str, Any],
    nominal_capacity: float,
    output_dir: Path,
) -> Optional[BatteryData]:
    """处理单个XJTU电池"""
    
    # 加载.mat文件
    mat_data = sio.loadmat(mat_file, squeeze_me=False)
    
    # 获取data数组（包含每个循环的数据）
    if 'data' not in mat_data:
        warnings.warn(f"No 'data' field in {mat_file.name}")
        return None
    
    data_array = mat_data['data']
    num_cycles = data_array.shape[1]
    
    # 获取summary（包含容量等统计信息）
    summary = mat_data.get('summary', None)
    discharge_capacities = None
    
    if summary is not None:
        try:
            discharge_capacities = summary['discharge_capacity_Ah'][0, 0].flatten()
        except:
            pass
    
    cycles = []
    
    for cycle_idx in range(num_cycles):
        try:
            cycle_data = data_array[0, cycle_idx]
            
            # 提取数据字段
            V = _extract_field(cycle_data, 'voltage_V')
            I = _extract_field(cycle_data, 'current_A')
            t_min = _extract_field(cycle_data, 'relative_time_min')
            
            if V is None or I is None or t_min is None:
                continue
            
            # 转换时间为秒
            t = t_min * 60.0
            
            # 确保数据长度一致
            min_len = min(len(V), len(I), len(t))
            if min_len < 10:
                continue
            
            V = V[:min_len]
            I = I[:min_len]
            t = t[:min_len]
            
            # 获取放电容量
            if discharge_capacities is not None and cycle_idx < len(discharge_capacities):
                discharge_capacity = float(discharge_capacities[cycle_idx])
            else:
                # 计算放电容量
                Qd = calc_capacity(I, t, is_charge=False)
                discharge_capacity = float(np.max(Qd)) if len(Qd) > 0 else 0.0
            
            # 提取充电段
            charging = find_charging_segment(V, I, t)
            
            if len(charging['voltage']) < 10:
                continue
            
            cycle = CycleData(
                cycle_number=cycle_idx + 1,
                voltage=charging['voltage'].astype(np.float32),
                current=charging['current'].astype(np.float32),
                time=charging['time'].astype(np.float32),
                capacity=discharge_capacity,
            )
            cycles.append(cycle)
            
        except Exception as e:
            warnings.warn(f"Error processing cycle {cycle_idx} in {mat_file.name}: {e}")
    
    if len(cycles) == 0:
        return None
    
    # 构建电池ID
    cell_name = mat_file.stem  # 如 "2C_battery-1"
    cell_id = f"XJTU_{cell_name}"
    
    battery = BatteryData(
        cell_id=cell_id,
        dataset='XJTU',
        nominal_capacity=nominal_capacity,
        cycles=cycles,
        chemistry='LFP',  # XJTU数据集使用LFP电池
        form_factor='prismatic',
        charge_cutoff_voltage=3.6,
        extra={
            'batch': batch_name,
            'charge_rate': batch_config['charge_rate'],
        }
    )
    
    # 计算SOH（基于第一次循环的容量）
    battery.compute_soh()
    battery.compute_eol()
    battery.save(output_dir / f'{battery.cell_id}.pkl')
    
    return battery


def _extract_field(struct: np.void, field_name: str) -> Optional[np.ndarray]:
    """
    从MATLAB结构体中提取字段
    
    Args:
        struct: numpy void类型的结构体
        field_name: 字段名
    
    Returns:
        numpy数组或None
    """
    try:
        # 获取字段
        field_data = struct[field_name]
        
        # 处理嵌套数组
        while isinstance(field_data, np.ndarray) and field_data.size == 1:
            field_data = field_data.flat[0]
        
        # 转换为1D数组
        if isinstance(field_data, np.ndarray):
            return field_data.flatten().astype(np.float64)
        
        return None
        
    except (KeyError, IndexError, TypeError):
        return None
