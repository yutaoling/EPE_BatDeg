"""
NASA数据集预处理 (NASA Ames Research Center)

数据格式：
- battery_alt_dataset.zip 包含加速寿命测试数据
- 三个子集：
  - regular_alt_batteries: 恒定负载测试
  - recommissioned_batteries: 不同寿命阶段测试
  - second_life_batteries: 二次寿命电池测试
- CSV列：start_time, time, mode (-1=放电, 0=静置, 1=充电), 
         voltage_charger, temperature_battery, voltage_load, current_load
- 2串电池组，电压约8-9V

注意：
- 此数据集为加速寿命测试（ALT），与传统循环测试不同
- 通过mode字段识别充放电阶段
- start_time代表测试日，每天可能包含多个充放电循环
"""

import zipfile
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm

from ..battery_data import BatteryData, CycleData
from .base import find_charging_segment, calc_capacity


# NASA电池组配置（2串）
NASA_CONFIG = {
    'cells_in_series': 2,
    'nominal_voltage_per_cell': 4.2,  # V
    'nominal_capacity': 2.6,  # Ah (估算单节容量)
}


def preprocess_nasa(
    raw_dir: str,
    output_dir: str,
    nominal_capacity: float = 2.6,
    verbose: bool = True,
) -> List[BatteryData]:
    """
    预处理NASA加速寿命测试数据集
    
    Args:
        raw_dir: 原始数据目录，包含battery_alt_dataset.zip
        output_dir: 输出目录
        nominal_capacity: 标称容量 (Ah)，默认2.6Ah
        verbose: 是否显示进度
    
    Returns:
        BatteryData列表
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    zip_file = raw_dir / 'battery_alt_dataset.zip'
    
    if not zip_file.exists():
        warnings.warn(f"NASA data file not found: {zip_file}")
        return []
    
    batteries = []
    
    with zipfile.ZipFile(zip_file, 'r') as zf:
        # 获取所有CSV文件（排除MACOSX）
        csv_files = sorted([
            n for n in zf.namelist() 
            if n.endswith('.csv') and 'MACOSX' not in n
        ])
        
        if verbose:
            print(f"[INFO] Found {len(csv_files)} battery files")
            csv_files_iter = tqdm(csv_files, desc='Processing NASA')
        else:
            csv_files_iter = csv_files
        
        for csv_file in csv_files_iter:
            try:
                with zf.open(csv_file) as f:
                    battery = _process_nasa_cell(
                        f,
                        csv_file,
                        nominal_capacity,
                        output_dir
                    )
                    if battery is not None and len(battery.cycles) > 5:
                        batteries.append(battery)
                        if verbose:
                            tqdm.write(f'Saved: {battery.cell_id} ({len(battery.cycles)} cycles)')
            except Exception as e:
                warnings.warn(f"Error processing {csv_file}: {e}")
    
    print(f"NASA: Processed {len(batteries)} batteries")
    return batteries


def _process_nasa_cell(
    file_handle,
    filename: str,
    nominal_capacity: float,
    output_dir: Path,
) -> Optional[BatteryData]:
    """处理单个NASA电池CSV文件"""
    
    # 读取CSV
    df = pd.read_csv(file_handle, low_memory=False)
    
    # 检查必要的列
    required_cols = ['time', 'mode', 'voltage_charger']
    for col in required_cols:
        if col not in df.columns:
            warnings.warn(f"Missing column {col} in {filename}")
            return None
    
    # 将电压转换为单节电压（2串电池组）
    df['voltage'] = df['voltage_charger'] / NASA_CONFIG['cells_in_series']
    
    # 使用电流信息
    # NASA数据中，充电时没有current_load，需要根据mode推断
    df['current'] = 0.0
    if 'current_load' in df.columns:
        # 放电电流（负值）
        df.loc[df['mode'] == -1, 'current'] = -df.loc[df['mode'] == -1, 'current_load'].fillna(0).abs()
    # 充电时假设固定电流（根据电压变化估算）
    df.loc[df['mode'] == 1, 'current'] = 1.0  # 假设1A充电电流
    
    # 识别充放电循环
    cycles_data = _extract_cycles(df)
    
    if len(cycles_data) == 0:
        return None
    
    cycles = []
    
    for cycle_idx, cycle_df in enumerate(cycles_data):
        if len(cycle_df) < 10:
            continue
        
        V = cycle_df['voltage'].values
        I = cycle_df['current'].values
        t = cycle_df['time'].values - cycle_df['time'].values.min()
        
        # 计算容量（优先使用放电数据，更准确）
        discharge_mask = cycle_df['mode'].values == -1
        charge_mask = cycle_df['mode'].values == 1
        
        capacity = None
        
        # 方法1：使用放电电流积分计算容量
        if discharge_mask.any() and 'current_load' in cycle_df.columns:
            discharge_df = cycle_df[discharge_mask]
            discharge_current = discharge_df['current_load'].abs().values
            discharge_time = discharge_df['time'].values
            if len(discharge_time) > 1:
                # 梯形积分：容量 = ∫|I|dt
                dt = np.diff(discharge_time) / 3600.0  # 转换为小时
                avg_current = (discharge_current[:-1] + discharge_current[1:]) / 2
                capacity = np.sum(avg_current * dt)
        
        # 方法2：退回到充电时间估算
        if capacity is None or capacity < 0.1:
            if charge_mask.any():
                charge_time = t[charge_mask]
                if len(charge_time) > 1:
                    charge_duration_hours = (charge_time.max() - charge_time.min()) / 3600.0
                    capacity = charge_duration_hours * 1.0  # 假设1A充电
        
        charge_capacity = capacity if capacity is not None and capacity > 0.1 else 0.0
        
        # 提取充电段
        charging = find_charging_segment(V, I, t, current_threshold=0.1)
        
        if len(charging['voltage']) < 10:
            continue
        
        cycle = CycleData(
            cycle_number=cycle_idx + 1,
            voltage=charging['voltage'].astype(np.float32),
            current=charging['current'].astype(np.float32),
            time=charging['time'].astype(np.float32),
            capacity=charge_capacity if charge_capacity > 0.1 else None,  # 过滤极小值
        )
        cycles.append(cycle)
    
    if len(cycles) == 0:
        return None
    
    # 从文件名提取电池ID和类别
    # 格式: battery_alt_dataset/regular_alt_batteries/battery00.csv
    parts = Path(filename).parts
    category = parts[-2] if len(parts) >= 2 else 'unknown'
    cell_name = Path(filename).stem  # battery00
    cell_id = f"NASA_{cell_name}"
    
    battery = BatteryData(
        cell_id=cell_id,
        dataset='NASA',
        nominal_capacity=nominal_capacity,
        cycles=cycles,
        chemistry='Li-ion',  # 具体化学体系未知
        form_factor='pack',  # 2串电池组
        charge_cutoff_voltage=4.2,
        extra={
            'category': category,
            'cells_in_series': NASA_CONFIG['cells_in_series'],
            'original_file': filename,
        }
    )
    
    # 计算SOH（基于第一次循环的容量）
    battery.compute_soh()
    battery.compute_eol()
    battery.save(output_dir / f'{battery.cell_id}.pkl')
    
    return battery


def _extract_cycles(df: pd.DataFrame) -> List[pd.DataFrame]:
    """
    从连续数据中提取充放电循环
    
    策略：
    1. 按start_time分组（代表不同的测试日）
    2. 在每个测试日内，找到完整的充电段
    """
    cycles = []
    
    # 按start_time分组
    for start_time, day_df in df.groupby('start_time'):
        day_df = day_df.sort_values('time').reset_index(drop=True)
        
        # 找到充电段（mode == 1）
        mode = day_df['mode'].values
        
        # 找到充电开始和结束的位置
        is_charging = mode == 1
        
        if not is_charging.any():
            continue
        
        # 找到连续的充电段
        diff = np.diff(is_charging.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        # 处理边界情况
        if len(starts) == 0 and is_charging[0]:
            starts = np.array([0])
        if len(ends) == 0 and is_charging[-1]:
            ends = np.array([len(mode)])
        
        if len(starts) == 0 or len(ends) == 0:
            continue
        
        if starts[0] > ends[0]:
            ends = ends[1:]
        if len(starts) > len(ends):
            starts = starts[:len(ends)]
        
        # 提取每个充电段
        for start, end in zip(starts, ends):
            # 扩展范围以包含部分静置和放电数据（用于完整的循环分析）
            # 但主要保留充电段
            extended_start = max(0, start - 100)  # 向前扩展
            extended_end = min(len(day_df), end + 100)  # 向后扩展
            
            cycle_df = day_df.iloc[extended_start:extended_end].copy()
            
            if len(cycle_df) >= 50:  # 至少50个数据点
                cycles.append(cycle_df)
    
    return cycles
