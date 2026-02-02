"""
HUST数据集预处理 (华中科技大学)

数据格式：
- hust_data.zip，解压后是pickle文件
- 每个pickle文件包含一个电池的数据
- 列：Current (mA), Time (s), Voltage (V)
"""

import pickle
import shutil
import zipfile
import numpy as np
import warnings
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from ..battery_data import BatteryData, CycleData
from .base import find_charging_segment, calc_capacity
import pandas as pd


def _extract_hust_cycle(df, cycle_num: int, nominal_capacity: float) -> Optional[CycleData]:
    """从DataFrame中提取单个循环"""
    # 尝试不同的列名
    if isinstance(df, pd.DataFrame):
        # 尝试不同的电流列名
        if 'Current (mA)' in df.columns:
            I = df['Current (mA)'].values / 1000.0
        elif 'current' in df.columns:
            I = df['current'].values
        elif 'Current' in df.columns:
            I = df['Current'].values
        elif 'I' in df.columns:
            I = df['I'].values
        else:
            return None
        
        # 尝试不同的时间列名
        if 'Time (s)' in df.columns:
            t = df['Time (s)'].values
        elif 'time' in df.columns:
            t = df['time'].values
        elif 'Time' in df.columns:
            t = df['Time'].values
        elif 't' in df.columns:
            t = df['t'].values
        else:
            return None
        
        # 尝试不同的电压列名
        if 'Voltage (V)' in df.columns:
            V = df['Voltage (V)'].values
        elif 'voltage' in df.columns:
            V = df['voltage'].values
        elif 'Voltage' in df.columns:
            V = df['Voltage'].values
        elif 'V' in df.columns:
            V = df['V'].values
        else:
            return None
    elif isinstance(df, dict):
        # 从dict中提取
        I = np.array(df.get('Current (mA)', df.get('current', df.get('I', [])))) 
        if 'Current (mA)' in df:
            I = I / 1000.0
        t = np.array(df.get('Time (s)', df.get('time', df.get('t', []))))
        V = np.array(df.get('Voltage (V)', df.get('voltage', df.get('V', []))))
        if len(I) == 0 or len(t) == 0 or len(V) == 0:
            return None
    else:
        return None
    
    Qd = calc_capacity(I, t, is_charge=False)
    discharge_capacity = float(np.max(Qd)) if len(Qd) > 0 else 0.0
    
    charging = find_charging_segment(V, I, t)
    
    if len(charging['voltage']) < 10:
        return None
    
    return CycleData(
        cycle_number=cycle_num,
        voltage=charging['voltage'].astype(np.float32),
        current=charging['current'].astype(np.float32),
        time=charging['time'].astype(np.float32),
        capacity=discharge_capacity,
    )


def preprocess_hust(
    raw_dir: str,
    output_dir: str,
    nominal_capacity: float = 1.1,
    verbose: bool = True,
) -> List[BatteryData]:
    """
    预处理HUST数据集
    
    Args:
        raw_dir: 原始数据目录
        output_dir: 输出目录
        nominal_capacity: 标称容量 (Ah)
        verbose: 是否显示进度
    
    Returns:
        BatteryData列表
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    zip_file = raw_dir / 'our_data.zip'
    
    if not zip_file.exists():
        warnings.warn(f"HUST data file not found: {zip_file}")
        return []
    
    with zipfile.ZipFile(zip_file, 'r') as zf:
        if verbose:
            for f in tqdm(zf.namelist(), desc='Extracting HUST'):
                zf.extract(f, raw_dir)
        else:
            zf.extractall(raw_dir)
    
    # 尝试多个可能的目录名
    possible_dirs = ['our_data', 'data', 'HUST', 'hust_data']
    data_dir = None
    for dir_name in possible_dirs:
        candidate = raw_dir / dir_name
        if candidate.exists():
            data_dir = candidate
            break
    
    if data_dir is None:
        # 直接在raw_dir中查找pkl文件
        data_dir = raw_dir
    
    cell_files = list(data_dir.glob('*.pkl'))
    
    # 如果没找到，尝试递归查找
    if len(cell_files) == 0:
        cell_files = list(data_dir.rglob('*.pkl'))
    
    if len(cell_files) == 0:
        warnings.warn(f"HUST: No pkl files found in {data_dir}")
        return []
    
    if verbose:
        print(f"HUST: Found {len(cell_files)} pkl files in {data_dir}")
    
    batteries = []
    pbar = tqdm(cell_files, desc='Processing HUST') if verbose else cell_files
    
    for cell_file in pbar:
        cell_id = cell_file.stem
        if verbose:
            pbar.set_description(f'Processing {cell_id}')
        
        try:
            battery = _process_hust_cell(cell_file, cell_id, nominal_capacity, output_dir)
            if battery is not None and len(battery.cycles) > 10:
                batteries.append(battery)
                if verbose:
                    tqdm.write(f'Saved: {battery.cell_id} ({len(battery.cycles)} cycles)')
        except Exception as e:
            warnings.warn(f"Error processing HUST {cell_id}: {e}")
    
    # 清理解压目录（但不删除raw_dir本身）
    if data_dir != raw_dir and data_dir.exists():
        try:
            shutil.rmtree(data_dir)
        except Exception:
            pass
    
    print(f"HUST: Processed {len(batteries)} batteries")
    return batteries


def _process_hust_cell(cell_file: Path, cell_id: str, nominal_capacity: float, output_dir: Path) -> Optional[BatteryData]:
    """处理单个HUST电池"""
    with open(cell_file, 'rb') as f:
        raw_data = pickle.load(f)
    
    # 尝试不同的数据结构
    cell_data = None
    if isinstance(raw_data, dict):
        if cell_id in raw_data and 'data' in raw_data[cell_id]:
            cell_data = raw_data[cell_id]['data']
        elif 'data' in raw_data:
            cell_data = raw_data['data']
        elif cell_id in raw_data:
            cell_data = raw_data[cell_id]
        else:
            # 尝试第一个key
            first_key = list(raw_data.keys())[0] if raw_data else None
            if first_key and isinstance(raw_data[first_key], dict):
                if 'data' in raw_data[first_key]:
                    cell_data = raw_data[first_key]['data']
                else:
                    cell_data = raw_data[first_key]
    
    if cell_data is None:
        warnings.warn(f"HUST {cell_id}: Cannot find data structure")
        return None
    
    cycles = []
    # 尝试不同的索引方式
    if isinstance(cell_data, dict):
        cycle_keys = sorted([k for k in cell_data.keys() if isinstance(k, int)])
        if not cycle_keys:
            cycle_keys = sorted([int(k) for k in cell_data.keys() if str(k).isdigit()])
        for cycle_num in cycle_keys:
            try:
                df = cell_data[cycle_num]
                cycles.append(_extract_hust_cycle(df, cycle_num, nominal_capacity))
            except Exception:
                continue
    elif isinstance(cell_data, list):
        for cycle_num, df in enumerate(cell_data):
            try:
                cycles.append(_extract_hust_cycle(df, cycle_num + 1, nominal_capacity))
            except Exception:
                continue
    
    # 过滤None
    cycles = [c for c in cycles if c is not None]
    
    cell_name = f'HUST_{cell_id}'
    if cell_name == 'HUST_7-5' and len(cycles) > 2:
        cycles = cycles[2:]
    
    if len(cycles) == 0:
        return None
    
    battery = BatteryData(
        cell_id=cell_name,
        dataset='HUST',
        nominal_capacity=nominal_capacity,
        cycles=cycles,
        chemistry='LFP',
        form_factor='18650',
        charge_cutoff_voltage=3.6,
    )
    
    # 计算SOH（基于第一次循环的容量）
    battery.compute_soh()
    battery.compute_eol()
    battery.save(output_dir / f'{battery.cell_id}.pkl')
    
    return battery
