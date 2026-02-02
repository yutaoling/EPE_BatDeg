"""
RWTH数据集预处理 (亚琛工业大学)

数据格式：
- RWTH.zip -> RWTH-2021-04545_818642/ -> Rawdata.zip
- Rawdata.zip -> Rohdaten/ -> *.zip (每个电池)
- 每个电池的zip解压后是CSV文件
- CSV列：Zeit(时间戳), Programmdauer(程序时间,ms), Strom(电流,A), Spannung(电压,V)

注意：
- 标称容量2.05Ah，但由于质量问题约为1.85Ah
- 循环在20%-80% SOC之间，有效容量约为1.85 * 0.6 = 1.11Ah
"""

import shutil
import zipfile
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from ..battery_data import BatteryData, CycleData
from .base import find_charging_segment, calc_capacity


def preprocess_rwth(
    raw_dir: str,
    output_dir: str,
    nominal_capacity: float = 1.11,
    verbose: bool = True,
) -> List[BatteryData]:
    """
    预处理RWTH数据集
    
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
    
    zip_file = raw_dir / 'RWTH.zip'
    
    if not zip_file.exists():
        warnings.warn(f"RWTH data file not found: {zip_file}")
        return []
    
    print('[INFO] Extracting RWTH.zip...')
    subdir = raw_dir / 'RWTH-2021-04545_818642'
    if not (subdir / 'Rawdata.zip').exists():
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(raw_dir)
    
    print('[INFO] Extracting Rawdata.zip...')
    rawdata_zip = subdir / 'Rawdata.zip'
    if not rawdata_zip.exists():
        warnings.warn(f"Rawdata.zip not found in {subdir}")
        return []
    
    with zipfile.ZipFile(rawdata_zip, 'r') as zf:
        files = zf.namelist()
        if verbose:
            files = tqdm(files, desc='Extracting cell data')
        for file in files:
            if "BOL" not in file and not (subdir / file).exists():
                zf.extract(file, subdir)
    
    datadir = subdir / 'Rohdaten'
    if not datadir.exists():
        warnings.warn(f"Rohdaten directory not found")
        return []
    
    print('[INFO] Extracting individual cell zip files...')
    cell_zips = list(datadir.glob('*.zip'))
    if verbose:
        cell_zips = tqdm(cell_zips, desc='Extracting cell zips')
    for cell_zip in cell_zips:
        csv_file = datadir / f'{cell_zip.stem}.csv'
        if not csv_file.exists():
            try:
                with zipfile.ZipFile(cell_zip, 'r') as zf:
                    zf.extractall(datadir)
            except Exception as e:
                warnings.warn(f"Error extracting {cell_zip}: {e}")
    
    batteries = []
    cells = [f'{i:03}' for i in range(2, 50)]
    
    if verbose:
        cells = tqdm(cells, desc='Processing RWTH cells')
    
    for cell in cells:
        cell_name = f'RWTH_{cell}'
        if verbose:
            cells.set_description(f'Processing {cell_name}')
        
        try:
            battery = _process_rwth_cell(datadir, cell, nominal_capacity, output_dir)
            if battery is not None and len(battery.cycles) > 10:
                batteries.append(battery)
                if verbose:
                    tqdm.write(f'Saved: {battery.cell_id} ({len(battery.cycles)} cycles)')
        except Exception as e:
            warnings.warn(f"Error processing RWTH {cell}: {e}")
    
    print('[INFO] Cleaning up extracted files...')
    try:
        shutil.rmtree(subdir)
    except Exception as e:
        warnings.warn(f"Error cleaning up: {e}")
    
    print(f"RWTH: Processed {len(batteries)} batteries")
    return batteries


def _process_rwth_cell(datadir: Path, cell: str, nominal_capacity: float, output_dir: Path) -> Optional[BatteryData]:
    """处理单个RWTH电池"""
    csv_files = list(datadir.glob(f'*{cell}=ZYK*Zyk*.csv'))
    
    if len(csv_files) == 0:
        return None
    
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, skiprows=[1])
            dfs.append(df)
        except Exception as e:
            warnings.warn(f"Error reading {f}: {e}")
    
    if len(dfs) == 0:
        return None
    
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates('Zeit').sort_values('Zeit')
    df = df[_find_time_anomalies(df['Programmdauer'].values)]
    df = df.reset_index(drop=True)
    
    cycle_ends = _find_cycle_ends(df['Strom'].values)
    cycle_end_indices = df['Strom'][cycle_ends].index.tolist()
    
    if len(cycle_end_indices) > 1:
        cycle_end_indices = cycle_end_indices[1:]
    
    if len(cycle_end_indices) < 2:
        return None
    
    cycles = []
    for i in range(1, len(cycle_end_indices)):
        start_idx = cycle_end_indices[i-1]
        end_idx = cycle_end_indices[i]
        
        cycle_data = df.iloc[start_idx:end_idx]
        
        if len(cycle_data) < 10:
            continue
        
        V = cycle_data['Spannung'].values
        I = cycle_data['Strom'].values
        t = cycle_data['Programmdauer'].values / 1000.0
        
        Qd = calc_capacity(I, t, is_charge=False)
        discharge_capacity = float(np.max(Qd))
        
        charging = find_charging_segment(V, I, t)
        
        if len(charging['voltage']) < 10:
            continue
        
        cycle = CycleData(
            cycle_number=i,
            voltage=charging['voltage'].astype(np.float32),
            current=charging['current'].astype(np.float32),
            time=charging['time'].astype(np.float32),
            capacity=discharge_capacity,
        )
        cycles.append(cycle)
    
    cycles = _clean_rwth_cycles(cycles)
    
    if len(cycles) == 0:
        return None
    
    cell_name = f'RWTH_{cell}'
    battery = BatteryData(
        cell_id=cell_name,
        dataset='RWTH',
        nominal_capacity=nominal_capacity,
        cycles=cycles,
        chemistry='NMC',
        form_factor='18650',
        charge_cutoff_voltage=3.9,
    )
    
    # 计算SOH（基于第一次循环的容量）
    battery.compute_soh()
    battery.compute_eol()
    battery.save(output_dir / f'{battery.cell_id}.pkl')
    
    return battery


def _find_cycle_ends(current: np.ndarray, lag: int = 10, tolerance: float = 0.1) -> np.ndarray:
    """找到循环结束点"""
    is_cycle_end = np.zeros(len(current), dtype=bool)
    enter_discharge_steps = 0
    nms_size = 500
    
    for i in range(len(current)):
        I = current[i]
        
        if 0 < i < len(current) - 1:
            if abs(current[i] - current[i-1]) > tolerance and abs(current[i] - current[i+1]) > tolerance:
                I = current[i+1]
        
        if I < 0:
            enter_discharge_steps += 1
        else:
            enter_discharge_steps = 0
        
        if enter_discharge_steps == lag:
            t = i - lag + 1
            if t > nms_size and np.max(is_cycle_end[t-nms_size:t]) > 0:
                continue
            is_cycle_end[t] = True
    
    return is_cycle_end


def _find_time_anomalies(time: np.ndarray, tolerance: float = 1e5) -> np.ndarray:
    """找到时间异常点"""
    result = np.ones(len(time), dtype=bool)
    prev = time[0]
    
    for i in range(1, len(time)):
        if time[i] - prev > tolerance:
            result[i] = False
        else:
            prev = time[i]
    
    return result


def _clean_rwth_cycles(cycles: List[CycleData], eps: float = 0.05, window: int = 5) -> List[CycleData]:
    """清理RWTH异常循环"""
    if len(cycles) <= 2 * window:
        return cycles
    
    Qd = np.array([c.capacity for c in cycles])
    to_remove = np.zeros(len(Qd), dtype=bool)
    
    for i in range(window, len(Qd) - window):
        prev_median = np.median(Qd[max(0, i-window):i])
        next_median = np.median(Qd[i:i+window])
        
        if abs(Qd[i] - prev_median) > eps and abs(Qd[i] - next_median) > eps:
            to_remove[i] = True
    
    clean_cycles = []
    new_idx = 1
    for i, cycle in enumerate(cycles):
        if not to_remove[i]:
            cycle.cycle_number = new_idx
            clean_cycles.append(cycle)
            new_idx += 1
    
    return clean_cycles
