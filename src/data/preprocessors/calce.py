"""
CALCE数据集预处理 (马里兰大学)

数据格式：
- ZIP文件，解压后是Excel或TXT文件
- 列：Cycle_Index, Test_Time(s), Current(A), Voltage(V)
- CS2_*: 标称容量1.1Ah
- CX2_*: 标称容量1.35Ah
"""

import os
import re
import shutil
import time
import zipfile
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from scipy.signal import medfilt
import gc

from ..battery_data import BatteryData, CycleData
from .base import find_charging_segment, calc_capacity


def preprocess_calce(
    raw_dir: str,
    output_dir: str,
    verbose: bool = True,
) -> List[BatteryData]:
    """
    预处理CALCE数据集
    
    Args:
        raw_dir: 原始数据目录
        output_dir: 输出目录
        verbose: 是否显示进度
    
    Returns:
        BatteryData列表
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    zip_files = list(raw_dir.glob('*.zip'))
    
    if len(zip_files) == 0:
        warnings.warn(f"No CALCE zip files found in {raw_dir}")
        return []
    
    batteries = []
    pbar = tqdm(zip_files, desc='Processing CALCE') if verbose else zip_files
    
    for zip_file in pbar:
        cell_name = zip_file.stem
        if verbose:
            pbar.set_description(f'Processing {cell_name}')
        
        try:
            battery = _process_calce_cell(zip_file, output_dir, verbose)
            if battery is not None and len(battery.cycles) > 10:
                batteries.append(battery)
                if verbose:
                    tqdm.write(f'Saved: {battery.cell_id} ({len(battery.cycles)} cycles)')
        except Exception as e:
            warnings.warn(f"Error processing {cell_name}: {e}")
    
    print(f"CALCE: Processed {len(batteries)} batteries")
    return batteries


def _process_calce_cell(zip_file: Path, output_dir: Path, verbose: bool = True) -> Optional[BatteryData]:
    """处理单个CALCE电池"""
    cell_name = zip_file.stem
    raw_dir = zip_file.parent
    data_dir = raw_dir / cell_name
    
    if not data_dir.exists():
        with zipfile.ZipFile(zip_file, 'r') as zf:
            zf.extractall(raw_dir)
        if cell_name == 'CX2_8' and (raw_dir / 'cx2_8').exists():
            os.rename(raw_dir / 'cx2_8', data_dir)
    
    if not data_dir.exists():
        return None
    
    data_files = list(data_dir.glob('*.xlsx')) + list(data_dir.glob('*.xls')) + list(data_dir.glob('*.txt'))
    
    if len(data_files) == 0:
        return None
    
    dfs = []
    for f in data_files:
        try:
            df = _load_calce_file(f)
            if df is not None and len(df) > 0:
                dfs.append(df)
        except Exception as e:
            warnings.warn(f"Error loading {f}: {e}")
    
    if len(dfs) == 0:
        return None
    
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(['date', 'Test_Time(s)'])
    
    df['Cycle_Index'] = _organize_cycle_index(df['Cycle_Index'].values)
    
    cycles = []
    for cycle_idx, (_, cycle_df) in enumerate(df.groupby(['date', 'Cycle_Index'])):
        I = cycle_df['Current(A)'].values
        t = cycle_df['Test_Time(s)'].values
        V = cycle_df['Voltage(V)'].values
        
        Qd = calc_capacity(I, t, is_charge=False)
        discharge_capacity = float(np.max(Qd))
        
        charging = find_charging_segment(V, I, t)
        
        if len(charging['voltage']) < 10:
            continue
        
        cycle = CycleData(
            cycle_number=cycle_idx,
            voltage=charging['voltage'].astype(np.float32),
            current=charging['current'].astype(np.float32),
            time=charging['time'].astype(np.float32),
            capacity=discharge_capacity,
        )
        cycles.append(cycle)
    
    cycles = _clean_calce_cycles(cycles, cell_name)
    
    if len(cycles) == 0:
        return None
    
    nominal_capacity = 1.1 if 'CS' in cell_name.upper() else 1.35
    
    battery = BatteryData(
        cell_id=f'CALCE_{cell_name}',
        dataset='CALCE',
        nominal_capacity=nominal_capacity,
        cycles=cycles,
        chemistry='LCO',
        form_factor='prismatic',
        charge_cutoff_voltage=4.2,
    )
    
    # 计算SOH（基于第一次循环的容量）
    battery.compute_soh()
    battery.compute_eol()
    battery.save(output_dir / f'{battery.cell_id}.pkl')
    
    # 强制垃圾回收，释放文件句柄
    gc.collect()
    
    # 重试删除目录（Windows可能需要时间释放文件句柄）
    for retry in range(3):
        try:
            shutil.rmtree(data_dir)
            break
        except PermissionError:
            time.sleep(0.5)
            gc.collect()
    
    return battery


def _load_calce_file(file: Path) -> Optional[pd.DataFrame]:
    """加载CALCE数据文件"""
    cache_file = file.with_name(file.stem + '_cache.csv')
    if cache_file.exists():
        return pd.read_csv(cache_file)
    
    if file.suffix == '.txt':
        df = pd.read_csv(file, sep='\t')
        date = _extract_calce_date(file.stem)
        result = pd.DataFrame({
            'date': date,
            'Cycle_Index': df['Charge count'] // 2 + 1,
            'Test_Time(s)': df['Time'],
            'Current(A)': df['mA'] / 1000.,
            'Voltage(V)': df['mV'] / 1000.,
        })
        return result
    
    elif file.suffix in ['.xlsx', '.xls']:
        excel_file = None
        try:
            # 尝试使用openpyxl引擎读取xlsx文件
            excel_file = pd.ExcelFile(file, engine='openpyxl')
        except ImportError:
            # 如果没有openpyxl，尝试使用xlrd引擎
            try:
                excel_file = pd.ExcelFile(file, engine='xlrd')
            except Exception:
                warnings.warn(f"Cannot read {file}: install openpyxl with 'pip install openpyxl'")
                return None
        except Exception as e:
            warnings.warn(f"Error reading {file}: {e}")
            return None
        
        try:
            channel_data = []
            for sheet_name in excel_file.sheet_names:
                if sheet_name.startswith('Channel') or sheet_name == 'Sheet1':
                    channel_data.append(excel_file.parse(sheet_name))
            
            if len(channel_data) == 0:
                return None
            
            df = pd.concat(channel_data, ignore_index=True)
            date = _extract_calce_date(file.stem)
            df['date'] = date
            
            columns_to_keep = ['date', 'Cycle_Index', 'Test_Time(s)', 'Current(A)', 'Voltage(V)']
            
            for col in columns_to_keep:
                if col not in df.columns:
                    return None
            
            result = df[columns_to_keep]
            
            result.to_csv(cache_file, index=False)
            
            return result
        finally:
            # 确保关闭Excel文件，释放文件句柄
            if excel_file is not None:
                excel_file.close()
    
    else:
        return None


def _extract_calce_date(filename: str) -> str:
    """从文件名提取日期"""
    filename = filename.upper()
    pat = r'C[XS]2?_\d+_(\d+)_(\d+)B?_(\d+)'
    matches = re.findall(pat, filename)
    if not matches:
        pat = r'(\d+)_(\d+)_(\d+)_CX2_32'
        matches = re.findall(pat, filename)
    if matches:
        month, day, year = matches[0]
        month, day, year = int(month), int(day), int(year)
        return f'{year:04}-{month:02}-{day:02}'
    return '0000-01-01'


def _organize_cycle_index(cycle_index: np.ndarray) -> np.ndarray:
    """重新组织循环索引"""
    result = cycle_index.copy()
    current_cycle = result[0]
    prev_value = result[0]
    
    for i in range(1, len(result)):
        if result[i] != prev_value:
            current_cycle += 1
            prev_value = result[i]
        result[i] = current_cycle
    
    return result


def _clean_calce_cycles(cycles: List[CycleData], cell_name: str) -> List[CycleData]:
    """清理CALCE循环数据，去除异常值"""
    if len(cycles) == 0:
        return cycles
    
    Qd = np.array([c.capacity for c in cycles])
    
    Qd_med = medfilt(Qd, min(21, len(Qd) // 2 * 2 + 1))
    ths = np.median(np.abs(Qd - Qd_med))
    should_keep = np.abs(Qd - Qd_med) < 3 * ths
    
    if cell_name == 'CX2_34' and len(should_keep) > 0:
        should_keep[0] = False
    if cell_name == 'CX2_16' and len(should_keep) > 0:
        should_keep[0] = False
    
    clean_cycles = []
    new_idx = 1
    for i, cycle in enumerate(cycles):
        if should_keep[i] and Qd[i] > 0.1:
            cycle.cycle_number = new_idx
            clean_cycles.append(cycle)
            new_idx += 1
    
    return clean_cycles
