"""
TJU数据集预处理 (天津大学)

数据格式：
- 3个数据集zip文件：NCA电池、NCM电池、NCM+NCA混合电池
- 每个zip解压后包含多个CSV文件，每个文件是一个电池
- CSV列：time/s, Ecell/V (电压), <I>/mA (电流), Q discharge/mA.h, Q charge/mA.h, cycle number
- 标称容量约3.5Ah (3500mAh)

数据集：
- Dataset_1_NCA_battery: NCA电池，25°C测试
- Dataset_2_NCM_battery: NCM电池
- Dataset_3_NCM_NCA_battery: NCM+NCA混合电池
"""

import zipfile
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm

from ..battery_data import BatteryData, CycleData
from .base import find_charging_segment


# TJU数据集配置
TJU_DATASETS = {
    'Dataset_1_NCA_battery': {
        'chemistry': 'NCA',
        'nominal_capacity': 3.5,  # Ah
    },
    'Dataset_2_NCM_battery': {
        'chemistry': 'NCM',
        'nominal_capacity': 3.5,
    },
    'Dataset_3_NCM_NCA_battery': {
        'chemistry': 'NCM+NCA',
        'nominal_capacity': 3.5,
    },
}


def preprocess_tju(
    raw_dir: str,
    output_dir: str,
    nominal_capacity: float = 3.5,
    verbose: bool = True,
) -> List[BatteryData]:
    """
    预处理TJU数据集
    
    Args:
        raw_dir: 原始数据目录，包含Dataset_*.zip文件
        output_dir: 输出目录
        nominal_capacity: 标称容量 (Ah)，默认3.5Ah
        verbose: 是否显示进度
    
    Returns:
        BatteryData列表
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    batteries = []
    
    # 处理每个数据集
    for dataset_name, config in TJU_DATASETS.items():
        zip_file = raw_dir / f'{dataset_name}.zip'
        
        if not zip_file.exists():
            if verbose:
                print(f"[INFO] {zip_file.name} not found, skipping...")
            continue
        
        if verbose:
            print(f"\n[INFO] Processing {dataset_name}...")
        
        # 解压并处理
        with zipfile.ZipFile(zip_file, 'r') as zf:
            csv_files = [n for n in zf.namelist() if n.endswith('.csv')]
            
            if verbose:
                csv_files_iter = tqdm(csv_files, desc=f'Processing {dataset_name}')
            else:
                csv_files_iter = csv_files
            
            for csv_file in csv_files_iter:
                try:
                    with zf.open(csv_file) as f:
                        battery = _process_tju_cell(
                            f, 
                            csv_file,
                            dataset_name,
                            config,
                            nominal_capacity,
                            output_dir
                        )
                        if battery is not None and len(battery.cycles) > 10:
                            batteries.append(battery)
                            if verbose:
                                tqdm.write(f'Saved: {battery.cell_id} ({len(battery.cycles)} cycles)')
                except Exception as e:
                    warnings.warn(f"Error processing {csv_file}: {e}")
    
    print(f"TJU: Processed {len(batteries)} batteries")
    return batteries


def _process_tju_cell(
    file_handle,
    filename: str,
    dataset_name: str,
    config: Dict,
    nominal_capacity: float,
    output_dir: Path,
) -> Optional[BatteryData]:
    """处理单个TJU电池CSV文件"""
    
    # 读取CSV
    df = pd.read_csv(file_handle)
    
    # 检查必要的列
    required_cols = ['time/s', 'Ecell/V', '<I>/mA', 'cycle number']
    for col in required_cols:
        if col not in df.columns:
            warnings.warn(f"Missing column {col} in {filename}")
            return None
    
    # 获取唯一循环号
    cycle_numbers = sorted(df['cycle number'].dropna().unique())
    
    if len(cycle_numbers) == 0:
        return None
    
    cycles = []
    
    for cycle_num in cycle_numbers:
        cycle_data = df[df['cycle number'] == cycle_num]
        
        if len(cycle_data) < 10:
            continue
        
        # 提取数据
        V = cycle_data['Ecell/V'].values
        I = cycle_data['<I>/mA'].values / 1000.0  # mA -> A
        t = cycle_data['time/s'].values
        
        # 获取容量（mAh -> Ah）
        discharge_capacity = cycle_data['Q discharge/mA.h'].max() / 1000.0 if 'Q discharge/mA.h' in cycle_data.columns else 0.0
        charge_capacity = cycle_data['Q charge/mA.h'].max() / 1000.0 if 'Q charge/mA.h' in cycle_data.columns else 0.0
        
        # 相对时间（从循环开始计算）
        t = t - t.min()
        
        # 提取充电段
        charging = find_charging_segment(V, I, t)
        
        if len(charging['voltage']) < 10:
            continue
        
        # 使用放电容量计算SOH（如果没有放电容量，使用充电容量）
        capacity = discharge_capacity if discharge_capacity > 0 else charge_capacity
        
        cycle = CycleData(
            cycle_number=int(cycle_num),
            voltage=charging['voltage'].astype(np.float32),
            current=charging['current'].astype(np.float32),
            time=charging['time'].astype(np.float32),
            capacity=capacity if capacity > 0 else None,
        )
        cycles.append(cycle)
    
    if len(cycles) == 0:
        return None
    
    # 从文件名提取电池ID
    # 格式: Dataset_1_NCA_battery/CY25-025_1-#1.csv
    cell_name = Path(filename).stem  # CY25-025_1-#1
    cell_id = f"TJU_{dataset_name.split('_')[1]}_{cell_name}"  # TJU_1_CY25-025_1-#1
    
    battery = BatteryData(
        cell_id=cell_id,
        dataset='TJU',
        nominal_capacity=nominal_capacity,
        cycles=cycles,
        chemistry=config['chemistry'],
        form_factor='cylindrical',
        charge_cutoff_voltage=4.2,
        extra={
            'dataset': dataset_name,
            'original_file': filename,
        }
    )
    
    # 计算SOH（基于第一次循环的容量）
    battery.compute_soh()
    battery.compute_eol()
    battery.save(output_dir / f'{battery.cell_id}.pkl')
    
    return battery
