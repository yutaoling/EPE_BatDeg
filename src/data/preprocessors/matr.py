"""
MATR数据集预处理 (MIT-Stanford-Toyota)

数据格式：
- MATLAB v7.3 HDF5格式 (.mat文件)
- 每个batch文件包含多个电池
- 结构：batch -> summary, cycles -> I, V, T, t, Qc, Qd...
"""

import h5py
import numpy as np
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from ..battery_data import BatteryData, CycleData
from .base import find_charging_segment, get_discharge_capacity


def preprocess_matr(
    raw_dir: str,
    output_dir: str,
    nominal_capacity: float = 1.1,
    verbose: bool = True,
) -> List[BatteryData]:
    """
    预处理MATR数据集
    
    Args:
        raw_dir: 原始数据目录，包含MATR_batch_*.mat文件
        output_dir: 输出目录
        nominal_capacity: 标称容量 (Ah)
        verbose: 是否显示进度
    
    Returns:
        BatteryData列表
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    batch_files = sorted([
        raw_dir / '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
        raw_dir / '2017-06-30_batchdata_updated_struct_errorcorrect.mat',
        raw_dir / '2018-04-12_batchdata_updated_struct_errorcorrect.mat',
        raw_dir / '2019-01-24_batchdata_updated_struct_errorcorrect.mat',
    ])
    
    batch_files = [f for f in batch_files if f.exists()]
    
    if len(batch_files) == 0:
        warnings.warn(f"No MATR batch files found in {raw_dir}")
        return []
    
    data_batches = []
    pbar = tqdm(enumerate(batch_files), total=len(batch_files), desc='Loading MATR batches') if verbose else enumerate(batch_files)
    
    for batch_idx, batch_file in pbar:
        if verbose:
            pbar.set_description(f'Loading {batch_file.name}')
        batch_data = _load_matr_batch(batch_file, batch_idx + 1)
        data_batches.append(batch_data)
    
    if len(data_batches) >= 2:
        _merge_matr_batches(data_batches)
    
    batch2_keys = {'b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16'}
    
    batteries = []
    total_cells = sum(len(batch) for batch in data_batches)
    
    pbar = tqdm(total=total_cells, desc='Processing MATR cells') if verbose else None
    
    for batch in data_batches:
        for cell_id, cell_data in batch.items():
            if pbar:
                pbar.update(1)
                pbar.set_description(f'Processing {cell_id}')
            
            if cell_id in batch2_keys:
                continue
            
            try:
                battery = _organize_matr_cell(cell_data, cell_id, nominal_capacity)
                if battery is not None and len(battery.cycles) > 10:
                    battery.save(output_dir / f'{battery.cell_id}.pkl')
                    batteries.append(battery)
                    if verbose:
                        tqdm.write(f'Saved: {battery.cell_id} ({len(battery.cycles)} cycles)')
            except Exception as e:
                warnings.warn(f"Error processing {cell_id}: {e}")
    
    if pbar:
        pbar.close()
    
    print(f"MATR: Processed {len(batteries)} batteries")
    return batteries


def _load_matr_batch(file: Path, batch_num: int) -> Dict[str, Any]:
    """加载单个MATR batch文件"""
    f = h5py.File(file, 'r')
    batch = f['batch']
    num_cells = batch['summary'].shape[0]
    bat_dict = {}
    
    for i in range(num_cells):
        try:
            cl = f[batch['cycle_life'][i, 0]][:]
            policy = f[batch['policy_readable'][i, 0]][:].tobytes()[::2].decode()
            
            summary = {}
            summary_group = f[batch['summary'][i, 0]]
            for key in ['IR', 'QCharge', 'QDischarge', 'Tavg', 'Tmin', 'Tmax', 'chargetime', 'cycle']:
                summary[key] = np.hstack(summary_group[key][0, :].tolist())
            
            cycles_group = f[batch['cycles'][i, 0]]
            cycle_dict = {}
            
            for j in range(cycles_group['I'].shape[0]):
                cd = {}
                for key in ['I', 'Qc', 'Qd', 'T', 'V', 't']:
                    cd[key] = np.hstack(f[cycles_group[key][j, 0]][:])
                if 'Qdlin' in cycles_group:
                    cd['Qdlin'] = np.hstack(f[cycles_group['Qdlin'][j, 0]][:])
                cycle_dict[str(j)] = cd
            
            cell_dict = {
                'cycle_life': cl,
                'charge_policy': policy,
                'summary': summary,
                'cycles': cycle_dict,
            }
            
            key = f'b{batch_num}c{i}'
            bat_dict[key] = cell_dict
            
        except Exception as e:
            warnings.warn(f"Error loading cell {i} from batch {batch_num}: {e}")
    
    f.close()
    return bat_dict


def _merge_matr_batches(data_batches: List[Dict]) -> None:
    """合并batch2延续到batch1的电池数据"""
    batch2_keys = ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']
    batch1_keys = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']
    add_len = [662, 981, 1060, 208, 482]
    
    for i, (b1k, b2k) in enumerate(zip(batch1_keys, batch2_keys)):
        if b1k not in data_batches[0] or b2k not in data_batches[1]:
            continue
        
        data_batches[0][b1k]['cycle_life'] = data_batches[0][b1k]['cycle_life'] + add_len[i]
        
        for key in data_batches[0][b1k]['summary'].keys():
            if key == 'cycle':
                data_batches[0][b1k]['summary'][key] = np.hstack((
                    data_batches[0][b1k]['summary'][key],
                    data_batches[1][b2k]['summary'][key] + len(data_batches[0][b1k]['summary'][key])
                ))
            else:
                data_batches[0][b1k]['summary'][key] = np.hstack((
                    data_batches[0][b1k]['summary'][key],
                    data_batches[1][b2k]['summary'][key]
                ))
        
        last_cycle = len(data_batches[0][b1k]['cycles'])
        for j, jk in enumerate(data_batches[1][b2k]['cycles'].keys()):
            data_batches[0][b1k]['cycles'][str(last_cycle + j)] = data_batches[1][b2k]['cycles'][jk]


def _organize_matr_cell(data: Dict, name: str, nominal_capacity: float) -> Optional[BatteryData]:
    """将MATR原始数据转换为BatteryData格式"""
    cycles = []
    
    for cycle_num in range(len(data['cycles'])):
        if cycle_num == 0:
            continue
        
        cur_data = data['cycles'][str(cycle_num)]
        
        V = cur_data['V']
        I = cur_data['I']
        t = cur_data['t']
        
        charging = find_charging_segment(V, I, t)
        
        discharge_capacity = get_discharge_capacity(I, t)
        
        if 'QDischarge' in data['summary'] and cycle_num < len(data['summary']['QDischarge']):
            discharge_capacity = data['summary']['QDischarge'][cycle_num]
        
        if len(charging['voltage']) < 10:
            continue
        
        cycle = CycleData(
            cycle_number=cycle_num,
            voltage=charging['voltage'].astype(np.float32),
            current=charging['current'].astype(np.float32),
            time=charging['time'].astype(np.float32),
            capacity=discharge_capacity,
        )
        cycles.append(cycle)
    
    if len(cycles) == 0:
        return None
    
    battery = BatteryData(
        cell_id=f'MATR_{name}',
        dataset='MATR',
        nominal_capacity=nominal_capacity,
        cycles=cycles,
        chemistry='LFP',
        form_factor='18650',
        charge_cutoff_voltage=3.6,
    )
    
    # 计算SOH（基于第一次循环的容量）
    battery.compute_soh()
    battery.compute_eol()
    
    return battery
