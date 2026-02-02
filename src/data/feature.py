"""
特征提取模块（q_norm + S1通道 + 前10循环baseline扣除）

目标：跨数据集/跨协议更稳健的表示。

- 进度轴：相对容量进度 q_norm ∈ [0,1]
- 通道：S1 = [v_norm, i_norm, q_norm]
- baseline：每块电池取前10个有效循环，计算其 v_norm/i_norm 的平均作为 baseline
- delta：对每个循环输出 v_delta/v_norm - baseline_v，i_delta/i_norm - baseline_i；q_norm 不扣除

输出接口保持不变：
- extract_features_from_cycle -> (16,)
- create_sequence -> (num_samples, 3)
- create_heatmap -> (window_size, num_samples, 3)

注意：
- 该实现会改变特征与输入定义，你计划重新生成processed数据，并将SOH改为前10循环容量均值作Q0。
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional

from .battery_data import CycleData, BatteryData

_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)

DEFAULT_NUM_SAMPLES = 200
DEFAULT_WINDOW_SIZE = 100
BASELINE_CYCLES = 10
EPS = 1e-8

# S1通道名（语义）
DEFAULT_CHANNELS = ['v_delta', 'i_delta', 'q_norm']
S1_CHANNELS = DEFAULT_CHANNELS

FEATURE_NAMES = [
    'v_mean', 'v_std', 'v_max', 'v_min', 'v_range', 'v_slope',
    'i_mean', 'i_std', 'i_max', 'i_min', 'i_range', 'i_slope',
    'charge_time', 'q_total', 'dv_dq_mean', 'di_dq_mean',
]


def _longest_true_run(mask: np.ndarray) -> Tuple[int, int]:
    """返回最长连续True段的[start,end)索引。"""
    if mask.size == 0:
        return 0, 0
    diff = np.diff(mask.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts = np.insert(starts, 0, 0)
    if mask[-1]:
        ends = np.append(ends, mask.size)
    if len(starts) == 0 or len(ends) == 0:
        return 0, mask.size
    lengths = ends[:len(starts)] - starts[:len(ends)]
    if len(lengths) == 0:
        return 0, mask.size
    k = int(np.argmax(lengths))
    return int(starts[k]), int(ends[min(k, len(ends) - 1)])


def _extract_full_charge_curve(cycle: CycleData, min_current: float = 0.01) -> Dict[str, np.ndarray]:
    """用电流阈值提取最长连续充电段，返回相对时间（从0开始）。"""
    v, i, t = cycle.voltage, cycle.current, cycle.time
    if v is None or len(v) == 0:
        return {'voltage': np.array([]), 'current': np.array([]), 'time': np.array([])}

    i = np.asarray(i, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32)

    mask = i > float(min_current)
    if int(mask.sum()) < 10:
        mask = np.abs(i) > float(min_current)
    if int(mask.sum()) < 10:
        mask = np.ones_like(i, dtype=bool)

    s, e = _longest_true_run(mask)
    v_seg = v[s:e]
    i_seg = i[s:e]
    t_seg = t[s:e]
    t_rel = t_seg - t_seg[0] if len(t_seg) > 0 else t_seg

    return {'voltage': v_seg, 'current': i_seg, 'time': t_rel}


def _compute_q_progress(i: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """计算 q(t) 与 q_norm(t)。"""
    if len(i) < 2 or len(t) < 2:
        q = np.zeros(len(i), dtype=np.float32)
        q_norm = np.linspace(0, 1, len(i), dtype=np.float32) if len(i) > 0 else q
        return q, q_norm

    # 保证时间单调（去重并排序）
    t = np.asarray(t, dtype=np.float32)
    i = np.asarray(i, dtype=np.float32)
    t_unique, idx = np.unique(t, return_index=True)
    if len(t_unique) < 2:
        q = np.zeros(len(i), dtype=np.float32)
        q_norm = np.linspace(0, 1, len(i), dtype=np.float32) if len(i) > 0 else q
        return q, q_norm

    t = t_unique
    i = i[idx]

    dt = np.diff(t)
    dt = np.maximum(dt, 1e-6)
    i_mid = (i[:-1] + i[1:]) / 2.0
    dq = np.abs(i_mid) * dt / 3600.0  # Ah
    q = np.concatenate([[0.0], np.cumsum(dq).astype(np.float32)])

    q_end = float(q[-1])
    if q_end < EPS:
        q_norm = np.linspace(0, 1, len(q), dtype=np.float32)
    else:
        q_norm = (q / q_end).astype(np.float32)

    return q.astype(np.float32), q_norm


def _interp_on_grid(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    if len(x) < 2 or len(y) < 2:
        return np.zeros_like(x_new, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    # 确保单调递增
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x_unique, idx = np.unique(x, return_index=True)
    y = y[idx]
    if len(x_unique) < 2:
        return np.zeros_like(x_new, dtype=np.float32)
    return np.interp(x_new, x_unique, y).astype(np.float32)


def _cycle_sequence_qnorm(cycle: CycleData, num_samples: int = DEFAULT_NUM_SAMPLES) -> Tuple[np.ndarray, float, float]:
    """返回该cycle的 (v_norm, i_norm, q_grid) 以及 q_end(Ah)、charge_time(s)。"""
    seg = _extract_full_charge_curve(cycle)
    v = seg['voltage']
    i = seg['current']
    t = seg['time']
    if len(v) < 2:
        q_grid = np.linspace(0, 1, num_samples, dtype=np.float32)
        seq = np.zeros((num_samples, 3), dtype=np.float32)
        seq[:, 2] = q_grid
        return seq, 0.0, 0.0

    q, q_norm = _compute_q_progress(i, t)
    # 对齐到q_norm长度
    min_len = min(len(v), len(q_norm))
    v = np.asarray(v[:min_len], dtype=np.float32)
    i = np.asarray(i[:min_len], dtype=np.float32)
    q_norm = np.asarray(q_norm[:min_len], dtype=np.float32)

    q_grid = np.linspace(0, 1, num_samples, dtype=np.float32)
    v_i = _interp_on_grid(q_norm, v, q_grid)
    i_i = _interp_on_grid(q_norm, i, q_grid)

    # 归一化（跨协议）
    v0 = float(v_i[0])
    v1 = float(v_i[-1])
    v_norm = (v_i - v0) / (abs(v1 - v0) + EPS)

    imax = float(np.max(np.abs(i_i)))
    i_norm = i_i / (imax + EPS)

    seq = np.stack([v_norm, i_norm, q_grid], axis=-1).astype(np.float32)

    q_end = float(q[-1]) if len(q) > 0 else 0.0
    charge_time = float(t[-1]) if len(t) > 0 else 0.0

    return seq, q_end, charge_time


def _compute_baseline_sequence(battery: BatteryData, num_samples: int = DEFAULT_NUM_SAMPLES, n_base: int = BASELINE_CYCLES) -> np.ndarray:
    """计算电池前n_base个有效cycle的baseline (num_samples,2)：只包含 v_norm/i_norm，不含q_norm。"""
    seqs = []
    for cyc in battery.cycles:
        seq, _, _ = _cycle_sequence_qnorm(cyc, num_samples=num_samples)
        # 认为有效：v通道非全0
        if np.any(seq[:, 0] != 0):
            seqs.append(seq[:, :2])
        if len(seqs) >= n_base:
            break
    if len(seqs) == 0:
        return np.zeros((num_samples, 2), dtype=np.float32)
    return np.mean(np.stack(seqs, axis=0), axis=0).astype(np.float32)


def create_sequence(
    cycle: CycleData,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    v_cutoff: float = 4.2,
    battery: Optional[BatteryData] = None,
    baseline: Optional[np.ndarray] = None,
) -> np.ndarray:
    """主线 sequence：返回 delta 序列 (num_samples,3)，q_norm不扣。"""
    # battery参数用于baseline（dataset层会传battery）
    if baseline is None:
        if battery is None:
            baseline = np.zeros((num_samples, 2), dtype=np.float32)
        else:
            baseline = _compute_baseline_sequence(battery, num_samples=num_samples)

    seq, _, _ = _cycle_sequence_qnorm(cycle, num_samples=num_samples)
    out = seq.copy()
    out[:, 0] = out[:, 0] - baseline[:, 0]
    out[:, 1] = out[:, 1] - baseline[:, 1]
    # out[:,2] 保持q_norm
    return out.astype(np.float32)


def extract_features_from_cycle(
    cycle: CycleData,
    v_cutoff: float = 4.2,
    battery: Optional[BatteryData] = None,
    baseline: Optional[np.ndarray] = None,
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> np.ndarray:
    """features：从 delta 序列提取16维统计特征（跨协议）。"""
    if baseline is None:
        if battery is None:
            baseline = np.zeros((num_samples, 2), dtype=np.float32)
        else:
            baseline = _compute_baseline_sequence(battery, num_samples=num_samples)

    seq, q_end, charge_time = _cycle_sequence_qnorm(cycle, num_samples=num_samples)
    v = seq[:, 0] - baseline[:, 0]
    i = seq[:, 1] - baseline[:, 1]
    q = seq[:, 2]

    # slope（对q轴）
    v_slope = float(np.polyfit(q, v, 1)[0]) if len(q) > 1 else 0.0
    i_slope = float(np.polyfit(q, i, 1)[0]) if len(q) > 1 else 0.0

    # dv/dq, di/dq
    if len(q) > 1:
        dq = np.diff(q)
        dq = np.maximum(dq, 1e-6)
        dv_dq = np.abs(np.diff(v) / dq)
        di_dq = np.abs(np.diff(i) / dq)
        # 稳健裁剪（同一参数）
        dv_lo, dv_hi = np.percentile(dv_dq, [10, 90])
        di_lo, di_hi = np.percentile(di_dq, [10, 90])
        dv_dq = np.clip(dv_dq, dv_lo, dv_hi)
        di_dq = np.clip(di_dq, di_lo, di_hi)
        dv_dq_mean = float(np.mean(dv_dq))
        di_dq_mean = float(np.mean(di_dq))
    else:
        dv_dq_mean = 0.0
        di_dq_mean = 0.0

    return np.array([
        float(np.mean(v)), float(np.std(v)), float(np.max(v)), float(np.min(v)), float(np.max(v) - np.min(v)), v_slope,
        float(np.mean(i)), float(np.std(i)), float(np.max(i)), float(np.min(i)), float(np.max(i) - np.min(i)), i_slope,
        float(charge_time), float(q_end), dv_dq_mean, di_dq_mean,
    ], dtype=np.float32)


def create_heatmap(
    battery: BatteryData,
    window_size: int = DEFAULT_WINDOW_SIZE,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    v_cutoff: float | None = None,
    start_cycle_idx: int | None = None,
    end_cycle_idx: int | None = None,
) -> np.ndarray:
    """image：堆叠 delta 序列并在循环轴插值到window_size。"""
    baseline = _compute_baseline_sequence(battery, num_samples=num_samples)
    start = start_cycle_idx if start_cycle_idx is not None else 0
    end = end_cycle_idx if end_cycle_idx is not None else len(battery)
    available = list(range(int(max(0, start)), int(min(len(battery), end))))

    if len(available) == 0:
        return np.zeros((window_size, num_samples, 3), dtype=np.float32)

    seqs = []
    for ci in available:
        seq = create_sequence(battery.cycles[ci], num_samples=num_samples, battery=battery, baseline=baseline)
        seqs.append(seq)

    seqs = np.stack(seqs, axis=0)  # (T, W, C)

    pos = np.linspace(0, len(available) - 1, window_size)
    idx0 = np.floor(pos).astype(int)
    idx1 = np.minimum(idx0 + 1, len(available) - 1)
    alpha = (pos - idx0).astype(np.float32)

    heatmap = (1.0 - alpha)[:, None, None] * seqs[idx0] + alpha[:, None, None] * seqs[idx1]
    return heatmap.astype(np.float32)


def extract_all_features_from_battery(battery: BatteryData, v_cutoff: float | None = None) -> np.ndarray:
    baseline = _compute_baseline_sequence(battery, num_samples=DEFAULT_NUM_SAMPLES)
    feats = [extract_features_from_cycle(c, battery=battery, baseline=baseline, num_samples=DEFAULT_NUM_SAMPLES) for c in battery.cycles]
    return np.stack(feats, axis=0)
