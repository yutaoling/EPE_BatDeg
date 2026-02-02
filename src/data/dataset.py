"""
PyTorch Dataset类（主线收敛版）

支持输入类型：
1. features: 统计特征 (16维向量)
2. sequence: 单循环时序数据 (num_samples, 3)
3. image: 多循环热力图 (window_size, num_samples, 3)

注意：
- `image` 的语义固定为：从循环0到当前cycle_idx（包含当前）的完整历史区间，在循环轴上插值到固定 window_size。
- 已移除 full_image/full_sequence 等历史入口。
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple

from .battery_data import BatteryData
from .feature import (
    extract_features_from_cycle,
    create_sequence,
    create_heatmap,
    _compute_baseline_sequence,
)
from .transforms import BaseTransform


class BatteryDataset(Dataset):
    """统一数据集

    关键点：
    - `training=True & random_truncate=True` 时：每次getitem动态随机截断点
    - 输出字段始终包含：feature/soh/rul/rul_mask/label/meta
    - `label` 根据 target_type 返回主训练标签（soh 或 rul 或 soh）
    """

    def __init__(
        self,
        batteries: List[BatteryData],
        input_type: str = 'sequence',
        target_type: str = 'soh',
        window_size: int = 100,
        num_samples: int = 200,
        v_cutoff: float | None = None,
        min_cycle_for_prediction: int = 10,
        feature_transform: BaseTransform | None = None,
        label_transform: BaseTransform | None = None,
        cache: bool = False,
        training: bool = True,
        random_truncate: bool = False,
        truncate_min_ratio: float = 0.2,
        truncate_max_ratio: float = 1.0,
        random_eol_threshold: bool = False,
        eol_threshold_margin: float = 0.01,
        default_eol_threshold: float = 0.8,
        rul_invalid_value: float = -1.0,
    ):
        self.batteries = batteries
        self.input_type = input_type
        self.target_type = target_type
        self.window_size = window_size
        self.num_samples = num_samples
        self.v_cutoff = v_cutoff
        self.min_cycle = min_cycle_for_prediction
        self.feature_transform = feature_transform
        self.label_transform = label_transform
        self.cache = cache

        self.training = training
        self.random_truncate = random_truncate
        self.truncate_min_ratio = truncate_min_ratio
        self.truncate_max_ratio = truncate_max_ratio

        self.random_eol_threshold = random_eol_threshold
        self.eol_threshold_margin = eol_threshold_margin
        self.default_eol_threshold = default_eol_threshold
        self.rul_invalid_value = rul_invalid_value

        self._cache = {} if cache else None
        self._baseline_cache: Dict[int, np.ndarray] = {}
        self._battery_soh_info = self._compute_battery_soh_info()
        self.samples = self._build_samples()

    def _compute_battery_soh_info(self) -> Dict[int, Dict[str, float]]:
        info: Dict[int, Dict[str, float]] = {}
        for idx, battery in enumerate(self.batteries):
            soh_array = battery.get_soh_array()
            if len(soh_array) > 0:
                info[idx] = {
                    'min_soh': float(np.min(soh_array)),
                    'max_soh': float(np.max(soh_array)),
                }
            else:
                info[idx] = {'min_soh': 0.7, 'max_soh': 1.0}
        return info

    def _build_samples(self) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []

        if self.training and self.random_truncate:
            for battery_idx, battery in enumerate(self.batteries):
                if battery.get_soh_array().size == 0:
                    battery.compute_soh()
                if len(battery) > self.min_cycle:
                    samples.append({
                        'battery_idx': battery_idx,
                        'cell_id': battery.cell_id,
                        'dataset': battery.dataset,
                        'total_cycles': len(battery),
                    })
            return samples

        min_required = max(self.min_cycle, self.window_size) if self.input_type == 'image' else self.min_cycle

        for battery_idx, battery in enumerate(self.batteries):
            if battery.get_soh_array().size == 0:
                battery.compute_soh()

            for cycle_idx in range(min_required, len(battery)):
                cycle = battery.cycles[cycle_idx]
                if cycle.soh is None:
                    continue
                samples.append({
                    'battery_idx': battery_idx,
                    'cycle_idx': cycle_idx,
                    'cell_id': battery.cell_id,
                    'cycle_number': cycle.cycle_number,
                    'soh': float(cycle.soh),
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_truncate_cycle_idx(self, battery: BatteryData) -> int:
        total_cycles = len(battery)
        min_idx = max(self.min_cycle, int(total_cycles * self.truncate_min_ratio))
        max_idx = min(total_cycles - 1, int(total_cycles * self.truncate_max_ratio))
        if min_idx >= max_idx:
            min_idx = self.min_cycle
            max_idx = total_cycles - 1
        return int(np.random.randint(min_idx, max_idx + 1))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        battery = self.batteries[sample['battery_idx']]

        if self.training and self.random_truncate:
            cycle_idx = self._sample_truncate_cycle_idx(battery)
            cycle = battery.cycles[cycle_idx]
            current_soh = float(cycle.soh) if cycle.soh is not None else 1.0
            meta = {
                'cell_id': sample['cell_id'],
                'dataset': sample.get('dataset', battery.dataset),
                'cycle_idx': cycle_idx,
                'cycle_number': cycle.cycle_number,
                'truncate_ratio': cycle_idx / max(1, len(battery)),
            }
        else:
            cycle_idx = sample['cycle_idx']
            current_soh = float(sample['soh'])
            meta = {
                'cell_id': sample['cell_id'],
                'cycle_idx': cycle_idx,
                'cycle_number': sample['cycle_number'],
            }

        # 为了实现“前10循环baseline”，我们在 dataset 层为每块电池预先缓存 baseline（v/i 两通道），
        # 并在 __getitem__ 返回中额外带出 v_delta/i_delta/t_delta（其中 t_delta 实际为 q_norm）。
        battery_idx = sample['battery_idx']
        baseline = self._baseline_cache.get(battery_idx)
        if baseline is None:
            baseline = _compute_baseline_sequence(battery, num_samples=self.num_samples)
            self._baseline_cache[battery_idx] = baseline

        if self._cache is not None and not (self.training and self.random_truncate):
            cache_key = f"{meta['cell_id']}_{cycle_idx}_{self.input_type}"
            if cache_key in self._cache:
                feature = self._cache[cache_key]
            else:
                feature = self._extract_feature(battery, cycle_idx, baseline)
                self._cache[cache_key] = feature
        else:
            feature = self._extract_feature(battery, cycle_idx, baseline)

        use_random_thr = self.training and self.random_eol_threshold
        if use_random_thr:
            rul_label, eol_thr = self._compute_random_rul(
                battery,
                cycle_idx,
                current_soh,
                sample_battery_idx=sample['battery_idx'],
            )
        else:
            rul_label, eol_thr = self._compute_fixed_rul(
                battery,
                cycle_idx,
                self.default_eol_threshold,
            )

        rul_mask = 1.0 if rul_label != self.rul_invalid_value else 0.0

        if self.target_type == 'soh':
            label = current_soh
        elif self.target_type == 'rul':
            label = rul_label
        else:
            label = current_soh

        if self.feature_transform is not None:
            feature = self.feature_transform(feature)
        if self.label_transform is not None:
            label = self.label_transform(np.array([label]))[0]

        # 额外输出 v_delta/i_delta/t_delta（t_delta 实际为 q_norm），便于后续特征分析或模型使用
        # 统一用当前 cycle 的 delta 序列（num_samples,3）切出三通道
        if self.input_type == 'sequence':
            seq_delta = feature
        else:
            seq_delta = create_sequence(
                battery.cycles[cycle_idx],
                self.num_samples,
                self.v_cutoff or battery.charge_cutoff_voltage,
                battery=battery,
                baseline=baseline,
            )

        v_delta = seq_delta[:, 0]
        i_delta = seq_delta[:, 1]
        t_delta = seq_delta[:, 2]

        return {
            'feature': torch.tensor(feature, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32),
            'soh': torch.tensor(current_soh, dtype=torch.float32),
            'rul': torch.tensor(rul_label, dtype=torch.float32),
            'rul_mask': torch.tensor(rul_mask, dtype=torch.float32),
            'eol_threshold': torch.tensor(eol_thr, dtype=torch.float32),
            'v_delta': torch.tensor(v_delta, dtype=torch.float32),
            'i_delta': torch.tensor(i_delta, dtype=torch.float32),
            't_delta': torch.tensor(t_delta, dtype=torch.float32),
            'meta': meta,
        }

    def _extract_feature(self, battery: BatteryData, cycle_idx: int, baseline: np.ndarray) -> np.ndarray:
        v_cutoff = self.v_cutoff or battery.charge_cutoff_voltage

        if self.input_type == 'features':
            return extract_features_from_cycle(
                battery.cycles[cycle_idx],
                v_cutoff,
                battery=battery,
                baseline=baseline,
                num_samples=self.num_samples,
            )

        if self.input_type == 'sequence':
            return create_sequence(
                battery.cycles[cycle_idx],
                self.num_samples,
                v_cutoff,
                battery=battery,
                baseline=baseline,
            )

        if self.input_type == 'image':
            # 完整历史：从0到cycle_idx（包含），映射到固定window_size
            return create_heatmap(
                battery=battery,
                window_size=self.window_size,
                num_samples=self.num_samples,
                v_cutoff=v_cutoff,
                start_cycle_idx=0,
                end_cycle_idx=cycle_idx + 1,
            )

        raise ValueError(f"Unknown input_type: {self.input_type}")

    def _compute_random_rul(
        self,
        battery: BatteryData,
        cycle_idx: int,
        current_soh: float,
        sample_battery_idx: int,
    ) -> Tuple[float, float]:
        min_soh = self._battery_soh_info[sample_battery_idx]['min_soh']
        margin = self.eol_threshold_margin

        hard_min = 0.65
        hard_max = 0.9

        threshold_min = max(hard_min, min_soh + margin)
        threshold_max = min(hard_max, current_soh - margin)

        if threshold_min >= threshold_max:
            threshold = float(np.clip(threshold_min, min_soh, current_soh))
        else:
            threshold = float(np.random.uniform(threshold_min, threshold_max))

        current_cycle_number = battery.cycles[cycle_idx].cycle_number

        for future_cycle in battery.cycles[cycle_idx:]:
            if future_cycle.soh is not None and future_cycle.soh < threshold:
                rul = future_cycle.cycle_number - current_cycle_number
                return float(max(0, rul)), float(threshold)

        return float(self.rul_invalid_value), float(threshold)

    def _compute_fixed_rul(
        self,
        battery: BatteryData,
        cycle_idx: int,
        threshold: float,
    ) -> Tuple[float, float]:
        current_cycle_number = battery.cycles[cycle_idx].cycle_number

        for future_cycle in battery.cycles[cycle_idx:]:
            if future_cycle.soh is not None and future_cycle.soh < threshold:
                rul = future_cycle.cycle_number - current_cycle_number
                return float(max(0, rul)), float(threshold)

        return float(self.rul_invalid_value), float(threshold)

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def get_feature_dim(self) -> Tuple:
        if self.input_type == 'features':
            return (16,)
        if self.input_type == 'sequence':
            return (self.num_samples, 3)
        if self.input_type == 'image':
            return (self.window_size, self.num_samples, 3)
        raise ValueError(f"Unknown input_type: {self.input_type}")


class MultiDomainDataset(Dataset):
    """将多个 domain 的 BatteryDataset 视为一个拼接后的大 Dataset。"""

    def __init__(
        self,
        domain_batteries: Dict[str, List[BatteryData]],
        input_type: str = 'image',
        target_type: str = 'soh',
        window_size: int = 100,
        num_samples: int = 200,
        training: bool = True,
        **dataset_kwargs,
    ):
        self._datasets: List[BatteryDataset] = []
        self._domain_of_dataset: List[str] = []

        for domain, bats in domain_batteries.items():
            ds = BatteryDataset(
                bats,
                input_type=input_type,
                target_type=target_type,
                window_size=window_size,
                num_samples=num_samples,
                training=training,
                **dataset_kwargs,
            )
            self._datasets.append(ds)
            self._domain_of_dataset.append(domain)

        self._offsets = np.cumsum([0] + [len(ds) for ds in self._datasets]).astype(int)

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ds_idx = int(np.searchsorted(self._offsets, idx, side='right') - 1)
        local_idx = int(idx - self._offsets[ds_idx])
        out = self._datasets[ds_idx][local_idx]
        meta = dict(out.get('meta', {}))
        meta['domain'] = self._domain_of_dataset[ds_idx]
        out['meta'] = meta
        return out


def create_multi_domain_dataloaders(
    domain_batteries: Dict[str, List[BatteryData]],
    input_type: str = 'image',
    target_type: str = 'soh',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
    balance_domains: bool = True,
    **dataset_kwargs,
) -> Tuple:
    """多域联合训练的 DataLoader 构造器。"""

    from torch.utils.data import DataLoader, WeightedRandomSampler

    np.random.seed(seed)

    train_domains: Dict[str, List[BatteryData]] = {}
    val_domains: Dict[str, List[BatteryData]] = {}
    test_domains: Dict[str, List[BatteryData]] = {}

    for domain, bats in domain_batteries.items():
        if len(bats) == 0:
            continue
        idx = np.random.permutation(len(bats))
        n_train = int(len(bats) * train_ratio)
        n_val = int(len(bats) * val_ratio)
        train_domains[domain] = [bats[i] for i in idx[:n_train]]
        val_domains[domain] = [bats[i] for i in idx[n_train:n_train + n_val]]
        test_domains[domain] = [bats[i] for i in idx[n_train + n_val:]]

    train_dataset = MultiDomainDataset(
        train_domains,
        input_type=input_type,
        target_type=target_type,
        training=True,
        **dataset_kwargs,
    )
    val_dataset = MultiDomainDataset(
        val_domains,
        input_type=input_type,
        target_type=target_type,
        training=False,
        **dataset_kwargs,
    )
    test_dataset = MultiDomainDataset(
        test_domains,
        input_type=input_type,
        target_type=target_type,
        training=False,
        **dataset_kwargs,
    )

    if balance_domains and len(train_dataset) > 0:
        offsets = train_dataset._offsets
        domain_counts: Dict[str, int] = {}
        for i, dom in enumerate(train_dataset._domain_of_dataset):
            domain_counts[dom] = domain_counts.get(dom, 0) + int(offsets[i + 1] - offsets[i])
        domain_counts = {d: max(1, c) for d, c in domain_counts.items()}
        domain_weight = {d: 1.0 / c for d, c in domain_counts.items()}

        weights = np.zeros(len(train_dataset), dtype=np.float32)
        for i, dom in enumerate(train_dataset._domain_of_dataset):
            start = int(offsets[i])
            end = int(offsets[i + 1])
            weights[start:end] = domain_weight[dom]

        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights).float(),
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    domain_info = {
        'train_domains': {k: len(v) for k, v in train_domains.items()},
        'val_domains': {k: len(v) for k, v in val_domains.items()},
        'test_domains': {k: len(v) for k, v in test_domains.items()},
    }

    return train_loader, val_loader, test_loader, domain_info


def create_dataloaders(
    batteries: List[BatteryData],
    input_type: str = 'sequence',
    target_type: str = 'soh',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
    weighted_sampling: bool = True,
    **dataset_kwargs,
) -> Tuple:
    """创建训练/验证/测试数据加载器（按电池级别划分）"""

    from torch.utils.data import DataLoader, WeightedRandomSampler

    np.random.seed(seed)
    indices = np.random.permutation(len(batteries))

    n_train = int(len(batteries) * train_ratio)
    n_val = int(len(batteries) * val_ratio)

    train_batteries = [batteries[i] for i in indices[:n_train]]
    val_batteries = [batteries[i] for i in indices[n_train:n_train + n_val]]
    test_batteries = [batteries[i] for i in indices[n_train + n_val:]]

    train_dataset = BatteryDataset(
        train_batteries,
        input_type=input_type,
        target_type=target_type,
        training=True,
        **dataset_kwargs,
    )
    val_dataset = BatteryDataset(
        val_batteries,
        input_type=input_type,
        target_type=target_type,
        training=False,
        **dataset_kwargs,
    )
    test_dataset = BatteryDataset(
        test_batteries,
        input_type=input_type,
        target_type=target_type,
        training=False,
        **dataset_kwargs,
    )

    if weighted_sampling and not (train_dataset.random_truncate and train_dataset.training) and len(train_dataset) > 0:
        labels = np.array([s.get('soh', 1.0) for s in train_dataset.samples], dtype=float)
        bins = np.linspace(labels.min() - 1e-3, labels.max() + 1e-3, 21)
        bin_indices = np.clip(np.digitize(labels, bins) - 1, 0, 19)
        bin_counts = np.bincount(bin_indices, minlength=20).astype(float)
        bin_counts = np.maximum(bin_counts, 1)
        bin_weights = 1.0 / bin_counts
        weights = bin_weights[bin_indices]

        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights).float(),
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
