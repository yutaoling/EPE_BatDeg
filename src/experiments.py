"""
实验管理模块

提供多数据集、多模型的实验配置、运行和结果管理。

本版本适配：
- 双头输出（SOH + RUL）
- 统一3通道（v_delta / i_delta / q_norm）
- 主推协议无关的 image（完整历史热力图：任意循环数插值到固定尺寸）
- 训练采用随机截断 + 随机EOL阈值RUL（由 BatteryDataset 提供）
- 不使用 early stopping / scheduler
"""

import numpy as np
import torch
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Any
from datetime import datetime

from .data import (
    create_dataloaders,
    load_processed_batteries,
)
from .models import get_model, MODELS


@dataclass
class ExperimentConfig:
    """实验配置"""

    name: str = "experiment"
    description: str = ""

    # 数据
    datasets: List[str] = field(default_factory=lambda: ['MATR'])
    input_type: str = 'image'  # 主推: image（统一完整历史+固定尺寸插值）
    num_samples: int = 200
    window_size: int = 100

    # 训练
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    batch_size: int = 16

    # 随机截断（训练策略）
    random_truncate: bool = True
    truncate_min_ratio: float = 0.2
    truncate_max_ratio: float = 1.0

    # RUL阈值策略
    random_eol_threshold: bool = True
    eol_threshold_margin: float = 0.01
    default_eol_threshold: float = 0.8

    # 模型
    models: List[str] = field(default_factory=lambda: ['cnn2d'])
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    target_type: str = 'both'  # 'soh' | 'rul' | 'both'
    soh_loss_weight: float = 1.0
    rul_loss_weight: float = 0.1

    # 设备
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 随机种子
    seed: int = 42

    # 输出
    output_dir: str = 'results'

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'ExperimentConfig':
        return cls(**d)

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


@dataclass
class ExperimentResult:
    model_name: str
    dataset_name: str
    input_type: str

    train_metrics: Dict[str, float] = field(default_factory=dict)
    val_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)

    train_history: Dict[str, list] = field(default_factory=dict)
    train_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _get_compatible_models(input_type: str) -> List[str]:
    input_type_map = {
        'features': ['mlp', 'pinn'],
        'sequence': ['cnn1d', 'lstm', 'bilstm', 'transformer', 'fno'],
        'image': ['cnn2d', 'vit', 'simplevit'],
    }
    return [m for m in input_type_map.get(input_type, []) if m in MODELS]


def prepare_data(config: ExperimentConfig, processed_dir: str = 'data/processed') -> Dict[str, Tuple]:
    """按数据集准备 train/val/test loaders"""
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    processed_dir = Path(processed_dir)
    data_loaders: Dict[str, Tuple] = {}

    for dataset_name in config.datasets:
        dataset_dir = processed_dir / dataset_name
        if not dataset_dir.exists():
            print(f"警告: 数据集目录不存在 {dataset_dir}")
            continue

        print(f"\n加载数据集: {dataset_name}")
        batteries = load_processed_batteries(str(dataset_dir))
        if len(batteries) == 0:
            print(f"警告: {dataset_name} 没有有效数据")
            continue

        # image 需要至少window_size个循环
        min_cycles = max(10, config.window_size) if config.input_type in ['image'] else 10
        batteries = [b for b in batteries if len(b) >= min_cycles]
        print(f"有效电池数: {len(batteries)}")
        if len(batteries) == 0:
            continue

        train_loader, val_loader, test_loader = create_dataloaders(
            batteries,
            input_type=config.input_type,
            target_type=config.target_type,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            batch_size=config.batch_size,
            num_samples=config.num_samples,
            window_size=config.window_size,
            seed=config.seed,
            # 训练策略
            random_truncate=config.random_truncate,
            truncate_min_ratio=config.truncate_min_ratio,
            truncate_max_ratio=config.truncate_max_ratio,
            random_eol_threshold=config.random_eol_threshold,
            eol_threshold_margin=config.eol_threshold_margin,
            default_eol_threshold=config.default_eol_threshold,
        )

        data_loaders[dataset_name] = (train_loader, val_loader, test_loader)
        print(f"训练/验证/测试样本: {len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")

    return data_loaders


def create_model(
    model_name: str,
    input_type: str,
    in_channels: int = 3,
    num_samples: int = 200,
    window_size: int = 100,
):
    """根据名称和输入类型创建模型（双头模型已统一）"""
    name = model_name.lower()

    if input_type == 'features':
        kwargs = {'input_dim': 16}
    elif input_type == 'sequence':
        kwargs = {'in_channels': in_channels}
    elif input_type == 'image':
        if name in ['cnn2d', 'resnet2d']:
            kwargs = {'in_channels': in_channels}
        else:
            kwargs = {
                'in_channels': in_channels,
                'img_height': window_size,
                'img_width': num_samples,
            }
            if name in ['vit', 'simplevit']:
                kwargs.update({
                    'patch_height': max(1, window_size // 10),
                    'patch_width': max(1, num_samples // 10),
                })
    else:
        raise ValueError(f"input_type={input_type} 不支持或不主推")

    return get_model(name, **kwargs)


def run_single_experiment(
    model_name: str,
    train_loader,
    val_loader,
    test_loader,
    config: ExperimentConfig,
    dataset_name: str,
) -> ExperimentResult:
    print(f"\n{'='*50}")
    print(f"训练模型: {model_name.upper()}")
    print(f"数据集: {dataset_name}")
    print(f"输入类型: {config.input_type}")
    print(f"目标: {config.target_type}")
    print(f"device: {config.device}")
    print(f"{'='*50}")

    model = create_model(
        model_name,
        config.input_type,
        in_channels=3,
        num_samples=config.num_samples,
        window_size=config.window_size,
    ).to(config.device)

    # 明确打印实际device（避免误以为在CPU）
    try:
        print(f"model param device: {next(model.parameters()).device}")
    except Exception:
        pass

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    start_time = time.time()

    save_path = Path(config.output_dir) / 'checkpoints' / f"{model_name}_{dataset_name}_{config.input_type}.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    history = model.fit(
        train_loader,
        val_loader,
        epochs=config.epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        save_path=str(save_path),
        verbose=True,
        target_type=config.target_type,
        soh_loss_weight=config.soh_loss_weight,
        rul_loss_weight=config.rul_loss_weight,
    )

    train_time = time.time() - start_time
    print(f"训练时间: {train_time:.1f}s")

    train_metrics = model.evaluate(train_loader)
    val_metrics = model.evaluate(val_loader)
    test_metrics = model.evaluate(test_loader)

    print("\n测试集结果:")
    print(f"  SOH_RMSE: {test_metrics.get('SOH_RMSE', 0):.4f}")
    print(f"  SOH_MAE:  {test_metrics.get('SOH_MAE', 0):.4f}")
    print(f"  RUL_RMSE: {test_metrics.get('RUL_RMSE', 0):.4f}")
    print(f"  RUL_MAE:  {test_metrics.get('RUL_MAE', 0):.4f}")

    return ExperimentResult(
        model_name=model_name,
        dataset_name=dataset_name,
        input_type=config.input_type,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        train_history=history,
        train_time=train_time,
        extra={'total_params': total_params, 'save_path': str(save_path)},
    )


def run_experiments(config: ExperimentConfig, processed_dir: str = 'data/processed') -> List[ExperimentResult]:
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    data_loaders = prepare_data(config, processed_dir)
    if len(data_loaders) == 0:
        print("错误: 没有有效的数据集")
        return []

    compatible = _get_compatible_models(config.input_type)
    models_to_run = [m for m in config.models if m.lower() in compatible]
    if len(models_to_run) == 0:
        print(f"错误: 没有与 {config.input_type} 输入兼容的模型")
        print(f"兼容模型: {compatible}")
        return []

    print(f"\n将运行的模型: {models_to_run}")

    results: List[ExperimentResult] = []

    for dataset_name, (train_loader, val_loader, test_loader) in data_loaders.items():
        for model_name in models_to_run:
            try:
                results.append(
                    run_single_experiment(
                        model_name,
                        train_loader,
                        val_loader,
                        test_loader,
                        config,
                        dataset_name,
                    )
                )
            except Exception as e:
                print(f"错误: {model_name} on {dataset_name} 失败: {e}")
                import traceback
                traceback.print_exc()

    return results


def save_results(results: List[ExperimentResult], config: ExperimentConfig, output_dir: str = None):
    output_dir = Path(output_dir or config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    config.save(str(output_dir / f'config_{timestamp}.json'))

    results_data = [r.to_dict() for r in results]
    with open(output_dir / f'results_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n结果已保存到: {output_dir}")


def print_results_table(results: List[ExperimentResult]):
    print("\n" + "=" * 90)
    print("实验结果汇总")
    print("=" * 90)
    print(f"{'Model':<12} {'Dataset':<10} {'SOH_RMSE':<10} {'RUL_RMSE':<10} {'Time(s)':<10}")
    print("-" * 90)

    for r in results:
        print(
            f"{r.model_name:<12} {r.dataset_name:<10} "
            f"{r.test_metrics.get('SOH_RMSE', 0):<10.4f} "
            f"{r.test_metrics.get('RUL_RMSE', 0):<10.4f} "
            f"{r.train_time:<10.1f}"
        )

    print("=" * 90)


def quick_experiment(
    datasets: List[str] = ['MATR'],
    models: List[str] = ['cnn2d'],
    input_type: str = 'image',
    epochs: int = 10,
    processed_dir: str = 'data/processed',
    output_dir: str = 'results',
) -> List[ExperimentResult]:
    config = ExperimentConfig(
        name=f"quick_{input_type}",
        datasets=datasets,
        models=models,
        input_type=input_type,
        epochs=epochs,
        output_dir=output_dir,
    )

    results = run_experiments(config, processed_dir)
    if len(results) > 0:
        print_results_table(results)
        save_results(results, config, output_dir)

    return results
