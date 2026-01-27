"""
训练工具函数
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader

from .data import BatteryData, BatteryDataset
from .models.base import BaseModel


def train_model(
    model: BaseModel,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = 'cuda',
    save_path: str = None,
    verbose: bool = True,
) -> Dict[str, list]:
    """
    训练模型的便捷函数
    
    Args:
        model: 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
        lr: 学习率
        device: 设备
        save_path: 保存路径
        verbose: 是否打印训练过程
    
    Returns:
        训练历史
    """
    model = model.to(device)
    history = model.fit(
        train_loader, val_loader,
        epochs=epochs, lr=lr,
        save_path=save_path, verbose=verbose,
    )
    return history


def cross_validate(
    model_class,
    model_kwargs: dict,
    batteries: List[BatteryData],
    n_folds: int = 5,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = 'cuda',
    input_type: str = 'sequence',
    batch_size: int = 32,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    K折交叉验证
    
    Args:
        model_class: 模型类
        model_kwargs: 模型参数
        batteries: 电池数据列表
        n_folds: 折数
        epochs: 每折训练轮数
        lr: 学习率
        device: 设备
        input_type: 输入类型
        batch_size: 批大小
        seed: 随机种子
        verbose: 是否打印
    
    Returns:
        各折的评估指标
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(batteries))
    fold_size = len(batteries) // n_folds
    
    results = {'RMSE': [], 'MAE': [], 'MAPE': []}
    
    for fold in range(n_folds):
        if verbose:
            print(f"\n===== Fold {fold + 1}/{n_folds} =====")
        
        # 划分数据
        val_start = fold * fold_size
        val_end = val_start + fold_size
        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
        
        train_batteries = [batteries[i] for i in train_indices]
        val_batteries = [batteries[i] for i in val_indices]
        
        # 创建数据集
        train_dataset = BatteryDataset(train_batteries, input_type=input_type)
        val_dataset = BatteryDataset(val_batteries, input_type=input_type)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 创建模型
        model = model_class(**model_kwargs).to(device)
        
        # 训练
        model.fit(train_loader, val_loader, epochs=epochs, lr=lr, verbose=verbose)
        
        # 评估
        metrics = model.evaluate(val_loader)
        
        for key in results:
            results[key].append(metrics[key])
        
        if verbose:
            print(f"Fold {fold + 1} - RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}")
    
    # 转换为numpy数组
    for key in results:
        results[key] = np.array(results[key])
    
    if verbose:
        print(f"\n===== Cross-Validation Results =====")
        for key in results:
            print(f"{key}: {results[key].mean():.4f} ± {results[key].std():.4f}")
    
    return results


def grid_search(
    model_class,
    param_grid: dict,
    batteries: List[BatteryData],
    train_ratio: float = 0.8,
    epochs: int = 50,
    device: str = 'cuda',
    input_type: str = 'sequence',
    batch_size: int = 32,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[dict, float]:
    """
    网格搜索超参数
    
    Args:
        model_class: 模型类
        param_grid: 参数网格 {'lr': [1e-3, 1e-4], 'hidden_size': [32, 64]}
        batteries: 电池数据列表
        train_ratio: 训练集比例
        epochs: 每次训练轮数
        device: 设备
        input_type: 输入类型
        batch_size: 批大小
        seed: 随机种子
        verbose: 是否打印
    
    Returns:
        最佳参数和最佳分数
    """
    from itertools import product
    
    # 划分数据
    np.random.seed(seed)
    indices = np.random.permutation(len(batteries))
    n_train = int(len(batteries) * train_ratio)
    
    train_batteries = [batteries[i] for i in indices[:n_train]]
    val_batteries = [batteries[i] for i in indices[n_train:]]
    
    train_dataset = BatteryDataset(train_batteries, input_type=input_type)
    val_dataset = BatteryDataset(val_batteries, input_type=input_type)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 生成所有参数组合
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))
    
    best_score = float('inf')
    best_params = None
    
    for combo in combinations:
        params = dict(zip(keys, combo))
        
        # 分离模型参数和训练参数
        model_params = {k: v for k, v in params.items() if k not in ['lr', 'weight_decay']}
        lr = params.get('lr', 1e-3)
        
        if verbose:
            print(f"\nTrying: {params}")
        
        # 创建和训练模型
        model = model_class(**model_params).to(device)
        model.fit(train_loader, val_loader, epochs=epochs, lr=lr, verbose=False)
        
        # 评估
        metrics = model.evaluate(val_loader)
        score = metrics['RMSE']
        
        if verbose:
            print(f"RMSE: {score:.4f}")
        
        if score < best_score:
            best_score = score
            best_params = params
    
    if verbose:
        print(f"\n===== Best Parameters =====")
        print(f"Params: {best_params}")
        print(f"RMSE: {best_score:.4f}")
    
    return best_params, best_score


def ensemble_predict(
    models: List[BaseModel],
    loader: DataLoader,
    weights: List[float] = None,
) -> np.ndarray:
    """
    模型集成预测
    
    Args:
        models: 模型列表
        loader: 数据加载器
        weights: 模型权重（默认等权重）
    
    Returns:
        集成预测结果
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    predictions = []
    for model, weight in zip(models, weights):
        pred = model.predict(loader)
        predictions.append(pred * weight)
    
    return np.sum(predictions, axis=0)
