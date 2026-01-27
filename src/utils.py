"""
工具函数模块

包含通用工具函数、绘图配置、模型管理等
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime


def setup_chinese_font():
    """
    配置matplotlib支持中文显示
    
    自动检测系统可用的中文字体并配置
    """
    import matplotlib.font_manager as fm
    
    # Windows常用中文字体（按优先级排序）
    chinese_fonts = [
        'Microsoft YaHei',      # 微软雅黑
        'SimHei',               # 黑体
        'SimSun',               # 宋体
        'KaiTi',                # 楷体
        'FangSong',             # 仿宋
        'Arial Unicode MS',
        'STSong',               # 华文宋体
        'STHeiti',              # 华文黑体
    ]
    
    # 获取系统已安装的字体
    available_fonts = set([f.name for f in fm.fontManager.ttflist])
    
    # 找到第一个可用的中文字体
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    if selected_font:
        plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print(f"Matplotlib font set to: {selected_font}")
    else:
        warnings.warn("No Chinese font found. Chinese characters may not display correctly.")
    
    return selected_font


def setup_plot_style(style='default', chinese=True):
    """
    配置绘图风格
    
    Args:
        style: 风格名称 ('default', 'paper', 'presentation')
        chinese: 是否启用中文支持
    """
    if chinese:
        setup_chinese_font()
    
    if style == 'paper':
        # 论文风格：高清、紧凑
        plt.rcParams.update({
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.figsize': (6, 4),
        })
    elif style == 'presentation':
        # 演示风格：大字体
        plt.rcParams.update({
            'figure.dpi': 100,
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.figsize': (10, 6),
        })
    else:
        # 默认风格
        plt.rcParams.update({
            'figure.dpi': 100,
            'font.size': 12,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'figure.figsize': (8, 5),
        })
    
    # 通用设置
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


# 便捷函数：初始化绘图环境
def init_plotting(chinese=True, style='default'):
    """
    初始化绘图环境（推荐在notebook开头调用）
    
    Usage:
        from src.utils import init_plotting
        init_plotting()
    """
    setup_plot_style(style=style, chinese=chinese)


# ============== 其他工具函数 ==============

def set_seed(seed=42):
    """设置随机种子以确保可重复性"""
    import random
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    """获取可用的计算设备"""
    import torch
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")
    return device


def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def print_model_summary(model, input_shape=None):
    """
    打印模型摘要
    
    Args:
        model: PyTorch模型
        input_shape: 输入形状（可选）
    """
    import torch
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("="*50)
    print(f"Model: {model.__class__.__name__}")
    print("="*50)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Input type:           {getattr(model, 'INPUT_TYPE', 'unknown')}")
    
    if input_shape:
        print(f"Input shape:          {input_shape}")
        try:
            x = torch.randn(1, *input_shape)
            with torch.no_grad():
                y = model(x)
            print(f"Output shape:         {tuple(y.shape)}")
        except Exception as e:
            print(f"Output shape:         (error: {e})")
    
    print("="*50)


# ============== 模型管理 ==============

@dataclass
class ModelMetadata:
    """模型元数据"""
    model_name: str
    dataset_name: str
    input_type: str
    epochs: int
    train_rmse: float
    val_rmse: float
    test_rmse: float
    train_time: float
    created_at: str
    extra_info: Dict[str, Any] = None
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'ModelMetadata':
        return cls(**d)


class ModelManager:
    """
    模型管理器 - 处理模型的保存、加载和缓存
    
    功能:
    - 自动根据模型名称和数据集生成唯一标识
    - 避免重复训练已存在的模型
    - 记录训练元数据（指标、时间等）
    
    Usage:
        manager = ModelManager('checkpoints')
        
        # 检查模型是否存在
        if manager.exists('lstm', 'MATR', 'sequence'):
            model = manager.load('lstm', 'MATR', 'sequence')
        else:
            # 训练模型...
            manager.save(model, 'lstm', 'MATR', 'sequence', metadata)
        
        # 列出所有模型
        manager.list_models()
    """
    
    def __init__(self, checkpoint_dir: str = 'checkpoints'):
        """
        初始化模型管理器
        
        Args:
            checkpoint_dir: 检查点目录
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.checkpoint_dir / 'metadata.json'
        self._load_metadata_index()
    
    def _load_metadata_index(self):
        """加载元数据索引"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self._index = json.load(f)
        else:
            self._index = {}
    
    def _save_metadata_index(self):
        """保存元数据索引"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self._index, f, indent=2, ensure_ascii=False)
    
    def _get_model_key(
        self, 
        model_name: str, 
        dataset_name: str, 
        input_type: str,
        extra_suffix: str = None,
    ) -> str:
        """生成模型唯一标识"""
        key = f"{model_name}_{dataset_name}_{input_type}"
        if extra_suffix:
            key = f"{key}_{extra_suffix}"
        return key.lower()
    
    def _get_model_path(self, key: str) -> Path:
        """获取模型文件路径"""
        return self.checkpoint_dir / f"{key}.pt"
    
    def exists(
        self, 
        model_name: str, 
        dataset_name: str, 
        input_type: str,
        extra_suffix: str = None,
    ) -> bool:
        """
        检查模型是否存在
        
        Args:
            model_name: 模型名称 (如 'lstm', 'cnn1d')
            dataset_name: 数据集名称 (如 'MATR', 'CALCE')
            input_type: 输入类型 (如 'sequence', 'image')
            extra_suffix: 额外后缀（用于区分不同配置）
        
        Returns:
            是否存在
        """
        key = self._get_model_key(model_name, dataset_name, input_type, extra_suffix)
        return key in self._index and self._get_model_path(key).exists()
    
    def save(
        self,
        model,
        model_name: str,
        dataset_name: str,
        input_type: str,
        metadata: ModelMetadata = None,
        extra_suffix: str = None,
    ) -> str:
        """
        保存模型
        
        Args:
            model: PyTorch模型
            model_name: 模型名称
            dataset_name: 数据集名称
            input_type: 输入类型
            metadata: 模型元数据
            extra_suffix: 额外后缀
        
        Returns:
            保存的key
        """
        import torch
        
        key = self._get_model_key(model_name, dataset_name, input_type, extra_suffix)
        path = self._get_model_path(key)
        
        # 保存模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'model_name': model_name,
            'dataset_name': dataset_name,
            'input_type': input_type,
        }, path)
        
        # 更新元数据索引
        if metadata is None:
            metadata = ModelMetadata(
                model_name=model_name,
                dataset_name=dataset_name,
                input_type=input_type,
                epochs=0,
                train_rmse=0,
                val_rmse=0,
                test_rmse=0,
                train_time=0,
                created_at=datetime.now().isoformat(),
            )
        
        self._index[key] = metadata.to_dict()
        self._save_metadata_index()
        
        print(f"模型已保存: {path}")
        return key
    
    def load(
        self,
        model_name: str,
        dataset_name: str,
        input_type: str,
        extra_suffix: str = None,
        device: str = 'cpu',
    ):
        """
        加载模型
        
        Args:
            model_name: 模型名称
            dataset_name: 数据集名称
            input_type: 输入类型
            extra_suffix: 额外后缀
            device: 目标设备
        
        Returns:
            加载的模型
        """
        import torch
        from .models import get_model
        
        key = self._get_model_key(model_name, dataset_name, input_type, extra_suffix)
        path = self._get_model_path(key)
        
        if not path.exists():
            raise FileNotFoundError(f"模型不存在: {key}")
        
        # 加载检查点
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        
        # 创建模型实例
        # 根据input_type确定参数
        if input_type == 'features':
            kwargs = {'input_dim': 16}
        elif input_type in ['sequence', 'full_sequence']:
            kwargs = {'in_channels': 4}
        elif input_type in ['image', 'full_image']:
            kwargs = {'in_channels': 4}
        else:
            kwargs = {}
        
        model = get_model(model_name, **kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        print(f"模型已加载: {key}")
        return model
    
    def get_metadata(
        self,
        model_name: str,
        dataset_name: str,
        input_type: str,
        extra_suffix: str = None,
    ) -> Optional[ModelMetadata]:
        """获取模型元数据"""
        key = self._get_model_key(model_name, dataset_name, input_type, extra_suffix)
        if key in self._index:
            return ModelMetadata.from_dict(self._index[key])
        return None
    
    def list_models(self, dataset_name: str = None, model_name: str = None) -> List[Dict]:
        """
        列出保存的模型
        
        Args:
            dataset_name: 过滤数据集
            model_name: 过滤模型类型
        
        Returns:
            模型信息列表
        """
        results = []
        for key, meta in self._index.items():
            if dataset_name and meta.get('dataset_name') != dataset_name:
                continue
            if model_name and meta.get('model_name') != model_name:
                continue
            results.append({
                'key': key,
                'path': str(self._get_model_path(key)),
                **meta
            })
        return results
    
    def print_models(self, dataset_name: str = None):
        """打印模型列表"""
        models = self.list_models(dataset_name=dataset_name)
        
        if not models:
            print("没有保存的模型")
            return
        
        print(f"\n{'='*80}")
        print(f"保存的模型 ({len(models)} 个)")
        print(f"{'='*80}")
        print(f"{'Key':<35} {'Test RMSE':<12} {'Created':<20}")
        print(f"{'-'*80}")
        
        for m in models:
            key = m['key']
            rmse = m.get('test_rmse', 0)
            created = m.get('created_at', '')[:19]
            print(f"{key:<35} {rmse:<12.6f} {created:<20}")
        
        print(f"{'='*80}")
    
    def delete(
        self,
        model_name: str,
        dataset_name: str,
        input_type: str,
        extra_suffix: str = None,
    ):
        """删除模型"""
        key = self._get_model_key(model_name, dataset_name, input_type, extra_suffix)
        path = self._get_model_path(key)
        
        if path.exists():
            path.unlink()
        
        if key in self._index:
            del self._index[key]
            self._save_metadata_index()
        
        print(f"模型已删除: {key}")
    
    def get_or_train(
        self,
        model_name: str,
        dataset_name: str,
        input_type: str,
        train_fn,
        extra_suffix: str = None,
        device: str = 'cuda',
        force_retrain: bool = False,
    ):
        """
        获取模型，如果不存在则训练
        
        Args:
            model_name: 模型名称
            dataset_name: 数据集名称
            input_type: 输入类型
            train_fn: 训练函数，签名: fn() -> (model, metadata)
            extra_suffix: 额外后缀
            device: 目标设备
            force_retrain: 强制重新训练
        
        Returns:
            (model, metadata, is_cached)
        """
        if not force_retrain and self.exists(model_name, dataset_name, input_type, extra_suffix):
            print(f"从缓存加载模型: {model_name}_{dataset_name}_{input_type}")
            model = self.load(model_name, dataset_name, input_type, extra_suffix, device)
            metadata = self.get_metadata(model_name, dataset_name, input_type, extra_suffix)
            return model, metadata, True
        
        print(f"训练新模型: {model_name}_{dataset_name}_{input_type}")
        model, metadata = train_fn()
        self.save(model, model_name, dataset_name, input_type, metadata, extra_suffix)
        return model, metadata, False


# 全局模型管理器实例
_default_manager = None

def get_model_manager(checkpoint_dir: str = 'checkpoints') -> ModelManager:
    """获取全局模型管理器"""
    global _default_manager
    if _default_manager is None or str(_default_manager.checkpoint_dir) != checkpoint_dir:
        _default_manager = ModelManager(checkpoint_dir)
    return _default_manager
