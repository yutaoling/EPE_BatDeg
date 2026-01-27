# 模型模块

from .base import BaseModel
from .mlp import MLP
from .cnn import CNN1D, CNN2D
from .lstm import LSTM, BiLSTM
from .transformer import TransformerModel

# 可选模型
try:
    from .pinn import PINN
except ImportError:
    PINN = None

try:
    from .vit import ViT, SimpleViT
except ImportError:
    ViT = None
    SimpleViT = None

try:
    from .fno import FNO
except ImportError:
    FNO = None


__all__ = [
    'BaseModel',
    'MLP',
    'CNN1D', 'CNN2D',
    'LSTM', 'BiLSTM',
    'TransformerModel',
    'PINN',
    'ViT',
    'SimpleViT',
    'FNO',
]


# 模型注册表（简化版）
MODELS = {
    'mlp': MLP,
    'cnn1d': CNN1D,
    'cnn2d': CNN2D,
    'lstm': LSTM,
    'bilstm': BiLSTM,
    'transformer': TransformerModel,
}

if PINN is not None:
    MODELS['pinn'] = PINN
if ViT is not None:
    MODELS['vit'] = ViT
if SimpleViT is not None:
    MODELS['simplevit'] = SimpleViT
if FNO is not None:
    MODELS['fno'] = FNO


def get_model(name: str, **kwargs):
    """根据名称获取模型"""
    name = name.lower()
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODELS.keys())}")
    return MODELS[name](**kwargs)
