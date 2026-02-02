# 数据模块

# 先处理pickle兼容性
import sys
if 'data' not in sys.modules:
    class _CompatModule:
        pass
    data_mod = _CompatModule()
    sys.modules['data'] = data_mod

from .battery_data import BatteryData, CycleData

# 注册模块别名以支持旧版pickle文件
sys.modules['data.battery_data'] = sys.modules['src.data.battery_data']

# 延迟导入需要torch的模块
try:
    from .dataset import (
        BatteryDataset,
        create_dataloaders,
        MultiDomainDataset,
        create_multi_domain_dataloaders,
    )
    from .transforms import (
        ZScoreTransform,
        LogScaleTransform,
        MinMaxTransform,
        SequentialTransform,
        ProtocolInvariantTransform,
        ProtocolAwareTransform,
    )
    _HAS_TORCH = True
except ImportError as e:
    import warnings
    warnings.warn(f"PyTorch modules not available: {e}")
    _HAS_TORCH = False
    BatteryDataset = None
    create_dataloaders = None
    MultiDomainDataset = None
    create_multi_domain_dataloaders = None
    ZScoreTransform = None
    LogScaleTransform = None
    MinMaxTransform = None
    SequentialTransform = None
    ProtocolInvariantTransform = None
    ProtocolAwareTransform = None
except Exception as e:
    import warnings
    warnings.warn(f"Error loading PyTorch modules: {e}")
    _HAS_TORCH = False
    BatteryDataset = None
    create_dataloaders = None
    MultiDomainDataset = None
    create_multi_domain_dataloaders = None
    ZScoreTransform = None
    LogScaleTransform = None
    MinMaxTransform = None
    SequentialTransform = None
    ProtocolInvariantTransform = None
    ProtocolAwareTransform = None

# 特征提取模块（主线）
from .feature import (
    FEATURE_NAMES,
    DEFAULT_CHANNELS,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_WINDOW_SIZE,
    extract_features_from_cycle,
    extract_all_features_from_battery,
    create_sequence,
    create_heatmap,
)

# 预处理模块（主线）
from .preprocessors import (
    preprocess_dataset,
    preprocess_all,
    preprocess_matr,
    preprocess_calce,
    preprocess_hust,
    preprocess_rwth,
    preprocess_xjtu,
    preprocess_tju,
    preprocess_nasa,
    load_processed_batteries,
    load_all_processed,
    PREPROCESSORS,
)

__all__ = [
    'BatteryData',
    'CycleData',
    'BatteryDataset',
    'create_dataloaders',
    'MultiDomainDataset',
    'create_multi_domain_dataloaders',
    'ZScoreTransform',
    'LogScaleTransform',
    'MinMaxTransform',
    'SequentialTransform',
    'ProtocolInvariantTransform',
    'ProtocolAwareTransform',
    'FEATURE_NAMES',
    'DEFAULT_CHANNELS',
    'DEFAULT_NUM_SAMPLES',
    'DEFAULT_WINDOW_SIZE',
    'extract_features_from_cycle',
    'extract_all_features_from_battery',
    'create_sequence',
    'create_heatmap',
    'preprocess_dataset',
    'preprocess_all',
    'preprocess_matr',
    'preprocess_calce',
    'preprocess_hust',
    'preprocess_rwth',
    'preprocess_xjtu',
    'preprocess_tju',
    'preprocess_nasa',
    'load_processed_batteries',
    'load_all_processed',
    'PREPROCESSORS',
]
