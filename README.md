# BatteryDegradation - 锂电池SOH/RUL预测框架

基于深度学习的锂电池健康状态(SOH)与剩余使用寿命(RUL)预测研究框架。

## 统一工作流

- **3通道**：voltage / current / time
- **双头输出**：SOH + RUL
- **主推协议无关**：`full_image`（完整充电曲线热力图）
- **训练策略**：随机截断 + 随机EOL阈值（由 `BatteryDataset` 提供）
- **不使用 early stopping / scheduler**

## 项目结构

```
BatteryDegradation/
├── data/
│   ├── raw/           # 原始数据 (7个数据集)
│   └── processed/     # 预处理后的.pkl文件
├── notebooks/         # 研究流程 (按顺序执行)
│   ├── 00_data_preprocessing.ipynb
│   ├── 01_feature_extraction.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_cross_domain_transfer.ipynb
├── src/
│   ├── data/          # 数据处理模块
│   ├── models/        # 模型定义
│   ├── experiments.py # 实验管理
│   ├── evaluate.py    # 评估工具
│   └── utils.py       # 工具函数
├── results/           # 实验结果
└── checkpoints/       # 模型检查点
```

## 支持的数据集

| 数据集 | 化学体系 | 电池数 | 截止电压 |
|--------|---------|--------|---------|
| MATR   | LFP     | 180    | 3.6V    |
| HUST   | LFP     | 77     | 3.6V    |
| XJTU   | LFP     | 56     | 3.6V    |
| CALCE  | LCO     | 11     | 4.2V    |
| TJU    | NCA/NCM | 130    | 4.2V    |
| RWTH   | NMC     | 48     | 3.9V    |
| NASA   | Li-ion  | 24     | 4.2V    |

## 支持的模型

| 类型 | 模型 | 输入类型 | 说明 |
|------|------|----------|------|
| 统计特征 | MLP, PINN | `features` | 16维统计特征 |
| 时序 | LSTM, CNN1D, Transformer, FNO | `sequence` | 单循环时序 |
| 视觉 | CNN2D, ViT | `image`, `full_image` | 多循环热力图 |

## 特征提取方法（3通道：V, I, time）

1. **`features`** `(16,)` - 16维协议无关统计特征
2. **`sequence`** `(200, 3)` - 单循环充电曲线（末端窗口）
3. **`image`** `(100, 200, 3)` - 多循环充电曲线热力图（末端窗口）
4. **`full_image`** `(100, 200, 3)` - **（主推）** 完整充电曲线热力图

## 快速开始

```bash
# 1. 创建环境
conda env create -f environment.yml
conda activate BatDeg

# 2. 按顺序运行notebooks
jupyter lab notebooks/
```

## 研究流程

```
00_data_preprocessing.ipynb  # (必须) 预处理原始数据
    ↓
01_feature_extraction.ipynb  # (可选) 理解特征提取方法
    ↓
02_model_training.ipynb      # (核心) 单数据集训练与评估
    ↓
03_cross_domain_transfer.ipynb # (核心) 跨数据集训练与评估
```

## RUL预测（基于SOH轨迹外推）

```python
from src.evaluate import RULPredictor, plot_rul_prediction

predictor = RULPredictor(threshold=0.8)
result = predictor.predict_from_soh_array(
    soh_array=battery.get_soh_array(),
    current_cycle=500
)
print(f"预测RUL: {result.predicted_rul} 循环")
```

## 依赖

- Python 3.11
- PyTorch 2.5+
- NumPy, Pandas, Matplotlib
- 详见 `environment.yml`
