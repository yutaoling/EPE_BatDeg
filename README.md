# BatteryDegradation - 锂电池SOH/RUL预测框架

基于深度学习的锂电池健康状态(SOH)与剩余使用寿命(RUL)预测研究框架。

## 统一工作流

- **3通道**：v_delta / i_delta / q_norm（容量进度轴）
- **双头输出**：SOH + RUL
- **主推协议无关**：`image`（delta热力图：完整历史→固定尺寸）
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
| 视觉 | CNN2D, ViT | `image` | 多循环热力图（完整历史→固定尺寸） |

## 特征与输入（3通道：v_delta, i_delta, q_norm）

本项目当前的三类输入均基于同一条主线：
- 以归一化容量进度轴 `q_norm∈[0,1]` 对齐单循环充电曲线
- 取每块电池前10个有效循环的均值序列作为 baseline（仅 v/i）
- 对每个循环做差得到 `v_delta/i_delta`（`q_norm` 不扣除）

1. **`features`** `(16,)` - 从 delta 序列统计得到的16维特征（导数为 `dv/dq`、`di/dq`）
2. **`sequence`** `(200, 3)` - 单循环 delta 序列 `[v_delta, i_delta, q_norm]`
3. **`image`** `(100, 200, 3)` - **（主推）** delta 热力图（历史0→当前，循环轴插值到固定 window_size）

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

## RUL 标签与训练方式（主线）

本项目当前的 RUL 不是通过“SOH 轨迹外推器”单独计算，而是在 `BatteryDataset` 中按训练策略动态生成：

- 训练时可启用 `random_truncate=True`：随机选择预测点（cycle_idx），使用从循环0到当前循环的历史输入。
- 训练时可启用 `random_eol_threshold=True`：对每个样本随机采样 RUL 阈值（SOH threshold），并计算从当前循环到首次低于该阈值的剩余循环数作为 RUL 标签。
- 阈值约束：阈值被限制在 `[0.65, 0.9]`，同时不低于电池最小 SOH、不高于当前 SOH（带 margin）。

建议通过 `src.experiments.ExperimentConfig` 直接训练：
- `target_type='rul'`（只做RUL）或 `target_type='both'`（SOH+RUL 双头）
- `input_type='image'`（完整历史热力图，输出尺寸恒定）

## 依赖

- Python 3.11
- PyTorch 2.5+
- NumPy, Pandas, Matplotlib
- 详见 `environment.yml`
