"""
评估工具函数
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from scipy.optimize import curve_fit
import warnings


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
    
    Returns:
        指标字典
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # MAPE
    mask = y_true != 0
    if mask.any():
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float('inf')
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Max Error
    max_error = np.max(np.abs(y_true - y_pred))
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'MaxError': max_error,
    }


def print_metrics(metrics: Dict[str, float], name: str = "Model"):
    """打印评估指标"""
    print(f"\n===== {name} Evaluation =====")
    print(f"RMSE:      {metrics['RMSE']:.6f}")
    print(f"MAE:       {metrics['MAE']:.6f}")
    print(f"MAPE:      {metrics['MAPE']:.2f}%")
    print(f"R²:        {metrics['R2']:.4f}")
    print(f"Max Error: {metrics['MaxError']:.6f}")


def compare_models(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
) -> None:
    """
    比较多个模型的结果
    
    Args:
        results: {model_name: {metric: value}}
        metrics: 要比较的指标列表
    """
    if metrics is None:
        metrics = ['RMSE', 'MAE', 'MAPE']
    
    print("\n===== Model Comparison =====")
    print(f"{'Model':<15}", end="")
    for m in metrics:
        print(f"{m:<12}", end="")
    print()
    print("-" * (15 + 12 * len(metrics)))
    
    for name, result in results.items():
        print(f"{name:<15}", end="")
        for m in metrics:
            value = result.get(m, 0)
            if m == 'MAPE':
                print(f"{value:<12.2f}", end="")
            else:
                print(f"{value:<12.6f}", end="")
        print()


# ============== 可视化函数 ==============

def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs True Values",
    save_path: str = None,
):
    """
    绘制预测值与真实值对比图
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 散点图
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.5, s=20)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predictions')
    ax.set_title('Predictions vs True Values')
    
    # 误差分布
    ax = axes[1]
    errors = y_pred - y_true
    ax.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--')
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Error Distribution (μ={errors.mean():.4f}, σ={errors.std():.4f})')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_training_history(
    history: Dict[str, list],
    title: str = "Training History",
    save_path: str = None,
):
    """
    绘制训练历史
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for key, values in history.items():
        ax.plot(values, label=key)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_degradation_curve(
    cycles: np.ndarray,
    soh_true: np.ndarray,
    soh_pred: np.ndarray = None,
    title: str = "Battery Degradation Curve",
    save_path: str = None,
):
    """
    绘制电池降解曲线
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(cycles, soh_true, 'b-', label='True SOH', linewidth=2)
    if soh_pred is not None:
        ax.plot(cycles, soh_pred, 'r--', label='Predicted SOH', linewidth=2)
    
    ax.axhline(y=0.8, color='gray', linestyle=':', label='EOL (80%)')
    
    ax.set_xlabel('Cycle Number')
    ax.set_ylabel('SOH')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.05])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_heatmap(
    heatmap: np.ndarray,
    channel: int = 0,
    channel_names: List[str] = None,
    title: str = "Charging Heatmap",
    save_path: str = None,
):
    """
    绘制充电热力图
    
    Args:
        heatmap: shape (window_size, num_samples, channels)
        channel: 显示哪个通道
        channel_names: 通道名称
    """
    if channel_names is None:
        channel_names = ['Voltage', 'Current', 'Time']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(heatmap[:, :, channel], aspect='auto', cmap='viridis')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Cycle Index')
    ax.set_title(f'{title} - {channel_names[channel]}')
    
    plt.colorbar(im, ax=ax, label=channel_names[channel])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = 'RMSE',
    title: str = None,
    save_path: str = None,
):
    """
    绘制模型对比柱状图
    """
    models = list(results.keys())
    values = [results[m][metric] for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    bars = ax.bar(models, values, color='steelblue', edgecolor='black')
    ax.set_ylabel(metric)
    ax.set_title(title or f'Model Comparison - {metric}')
    
    # 添加数值标签
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def create_results_table(
    results: Dict[str, Dict[str, float]],
    save_path: str = None,
) -> str:
    """
    创建结果表格（Markdown格式）
    """
    metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
    
    # 表头
    table = "| Model | " + " | ".join(metrics) + " |\n"
    table += "|" + "|".join(["---"] * (len(metrics) + 1)) + "|\n"
    
    # 数据行
    for model, result in results.items():
        row = f"| {model} |"
        for m in metrics:
            value = result.get(m, 0)
            if m == 'MAPE':
                row += f" {value:.2f}% |"
            else:
                row += f" {value:.4f} |"
        table += row + "\n"
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(table)
    
    return table


# ============== RUL预测 (基于SOH轨迹外推) ==============

@dataclass
class RULPredictionResult:
    """RUL预测结果"""
    current_cycle: int              # 当前循环
    current_soh: float              # 当前SOH
    predicted_rul: int              # 预测的RUL
    predicted_eol: int              # 预测的EOL循环
    true_rul: Optional[int]         # 真实RUL (如果已知)
    true_eol: Optional[int]         # 真实EOL (如果已知)
    soh_trajectory: np.ndarray      # 预测的SOH轨迹
    cycle_trajectory: np.ndarray    # 对应的循环数
    fit_method: str                 # 使用的拟合方法
    fit_params: dict                # 拟合参数
    confidence: float               # 置信度 (拟合质量)


class RULPredictor:
    """
    RUL预测器 - 基于SOH轨迹外推
    
    工作原理:
    1. 使用SOH预测模型预测每个循环的SOH
    2. 对预测的SOH轨迹进行曲线拟合（指数衰减、线性、多项式等）
    3. 外推拟合曲线，预测SOH降到阈值以下的循环数
    4. RUL = 预测EOL - 当前循环
    
    支持的衰减模型:
    - exponential: SOH(n) = a * exp(-b * n) + c
    - linear: SOH(n) = a * n + b
    - polynomial: SOH(n) = a * n^2 + b * n + c
    - power: SOH(n) = a * n^b + c
    
    Example:
        >>> predictor = RULPredictor(threshold=0.8)
        >>> result = predictor.predict_from_soh_array(
        ...     soh_array=battery.get_soh_array(),
        ...     current_cycle=500
        ... )
        >>> print(f"预测RUL: {result.predicted_rul} 循环")
    """
    
    # 衰减模型定义
    @staticmethod
    def _exponential(n, a, b, c):
        """指数衰减: SOH = a * exp(-b * n) + c"""
        return a * np.exp(-b * n) + c
    
    @staticmethod
    def _linear(n, a, b):
        """线性衰减: SOH = a * n + b"""
        return a * n + b
    
    @staticmethod
    def _polynomial(n, a, b, c):
        """二次多项式: SOH = a * n^2 + b * n + c"""
        return a * n**2 + b * n + c
    
    @staticmethod
    def _power(n, a, b, c):
        """幂律衰减: SOH = a * n^b + c"""
        return a * np.power(n + 1, b) + c
    
    def __init__(
        self,
        threshold: float = 0.8,
        fit_method: str = 'auto',
        max_extrapolate_cycles: int = 5000,
    ):
        """
        初始化RUL预测器
        
        Args:
            threshold: SOH阈值，默认0.8 (80%)
            fit_method: 拟合方法 ('exponential', 'linear', 'polynomial', 'power', 'auto')
                       'auto' 自动选择最佳拟合
            max_extrapolate_cycles: 最大外推循环数
        """
        self.threshold = threshold
        self.fit_method = fit_method
        self.max_extrapolate_cycles = max_extrapolate_cycles
        
        # 拟合函数映射
        self._fit_funcs = {
            'exponential': (self._exponential, [0.2, 0.001, 0.8], [(0, 1), (0, 0.1), (0, 1)]),
            'linear': (self._linear, [-0.0001, 1.0], [(-0.01, 0), (0.5, 1.5)]),
            'polynomial': (self._polynomial, [-1e-8, -1e-4, 1.0], [(-1e-6, 0), (-0.01, 0), (0.5, 1.5)]),
            'power': (self._power, [-0.1, 0.5, 1.0], [(-1, 0), (0, 1), (0.5, 1.5)]),
        }
    
    def fit_degradation_curve(
        self,
        cycles: np.ndarray,
        soh: np.ndarray,
        method: str = None,
    ) -> Tuple[callable, dict, float]:
        """
        拟合退化曲线
        
        Args:
            cycles: 循环数数组
            soh: SOH数组
            method: 拟合方法
        
        Returns:
            (拟合函数, 参数字典, R²)
        """
        method = method or self.fit_method
        
        if method == 'auto':
            # 尝试所有方法，选择最佳
            best_r2 = -np.inf
            best_result = None
            
            for m in ['exponential', 'linear', 'polynomial']:
                try:
                    func, params, r2 = self._fit_single_method(cycles, soh, m)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_result = (func, params, r2, m)
                except Exception:
                    continue
            
            if best_result is None:
                raise ValueError("无法拟合退化曲线")
            
            return best_result[0], best_result[1], best_result[2], best_result[3]
        else:
            func, params, r2 = self._fit_single_method(cycles, soh, method)
            return func, params, r2, method
    
    def _fit_single_method(
        self,
        cycles: np.ndarray,
        soh: np.ndarray,
        method: str,
    ) -> Tuple[callable, dict, float]:
        """单一方法拟合"""
        func_template, p0, bounds = self._fit_funcs[method]
        
        # 转换bounds格式
        lower_bounds = [b[0] for b in bounds]
        upper_bounds = [b[1] for b in bounds]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                func_template,
                cycles,
                soh,
                p0=p0,
                bounds=(lower_bounds, upper_bounds),
                maxfev=5000,
            )
        
        # 计算R²
        y_pred = func_template(cycles, *popt)
        ss_res = np.sum((soh - y_pred) ** 2)
        ss_tot = np.sum((soh - soh.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        
        # 创建参数字典
        param_names = {
            'exponential': ['a', 'b', 'c'],
            'linear': ['a', 'b'],
            'polynomial': ['a', 'b', 'c'],
            'power': ['a', 'b', 'c'],
        }
        params = dict(zip(param_names[method], popt))
        
        # 返回绑定参数的函数
        def fitted_func(n):
            return func_template(n, *popt)
        
        return fitted_func, params, r2
    
    def predict_eol(
        self,
        fit_func: callable,
        current_cycle: int,
    ) -> int:
        """
        预测EOL循环数
        
        Args:
            fit_func: 拟合后的退化函数
            current_cycle: 当前循环
        
        Returns:
            预测的EOL循环数
        """
        # 从当前循环开始搜索
        for n in range(current_cycle, current_cycle + self.max_extrapolate_cycles):
            soh = fit_func(n)
            if soh < self.threshold:
                return n
        
        # 未找到，返回最大值
        return current_cycle + self.max_extrapolate_cycles
    
    def predict_from_soh_array(
        self,
        soh_array: np.ndarray,
        current_cycle: int = None,
        true_eol: int = None,
        min_history: int = 50,
    ) -> RULPredictionResult:
        """
        从SOH数组预测RUL
        
        Args:
            soh_array: 历史SOH数组
            current_cycle: 当前循环（默认为数组最后一个）
            true_eol: 真实EOL（用于评估）
            min_history: 最少需要的历史循环数
        
        Returns:
            RULPredictionResult
        """
        soh_array = np.asarray(soh_array)
        
        # 过滤无效值
        valid_mask = ~np.isnan(soh_array) & (soh_array > 0) & (soh_array <= 1.5)
        soh_valid = soh_array[valid_mask]
        cycles_valid = np.arange(len(soh_array))[valid_mask]
        
        if len(soh_valid) < min_history:
            raise ValueError(f"历史数据不足: {len(soh_valid)} < {min_history}")
        
        if current_cycle is None:
            current_cycle = int(cycles_valid[-1])
        
        # 使用当前循环之前的数据拟合
        mask = cycles_valid <= current_cycle
        fit_cycles = cycles_valid[mask]
        fit_soh = soh_valid[mask]
        
        # 拟合退化曲线
        fit_func, fit_params, r2, method = self.fit_degradation_curve(fit_cycles, fit_soh)
        
        # 预测EOL
        predicted_eol = self.predict_eol(fit_func, current_cycle)
        predicted_rul = predicted_eol - current_cycle
        
        # 生成预测轨迹
        trajectory_cycles = np.arange(current_cycle, predicted_eol + 50)
        trajectory_soh = np.array([fit_func(n) for n in trajectory_cycles])
        
        # 计算真实RUL
        true_rul = None
        if true_eol is not None:
            true_rul = max(0, true_eol - current_cycle)
        
        return RULPredictionResult(
            current_cycle=current_cycle,
            current_soh=float(fit_func(current_cycle)),
            predicted_rul=predicted_rul,
            predicted_eol=predicted_eol,
            true_rul=true_rul,
            true_eol=true_eol,
            soh_trajectory=trajectory_soh,
            cycle_trajectory=trajectory_cycles,
            fit_method=method,
            fit_params=fit_params,
            confidence=max(0, r2),
        )
    
    def predict_from_model(
        self,
        model,
        battery,
        current_cycle_idx: int,
        input_type: str = 'sequence',
        num_samples: int = 200,
        device: str = 'cuda',
    ) -> RULPredictionResult:
        """
        使用SOH预测模型预测RUL
        
        工作流程:
        1. 使用模型预测每个历史循环的SOH
        2. 基于预测SOH拟合退化曲线
        3. 外推得到RUL
        
        Args:
            model: 训练好的SOH预测模型
            battery: BatteryData对象
            current_cycle_idx: 当前循环索引
            input_type: 输入类型
            num_samples: 采样点数
            device: 计算设备
        
        Returns:
            RULPredictionResult
        """
        import torch
        from .data.feature import create_sequence, create_full_sequence
        
        model.eval()
        predicted_soh = []
        
        # 预测每个循环的SOH
        with torch.no_grad():
            for idx in range(min(current_cycle_idx + 1, len(battery))):
                cycle = battery.cycles[idx]
                
                # 提取特征
                if input_type == 'sequence':
                    feat = create_sequence(cycle, num_samples, battery.charge_cutoff_voltage)
                elif input_type == 'full_sequence':
                    feat = create_full_sequence(cycle, num_samples)
                else:
                    raise ValueError(f"不支持的input_type: {input_type}")
                
                # 预测
                x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
                pred = model(x).cpu().numpy().flatten()[0]
                predicted_soh.append(pred)
        
        predicted_soh = np.array(predicted_soh)
        
        # 计算真实EOL
        true_eol = battery.compute_eol(self.threshold)
        
        return self.predict_from_soh_array(
            soh_array=predicted_soh,
            current_cycle=current_cycle_idx,
            true_eol=true_eol,
        )
    
    def evaluate_on_battery(
        self,
        battery,
        eval_points: List[float] = [0.3, 0.5, 0.7, 0.9],
    ) -> Dict[str, any]:
        """
        在电池上评估RUL预测性能
        
        Args:
            battery: BatteryData对象
            eval_points: 评估点（寿命比例）
        
        Returns:
            评估结果字典
        """
        soh_array = battery.get_soh_array()
        true_eol = battery.compute_eol(self.threshold)
        
        if true_eol is None:
            return {'error': 'Battery has not reached EOL'}
        
        results = []
        
        for ratio in eval_points:
            eval_cycle = int(true_eol * ratio)
            
            if eval_cycle < 50:
                continue
            
            try:
                result = self.predict_from_soh_array(
                    soh_array[:eval_cycle],
                    current_cycle=eval_cycle,
                    true_eol=true_eol,
                )
                
                error = result.predicted_rul - result.true_rul
                relative_error = abs(error) / max(1, result.true_rul) * 100
                
                results.append({
                    'eval_ratio': ratio,
                    'current_cycle': eval_cycle,
                    'predicted_rul': result.predicted_rul,
                    'true_rul': result.true_rul,
                    'error': error,
                    'relative_error': relative_error,
                    'confidence': result.confidence,
                    'fit_method': result.fit_method,
                })
            except Exception as e:
                results.append({
                    'eval_ratio': ratio,
                    'error_msg': str(e),
                })
        
        return {
            'battery_id': battery.cell_id,
            'true_eol': true_eol,
            'threshold': self.threshold,
            'evaluations': results,
        }


def plot_rul_prediction(
    result: RULPredictionResult,
    historical_soh: np.ndarray = None,
    historical_cycles: np.ndarray = None,
    title: str = None,
    save_path: str = None,
):
    """
    可视化RUL预测结果
    
    Args:
        result: RUL预测结果
        historical_soh: 历史SOH数组（可选）
        historical_cycles: 历史循环数（可选）
        title: 图表标题
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制历史数据
    if historical_soh is not None:
        if historical_cycles is None:
            historical_cycles = np.arange(len(historical_soh))
        ax.scatter(historical_cycles, historical_soh, alpha=0.5, s=10, label='历史SOH', color='blue')
    
    # 绘制预测轨迹
    ax.plot(result.cycle_trajectory, result.soh_trajectory, 
            'r-', linewidth=2, label=f'预测轨迹 ({result.fit_method})')
    
    # 绘制阈值线
    ax.axhline(y=0.8, color='green', linestyle='--', linewidth=2, label='EOL阈值 (80%)')
    
    # 标记当前点
    ax.axvline(x=result.current_cycle, color='orange', linestyle=':', label=f'当前循环 ({result.current_cycle})')
    
    # 标记预测EOL
    ax.axvline(x=result.predicted_eol, color='red', linestyle=':', label=f'预测EOL ({result.predicted_eol})')
    
    # 标记真实EOL
    if result.true_eol is not None:
        ax.axvline(x=result.true_eol, color='purple', linestyle=':', label=f'真实EOL ({result.true_eol})')
    
    ax.set_xlabel('循环数')
    ax.set_ylabel('SOH')
    ax.set_ylim(0.6, 1.05)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 标题
    if title is None:
        title = f'RUL预测: 预测={result.predicted_rul}循环'
        if result.true_rul is not None:
            title += f', 真实={result.true_rul}循环, 误差={result.predicted_rul - result.true_rul}'
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def evaluate_rul_predictions(
    batteries: list,
    predictor: RULPredictor = None,
    eval_points: List[float] = [0.3, 0.5, 0.7],
) -> Dict[str, any]:
    """
    批量评估RUL预测
    
    Args:
        batteries: 电池列表
        predictor: RUL预测器
        eval_points: 评估点
    
    Returns:
        汇总评估结果
    """
    if predictor is None:
        predictor = RULPredictor(threshold=0.8)
    
    all_errors = {p: [] for p in eval_points}
    all_relative_errors = {p: [] for p in eval_points}
    
    for battery in batteries:
        result = predictor.evaluate_on_battery(battery, eval_points)
        
        if 'error' in result:
            continue
        
        for eval_result in result['evaluations']:
            if 'error_msg' in eval_result:
                continue
            
            ratio = eval_result['eval_ratio']
            all_errors[ratio].append(eval_result['error'])
            all_relative_errors[ratio].append(eval_result['relative_error'])
    
    # 汇总统计
    summary = {}
    for ratio in eval_points:
        errors = np.array(all_errors[ratio])
        rel_errors = np.array(all_relative_errors[ratio])
        
        if len(errors) > 0:
            summary[f'{int(ratio*100)}%_life'] = {
                'count': len(errors),
                'mean_error': float(np.mean(errors)),
                'std_error': float(np.std(errors)),
                'mae': float(np.mean(np.abs(errors))),
                'mean_relative_error': float(np.mean(rel_errors)),
            }
    
    return summary
