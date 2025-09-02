"""
模型评估模块
包含各种评估指标和可视化功能
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from pathlib import Path


def evaluate_single_task(model: nn.Module,
                        test_loader: DataLoader,
                        criterion: nn.Module,
                        device: torch.device,
                        task_name: str,
                        save_results: bool = True,
                        results_dir: str = "outputs/results") -> Dict[str, Any]:
    """
    单任务模型评估
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 设备
        task_name: 任务名称
        save_results: 是否保存结果
        results_dir: 结果保存目录
        
    Returns:
        评估结果字典
    """
    print(f"\n=== 开始评估 {task_name} ===")
    print(f"测试样本数: {len(test_loader.dataset)}")
    
    model = model.to(device)
    model.eval()
    
    # 确定任务类型
    is_classification = isinstance(criterion, nn.CrossEntropyLoss)
    
    all_predictions = []
    all_targets = []
    all_outputs = []
    test_loss = 0.0
    
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            
            outputs = model(features)
            
            # 计算损失
            if is_classification:
                loss = criterion(outputs, targets)
                predictions = torch.argmax(outputs, dim=1)
                all_outputs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            else:  # regression
                outputs = outputs.squeeze()
                loss = criterion(outputs, targets)
                predictions = outputs
                all_outputs.extend(outputs.cpu().numpy())
            
            test_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    avg_test_loss = test_loss / len(test_loader)
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_outputs = np.array(all_outputs)
    
    # 计算评估指标
    if is_classification:
        results = _evaluate_classification(all_targets, all_predictions, all_outputs, avg_test_loss)
    else:
        results = _evaluate_regression(all_targets, all_predictions, avg_test_loss)
    
    results['task_name'] = task_name
    results['num_samples'] = len(all_targets)
    
    # 打印结果
    _print_evaluation_results(results, is_classification)
    
    # 保存结果和可视化
    if save_results:
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        # 保存预测结果
        predictions_df = pd.DataFrame({
            'true_values': all_targets,
            'predictions': all_predictions
        })
        if is_classification:
            predictions_df['probabilities'] = [prob.tolist() for prob in all_outputs]
        
        predictions_df.to_csv(results_path / f"{task_name}_predictions.csv", index=False)
        
        # 保存评估指标
        metrics_df = pd.DataFrame([results])
        metrics_df.to_csv(results_path / f"{task_name}_metrics.csv", index=False)
        
        # 生成可视化
        _create_evaluation_plots(all_targets, all_predictions, all_outputs, 
                                is_classification, results_path, task_name)
        
        print(f"评估结果已保存至: {results_path}")
    
    return results


def _evaluate_classification(true_labels: np.ndarray, 
                           predictions: np.ndarray, 
                           probabilities: np.ndarray,
                           test_loss: float) -> Dict[str, Any]:
    """
    分类任务评估
    
    Args:
        true_labels: 真实标签
        predictions: 预测标签
        probabilities: 预测概率
        test_loss: 测试损失
        
    Returns:
        评估结果字典
    """
    results = {
        'test_loss': test_loss,
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions, average='weighted'),
        'recall': recall_score(true_labels, predictions, average='weighted'),
        'f1_score': f1_score(true_labels, predictions, average='weighted'),
    }
    
    # 计算各类别的指标
    precision_per_class = precision_score(true_labels, predictions, average=None)
    recall_per_class = recall_score(true_labels, predictions, average=None)
    f1_per_class = f1_score(true_labels, predictions, average=None)
    
    results['precision_per_class'] = precision_per_class.tolist()
    results['recall_per_class'] = recall_per_class.tolist()
    results['f1_per_class'] = f1_per_class.tolist()
    
    # 混淆矩阵
    cm = confusion_matrix(true_labels, predictions)
    results['confusion_matrix'] = cm.tolist()
    
    # ROC曲线 (仅支持二分类)
    if len(np.unique(true_labels)) == 2:
        fpr, tpr, _ = roc_curve(true_labels, probabilities[:, 1])
        roc_auc = auc(fpr, tpr)
        results['roc_auc'] = roc_auc
        results['fpr'] = fpr.tolist()
        results['tpr'] = tpr.tolist()
    
    return results


def _evaluate_regression(true_values: np.ndarray, 
                        predictions: np.ndarray,
                        test_loss: float) -> Dict[str, Any]:
    """
    回归任务评估
    
    Args:
        true_values: 真实值
        predictions: 预测值
        test_loss: 测试损失
        
    Returns:
        评估结果字典
    """
    results = {
        'test_loss': test_loss,
        'mse': mean_squared_error(true_values, predictions),
        'rmse': np.sqrt(mean_squared_error(true_values, predictions)),
        'mae': mean_absolute_error(true_values, predictions),
        'r2_score': r2_score(true_values, predictions),
    }
    
    # 计算残差统计
    residuals = predictions - true_values
    results['mean_residual'] = np.mean(residuals)
    results['std_residual'] = np.std(residuals)
    results['max_residual'] = np.max(np.abs(residuals))
    
    # 计算相关系数
    correlation = np.corrcoef(true_values, predictions)[0, 1]
    results['correlation'] = correlation
    
    return results


def _print_evaluation_results(results: Dict[str, Any], is_classification: bool):
    """
    打印评估结果
    
    Args:
        results: 评估结果字典
        is_classification: 是否为分类任务
    """
    print(f"\n评估结果:")
    print(f"测试损失: {results['test_loss']:.4f}")
    
    if is_classification:
        print(f"准确率: {results['accuracy']:.4f}")
        print(f"精确率: {results['precision']:.4f}")
        print(f"召回率: {results['recall']:.4f}")
        print(f"F1分数: {results['f1_score']:.4f}")
        
        if 'roc_auc' in results:
            print(f"ROC AUC: {results['roc_auc']:.4f}")
        
        print(f"\n各类别详细指标:")
        for i, (p, r, f1) in enumerate(zip(results['precision_per_class'], 
                                          results['recall_per_class'], 
                                          results['f1_per_class'])):
            print(f"  类别 {i}: 精确率={p:.4f}, 召回率={r:.4f}, F1={f1:.4f}")
    
    else:
        print(f"R²分数: {results['r2_score']:.4f}")
        print(f"均方误差 (MSE): {results['mse']:.4f}")
        print(f"均方根误差 (RMSE): {results['rmse']:.4f}")
        print(f"平均绝对误差 (MAE): {results['mae']:.4f}")
        print(f"相关系数: {results['correlation']:.4f}")
        print(f"残差统计: 均值={results['mean_residual']:.4f}, 标准差={results['std_residual']:.4f}")


def _create_evaluation_plots(true_values: np.ndarray,
                           predictions: np.ndarray,
                           outputs: np.ndarray,
                           is_classification: bool,
                           save_dir: Path,
                           task_name: str):
    """
    创建评估可视化图表
    
    Args:
        true_values: 真实值
        predictions: 预测值
        outputs: 模型输出
        is_classification: 是否为分类任务
        save_dir: 保存目录
        task_name: 任务名称
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    if is_classification:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{task_name} 分类结果分析', fontsize=16)
        
        # 混淆矩阵
        cm = confusion_matrix(true_values, predictions)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 0], cmap='Blues')
        axes[0, 0].set_title('混淆矩阵')
        axes[0, 0].set_xlabel('预测标签')
        axes[0, 0].set_ylabel('真实标签')
        
        # 预测概率分布
        axes[0, 1].hist(outputs[:, 1], bins=30, alpha=0.7, label='正类概率')
        axes[0, 1].set_title('预测概率分布')
        axes[0, 1].set_xlabel('预测概率')
        axes[0, 1].set_ylabel('频数')
        axes[0, 1].legend()
        
        # ROC曲线 (仅二分类)
        if len(np.unique(true_values)) == 2:
            fpr, tpr, _ = roc_curve(true_values, outputs[:, 1])
            roc_auc = auc(fpr, tpr)
            axes[1, 0].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
            axes[1, 0].plot([0, 1], [0, 1], 'k--')
            axes[1, 0].set_title('ROC曲线')
            axes[1, 0].set_xlabel('假正率')
            axes[1, 0].set_ylabel('真正率')
            axes[1, 0].legend()
        
        # 预测结果对比
        x = np.arange(len(true_values[:50]))
        axes[1, 1].scatter(x, true_values[:50], label='真实值', alpha=0.7)
        axes[1, 1].scatter(x, predictions[:50], label='预测值', alpha=0.7)
        axes[1, 1].set_title('预测结果对比 (前50个样本)')
        axes[1, 1].set_xlabel('样本索引')
        axes[1, 1].set_ylabel('类别')
        axes[1, 1].legend()
        
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{task_name} 回归结果分析', fontsize=16)
        
        # 真实值 vs 预测值散点图
        axes[0, 0].scatter(true_values, predictions, alpha=0.6)
        min_val, max_val = min(true_values.min(), predictions.min()), max(true_values.max(), predictions.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[0, 0].set_title('真实值 vs 预测值')
        axes[0, 0].set_xlabel('真实值')
        axes[0, 0].set_ylabel('预测值')
        
        # 残差分布
        residuals = predictions - true_values
        axes[0, 1].hist(residuals, bins=30, alpha=0.7)
        axes[0, 1].set_title('残差分布')
        axes[0, 1].set_xlabel('残差')
        axes[0, 1].set_ylabel('频数')
        
        # 残差 vs 预测值
        axes[1, 0].scatter(predictions, residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_title('残差 vs 预测值')
        axes[1, 0].set_xlabel('预测值')
        axes[1, 0].set_ylabel('残差')
        
        # Q-Q图（残差正态性检验）
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('残差Q-Q图')
    
    plt.tight_layout()
    plt.savefig(save_dir / f"{task_name}_evaluation.png", dpi=300, bbox_inches='tight')
    plt.show()


def compare_models(results_list: List[Dict[str, Any]], 
                  model_names: List[str],
                  save_path: Optional[str] = None) -> pd.DataFrame:
    """
    比较多个模型的性能
    
    Args:
        results_list: 模型结果列表
        model_names: 模型名称列表
        save_path: 保存路径
        
    Returns:
        比较结果DataFrame
    """
    comparison_data = []
    
    for results, name in zip(results_list, model_names):
        row = {'Model': name}
        row.update(results)
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"模型比较结果已保存至: {save_path}")
    
    return df