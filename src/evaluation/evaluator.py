"""
模型评估器
统一处理深度学习和传统ML模型的评估
包含可视化、交叉验证、SHAP分析等高级功能
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
import warnings
warnings.filterwarnings('ignore')

# 尝试导入SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

from ..models.traditional_ml import TraditionalMLWrapper


class ModelEvaluator:
    """统一的模型评估器"""
    
    def __init__(self, output_dir: str = "outputs/evaluation"):
        """
        初始化评估器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置绘图样式
        plt.style.use('default')
        sns.set_palette("husl")
    
    def evaluate_deep_learning_model(self,
                                   model: nn.Module,
                                   test_loader: DataLoader,
                                   criterion: nn.Module,
                                   device: torch.device,
                                   task_type: str) -> Dict[str, Any]:
        """
        评估深度学习模型
        
        Args:
            model: 模型
            test_loader: 测试数据加载器
            criterion: 损失函数
            device: 设备
            task_type: 任务类型
            
        Returns:
            评估结果字典
        """
        model.eval()
        model.to(device)
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        total_loss = 0.0
        inference_times = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                # 测量推理时间
                start_time = time.time()
                output = model(data)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # 计算损失
                loss = criterion(output, target)
                total_loss += loss.item()
                
                # 收集预测结果
                if task_type == "classification":
                    probabilities = torch.softmax(output, dim=1)
                    predictions = output.argmax(dim=1)
                    all_probabilities.extend(probabilities.cpu().numpy())
                else:  # regression
                    predictions = output.squeeze()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # 转换为numpy数组
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # 计算评估指标
        results = {
            'test_loss': total_loss / len(test_loader),
            'avg_inference_time': np.mean(inference_times),
            'total_inference_time': np.sum(inference_times)
        }
        
        if task_type == "classification":
            results.update(self._compute_classification_metrics(
                targets, predictions, np.array(all_probabilities)
            ))
        else:
            results.update(self._compute_regression_metrics(targets, predictions))
        
        return results
    
    def evaluate_traditional_ml_model(self,
                                    model: TraditionalMLWrapper,
                                    X_test: np.ndarray,
                                    y_test: np.ndarray,
                                    task_type: str) -> Dict[str, Any]:
        """
        评估传统机器学习模型
        
        Args:
            model: 模型包装器
            X_test: 测试特征
            y_test: 测试标签
            task_type: 任务类型
            
        Returns:
            评估结果字典
        """
        # 测量推理时间
        start_time = time.time()
        predictions = model.predict(X_test)
        inference_time = time.time() - start_time
        
        results = {
            'avg_inference_time': inference_time / len(X_test),
            'total_inference_time': inference_time
        }
        
        if task_type == "classification":
            # 获取预测概率
            probabilities = model.predict_proba(X_test)
            results.update(self._compute_classification_metrics(
                y_test, predictions, probabilities
            ))
        else:
            results.update(self._compute_regression_metrics(y_test, predictions))
        
        return results
    
    def _compute_classification_metrics(self,
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """计算分类任务指标"""
        metrics = {}
        
        # 基本指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # ROC-AUC和PR-AUC（如果有概率预测）
        if y_prob is not None:
            try:
                if y_prob.shape[1] == 2:  # 二分类
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
                    metrics['pr_auc'] = average_precision_score(y_true, y_prob[:, 1])
                else:  # 多分类
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
                    metrics['pr_auc'] = average_precision_score(y_true, y_prob, average='weighted')
            except:
                metrics['roc_auc'] = None
                metrics['pr_auc'] = None
        
        # 混淆矩阵
        try:
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
        except:
            metrics['confusion_matrix'] = None
        
        return metrics
    
    def _compute_regression_metrics(self,
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray) -> Dict[str, float]:
        """计算回归任务指标"""
        metrics = {}
        
        # 基本指标
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2_score'] = r2_score(y_true, y_pred)
        
        # MAPE（平均绝对百分比误差）
        try:
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            metrics['mean_absolute_percentage_error'] = mape
        except:
            metrics['mean_absolute_percentage_error'] = None
        
        return metrics
    
    def compare_models(self,
                      results_dict: Dict[str, Dict[str, Any]],
                      task_type: str,
                      primary_metric: Optional[str] = None) -> Dict[str, Any]:
        """
        比较多个模型的性能
        
        Args:
            results_dict: 模型结果字典
            task_type: 任务类型
            primary_metric: 主要比较指标
            
        Returns:
            比较结果字典
        """
        if primary_metric is None:
            if task_type == "classification":
                primary_metric = "f1_score"
            else:
                primary_metric = "r2_score"
        
        # 提取主要指标
        model_scores = {}
        for model_name, results in results_dict.items():
            if primary_metric in results:
                model_scores[model_name] = results[primary_metric]
        
        # 排序
        if task_type == "classification" or primary_metric in ["r2_score", "accuracy"]:
            # 越大越好
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        else:
            # 越小越好（MSE, MAE等）
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1])
        
        comparison_result = {
            "primary_metric": primary_metric,
            "rankings": sorted_models,
            "best_model": sorted_models[0][0] if sorted_models else None,
            "best_score": sorted_models[0][1] if sorted_models else None,
            "score_differences": {}
        }
        
        # 计算与最佳模型的差距
        if sorted_models:
            best_score = sorted_models[0][1]
            for model_name, score in sorted_models[1:]:
                comparison_result["score_differences"][model_name] = abs(best_score - score)
        
        return comparison_result
    
    def generate_evaluation_report(self,
                                 model_results: Dict[str, Any],
                                 model_name: str,
                                 task_type: str) -> str:
        """
        生成评估报告
        
        Args:
            model_results: 模型评估结果
            model_name: 模型名称
            task_type: 任务类型
            
        Returns:
            格式化的评估报告字符串
        """
        report_lines = [
            f"=== {model_name} Evaluation Report ===",
            f"Task Type: {task_type.title()}",
            ""
        ]
        
        # 基本性能指标
        if task_type == "classification":
            report_lines.extend([
                "Classification Metrics:",
                f"  Accuracy: {model_results.get('accuracy', 'N/A'):.4f}",
                f"  Precision: {model_results.get('precision', 'N/A'):.4f}",
                f"  Recall: {model_results.get('recall', 'N/A'):.4f}",
                f"  F1-Score: {model_results.get('f1_score', 'N/A'):.4f}",
                f"  ROC-AUC: {model_results.get('roc_auc', 'N/A'):.4f}",
                f"  PR-AUC: {model_results.get('pr_auc', 'N/A'):.4f}",
                ""
            ])
        else:
            report_lines.extend([
                "Regression Metrics:",
                f"  R² Score: {model_results.get('r2_score', 'N/A'):.4f}",
                f"  MSE: {model_results.get('mse', 'N/A'):.4f}",
                f"  RMSE: {model_results.get('rmse', 'N/A'):.4f}",
                f"  MAE: {model_results.get('mae', 'N/A'):.4f}",
                f"  MAPE: {model_results.get('mean_absolute_percentage_error', 'N/A'):.2f}%",
                ""
            ])
        
        # 性能指标
        report_lines.extend([
            "Performance Metrics:",
            f"  Avg Inference Time: {model_results.get('avg_inference_time', 'N/A'):.6f}s",
            f"  Total Inference Time: {model_results.get('total_inference_time', 'N/A'):.4f}s",
            ""
        ])
        
        return "\n".join(report_lines)
    
    def plot_confusion_matrix(self, 
                             y_true: np.ndarray, 
                             y_pred: np.ndarray,
                             model_name: str,
                             save_path: Optional[str] = None) -> str:
        """
        绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            model_name: 模型名称
            save_path: 保存路径
            
        Returns:
            保存的图片路径
        """
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=np.unique(y_true),
                   yticklabels=np.unique(y_true))
        
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path is None:
            save_path = self.output_dir / f"{model_name}_confusion_matrix.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_roc_curve(self,
                       y_true: np.ndarray,
                       y_prob: np.ndarray,
                       model_name: str,
                       save_path: Optional[str] = None) -> str:
        """
        绘制ROC曲线
        
        Args:
            y_true: 真实标签
            y_prob: 预测概率（正类）
            model_name: 模型名称
            save_path: 保存路径
            
        Returns:
            保存的图片路径
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = self.output_dir / f"{model_name}_roc_curve.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_pr_curve(self,
                      y_true: np.ndarray,
                      y_prob: np.ndarray,
                      model_name: str,
                      save_path: Optional[str] = None) -> str:
        """
        绘制Precision-Recall曲线
        
        Args:
            y_true: 真实标签
            y_prob: 预测概率（正类）
            model_name: 模型名称
            save_path: 保存路径
            
        Returns:
            保存的图片路径
        """
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'{model_name} (AP = {ap:.3f})')
        
        # 基线
        baseline = np.sum(y_true) / len(y_true)
        plt.axhline(y=baseline, color='k', linestyle='--', 
                   label=f'Baseline (AP = {baseline:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = self.output_dir / f"{model_name}_pr_curve.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_residuals(self,
                       y_true: np.ndarray,
                       y_pred: np.ndarray,
                       model_name: str,
                       save_path: Optional[str] = None) -> str:
        """
        绘制残差图（回归任务）
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            model_name: 模型名称
            save_path: 保存路径
            
        Returns:
            保存的图片路径
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 残差vs预测值
        axes[0].scatter(y_pred, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title(f'Residuals vs Predicted - {model_name}')
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q图
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title(f'Q-Q Plot - {model_name}')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f"{model_name}_residuals.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def cross_validate_model(self,
                            model,
                            X: np.ndarray,
                            y: np.ndarray,
                            task_type: str,
                            cv_folds: int = 5,
                            random_state: int = 42) -> Dict[str, Any]:
        """
        交叉验证模型
        
        Args:
            model: 模型（sklearn兼容）
            X: 特征数据
            y: 标签数据
            task_type: 任务类型
            cv_folds: 交叉验证折数
            random_state: 随机种子
            
        Returns:
            交叉验证结果
        """
        print(f"开始{cv_folds}折交叉验证...")
        
        if task_type == "classification":
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        
        cv_results = {}
        for score in scoring:
            scores = cross_val_score(model, X, y, cv=cv, scoring=score, n_jobs=-1)
            cv_results[score] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
        
        print(f"交叉验证完成: {len(scoring)}个指标")
        return cv_results
    
    def explain_model_shap(self,
                          model,
                          X_train: np.ndarray,
                          X_test: np.ndarray,
                          model_name: str,
                          feature_names: Optional[List[str]] = None,
                          max_display: int = 20) -> Dict[str, Any]:
        """
        使用SHAP解释模型
        
        Args:
            model: 模型
            X_train: 训练数据
            X_test: 测试数据
            model_name: 模型名称
            feature_names: 特征名称
            max_display: 最大显示特征数
            
        Returns:
            SHAP分析结果
        """
        if not SHAP_AVAILABLE:
            return {"error": "SHAP not available"}
        
        print("开始SHAP分析...")
        
        try:
            # 选择合适的解释器
            if hasattr(model, 'predict_proba'):  # sklearn模型
                explainer = shap.Explainer(model, X_train)
            else:  # 其他模型
                explainer = shap.KernelExplainer(model.predict, X_train[:100])  # 使用前100个样本作为背景
            
            # 计算SHAP值
            shap_values = explainer(X_test[:100])  # 对前100个测试样本计算SHAP值
            
            # 生成图表
            shap_plots = {}
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names,
                             max_display=max_display, show=False)
            summary_path = self.output_dir / f"{model_name}_shap_summary.png"
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            plt.close()
            shap_plots['summary'] = str(summary_path)
            
            # Feature importance
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names,
                             plot_type="bar", max_display=max_display, show=False)
            importance_path = self.output_dir / f"{model_name}_shap_importance.png"
            plt.savefig(importance_path, dpi=300, bbox_inches='tight')
            plt.close()
            shap_plots['importance'] = str(importance_path)
            
            # 计算特征重要性分数
            importance_values = np.abs(shap_values.values).mean(0)
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importance_values))]
            
            feature_importance = dict(zip(feature_names, importance_values))
            feature_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True))
            
            print(f"SHAP分析完成，生成了{len(shap_plots)}个图表")
            
            return {
                "plots": shap_plots,
                "feature_importance": feature_importance,
                "shap_values_shape": shap_values.values.shape,
                "background_size": X_train.shape[0]
            }
            
        except Exception as e:
            print(f"SHAP分析失败: {e}")
            return {"error": str(e)}
    
    def compute_statistical_tests(self,
                                 results1: Dict[str, float],
                                 results2: Dict[str, float],
                                 metric: str,
                                 confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        计算两个模型之间的统计显著性测试
        
        Args:
            results1: 第一个模型的CV结果
            results2: 第二个模型的CV结果
            metric: 要比较的指标
            confidence_level: 置信水平
            
        Returns:
            统计测试结果
        """
        from scipy import stats
        
        if metric not in results1 or metric not in results2:
            return {"error": f"Metric '{metric}' not found in results"}
        
        scores1 = np.array(results1[metric]['scores'])
        scores2 = np.array(results2[metric]['scores'])
        
        # 配对t检验
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        
        # 置信区间
        diff = scores1 - scores2
        mean_diff = diff.mean()
        std_diff = diff.std()
        n = len(diff)
        
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, n-1)
        margin_error = t_critical * std_diff / np.sqrt(n)
        
        ci_lower = mean_diff - margin_error
        ci_upper = mean_diff + margin_error
        
        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "mean_difference": mean_diff,
            "confidence_interval": (ci_lower, ci_upper),
            "confidence_level": confidence_level,
            "is_significant": p_value < (1 - confidence_level),
            "better_model": "model1" if mean_diff > 0 else "model2"
        }
    
    def generate_comprehensive_report(self,
                                    model_results: Dict[str, Any],
                                    model_name: str,
                                    task_type: str,
                                    include_plots: bool = True) -> Dict[str, Any]:
        """
        生成综合评估报告
        
        Args:
            model_results: 模型评估结果
            model_name: 模型名称
            task_type: 任务类型
            include_plots: 是否包含图表
            
        Returns:
            综合报告字典
        """
        report = {
            "model_name": model_name,
            "task_type": task_type,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": {},
            "plots": [],
            "summary": ""
        }
        
        # 提取关键指标
        if task_type == "classification":
            key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
        else:
            key_metrics = ['r2_score', 'mse', 'rmse', 'mae', 'mean_absolute_percentage_error']
        
        for metric in key_metrics:
            if metric in model_results:
                report["metrics"][metric] = model_results[metric]
        
        # 性能指标
        perf_metrics = ['avg_inference_time', 'total_inference_time']
        for metric in perf_metrics:
            if metric in model_results:
                report["metrics"][metric] = model_results[metric]
        
        # 生成文本报告
        report["summary"] = self.generate_evaluation_report(
            model_results, model_name, task_type
        )
        
        # 保存报告到文件
        report_path = self.output_dir / f"{model_name}_comprehensive_report.json"
        import json
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"综合报告已保存到: {report_path}")
        
        return report


if __name__ == "__main__":
    # 测试代码
    print("Testing model evaluator...")
    
    evaluator = ModelEvaluator()
    
    # 测试分类指标计算
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6], 
                       [0.2, 0.8], [0.6, 0.4], [0.9, 0.1]])
    
    clf_metrics = evaluator._compute_classification_metrics(y_true, y_pred, y_prob)
    print(f"Classification metrics: {clf_metrics}")
    
    # 测试回归指标计算
    y_true_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred_reg = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
    
    reg_metrics = evaluator._compute_regression_metrics(y_true_reg, y_pred_reg)
    print(f"Regression metrics: {reg_metrics}")
    
    # 测试报告生成
    report = evaluator.generate_evaluation_report(
        clf_metrics, "TestModel", "classification"
    )
    print(f"\nEvaluation report:\n{report}")
    
    print("Evaluator test completed!")