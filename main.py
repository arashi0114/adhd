#!/usr/bin/env python3
"""
ADHD认知障碍预测项目主入口
支持批量训练、模型比较、可控流程
"""
import os
import sys
import argparse
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config.config_manager import ConfigManager
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.dataset import create_datasets
from src.models.deep_learning import create_deep_learning_model
from src.models.traditional_ml import create_traditional_ml_model
from src.evaluation.evaluator import ModelEvaluator
from src.training.train_dl import train_deep_learning_model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import json
import warnings
warnings.filterwarnings('ignore')


class ExperimentRunner:
    """实验运行器 - 统一管理整个实验流程"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化实验运行器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = ConfigManager(config_path)
        self.data_loader = None
        self.preprocessor = None
        self.datasets = {}
        self.evaluator = ModelEvaluator(str(self.config.get_output_path('plots')))
        self.results = {}
        
        # 设置随机种子
        self._set_random_seed()
        
        print("=== ADHD认知障碍预测实验系统 ===")
        print(f"实验名称: {self.config.get('experiment.name')}")
        
    def _set_random_seed(self):
        """设置随机种子确保可复现性"""
        seed = self.config.get('random_seed', 42)
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        print(f"随机种子设置为: {seed}")
    
    def run_experiment(self, 
                      steps: Optional[List[str]] = None,
                      tasks: Optional[List[str]] = None,
                      models: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        运行完整实验
        
        Args:
            steps: 要执行的步骤列表 ['data', 'training', 'evaluation']
            tasks: 要运行的任务列表
            models: 要运行的模型列表
            
        Returns:
            实验结果摘要
        """
        start_time = time.time()
        
        if steps is None:
            steps = ['data', 'training', 'evaluation']
        
        if tasks is None:
            tasks = self.config.get_enabled_tasks()
        
        print(f"\n执行步骤: {steps}")
        print(f"任务: {tasks}")
        print(f"总模型数: {len(self.config.get_enabled_models())}")
        
        try:
            # 步骤1: 数据准备
            if 'data' in steps:
                self.prepare_data()
            
            # 步骤2: 模型训练
            if 'training' in steps:
                self.train_all_models(tasks, models)
            
            # 步骤3: 结果分析
            if 'evaluation' in steps:
                self.analyze_results(tasks)
            
            # 生成最终报告
            summary = self.generate_final_report()
            
            total_time = time.time() - start_time
            print(f"\n=== 实验完成 ===")
            print(f"总耗时: {total_time:.2f}秒")
            
            return summary
            
        except Exception as e:
            print(f"\n❌ 实验失败: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "failed", "error": str(e)}
    
    def prepare_data(self):
        """数据准备步骤"""
        print("\n=== 步骤1: 数据准备 ===")
        
        # 1. 加载原始数据
        print("1.1 加载数据...")
        data_config = self.config.get_data_config()
        self.data_loader = DataLoader(data_config.get('data_dir', 'data/cleaned'))
        load_info = self.data_loader.load_latest_data()
        
        # 2. 数据预处理
        print("1.2 数据预处理...")
        preprocess_config = self.config.get_preprocessing_config()
        
        self.preprocessor = DataPreprocessor()
        # 配置预处理参数
        imputation_enabled = preprocess_config.get('multiple_imputation', {}).get('enabled', True)
        n_imputations = preprocess_config.get('multiple_imputation', {}).get('n_imputations', 5)
        outlier_method = preprocess_config.get('outlier_detection', {}).get('method', 'iqr')
        scaling_method = preprocess_config.get('scaling', {}).get('method', 'standard')
        
        # 使用一个任务的数据来拟合预处理器
        available_tasks = self.config.get_enabled_tasks()
        if available_tasks:
            sample_task = available_tasks[0]
            features, _ = self.data_loader.get_aligned_data(sample_task)
            
            self.preprocessor.fit(
                features,
                imputation_iterations=n_imputations if imputation_enabled else 0,
                outlier_method=outlier_method,
                scaling_method=scaling_method
            )
        
        # 保存数据准备报告
        data_report = {
            "load_info": load_info,
            "preprocessing_report": self.preprocessor.get_preprocessing_report()
        }
        
        report_path = self.config.get_output_path('logs', 'data_preparation_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(data_report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"数据准备完成，报告保存至: {report_path}")
    
    def train_all_models(self, tasks: List[str], model_filter: Optional[List[str]] = None):
        """训练所有模型"""
        print("\n=== 步骤2: 模型训练 ===")
        
        enabled_models = self.config.get_enabled_models()
        
        if model_filter:
            enabled_models = {k: v for k, v in enabled_models.items() if k in model_filter}
        
        print(f"将训练 {len(enabled_models)} 个模型")
        
        for task_type in tasks:
            print(f"\n--- 训练 {task_type} 任务的模型 ---")
            
            if task_type not in self.results:
                self.results[task_type] = {}
            
            # 创建数据集（每个任务单独创建）
            task_datasets = create_datasets(
                data_loader=self.data_loader,
                preprocessor=self.preprocessor,
                task_type=task_type,
                test_size=self.config.get('data.test_ratio', 0.1),
                val_size=self.config.get('data.val_ratio', 0.1),
                random_state=self.config.get('random_seed', 42)
            )
            
            for model_id, model_info in enabled_models.items():
                print(f"\n训练模型: {model_id}")
                
                try:
                    result = self._train_single_model(
                        model_id, model_info, task_type, task_datasets
                    )
                    self.results[task_type][model_id] = result
                    print(f"✅ {model_id} 训练完成")
                    
                except Exception as e:
                    print(f"❌ {model_id} 训练失败: {e}")
                    self.results[task_type][model_id] = {
                        "status": "failed",
                        "error": str(e)
                    }
    
    def _train_single_model(self, model_id: str, model_info: Dict[str, Any], 
                           task_type: str, datasets: Dict) -> Dict[str, Any]:
        """训练单个模型"""
        start_time = time.time()
        model_category = model_info['category']
        model_name = model_info['name']
        
        # 获取数据集信息
        train_dataset = datasets['train']
        val_dataset = datasets['val'] 
        test_dataset = datasets['test']
        
        num_features = train_dataset.get_feature_dim()
        if task_type == "classification":
            output_dim = train_dataset.get_num_classes()
        else:
            output_dim = 1
        
        # 准备模型配置
        model_config = model_info['config']['params'].copy()
        model_config.update({
            'input_dim': num_features,
            'output_dim': output_dim,
            'task_type': task_type
        })
        
        if model_category == 'deep_learning':
            return self._train_deep_learning_model(
                model_name, model_config, datasets, model_id, task_type, start_time
            )
        else:  # traditional_ml
            return self._train_traditional_ml_model(
                model_name, model_config, datasets, model_id, task_type, start_time
            )
    
    def _train_deep_learning_model(self, model_name: str, model_config: Dict[str, Any],
                                 datasets: Dict, model_id: str, task_type: str,
                                 start_time: float) -> Dict[str, Any]:
        """训练深度学习模型"""
        # 创建模型
        model = create_deep_learning_model(model_name, model_config)
        
        # 训练配置
        train_config = self.config.get_training_config('deep_learning')
        device = torch.device(train_config.get('device', 'cpu'))
        model.to(device)
        
        # 创建数据加载器
        batch_size = train_config.get('batch_size', 64)
        train_loader = TorchDataLoader(datasets['train'], batch_size=batch_size, shuffle=True)
        val_loader = TorchDataLoader(datasets['val'], batch_size=batch_size, shuffle=False)
        test_loader = TorchDataLoader(datasets['test'], batch_size=batch_size, shuffle=False)
        
        # 损失函数和优化器
        if task_type == "classification":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=train_config.get('learning_rate', 0.001)
        )
        
        # 训练模型
        history = train_deep_learning_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=train_config.get('num_epochs', 100),
            patience=train_config.get('early_stopping', {}).get('patience', 15)
        )
        
        # 保存模型
        model_path = self.config.get_output_path('models', f"{model_id}_{task_type}.pth")
        torch.save(model.state_dict(), model_path)
        
        # 评估模型
        eval_results = self.evaluator.evaluate_deep_learning_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            task_type=task_type
        )
        
        training_time = time.time() - start_time
        
        return {
            "status": "completed",
            "model_type": "deep_learning",
            "model_name": model_name,
            "training_history": history,
            "evaluation_results": eval_results,
            "training_time": training_time,
            "model_path": str(model_path)
        }
    
    def _train_traditional_ml_model(self, model_name: str, model_config: Dict[str, Any],
                                   datasets: Dict, model_id: str, task_type: str,
                                   start_time: float) -> Dict[str, Any]:
        """训练传统机器学习模型"""
        # 创建模型
        model = create_traditional_ml_model(model_name, {
            'task_type': task_type,
            'hyperparameters': model_config
        })
        
        # 准备数据
        train_dataset = datasets['train']
        val_dataset = datasets['val']
        test_dataset = datasets['test']
        
        X_train, y_train = train_dataset.X, train_dataset.y
        X_val, y_val = val_dataset.X, val_dataset.y
        X_test, y_test = test_dataset.X, test_dataset.y
        
        # 训练模型
        train_result = model.fit(X_train, y_train, X_val, y_val)
        
        # 保存模型
        model_path = self.config.get_output_path('models', f"{model_id}_{task_type}.joblib")
        model.save_model(str(model_path))
        
        # 评估模型
        eval_results = self.evaluator.evaluate_traditional_ml_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            task_type=task_type
        )
        
        training_time = time.time() - start_time
        
        return {
            "status": "completed",
            "model_type": "traditional_ml",
            "model_name": model_name,
            "training_result": train_result,
            "evaluation_results": eval_results,
            "training_time": training_time,
            "model_path": str(model_path)
        }
    
    def analyze_results(self, tasks: List[str]):
        """分析实验结果"""
        print("\n=== 步骤3: 结果分析 ===")
        
        for task_type in tasks:
            if task_type not in self.results:
                continue
                
            print(f"\n--- 分析 {task_type} 结果 ---")
            
            # 收集成功的结果
            successful_results = {
                model_id: result for model_id, result in self.results[task_type].items()
                if result.get("status") == "completed"
            }
            
            if not successful_results:
                print(f"❌ {task_type} 任务没有成功的模型")
                continue
            
            # 创建比较表格
            comparison_data = []
            for model_id, result in successful_results.items():
                eval_results = result.get("evaluation_results", {})
                row = {
                    "model_id": model_id,
                    "model_type": result.get("model_type"),
                    "model_name": result.get("model_name"),
                    "training_time": result.get("training_time", 0)
                }
                
                # 添加评估指标
                if task_type == "classification":
                    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
                else:
                    metrics = ["mse", "rmse", "mae", "r2_score"]
                
                for metric in metrics:
                    row[metric] = eval_results.get(metric, None)
                
                comparison_data.append(row)
            
            # 创建比较DataFrame
            comparison_df = pd.DataFrame(comparison_data)
            
            # 保存比较结果
            comparison_path = self.config.get_output_path('results', f"{task_type}_model_comparison.csv")
            comparison_df.to_csv(comparison_path, index=False)
            
            # 找出最佳模型
            if task_type == "classification":
                best_idx = comparison_df["f1_score"].idxmax()
                primary_metric = "f1_score"
            else:
                best_idx = comparison_df["r2_score"].idxmax()
                primary_metric = "r2_score"
            
            best_model = comparison_df.loc[best_idx, "model_id"]
            best_score = comparison_df.loc[best_idx, primary_metric]
            
            print(f"✅ 最佳 {task_type} 模型: {best_model} ({primary_metric}={best_score:.4f})")
            print(f"比较结果保存至: {comparison_path}")
    
    def generate_final_report(self) -> Dict[str, Any]:
        """生成最终实验报告"""
        print("\n=== 生成最终报告 ===")
        
        # 统计结果
        total_models = 0
        successful_models = 0
        failed_models = 0
        
        task_summaries = {}
        
        for task_type, task_results in self.results.items():
            total_task_models = len(task_results)
            successful_task_models = len([r for r in task_results.values() if r.get("status") == "completed"])
            failed_task_models = total_task_models - successful_task_models
            
            task_summaries[task_type] = {
                "total_models": total_task_models,
                "successful_models": successful_task_models,
                "failed_models": failed_task_models
            }
            
            total_models += total_task_models
            successful_models += successful_task_models
            failed_models += failed_task_models
        
        # 生成报告
        final_report = {
            "experiment_info": {
                "name": self.config.get('experiment.name'),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config_summary": self.config.get_summary()
            },
            "overall_statistics": {
                "total_models_trained": total_models,
                "successful_models": successful_models,
                "failed_models": failed_models,
                "success_rate": successful_models / total_models if total_models > 0 else 0
            },
            "task_summaries": task_summaries,
            "output_directory": str(self.config.output_dirs['base'])
        }
        
        # 保存最终报告
        report_path = self.config.get_output_path('results', 'final_experiment_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"最终报告已保存: {report_path}")
        print(f"成功训练模型: {successful_models}/{total_models}")
        
        return final_report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ADHD认知障碍预测实验系统')
    parser.add_argument('--config', '-c', default='config/config.yaml', 
                       help='配置文件路径')
    parser.add_argument('--steps', nargs='+', 
                       choices=['data', 'training', 'evaluation'],
                       default=['data', 'training', 'evaluation'],
                       help='要执行的步骤')
    parser.add_argument('--tasks', nargs='+',
                       choices=['classification', 'regression'],
                       help='要运行的任务（默认运行配置中启用的所有任务）')
    parser.add_argument('--models', nargs='+',
                       help='要训练的模型ID（默认训练所有启用的模型）')
    
    args = parser.parse_args()
    
    # 检查配置文件
    if not Path(args.config).exists():
        print(f"❌ 配置文件不存在: {args.config}")
        return 1
    
    try:
        # 创建实验运行器
        runner = ExperimentRunner(args.config)
        
        # 运行实验
        summary = runner.run_experiment(
            steps=args.steps,
            tasks=args.tasks,
            models=args.models
        )
        
        if summary.get('status') == 'failed':
            return 1
            
        print("\n🎉 实验成功完成！")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断实验")
        return 130
    except Exception as e:
        print(f"\n💥 实验异常终止: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())