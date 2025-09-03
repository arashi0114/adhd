"""
统一数据加载模块
从data/cleaned目录加载最新的CSV文件，并提取认知障碍相关标签
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime


class DataLoader:
    """
    数据加载器 - 专门处理ELSA认知障碍数据
    """
    
    def __init__(self, data_dir: str = "data/cleaned"):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.data = None
        self.feature_cols = None
        self.labels = {}
        
    def load_latest_data(self, file_pattern: str = "*.csv") -> Dict[str, Any]:
        """
        加载最新的CSV数据文件
        
        Args:
            file_pattern: 文件匹配模式
            
        Returns:
            数据加载信息字典
        """
        print("=== 开始数据加载 ===")
        
        # 查找最新的CSV文件
        csv_files = list(self.data_dir.glob(file_pattern))
        if not csv_files:
            raise FileNotFoundError(f"在 {self.data_dir} 中没有找到CSV文件")
        
        # 按修改时间排序，选择最新的
        latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
        print(f"选择文件: {latest_file}")
        
        # 加载数据
        self.data = pd.read_csv(latest_file)
        print(f"数据形状: {self.data.shape}")
        
        # 提取标签
        self._extract_labels()
        
        # 提取特征列
        self._extract_features()
        
        # 数据质量检查
        quality_info = self._check_data_quality()
        
        return {
            "file_path": str(latest_file),
            "data_shape": self.data.shape,
            "available_labels": list(self.labels.keys()),
            "feature_count": len(self.feature_cols),
            "quality_info": quality_info,
            "load_time": datetime.now().isoformat()
        }
    
    def _extract_labels(self):
        """提取认知障碍相关标签"""
        # 早筛标签 (二分类): racogimp_label
        if 'racogimp_label' in self.data.columns:
            # 转换为二分类：-1(从未发病)->0, 其他->1
            binary_labels = (self.data['racogimp_label'] != -1).astype(int)
            # 排除缺失值
            valid_idx = ~self.data['racogimp_label'].isna()
            self.labels['classification'] = pd.Series(
                binary_labels, 
                index=self.data.index
            )[valid_idx]
            print(f"分类任务标签: {self.labels['classification'].value_counts().to_dict()}")
        
        # 发病时间预测标签 (回归): 寻找r{wave}cogimpt_label列
        regression_cols = [col for col in self.data.columns 
                          if col.startswith('r') and col.endswith('cogimpt_label')]
        
        if regression_cols:
            # 使用第一个wave的认知障碍时间标签作为回归目标
            reg_col = regression_cols[0]  # 例如 'r1cogimpt_label'
            valid_idx = ~self.data[reg_col].isna()
            self.labels['regression'] = self.data[reg_col][valid_idx]
            print(f"回归任务标签列: {reg_col}, 有效样本: {len(self.labels['regression'])}")
            print(f"回归标签范围: {self.labels['regression'].min():.2f} - {self.labels['regression'].max():.2f}")
    
    def _extract_features(self):
        """提取特征列"""
        # 排除标签列和ID列
        exclude_patterns = ['label', 'id', 'ID', 'index']
        self.feature_cols = []
        
        for col in self.data.columns:
            # 跳过明显的标签和ID列
            if any(pattern.lower() in col.lower() for pattern in exclude_patterns):
                continue
            # 跳过非数值列
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                continue
            self.feature_cols.append(col)
        
        print(f"特征列数量: {len(self.feature_cols)}")
        print(f"特征列示例: {self.feature_cols[:5]}")
    
    def _check_data_quality(self) -> Dict[str, Any]:
        """检查数据质量"""
        if self.data is None or not self.feature_cols:
            return {}
        
        feature_data = self.data[self.feature_cols]
        
        quality_info = {
            "total_samples": len(self.data),
            "total_features": len(self.feature_cols),
            "missing_rate_per_sample": feature_data.isnull().sum(axis=1).mean() / len(self.feature_cols),
            "missing_rate_per_feature": feature_data.isnull().sum(axis=0).mean() / len(feature_data),
            "complete_samples": (feature_data.isnull().sum(axis=1) == 0).sum(),
            "complete_features": (feature_data.isnull().sum(axis=0) == 0).sum()
        }
        
        print("数据质量报告:")
        print(f"  - 总样本数: {quality_info['total_samples']}")
        print(f"  - 总特征数: {quality_info['total_features']}")
        print(f"  - 平均样本缺失率: {quality_info['missing_rate_per_sample']:.2%}")
        print(f"  - 平均特征缺失率: {quality_info['missing_rate_per_feature']:.2%}")
        print(f"  - 完整样本数: {quality_info['complete_samples']}")
        
        return quality_info
    
    def get_features(self) -> pd.DataFrame:
        """获取特征数据"""
        if self.data is None or not self.feature_cols:
            raise ValueError("数据未加载或特征未提取")
        return self.data[self.feature_cols].copy()
    
    def get_labels(self, task_type: str) -> pd.Series:
        """
        获取指定任务的标签
        
        Args:
            task_type: 任务类型 ('classification' 或 'regression')
            
        Returns:
            标签Series
        """
        if task_type not in self.labels:
            raise ValueError(f"任务类型 '{task_type}' 不可用。可用类型: {list(self.labels.keys())}")
        return self.labels[task_type].copy()
    
    def get_aligned_data(self, task_type: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        获取对齐的特征和标签数据
        
        Args:
            task_type: 任务类型
            
        Returns:
            (特征DataFrame, 标签Series)
        """
        labels = self.get_labels(task_type)
        features = self.get_features()
        
        # 获取共同的索引
        common_idx = features.index.intersection(labels.index)
        
        aligned_features = features.loc[common_idx]
        aligned_labels = labels.loc[common_idx]
        
        print(f"{task_type}任务对齐后数据形状:")
        print(f"  - 特征: {aligned_features.shape}")
        print(f"  - 标签: {aligned_labels.shape}")
        
        return aligned_features, aligned_labels
    
    def get_data_summary(self) -> Dict[str, Any]:
        """获取数据摘要"""
        if self.data is None:
            return {"status": "数据未加载"}
        
        summary = {
            "loaded": True,
            "total_shape": self.data.shape,
            "feature_count": len(self.feature_cols) if self.feature_cols else 0,
            "available_tasks": list(self.labels.keys()),
            "memory_usage_mb": self.data.memory_usage(deep=True).sum() / (1024**2)
        }
        
        # 添加每个任务的标签统计
        for task_type, labels in self.labels.items():
            if task_type == 'classification':
                summary[f"{task_type}_distribution"] = labels.value_counts().to_dict()
            else:
                summary[f"{task_type}_stats"] = {
                    "count": len(labels),
                    "mean": labels.mean(),
                    "std": labels.std(),
                    "min": labels.min(),
                    "max": labels.max()
                }
        
        return summary


def load_data(data_dir: str = "data/cleaned") -> DataLoader:
    """
    便捷函数：加载数据
    
    Args:
        data_dir: 数据目录
        
    Returns:
        已加载数据的DataLoader实例
    """
    loader = DataLoader(data_dir)
    loader.load_latest_data()
    return loader


if __name__ == "__main__":
    # 测试代码
    try:
        print("测试数据加载模块...")
        
        # 加载数据
        loader = load_data()
        
        # 打印摘要
        summary = loader.get_data_summary()
        print(f"\n数据摘要: {summary}")
        
        # 测试获取对齐数据
        if 'classification' in loader.labels:
            features, labels = loader.get_aligned_data('classification')
            print(f"\n分类任务数据对齐完成: {features.shape}, {labels.shape}")
        
        if 'regression' in loader.labels:
            features, labels = loader.get_aligned_data('regression')
            print(f"回归任务数据对齐完成: {features.shape}, {labels.shape}")
        
        print("\n✅ 数据加载模块测试成功!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()