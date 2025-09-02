"""
认知数据集类
实现PyTorch Dataset接口，用于ADHD项目的数据加载和处理
"""
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List, Union


class CognitiveDataset(Dataset):
    """
    认知数据集类
    支持分类和回归任务的数据加载
    """
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 split: str = "train",
                 task_type: str = "classification",
                 target_col: str = None,
                 exclude_cols: List[str] = None,
                 test_size: float = 0.2,
                 val_size: float = 0.1,
                 random_state: int = 42):
        """
        初始化数据集
        
        Args:
            data: 完整的数据框
            split: 数据分割类型 ("train", "val", "test")
            task_type: 任务类型 ("classification", "regression")  
            target_col: 目标变量列名，如果为None则自动选择
            exclude_cols: 需要排除的列名列表
            test_size: 测试集比例
            val_size: 验证集比例
            random_state: 随机种子
        """
        self.task_type = task_type
        self.split = split
        self.random_state = random_state
        
        # 自动选择目标变量
        if target_col is None:
            if task_type == "classification":
                target_col = "cognitive_impairment_label"
            else:  # regression
                target_col = "global_cognitive_z_score"
        
        self.target_col = target_col
        
        # 检查目标列是否存在
        if target_col not in data.columns:
            raise ValueError(f"目标列 '{target_col}' 不存在于数据中")
        
        # 设置默认排除列
        if exclude_cols is None:
            exclude_cols = []
        
        # 始终排除两个认知相关列中的另一个
        cognitive_cols = ["cognitive_impairment_label", "global_cognitive_z_score"]
        for col in cognitive_cols:
            if col != target_col and col not in exclude_cols:
                exclude_cols.append(col)
        
        self.exclude_cols = exclude_cols
        
        # 准备特征和目标变量
        self.features, self.targets = self._prepare_data(data)
        
        # 数据分割
        self.X, self.y = self._split_data(test_size, val_size)
        
        print(f"{split.upper()} dataset initialized:")
        print(f"  - Task type: {task_type}")
        print(f"  - Target column: {target_col}")
        print(f"  - Samples: {len(self.X)}")
        print(f"  - Features: {self.X.shape[1]}")
        print(f"  - Target distribution: {self._get_target_stats()}")
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备特征和目标数据
        
        Args:
            data: 原始数据框
            
        Returns:
            特征数组和目标数组
        """
        # 移除缺失的目标值
        data_clean = data[data[self.target_col].notna()].copy()
        print(f"移除目标值缺失后剩余样本数: {len(data_clean)}")
        
        # 分离特征和目标
        feature_cols = [col for col in data_clean.columns 
                       if col != self.target_col and col not in self.exclude_cols]
        
        X = data_clean[feature_cols].values
        y = data_clean[self.target_col].values
        
        # 处理特征中的缺失值（用均值填充）
        from sklearn.impute import SimpleImputer
        if np.isnan(X).any():
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)
            print("已用均值填充特征中的缺失值")
        
        print(f"特征维度: {X.shape}")
        print(f"目标维度: {y.shape}")
        
        return X, y
    
    def _split_data(self, test_size: float, val_size: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        分割数据
        
        Args:
            test_size: 测试集比例
            val_size: 验证集比例
            
        Returns:
            对应分割的特征和目标数组
        """
        # 首先分离出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.features, self.targets,
            test_size=test_size,
            random_state=self.random_state,
            stratify=self.targets if self.task_type == "classification" else None
        )
        
        # 然后从剩余数据中分离训练集和验证集
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=y_temp if self.task_type == "classification" else None
        )
        
        # 根据split参数返回对应的数据
        if self.split == "train":
            return X_train, y_train
        elif self.split == "val":
            return X_val, y_val
        elif self.split == "test":
            return X_test, y_test
        else:
            raise ValueError(f"不支持的split类型: {self.split}")
    
    def _get_target_stats(self) -> str:
        """
        获取目标变量的统计信息
        
        Returns:
            统计信息字符串
        """
        if self.task_type == "classification":
            unique, counts = np.unique(self.y, return_counts=True)
            return f"Class distribution: {dict(zip(unique, counts))}"
        else:
            return f"Mean: {self.y.mean():.3f}, Std: {self.y.std():.3f}, Range: [{self.y.min():.3f}, {self.y.max():.3f}]"
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            特征张量和目标张量
        """
        features = torch.FloatTensor(self.X[idx])
        
        if self.task_type == "classification":
            target = torch.LongTensor([self.y[idx]])[0]  # 分类任务使用LongTensor
        else:
            target = torch.FloatTensor([self.y[idx]])[0]  # 回归任务使用FloatTensor
        
        return features, target
    
    def get_feature_dim(self) -> int:
        """返回特征维度"""
        return self.X.shape[1]
    
    def get_num_classes(self) -> int:
        """返回类别数（仅适用于分类任务）"""
        if self.task_type != "classification":
            raise ValueError("get_num_classes仅适用于分类任务")
        return len(np.unique(self.y))
    
    def get_class_weights(self) -> torch.Tensor:
        """
        计算类别权重（用于处理类别不平衡）
        
        Returns:
            类别权重张量
        """
        if self.task_type != "classification":
            raise ValueError("get_class_weights仅适用于分类任务")
        
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(self.y)
        weights = compute_class_weight('balanced', classes=classes, y=self.y)
        
        return torch.FloatTensor(weights)
    
    def get_dataset_info(self) -> dict:
        """
        获取数据集详细信息
        
        Returns:
            数据集信息字典
        """
        info = {
            'split': self.split,
            'task_type': self.task_type,
            'target_column': self.target_col,
            'num_samples': len(self.X),
            'num_features': self.X.shape[1],
            'target_stats': self._get_target_stats()
        }
        
        if self.task_type == "classification":
            info['num_classes'] = self.get_num_classes()
            info['class_weights'] = self.get_class_weights().tolist()
        
        return info