"""
认知数据集类
实现PyTorch Dataset接口，用于ADHD项目的数据加载和处理
支持新的数据加载器和预处理器
"""
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, List, Union

from .loader import DataLoader
from .preprocessor import DataPreprocessor


class CognitiveDataset(Dataset):
    """
    认知数据集类
    支持分类和回归任务的数据加载
    整合新的数据加载器和预处理器
    """
    
    def __init__(self, 
                 data_loader: DataLoader = None,
                 preprocessor: DataPreprocessor = None,
                 split: str = "train",
                 task_type: str = "classification",
                 test_size: float = 0.1,
                 val_size: float = 0.1,
                 random_state: int = 42,
                 # 向后兼容的参数
                 data: pd.DataFrame = None,
                 target_col: str = None,
                 exclude_cols: List[str] = None):
        """
        初始化数据集
        
        Args:
            data_loader: 数据加载器实例（新方式）
            preprocessor: 数据预处理器实例（可选）
            split: 数据分割类型 ("train", "val", "test")
            task_type: 任务类型 ("classification", "regression")  
            test_size: 测试集比例
            val_size: 验证集比例
            random_state: 随机种子
            data: 完整的数据框（向后兼容）
            target_col: 目标变量列名（向后兼容）
            exclude_cols: 需要排除的列名列表（向后兼容）
        """
        self.task_type = task_type
        self.split = split
        self.random_state = random_state
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        
        # 根据不同的初始化方式准备数据
        if data_loader is not None:
            # 新方式：使用DataLoader
            self._prepare_from_loader()
        elif data is not None:
            # 向后兼容：使用传入的DataFrame
            self._prepare_from_dataframe(data, target_col, exclude_cols)
        else:
            raise ValueError("必须提供data_loader或data参数")
        
        # 数据分割
        self.X, self.y = self._split_data(test_size, val_size)
        
        print(f"{split.upper()} dataset initialized:")
        print(f"  - Task type: {task_type}")
        print(f"  - Samples: {len(self.X)}")
        print(f"  - Features: {self.X.shape[1]}")
        print(f"  - Target distribution: {self._get_target_stats()}")
    
    def _prepare_from_loader(self):
        """从DataLoader准备数据"""
        if self.data_loader is None or not hasattr(self.data_loader, 'labels'):
            raise ValueError("DataLoader未正确加载数据")
        
        # 获取对齐的特征和标签
        features, labels = self.data_loader.get_aligned_data(self.task_type)
        
        # 预处理特征数据
        if self.preprocessor is not None:
            features = self.preprocessor.fit_transform(features)
        else:
            # 简单的预处理：填充缺失值
            features = features.fillna(features.mean())
        
        # 转换为numpy数组
        self.features = features.values.astype(np.float32)
        self.targets = labels.values
        
        print(f"从DataLoader准备数据完成: {self.features.shape}, {self.targets.shape}")
    
    def _prepare_from_dataframe(self, data: pd.DataFrame, target_col: str, exclude_cols: List[str]):
        """从DataFrame准备数据（向后兼容）"""
        # 保存参数用于后续使用
        self.target_col = target_col
        self.exclude_cols = exclude_cols or []
        
        # 自动选择目标变量
        if target_col is None:
            if self.task_type == "classification":
                target_col = "racogimp_label"
            else:  # regression
                regression_cols = [col for col in data.columns if col.endswith('cogimpt_label') and col.startswith('r')]
                if regression_cols:
                    target_col = regression_cols[0]
                else:
                    raise ValueError("未找到回归标签列")
            self.target_col = target_col
        
        # 检查目标列是否存在
        if target_col not in data.columns:
            raise ValueError(f"目标列 '{target_col}' 不存在于数据中")
        
        # 排除所有其他标签列
        import re
        all_label_cols = [col for col in data.columns 
                         if col == "racogimp_label" or re.match(r'r\d+cogimpt_label', col)]
        for col in all_label_cols:
            if col != target_col and col not in self.exclude_cols:
                self.exclude_cols.append(col)
        
        # 准备特征和目标变量
        self.features, self.targets = self._prepare_legacy_data(data)
    
    def _prepare_legacy_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备特征和目标数据（向后兼容版本）
        
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
        
        y = data_clean[self.target_col].values
        
        # 处理标签值
        if self.task_type == "classification" and self.target_col == "racogimp_label":
            # 将二分类标签转换：-1 -> 0 (未发病), 其他 -> 1 (发病)
            y = (y != -1).astype(int)
            print(f"二分类标签转换完成: 阴性={np.sum(y==0)}, 阳性={np.sum(y==1)}")
        
        # 简化的数据预处理：只处理数值型数据
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import LabelEncoder
        
        # 先处理非数值型数据：转换为数值型或删除
        processed_data = data_clean[feature_cols].copy()
        
        # 对每列单独处理
        for col in processed_data.columns:
            if processed_data[col].dtype == 'object':
                # 分类变量：标签编码
                le = LabelEncoder()
                # 先填充缺失值，然后编码
                processed_data[col] = processed_data[col].fillna('Missing')
                processed_data[col] = le.fit_transform(processed_data[col].astype(str))
            else:
                # 数值变量：均值填充
                processed_data[col] = processed_data[col].fillna(processed_data[col].mean())
        
        # 转换为numpy数组
        X = processed_data.values.astype(np.float32)
        
        print(f"处理了 {len(feature_cols)} 个特征")
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


def create_datasets(data_loader: DataLoader, 
                   preprocessor: DataPreprocessor = None,
                   task_type: str = "classification",
                   test_size: float = 0.1,
                   val_size: float = 0.1,
                   random_state: int = 42) -> dict:
    """
    便捷函数：创建训练、验证和测试数据集
    
    Args:
        data_loader: 数据加载器
        preprocessor: 数据预处理器
        task_type: 任务类型
        test_size: 测试集比例
        val_size: 验证集比例
        random_state: 随机种子
        
    Returns:
        包含三个数据集的字典
    """
    datasets = {
        'train': CognitiveDataset(
            data_loader=data_loader,
            preprocessor=preprocessor,
            split="train",
            task_type=task_type,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state
        ),
        'val': CognitiveDataset(
            data_loader=data_loader,
            preprocessor=preprocessor,
            split="val",
            task_type=task_type,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state
        ),
        'test': CognitiveDataset(
            data_loader=data_loader,
            preprocessor=preprocessor,
            split="test",
            task_type=task_type,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state
        )
    }
    
    return datasets