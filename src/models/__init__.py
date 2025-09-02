"""
模型模块
包含分类和回归模型定义
"""
from .classification import ClassificationModel
from .regression import RegressionModel

__all__ = ['ClassificationModel', 'RegressionModel']