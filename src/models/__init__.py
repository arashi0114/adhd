"""
模型模块
包含深度学习和传统机器学习模型
"""
from .deep_learning import create_deep_learning_model
from .traditional_ml import create_traditional_ml_model

__all__ = [
    'create_deep_learning_model',
    'create_traditional_ml_model'
]