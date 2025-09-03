"""
传统机器学习模型库
实现XGBoost、SVM、随机森林、线性回归等模型
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.base import BaseEstimator
import joblib
import os

# 尝试导入XGBoost，如果未安装则提供警告
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Please install with: pip install xgboost")


class TraditionalMLWrapper:
    """
    传统机器学习模型包装器
    提供统一的训练和预测接口
    """
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        初始化传统ML模型
        
        Args:
            model_name: 模型名称
            config: 模型配置
        """
        self.model_name = model_name
        self.config = config
        self.task_type = config.get('task_type', 'classification')
        self.model = self._create_model()
        self.is_fitted = False
        
        print(f"Initialized {model_name} for {self.task_type}")
    
    def _create_model(self) -> BaseEstimator:
        """
        根据配置创建模型
        
        Returns:
            sklearn模型实例
        """
        hyperparams = self.config.get('hyperparameters', {})
        
        if self.model_name == 'xgboost':
            return self._create_xgboost_model(hyperparams)
        elif self.model_name == 'svm':
            return self._create_svm_model(hyperparams)
        elif self.model_name == 'random_forest':
            return self._create_random_forest_model(hyperparams)
        elif self.model_name == 'linear':
            return self._create_linear_model(hyperparams)
        elif self.model_name == 'gradient_boosting':
            return self._create_gradient_boosting_model(hyperparams)
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")
    
    def _create_xgboost_model(self, hyperparams: Dict[str, Any]) -> BaseEstimator:
        """创建XGBoost模型"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available")
        
        params = {
            'n_estimators': hyperparams.get('n_estimators', 100),
            'max_depth': hyperparams.get('max_depth', 6),
            'learning_rate': hyperparams.get('learning_rate', 0.1),
            'subsample': hyperparams.get('subsample', 0.8),
            'colsample_bytree': hyperparams.get('colsample_bytree', 0.8),
            'reg_alpha': hyperparams.get('reg_alpha', 0),
            'reg_lambda': hyperparams.get('reg_lambda', 1),
            'random_state': hyperparams.get('random_state', 42),
            'n_jobs': -1
        }
        
        if self.task_type == 'classification':
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'logloss'
            return xgb.XGBClassifier(**params)
        else:
            params['objective'] = 'reg:squarederror'
            params['eval_metric'] = 'rmse'
            return xgb.XGBRegressor(**params)
    
    def _create_svm_model(self, hyperparams: Dict[str, Any]) -> BaseEstimator:
        """创建SVM模型"""
        base_params = {
            'C': hyperparams.get('C', 1.0),
            'kernel': hyperparams.get('kernel', 'rbf'),
            'gamma': hyperparams.get('gamma', 'scale'),
            'degree': hyperparams.get('degree', 3)
        }
        
        if self.task_type == 'classification':
            base_params['probability'] = True  # 启用概率预测
            base_params['random_state'] = hyperparams.get('random_state', 42)
            return SVC(**base_params)
        else:
            # SVR不支持random_state参数
            return SVR(**base_params)
    
    def _create_random_forest_model(self, hyperparams: Dict[str, Any]) -> BaseEstimator:
        """创建随机森林模型"""
        params = {
            'n_estimators': hyperparams.get('n_estimators', 100),
            'max_depth': hyperparams.get('max_depth', None),
            'min_samples_split': hyperparams.get('min_samples_split', 2),
            'min_samples_leaf': hyperparams.get('min_samples_leaf', 1),
            'max_features': hyperparams.get('max_features', 'sqrt'),
            'bootstrap': hyperparams.get('bootstrap', True),
            'random_state': hyperparams.get('random_state', 42),
            'n_jobs': -1
        }
        
        if self.task_type == 'classification':
            return RandomForestClassifier(**params)
        else:
            return RandomForestRegressor(**params)
    
    def _create_linear_model(self, hyperparams: Dict[str, Any]) -> BaseEstimator:
        """创建线性模型"""
        if self.task_type == 'classification':
            params = {
                'C': hyperparams.get('C', 1.0),
                'penalty': hyperparams.get('penalty', 'l2'),
                'solver': hyperparams.get('solver', 'lbfgs'),
                'max_iter': hyperparams.get('max_iter', 1000),
                'random_state': hyperparams.get('random_state', 42)
            }
            return LogisticRegression(**params)
        else:
            # 使用Ridge回归而不是普通线性回归以增加正则化
            if hyperparams.get('alpha', 1.0) > 0:
                params = {
                    'alpha': hyperparams.get('alpha', 1.0),
                    'fit_intercept': hyperparams.get('fit_intercept', True),
                    'random_state': hyperparams.get('random_state', 42)
                }
                return Ridge(**params)
            else:
                params = {
                    'fit_intercept': hyperparams.get('fit_intercept', True),
                }
                return LinearRegression(**params)
    
    def _create_gradient_boosting_model(self, hyperparams: Dict[str, Any]) -> BaseEstimator:
        """创建梯度提升模型"""
        params = {
            'n_estimators': hyperparams.get('n_estimators', 100),
            'learning_rate': hyperparams.get('learning_rate', 0.1),
            'max_depth': hyperparams.get('max_depth', 3),
            'min_samples_split': hyperparams.get('min_samples_split', 2),
            'min_samples_leaf': hyperparams.get('min_samples_leaf', 1),
            'subsample': hyperparams.get('subsample', 1.0),
            'max_features': hyperparams.get('max_features', None),
            'random_state': hyperparams.get('random_state', 42)
        }
        
        if self.task_type == 'classification':
            return GradientBoostingClassifier(**params)
        else:
            return GradientBoostingRegressor(**params)
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            X: 训练特征
            y: 训练标签
            X_val: 验证特征（可选）
            y_val: 验证标签（可选）
            
        Returns:
            训练结果字典
        """
        print(f"Training {self.model_name} on {X.shape[0]} samples with {X.shape[1]} features")
        
        # XGBoost支持验证集
        if self.model_name == 'xgboost' and X_val is not None and y_val is not None:
            eval_set = [(X, y), (X_val, y_val)]
            self.model.fit(
                X, y, 
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X, y)
        
        self.is_fitted = True
        
        # 计算训练集性能
        train_score = self.model.score(X, y)
        result = {
            'train_score': train_score,
            'model_params': self.model.get_params(),
            'feature_count': X.shape[1]
        }
        
        # 计算验证集性能
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            result['val_score'] = val_score
        
        print(f"Training completed. Train score: {train_score:.4f}")
        return result
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 输入特征
            
        Returns:
            预测结果
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        预测概率（仅分类任务）
        
        Args:
            X: 输入特征
            
        Returns:
            预测概率，如果模型不支持则返回None
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if self.task_type != 'classification':
            return None
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            return None
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        获取特征重要性
        
        Returns:
            特征重要性数组，如果模型不支持则返回None
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # 对于线性模型，使用系数的绝对值作为特征重要性
            coef = self.model.coef_
            if len(coef.shape) > 1:
                coef = coef.flatten()
            return np.abs(coef)
        else:
            return None
    
    def save_model(self, filepath: str) -> None:
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_dict = {
            'model': self.model,
            'model_name': self.model_name,
            'config': self.config,
            'task_type': self.task_type,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(save_dict, filepath)
        print(f"Model saved to: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'TraditionalMLWrapper':
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
            
        Returns:
            加载的模型实例
        """
        save_dict = joblib.load(filepath)
        
        # 创建新实例
        instance = cls(save_dict['model_name'], save_dict['config'])
        instance.model = save_dict['model']
        instance.task_type = save_dict['task_type']
        instance.is_fitted = save_dict['is_fitted']
        
        print(f"Model loaded from: {filepath}")
        return instance
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        info = {
            'model_name': self.model_name,
            'task_type': self.task_type,
            'is_fitted': self.is_fitted,
            'config': self.config
        }
        
        if self.is_fitted:
            # 添加模型参数
            info['model_params'] = self.model.get_params()
            
            # 添加特征重要性信息
            feature_importance = self.get_feature_importance()
            if feature_importance is not None:
                info['has_feature_importance'] = True
                info['feature_importance_shape'] = feature_importance.shape
            else:
                info['has_feature_importance'] = False
        
        return info


def create_traditional_ml_model(model_name: str, config: Dict[str, Any]) -> TraditionalMLWrapper:
    """
    创建传统机器学习模型
    
    Args:
        model_name: 模型名称
        config: 模型配置
        
    Returns:
        模型包装器实例
    """
    return TraditionalMLWrapper(model_name, config)


if __name__ == "__main__":
    # 测试代码
    import numpy as np
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    
    # 测试分类任务
    print("=== Testing Classification Models ===")
    X_clf, y_clf = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )
    
    clf_models = ['random_forest', 'svm', 'linear', 'gradient_boosting']
    if XGBOOST_AVAILABLE:
        clf_models.append('xgboost')
    
    for model_name in clf_models:
        print(f"\n--- Testing {model_name} (Classification) ---")
        config = {
            'task_type': 'classification',
            'hyperparameters': {'random_state': 42}
        }
        
        try:
            model = create_traditional_ml_model(model_name, config)
            result = model.fit(X_train_clf, y_train_clf, X_test_clf, y_test_clf)
            predictions = model.predict(X_test_clf)
            proba = model.predict_proba(X_test_clf)
            
            print(f"  Train score: {result['train_score']:.4f}")
            if 'val_score' in result:
                print(f"  Val score: {result['val_score']:.4f}")
            print(f"  Predictions shape: {predictions.shape}")
            if proba is not None:
                print(f"  Probabilities shape: {proba.shape}")
            
            # 测试特征重要性
            importance = model.get_feature_importance()
            if importance is not None:
                print(f"  Feature importance shape: {importance.shape}")
                print(f"  Top 3 important features: {np.argsort(importance)[-3:]}")
        
        except Exception as e:
            print(f"  Error: {e}")
    
    # 测试回归任务
    print("\n=== Testing Regression Models ===")
    X_reg, y_reg = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    reg_models = ['random_forest', 'svm', 'linear', 'gradient_boosting']
    if XGBOOST_AVAILABLE:
        reg_models.append('xgboost')
    
    for model_name in reg_models:
        print(f"\n--- Testing {model_name} (Regression) ---")
        config = {
            'task_type': 'regression',
            'hyperparameters': {'random_state': 42}
        }
        
        try:
            model = create_traditional_ml_model(model_name, config)
            result = model.fit(X_train_reg, y_train_reg, X_test_reg, y_test_reg)
            predictions = model.predict(X_test_reg)
            
            print(f"  Train score: {result['train_score']:.4f}")
            if 'val_score' in result:
                print(f"  Val score: {result['val_score']:.4f}")
            print(f"  Predictions shape: {predictions.shape}")
            
            # 测试特征重要性
            importance = model.get_feature_importance()
            if importance is not None:
                print(f"  Feature importance shape: {importance.shape}")
        
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n=== All tests completed ===")