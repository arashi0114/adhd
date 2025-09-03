"""
数据预处理模块
实现多重插补、异常值处理、标准化等预处理功能
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    数据预处理器
    支持多重插补、异常值处理、标准化等功能
    """
    
    def __init__(self):
        self.imputers = []  # 多重插补器列表
        self.scalers = {}   # 标准化器字典
        self.outlier_bounds = {}  # 异常值边界
        self.feature_stats = {}   # 特征统计信息
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, 
            imputation_iterations: int = 5,
            outlier_method: str = 'iqr',
            scaling_method: str = 'standard',
            outlier_threshold: float = 1.5) -> 'DataPreprocessor':
        """
        拟合预处理器
        
        Args:
            X: 特征数据
            imputation_iterations: 多重插补迭代次数
            outlier_method: 异常值检测方法 ('iqr', 'z_score', 'isolation_forest')
            scaling_method: 标准化方法 ('standard', 'robust', 'none')
            outlier_threshold: 异常值阈值
            
        Returns:
            拟合后的预处理器
        """
        print("=== 开始拟合数据预处理器 ===")
        print(f"原始数据形状: {X.shape}")
        
        X_copy = X.copy()
        
        # 1. 计算特征统计信息
        self._compute_feature_stats(X_copy)
        
        # 2. 拟合异常值检测
        self._fit_outlier_detection(X_copy, method=outlier_method, threshold=outlier_threshold)
        
        # 3. 拟合多重插补器
        self._fit_imputers(X_copy, n_imputations=imputation_iterations)
        
        # 4. 拟合标准化器
        self._fit_scaler(X_copy, method=scaling_method)
        
        self.is_fitted = True
        print("✅ 数据预处理器拟合完成")
        return self
    
    def transform(self, X: pd.DataFrame, 
                  handle_outliers: bool = True,
                  impute: bool = True,
                  scale: bool = True) -> pd.DataFrame:
        """
        转换数据
        
        Args:
            X: 输入数据
            handle_outliers: 是否处理异常值
            impute: 是否进行插补
            scale: 是否进行标准化
            
        Returns:
            预处理后的数据
        """
        if not self.is_fitted:
            raise ValueError("预处理器未拟合，请先调用fit()方法")
        
        print("=== 开始数据预处理转换 ===")
        print(f"输入数据形状: {X.shape}")
        
        X_processed = X.copy()
        
        # 1. 异常值处理
        if handle_outliers:
            X_processed = self._handle_outliers(X_processed)
            print(f"异常值处理后形状: {X_processed.shape}")
        
        # 2. 多重插补
        if impute:
            X_processed = self._impute_missing_values(X_processed)
            print(f"插补后形状: {X_processed.shape}")
        
        # 3. 标准化
        if scale:
            X_processed = self._scale_features(X_processed)
            print(f"标准化后形状: {X_processed.shape}")
        
        print("✅ 数据预处理转换完成")
        return X_processed
    
    def fit_transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """拟合并转换数据"""
        return self.fit(X, **kwargs).transform(X)
    
    def _compute_feature_stats(self, X: pd.DataFrame):
        """计算特征统计信息"""
        print("计算特征统计信息...")
        
        self.feature_stats = {
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'missing_rates': X.isnull().sum() / len(X),
            'dtypes': X.dtypes,
            'numeric_features': X.select_dtypes(include=[np.number]).columns.tolist()
        }
        
        # 计算数值特征的统计信息
        numeric_X = X.select_dtypes(include=[np.number])
        if not numeric_X.empty:
            self.feature_stats.update({
                'means': numeric_X.mean(),
                'stds': numeric_X.std(),
                'mins': numeric_X.min(),
                'maxs': numeric_X.max(),
                'q25': numeric_X.quantile(0.25),
                'q75': numeric_X.quantile(0.75)
            })
        
        missing_count = X.isnull().sum().sum()
        print(f"  - 数值特征数: {len(self.feature_stats['numeric_features'])}")
        print(f"  - 总缺失值: {missing_count} ({missing_count/(X.shape[0]*X.shape[1]):.2%})")
    
    def _fit_outlier_detection(self, X: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5):
        """拟合异常值检测"""
        print(f"拟合异常值检测 (方法: {method})...")
        
        numeric_X = X.select_dtypes(include=[np.number])
        
        if method == 'iqr':
            for col in numeric_X.columns:
                Q1 = numeric_X[col].quantile(0.25)
                Q3 = numeric_X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                self.outlier_bounds[col] = (lower_bound, upper_bound)
        
        elif method == 'z_score':
            for col in numeric_X.columns:
                mean = numeric_X[col].mean()
                std = numeric_X[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
                self.outlier_bounds[col] = (lower_bound, upper_bound)
        
        outlier_features = len(self.outlier_bounds)
        print(f"  - 配置了 {outlier_features} 个特征的异常值检测")
    
    def _fit_imputers(self, X: pd.DataFrame, n_imputations: int = 5):
        """拟合多重插补器"""
        print(f"拟合多重插补器 (插补次数: {n_imputations})...")
        
        numeric_X = X.select_dtypes(include=[np.number])
        
        if numeric_X.empty:
            print("  - 无数值特征，跳过插补")
            return
        
        # 创建多个插补器
        self.imputers = []
        for i in range(n_imputations):
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(n_estimators=10, random_state=42+i),
                max_iter=10,
                random_state=42+i,
                verbose=0
            )
            
            # 拟合插补器
            imputer.fit(numeric_X)
            self.imputers.append(imputer)
        
        print(f"  - 创建了 {len(self.imputers)} 个插补器")
    
    def _fit_scaler(self, X: pd.DataFrame, method: str = 'standard'):
        """拟合标准化器"""
        if method == 'none':
            print("跳过标准化")
            return
            
        print(f"拟合标准化器 (方法: {method})...")
        
        numeric_X = X.select_dtypes(include=[np.number])
        
        if numeric_X.empty:
            print("  - 无数值特征，跳过标准化")
            return
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
        
        # 拟合标准化器（使用无缺失值的数据）
        complete_data = numeric_X.dropna()
        if not complete_data.empty:
            scaler.fit(complete_data)
            self.scalers['main'] = scaler
            print(f"  - 使用 {len(complete_data)} 个完整样本拟合标准化器")
        else:
            print("  - 警告: 无完整样本，将在插补后再拟合标准化器")
    
    def _handle_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """处理异常值"""
        if not self.outlier_bounds:
            return X
        
        X_clean = X.copy()
        outlier_count = 0
        
        for col, (lower, upper) in self.outlier_bounds.items():
            if col in X_clean.columns:
                # 将异常值设为NaN，后续通过插补处理
                outliers = (X_clean[col] < lower) | (X_clean[col] > upper)
                outlier_count += outliers.sum()
                X_clean.loc[outliers, col] = np.nan
        
        print(f"  - 处理了 {outlier_count} 个异常值")
        return X_clean
    
    def _impute_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """多重插补缺失值"""
        if not self.imputers:
            return X
        
        numeric_X = X.select_dtypes(include=[np.number])
        non_numeric_X = X.select_dtypes(exclude=[np.number])
        
        if numeric_X.empty:
            return X
        
        # 多重插补并取平均
        imputed_arrays = []
        for imputer in self.imputers:
            imputed = imputer.transform(numeric_X)
            imputed_arrays.append(imputed)
        
        # 对多次插补结果取平均
        mean_imputed = np.mean(imputed_arrays, axis=0)
        
        # 重建DataFrame
        imputed_df = pd.DataFrame(
            mean_imputed, 
            columns=numeric_X.columns, 
            index=numeric_X.index
        )
        
        # 合并非数值列
        if not non_numeric_X.empty:
            result = pd.concat([imputed_df, non_numeric_X], axis=1)
            # 保持原始列顺序
            result = result[X.columns]
        else:
            result = imputed_df
        
        missing_before = numeric_X.isnull().sum().sum()
        missing_after = imputed_df.isnull().sum().sum()
        print(f"  - 插补前缺失值: {missing_before}, 插补后缺失值: {missing_after}")
        
        return result
    
    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """标准化特征"""
        if 'main' not in self.scalers:
            # 如果之前没有拟合标准化器，现在拟合
            numeric_X = X.select_dtypes(include=[np.number])
            if not numeric_X.empty:
                scaler = StandardScaler()
                scaler.fit(numeric_X)
                self.scalers['main'] = scaler
            else:
                return X
        
        scaler = self.scalers['main']
        
        numeric_X = X.select_dtypes(include=[np.number])
        non_numeric_X = X.select_dtypes(exclude=[np.number])
        
        if numeric_X.empty:
            return X
        
        # 标准化数值特征
        scaled_array = scaler.transform(numeric_X)
        scaled_df = pd.DataFrame(
            scaled_array,
            columns=numeric_X.columns,
            index=numeric_X.index
        )
        
        # 合并非数值列
        if not non_numeric_X.empty:
            result = pd.concat([scaled_df, non_numeric_X], axis=1)
            result = result[X.columns]  # 保持原始列顺序
        else:
            result = scaled_df
        
        print(f"  - 标准化了 {len(numeric_X.columns)} 个数值特征")
        return result
    
    def get_preprocessing_report(self) -> Dict[str, Any]:
        """获取预处理报告"""
        if not self.is_fitted:
            return {"status": "未拟合"}
        
        report = {
            "fitted": True,
            "feature_count": self.feature_stats.get('n_features', 0),
            "sample_count": self.feature_stats.get('n_samples', 0),
            "numeric_features": len(self.feature_stats.get('numeric_features', [])),
            "imputers_count": len(self.imputers),
            "outlier_detection_features": len(self.outlier_bounds),
            "scaling_enabled": 'main' in self.scalers
        }
        
        # 添加缺失值统计
        if 'missing_rates' in self.feature_stats:
            missing_rates = self.feature_stats['missing_rates']
            report['missing_value_stats'] = {
                "features_with_missing": (missing_rates > 0).sum(),
                "max_missing_rate": missing_rates.max(),
                "mean_missing_rate": missing_rates.mean()
            }
        
        return report


def create_preprocessor(imputation_iterations: int = 5,
                       outlier_method: str = 'iqr',
                       scaling_method: str = 'standard') -> DataPreprocessor:
    """
    便捷函数：创建预处理器
    
    Args:
        imputation_iterations: 插补迭代次数
        outlier_method: 异常值检测方法
        scaling_method: 标准化方法
        
    Returns:
        预处理器实例
    """
    preprocessor = DataPreprocessor()
    return preprocessor


if __name__ == "__main__":
    # 测试代码
    print("测试数据预处理模块...")
    
    # 创建测试数据
    np.random.seed(42)
    n_samples, n_features = 1000, 20
    
    # 生成基础数据
    X_test = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # 添加缺失值
    missing_mask = np.random.random((n_samples, n_features)) < 0.1
    X_test = X_test.mask(missing_mask)
    
    # 添加一些异常值
    for col in X_test.columns[:5]:
        outlier_idx = np.random.choice(n_samples, 10, replace=False)
        X_test.loc[outlier_idx, col] = np.random.randn(10) * 10
    
    print(f"测试数据形状: {X_test.shape}")
    print(f"缺失值数量: {X_test.isnull().sum().sum()}")
    
    # 测试预处理
    preprocessor = create_preprocessor(
        imputation_iterations=3,
        outlier_method='iqr',
        scaling_method='standard'
    )
    
    X_processed = preprocessor.fit_transform(X_test)
    
    print(f"\n处理后数据形状: {X_processed.shape}")
    print(f"处理后缺失值: {X_processed.isnull().sum().sum()}")
    print(f"处理后数据范围: [{X_processed.min().min():.3f}, {X_processed.max().max():.3f}]")
    
    # 生成报告
    report = preprocessor.get_preprocessing_report()
    print(f"\n预处理报告: {report}")
    
    print("\n✅ 数据预处理模块测试成功!")