"""
数据清洗和缺失值处理模块
包含缺失值分析、多重插补、数据标准化和异常值处理功能
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def analyze_missing_values(df, show_plots=True, save_path=None):
    """
    分析数据集的缺失值情况
    
    Args:
        df: 数据框
        show_plots: 是否显示可视化图表
        save_path: 保存图表的路径
        
    Returns:
        dict: 缺失值分析结果
    """
    print("=== 缺失值分析 ===")
    
    # 基本统计
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()
    missing_percentage = (missing_cells / total_cells) * 100
    
    print(f"数据集形状: {df.shape}")
    print(f"总单元格数: {total_cells}")
    print(f"缺失单元格数: {missing_cells}")
    print(f"总缺失率: {missing_percentage:.2f}%")
    
    # 按变量统计缺失值
    missing_by_var = df.isna().sum()
    missing_pct_by_var = (missing_by_var / len(df)) * 100
    
    # 按样本统计缺失值
    missing_by_sample = df.isna().sum(axis=1)
    missing_pct_by_sample = (missing_by_sample / df.shape[1]) * 100
    
    # 统计结果
    var_missing_stats = {
        'no_missing': (missing_by_var == 0).sum(),
        'less_than_5pct': ((missing_pct_by_var > 0) & (missing_pct_by_var < 5)).sum(),
        'between_5_20pct': ((missing_pct_by_var >= 5) & (missing_pct_by_var < 20)).sum(),
        'between_20_50pct': ((missing_pct_by_var >= 20) & (missing_pct_by_var < 50)).sum(),
        'more_than_50pct': (missing_pct_by_var >= 50).sum()
    }
    
    sample_missing_stats = {
        'no_missing': (missing_by_sample == 0).sum(),
        'less_than_10pct': ((missing_pct_by_sample > 0) & (missing_pct_by_sample < 10)).sum(),
        'between_10_30pct': ((missing_pct_by_sample >= 10) & (missing_pct_by_sample < 30)).sum(),
        'between_30_70pct': ((missing_pct_by_sample >= 30) & (missing_pct_by_sample < 70)).sum(),
        'more_than_70pct': (missing_pct_by_sample >= 70).sum()
    }
    
    print(f"\n变量缺失值分布:")
    for category, count in var_missing_stats.items():
        print(f"  {category}: {count} 个变量")
    
    print(f"\n样本缺失值分布:")
    for category, count in sample_missing_stats.items():
        print(f"  {category}: {count} 个样本")
    
    # 找出缺失值最多的变量和样本
    most_missing_vars = missing_pct_by_var.nlargest(10)
    most_missing_samples_idx = missing_pct_by_sample.nlargest(10).index
    
    print(f"\n缺失值最多的10个变量:")
    for var, pct in most_missing_vars.items():
        print(f"  {var}: {pct:.1f}%")
    
    # 可视化
    if show_plots:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 变量缺失值分布
        missing_pct_by_var.hist(bins=50, ax=axes[0,0])
        axes[0,0].set_title('变量缺失值百分比分布')
        axes[0,0].set_xlabel('缺失值百分比')
        axes[0,0].set_ylabel('变量数量')
        
        # 样本缺失值分布
        missing_pct_by_sample.hist(bins=50, ax=axes[0,1])
        axes[0,1].set_title('样本缺失值百分比分布')
        axes[0,1].set_xlabel('缺失值百分比')
        axes[0,1].set_ylabel('样本数量')
        
        # 缺失值模式（前20个变量）
        if df.shape[1] <= 50:
            df_subset = df.iloc[:, :min(50, df.shape[1])]
        else:
            # 选择缺失值最多的50个变量
            top_missing_vars = missing_pct_by_var.nlargest(50).index
            df_subset = df[top_missing_vars]
            
        missing_matrix = df_subset.isna().astype(int)
        sns.heatmap(missing_matrix.iloc[:100, :20], 
                   cbar=True, ax=axes[1,0], cmap='viridis')
        axes[1,0].set_title('缺失值模式（前100个样本，前20个变量）')
        axes[1,0].set_xlabel('变量')
        axes[1,0].set_ylabel('样本')
        
        # 缺失值最多的变量
        most_missing_vars.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('缺失值最多的10个变量')
        axes[1,1].set_xlabel('变量')
        axes[1,1].set_ylabel('缺失值百分比')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"缺失值分析图表已保存至: {save_path}")
        
        if show_plots:
            plt.show()
    
    # 返回分析结果
    analysis_result = {
        'total_missing_rate': missing_percentage,
        'var_missing_stats': var_missing_stats,
        'sample_missing_stats': sample_missing_stats,
        'most_missing_vars': most_missing_vars.to_dict(),
        'missing_by_var': missing_pct_by_var,
        'missing_by_sample': missing_pct_by_sample
    }
    
    return analysis_result


def multiple_imputation(df, n_estimators=10, max_iter=10, random_state=42, verbose=True):
    """
    使用多重插补处理缺失值
    
    Args:
        df: 包含缺失值的数据框
        n_estimators: 随机森林估计器数量
        max_iter: 最大迭代次数
        random_state: 随机种子
        verbose: 是否显示详细信息
        
    Returns:
        pd.DataFrame: 插补后的数据框
    """
    print("=== 开始多重插补 ===")
    
    # 分离数值型和非数值型变量
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"数值型变量: {len(numeric_cols)} 个")
    print(f"非数值型变量: {len(non_numeric_cols)} 个")
    
    if len(numeric_cols) == 0:
        print("警告: 没有找到数值型变量，跳过插补")
        return df.copy()
    
    # 记录插补前的统计信息
    original_missing = df[numeric_cols].isna().sum().sum()
    total_cells = len(df) * len(numeric_cols)
    missing_rate = (original_missing / total_cells) * 100
    
    print(f"插补前缺失值统计:")
    print(f"- 总缺失值: {original_missing}")
    print(f"- 缺失率: {missing_rate:.2f}%")
    
    # 仅对数值型变量进行插补
    df_numeric = df[numeric_cols].copy()
    
    # 使用IterativeImputer进行多重插补
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=n_estimators, random_state=random_state),
        max_iter=max_iter,
        random_state=random_state,
        verbose=2 if verbose else 0
    )
    
    print("正在执行多重插补...")
    df_imputed_numeric = pd.DataFrame(
        imputer.fit_transform(df_numeric),
        columns=numeric_cols,
        index=df.index
    )
    
    # 将插补结果与非数值型变量合并
    if non_numeric_cols:
        df_imputed = pd.concat([df_imputed_numeric, df[non_numeric_cols]], axis=1)
        # 恢复原始列顺序
        df_imputed = df_imputed[df.columns]
    else:
        df_imputed = df_imputed_numeric
    
    # 验证插补效果
    final_missing = df_imputed.isna().sum().sum()
    imputation_success_rate = ((original_missing - final_missing) / original_missing) * 100 if original_missing > 0 else 100
    
    print(f"\n插补完成统计:")
    print(f"- 插补前缺失值: {original_missing}")
    print(f"- 插补后缺失值: {final_missing}")
    print(f"- 插补成功率: {imputation_success_rate:.2f}%")
    
    # 检查插补后的数据质量
    print(f"\n数据质量检查:")
    print(f"- 数据框形状: {df_imputed.shape}")
    print(f"- 是否包含无穷值: {np.isinf(df_imputed.select_dtypes(include=[np.number])).any().any()}")
    
    return df_imputed


def standardize_data(df, exclude_cols=None, method='zscore'):
    """
    标准化数据
    
    Args:
        df: 数据框
        exclude_cols: 不需要标准化的列
        method: 标准化方法 ('zscore', 'minmax')
        
    Returns:
        tuple: (标准化后的数据框, 标准化器对象)
    """
    print(f"=== 数据标准化 (方法: {method}) ===")
    
    if exclude_cols is None:
        exclude_cols = []
    
    # 识别数值型变量
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_standardize = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"需要标准化的变量数: {len(cols_to_standardize)}")
    print(f"排除的变量数: {len(exclude_cols)}")
    
    if not cols_to_standardize:
        print("警告: 没有需要标准化的数值型变量")
        return df.copy(), None
    
    # 创建标准化器
    if method == 'zscore':
        scaler = StandardScaler()
    else:
        raise ValueError(f"不支持的标准化方法: {method}")
    
    # 应用标准化
    df_standardized = df.copy()
    standardized_data = scaler.fit_transform(df[cols_to_standardize])
    df_standardized[cols_to_standardize] = standardized_data
    
    # 验证标准化效果
    print(f"\n标准化效果验证:")
    for col in cols_to_standardize[:5]:  # 只检查前5个变量
        mean_val = df_standardized[col].mean()
        std_val = df_standardized[col].std()
        print(f"  {col}: mean={mean_val:.6f}, std={std_val:.6f}")
    
    return df_standardized, scaler


def detect_outliers(df, method='iqr', threshold=1.5, z_threshold=3):
    """
    检测异常值
    
    Args:
        df: 数据框
        method: 检测方法 ('iqr', 'zscore', 'both')
        threshold: IQR方法的倍数阈值
        z_threshold: Z-score方法的阈值
        
    Returns:
        dict: 异常值检测结果
    """
    print(f"=== 异常值检测 (方法: {method}) ===")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_results = {}
    
    for col in numeric_cols:
        data = df[col].dropna()
        outliers_iqr = []
        outliers_zscore = []
        
        if method in ['iqr', 'both']:
            # IQR方法
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers_iqr = data[(data < lower_bound) | (data > upper_bound)].index.tolist()
        
        if method in ['zscore', 'both']:
            # Z-score方法
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers_zscore = data[z_scores > z_threshold].index.tolist()
        
        # 合并异常值
        if method == 'iqr':
            outliers = outliers_iqr
        elif method == 'zscore':
            outliers = outliers_zscore
        else:  # both
            outliers = list(set(outliers_iqr + outliers_zscore))
        
        outlier_results[col] = {
            'count': len(outliers),
            'percentage': len(outliers) / len(data) * 100,
            'indices': outliers,
            'bounds': {
                'iqr_lower': Q1 - threshold * IQR if method in ['iqr', 'both'] else None,
                'iqr_upper': Q3 + threshold * IQR if method in ['iqr', 'both'] else None,
                'z_threshold': z_threshold if method in ['zscore', 'both'] else None
            }
        }
    
    # 统计总体情况
    total_outliers = sum(result['count'] for result in outlier_results.values())
    vars_with_outliers = sum(1 for result in outlier_results.values() if result['count'] > 0)
    
    print(f"异常值检测结果:")
    print(f"- 检测的变量数: {len(numeric_cols)}")
    print(f"- 有异常值的变量数: {vars_with_outliers}")
    print(f"- 总异常值数: {total_outliers}")
    
    # 显示异常值最多的前10个变量
    sorted_vars = sorted(outlier_results.items(), key=lambda x: x[1]['count'], reverse=True)
    print(f"\n异常值最多的前10个变量:")
    for var, result in sorted_vars[:10]:
        print(f"  {var}: {result['count']} ({result['percentage']:.2f}%)")
    
    return outlier_results


def handle_outliers(df, outlier_results, method='clip', percentile_range=(1, 99)):
    """
    处理异常值
    
    Args:
        df: 数据框
        outlier_results: 异常值检测结果
        method: 处理方法 ('clip', 'remove', 'winsorize')
        percentile_range: 百分位数范围（用于winsorize方法）
        
    Returns:
        pd.DataFrame: 处理后的数据框
    """
    print(f"=== 异常值处理 (方法: {method}) ===")
    
    df_processed = df.copy()
    
    if method == 'clip':
        # 截断异常值到合理范围
        for col, result in outlier_results.items():
            if result['count'] > 0 and result['bounds']['iqr_lower'] is not None:
                lower_bound = result['bounds']['iqr_lower']
                upper_bound = result['bounds']['iqr_upper']
                df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
                
    elif method == 'winsorize':
        # 使用百分位数截断
        for col in outlier_results.keys():
            if outlier_results[col]['count'] > 0:
                lower_pct = np.percentile(df[col].dropna(), percentile_range[0])
                upper_pct = np.percentile(df[col].dropna(), percentile_range[1])
                df_processed[col] = df_processed[col].clip(lower=lower_pct, upper=upper_pct)
                
    elif method == 'remove':
        # 删除包含异常值的行（谨慎使用）
        all_outlier_indices = set()
        for result in outlier_results.values():
            all_outlier_indices.update(result['indices'])
        
        original_count = len(df_processed)
        df_processed = df_processed.drop(index=all_outlier_indices)
        removed_count = original_count - len(df_processed)
        
        print(f"删除了 {removed_count} 个包含异常值的样本")
    
    print(f"异常值处理完成")
    print(f"- 处理前样本数: {len(df)}")
    print(f"- 处理后样本数: {len(df_processed)}")
    
    return df_processed


def validate_data_quality(df_original, df_processed, critical_cols=None):
    """
    验证数据质量
    
    Args:
        df_original: 原始数据框
        df_processed: 处理后数据框
        critical_cols: 关键列名列表
        
    Returns:
        dict: 数据质量报告
    """
    print("=== 数据质量验证 ===")
    
    report = {}
    
    # 基本形状比较
    report['shape_comparison'] = {
        'original': df_original.shape,
        'processed': df_processed.shape,
        'rows_lost': df_original.shape[0] - df_processed.shape[0],
        'cols_lost': df_original.shape[1] - df_processed.shape[1]
    }
    
    # 缺失值比较
    original_missing = df_original.isna().sum().sum()
    processed_missing = df_processed.isna().sum().sum()
    
    report['missing_values'] = {
        'original': original_missing,
        'processed': processed_missing,
        'reduction': original_missing - processed_missing,
        'reduction_rate': ((original_missing - processed_missing) / original_missing * 100) if original_missing > 0 else 0
    }
    
    # 数据类型一致性
    original_dtypes = df_original.dtypes
    processed_dtypes = df_processed.dtypes
    
    dtype_changes = []
    for col in df_original.columns:
        if col in df_processed.columns:
            if original_dtypes[col] != processed_dtypes[col]:
                dtype_changes.append({
                    'column': col,
                    'original': str(original_dtypes[col]),
                    'processed': str(processed_dtypes[col])
                })
    
    report['dtype_changes'] = dtype_changes
    
    # 关键列检查
    if critical_cols:
        missing_critical_cols = [col for col in critical_cols if col not in df_processed.columns]
        report['critical_cols_status'] = {
            'missing': missing_critical_cols,
            'all_present': len(missing_critical_cols) == 0
        }
    
    # 数值稳定性检查（检查是否有无穷值或极大值）
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    quality_issues = {}
    
    for col in numeric_cols:
        issues = []
        if df_processed[col].isna().any():
            issues.append('has_nan')
        if np.isinf(df_processed[col]).any():
            issues.append('has_inf')
        if (df_processed[col].abs() > 1e10).any():
            issues.append('has_extreme_values')
        
        if issues:
            quality_issues[col] = issues
    
    report['quality_issues'] = quality_issues
    
    # 打印报告
    print(f"数据质量验证结果:")
    print(f"- 原始数据形状: {report['shape_comparison']['original']}")
    print(f"- 处理后数据形状: {report['shape_comparison']['processed']}")
    print(f"- 丢失样本数: {report['shape_comparison']['rows_lost']}")
    print(f"- 缺失值减少: {report['missing_values']['reduction']} ({report['missing_values']['reduction_rate']:.1f}%)")
    
    if dtype_changes:
        print(f"- 数据类型变化: {len(dtype_changes)} 个变量")
    
    if quality_issues:
        print(f"- 数据质量问题: {len(quality_issues)} 个变量存在问题")
    else:
        print(f"- 数据质量: 良好，无明显问题")
    
    return report


def save_processed_data(df, output_path, format='csv', compression=None):
    """
    保存处理后的数据
    
    Args:
        df: 数据框
        output_path: 输出路径
        format: 保存格式 ('csv', 'parquet', 'pickle')
        compression: 压缩方式
    """
    print(f"=== 保存处理后的数据 ===")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'csv':
        df.to_csv(output_path, index=False, compression=compression)
    elif format == 'parquet':
        df.to_parquet(output_path, index=False, compression=compression)
    elif format == 'pickle':
        df.to_pickle(output_path, compression=compression)
    else:
        raise ValueError(f"不支持的保存格式: {format}")
    
    file_size = output_path.stat().st_size / (1024 * 1024)  # MB
    print(f"数据已保存至: {output_path}")
    print(f"文件大小: {file_size:.2f} MB")
    print(f"数据形状: {df.shape}")


def preprocess_data(df, 
                   perform_imputation=True,
                   perform_standardization=True,
                   handle_outliers_flag=True,
                   outlier_method='iqr',
                   exclude_from_standardization=None,
                   output_path=None):
    """
    完整的数据预处理流程
    
    Args:
        df: 原始数据框
        perform_imputation: 是否进行缺失值插补
        perform_standardization: 是否进行标准化
        handle_outliers_flag: 是否处理异常值
        outlier_method: 异常值检测方法
        exclude_from_standardization: 不需要标准化的列
        output_path: 输出路径
        
    Returns:
        tuple: (处理后的数据框, 处理过程信息)
    """
    print("=== 开始数据预处理流程 ===")
    print(f"原始数据形状: {df.shape}")
    
    df_processed = df.copy()
    process_info = {}
    
    # 1. 缺失值分析
    print("\n" + "="*50)
    missing_analysis = analyze_missing_values(df_processed, show_plots=False)
    process_info['missing_analysis'] = missing_analysis
    
    # 2. 缺失值插补
    if perform_imputation:
        print("\n" + "="*50)
        df_processed = multiple_imputation(df_processed)
        process_info['imputation_performed'] = True
    else:
        print("\n跳过缺失值插补")
        process_info['imputation_performed'] = False
    
    # 3. 异常值检测和处理
    if handle_outliers_flag:
        print("\n" + "="*50)
        outlier_results = detect_outliers(df_processed, method=outlier_method)
        df_processed = handle_outliers(df_processed, outlier_results, method='clip')
        process_info['outlier_detection'] = outlier_results
        process_info['outlier_handling_performed'] = True
    else:
        print("\n跳过异常值处理")
        process_info['outlier_handling_performed'] = False
    
    # 4. 数据标准化
    if perform_standardization:
        print("\n" + "="*50)
        if exclude_from_standardization is None:
            # 默认不标准化的列（通常是标签列和ID列）
            exclude_from_standardization = ['cognitive_impairment_label', 'global_cognitive_z_score']
        
        df_processed, scaler = standardize_data(df_processed, exclude_cols=exclude_from_standardization)
        process_info['standardization_performed'] = True
        process_info['scaler'] = scaler
    else:
        print("\n跳过数据标准化")
        process_info['standardization_performed'] = False
    
    # 5. 数据质量验证
    print("\n" + "="*50)
    quality_report = validate_data_quality(df, df_processed)
    process_info['quality_report'] = quality_report
    
    # 6. 保存处理后的数据
    if output_path:
        print("\n" + "="*50)
        save_processed_data(df_processed, output_path)
        process_info['output_path'] = output_path
    
    print("\n=== 数据预处理完成 ===")
    print(f"最终数据形状: {df_processed.shape}")
    
    return df_processed, process_info