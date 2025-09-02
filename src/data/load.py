"""
数据加载和筛选模块
包含数据加载、变量筛选和样本过滤功能
"""
import pandas as pd
import numpy as np
import re
from pathlib import Path


def load_raw_data(file_path):
    """
    加载原始数据
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        pd.DataFrame: 原始数据
    """
    print(f"正在加载数据: {file_path}")
    df = pd.read_stata(file_path)
    print(f"数据形状: {df.shape}")
    return df


def count_unique_vars(var_list):
    """
    统计变量名种类数量
    
    Args:
        var_list: 变量名列表
        
    Returns:
        tuple: (变量名列表, 变量种类数)
    """
    unique_vars = set()
    
    for var in var_list:
        match = re.match(r'r\d+(.+)', var)
        if match:
            unique_vars.add(match.group(1))
        else:
            unique_vars.add(var)
    
    return sorted(unique_vars), len(unique_vars)


def remove_multiple_ranges(var_list, ranges_to_remove):
    """
    从变量列表中删除多个范围的变量
    
    Args:
        var_list: 原始变量列表
        ranges_to_remove: 要删除的范围列表，每个元素为(start_var, end_var)
        
    Returns:
        list: 过滤后的变量列表
    """
    result = var_list.copy()
    
    for start_var, end_var in ranges_to_remove:
        try:
            start_idx = result.index(start_var)
            end_idx = result.index(end_var)
            # 删除范围内的变量
            for _ in range(end_idx - start_idx + 1):
                result.pop(start_idx)
        except ValueError:
            print(f"警告: 未找到变量 {start_var} 或 {end_var}")
    
    return result


def remove_multiple_ranges_with_keep(var_list, ranges_to_remove, ranges_to_keep):
    """
    从变量列表中删除多个范围的变量，但保留指定变量
    
    Args:
        var_list: 原始变量列表
        ranges_to_remove: 要删除的范围列表
        ranges_to_keep: 要保留的范围列表
        
    Returns:
        list: 过滤后的变量列表
    """
    # 收集要保留的变量
    vars_to_keep = []
    for start_var, end_var in ranges_to_keep:
        try:
            start_idx = var_list.index(start_var)
            end_idx = var_list.index(end_var)
            vars_to_keep.extend(var_list[start_idx:end_idx+1])
        except ValueError:
            print(f"警告: 未找到要保留的变量范围 {start_var} 到 {end_var}")
    
    # 先删除范围
    result = remove_multiple_ranges(var_list, ranges_to_remove)
    
    # 然后添加要保留的变量
    result.extend(vars_to_keep)
    
    return result


def filter_vars_by_sections(df, sections_config=None):
    """
    根据ELSA数据集的section筛选变量
    
    Args:
        df: 原始数据框
        sections_config: section配置，如果为None则使用默认配置
        
    Returns:
        list: 筛选后的变量列表
    """
    cols = list(df.columns)
    
    if sections_config is None:
        # 默认配置
        sections_config = {
            'marriage': ('r1mstat', 'r9mstat'),
            'health': ('r1shlt', 's9memrys'),
            'physical_measures': ('r1wspeed1', 's6chrothr'),
            'assistance_caregiving': ('r6dresshlp', 's9gcaresckhpw'),
            'stress': ('r2satjob', 's5dcother'),
            'psychosocial': ('r1depres', 's7slfneg2')
        }
    
    # 基础变量
    base_vars = ["ragender", "rabyear", "raeducl"]
    
    # 提取各section变量
    all_vars = base_vars.copy()
    
    for section_name, (start_var, end_var) in sections_config.items():
        try:
            start_idx = cols.index(start_var)
            end_idx = cols.index(end_var) + 1
            section_vars = cols[start_idx:end_idx]
            # 只保留以'r'开头的变量（本人相关，排除配偶）
            section_vars = [var for var in section_vars if var.startswith('r')]
            all_vars.extend(section_vars)
            print(f"Section {section_name}: 添加了 {len(section_vars)} 个变量")
        except ValueError:
            print(f"警告: Section {section_name} 的变量范围 {start_var}-{end_var} 未找到")
    
    return all_vars


def remove_data_leakage_vars(var_list):
    """
    删除可能导致数据泄露的变量（如阿兹海默诊断相关）
    
    Args:
        var_list: 变量列表
        
    Returns:
        list: 过滤后的变量列表
    """
    leakage_ranges = [
        ("r1alzhe", "radiagdemen"),
        ("r2alzhs", "r9memrys")
    ]
    
    return remove_multiple_ranges(var_list, leakage_ranges)


def remove_irrelevant_vars(var_list):
    """
    删除不相关变量
    
    Args:
        var_list: 变量列表
        
    Returns:
        list: 过滤后的变量列表
    """
    # 删除社会阶级地位相关变量
    irrelevant_ranges = [("r1cantril", "r3cantrilc")]
    
    return remove_multiple_ranges(var_list, irrelevant_ranges)


def keep_summary_scores_only(var_list):
    """
    保留汇总得分，删除单个问题项
    
    Args:
        var_list: 变量列表
        
    Returns:
        list: 过滤后的变量列表
    """
    result = var_list.copy()
    
    # 定义要处理的范围和要保留的变量
    processing_configs = [
        # ADL相关
        {
            'remove_ranges': [("r1walkra", "r9nagi8a")],
            'keep_ranges': [("r1adltot6", "r9adltot6"), ("r1iadltot1m_e", "r9iadltot1m_e"), ("r1nagi10", "r9nagi10")]
        },
        # Falls相关
        {
            'remove_ranges': [("r1fall", "r9hip")],
            'keep_ranges': [("r1fallnum", "r9fallnum")]
        },
        # 身体测量相关
        {
            'remove_ranges': [
                ("r1wspeed1", "r9wspeed2"), ("r1walksft", "r9walkothr"),
                ("r2systo1", "r8systo3"), ("r2diasto1", "r8diasto3"),
                ("r2pulse1", "r8pulse3"), ("r2bpsft", "r8bpothr"),
                ("r2lgrip1", "r8lgrip3"), ("r2rgrip1", "r8rgrip3"),
                ("r2gripsft", "r8gripothr"), ("r2hghtsft", "r9wghtothr"),
                ("r2puff1", "r4puff3"), ("r2fvc1", "r4fvc3"),
                ("r2fev1", "r4fev3"), ("r2puffsft", "r6puffothr_e"),
                ("r2legrsft", "r6legrothr"), ("r2chrsft", "r6chrothr")
            ],
            'keep_ranges': [("r2balance_e", "r6balance_e")]
        },
        # 社会支持相关
        {
            'remove_ranges': [
                ("r1sustdfe", "r9ssupportm"),
                ("r1kustdfe", "r9ksupportm"),
                ("r1oustdfe", "r9osupportm"),
                ("r1fustdfe", "r9fsupportm")
            ],
            'keep_ranges': [
                ("r1ssupport6", "r9ssupport6"),
                ("r1ksupport6", "r9ksupport6"),
                ("r1osupport6", "r9osupport6"),
                ("r1fsupport6", "r9fsupport6")
            ]
        },
        # 心理健康相关
        {
            'remove_ranges': [
                ("r1depres", "r9cesdm"),
                ("r2lideal", "r9satlifez"),
                ("r1ageprv", "r9casp12")
            ],
            'keep_ranges': [
                ("r1cesd", "r9cesd"),
                ("r2lsatsc", "r9lsatsc"),
                ("r1cntrlndx6", "r9cntrlndx6"),
                ("r1autondx5", "r9autondx5"),
                ("r1plsrndx4", "r9plsrndx4"),
                ("r1slfrlndx4", "r9slfrlndx4"),
                ("r1casp19", "r9casp19")
            ]
        }
    ]
    
    # 应用所有配置
    for config in processing_configs:
        result = remove_multiple_ranges_with_keep(
            result,
            config['remove_ranges'],
            config.get('keep_ranges', [])
        )
    
    # 单独处理一些简单的删除
    simple_removes = [
        ("r1painfr", "r9painfr"),
        ("r2shltc", "r9hips"),
        ("r6racany", "r9rcany")
    ]
    
    for remove_range in simple_removes:
        result = remove_multiple_ranges(result, [remove_range])
    
    return result


def count_wave_numbers(var_list):
    """
    统计各波次的变量数量
    
    Args:
        var_list: 变量列表
        
    Returns:
        dict: 各波次变量数量统计
    """
    wave_counts = {}
    
    for var in var_list:
        match = re.match(r'r(\d+)', var)
        if match:
            wave = int(match.group(1))
            wave_counts[wave] = wave_counts.get(wave, 0) + 1
    
    return wave_counts


def filter_vars_by_wave_count(var_list, min_waves=8):
    """
    过滤变量，只保留出现在足够多波次中的变量
    
    Args:
        var_list: 变量列表
        min_waves: 最少出现的波次数
        
    Returns:
        tuple: (过滤后的变量列表, 被移除的变量列表)
    """
    # 分离有波次的变量和无波次的变量
    wave_vars = {}  # {变量名: [出现的波次]}
    non_wave_vars = []

    for col in var_list:
        match = re.match(r'r(\d+)([a-zA-Z0-9_]+)', col.lower())
        if match:
            wave = int(match.group(1))
            var_name = match.group(2)
            if var_name not in wave_vars:
                wave_vars[var_name] = []
            wave_vars[var_name].append(wave)
        else:
            non_wave_vars.append(col)

    # 统计每个变量出现在多少个波次中
    var_wave_counts = {var: len(waves) for var, waves in wave_vars.items()}

    # 筛选出现在min_waves个或更多波次的变量
    valid_vars = [var for var, count in var_wave_counts.items() if count >= min_waves]
    removed_vars = [var for var, count in var_wave_counts.items() if count < min_waves]

    # 重建符合条件的变量列表
    filtered_cols = []
    for var in valid_vars:
        for wave in wave_vars[var]:
            filtered_cols.append(f"r{wave}{var}")

    # 添加非波次变量（这些变量保留）
    filtered_cols.extend(non_wave_vars)

    # 保持原始顺序
    filtered_cols = [col for col in var_list if col in filtered_cols]
    removed_cols = [col for col in var_list if col not in filtered_cols]

    print(f"波次过滤统计:")
    print(f"- 原始变量总数: {len(var_list)}")
    print(f"- 有波次的变量数: {len(wave_vars)}")
    print(f"- 非波次变量数: {len(non_wave_vars)}")
    print(f"- 出现在{min_waves}个或更多波次的变量数: {len(valid_vars)}")
    print(f"- 被移除的变量数: {len(removed_vars)}")
    print(f"- 过滤后变量总数: {len(filtered_cols)}")

    return filtered_cols, removed_cols


def calculate_cognitive_impairment_label(df_cognition):
    """
    计算认知障碍标签
    基于三个认知领域：记忆、定向、执行功能
    
    Args:
        df_cognition: 包含认知变量的数据框
        
    Returns:
        pd.DataFrame: 包含认知z分数和标签的数据框
    """
    # 1. 识别三个认知领域的变量
    all_cols = df_cognition.columns.tolist()
    
    # 记忆测试变量 - 词语回忆任务 (tr20相关)
    memory_vars = [col for col in all_cols if 'tr20' in col.lower()]
    
    # 定向测试变量 - 日期/时间定向
    orientation_vars = [col for col in all_cols if 'orient' in col.lower()]
    
    # 执行功能变量 - 动物命名流畅性
    executive_vars = [col for col in all_cols if 'verbf' in col.lower()]
    
    print(f"识别到认知变量:")
    print(f"- 记忆: {len(memory_vars)} 个变量")
    print(f"- 定向: {len(orientation_vars)} 个变量") 
    print(f"- 执行功能: {len(executive_vars)} 个变量")
    
    # 2. 计算各领域的z分数
    def calculate_domain_z_score(variables):
        if not variables:
            return pd.Series(index=df_cognition.index, dtype=float)
        
        domain_data = df_cognition[variables].copy()
        z_scores = pd.DataFrame(index=df_cognition.index)
        
        # 对每个变量计算z分数（基于该变量的全局均值和标准差）
        for var in variables:
            if var in domain_data.columns:
                var_data = domain_data[var].dropna()
                if len(var_data) > 0:
                    mean_val = var_data.mean()
                    std_val = var_data.std()
                    if std_val > 0:
                        z_scores[var] = (domain_data[var] - mean_val) / std_val
                    else:
                        z_scores[var] = 0
        
        # 返回该领域的平均z分数
        return z_scores.mean(axis=1, skipna=True)
    
    # 计算三个领域的z分数
    memory_z = calculate_domain_z_score(memory_vars)
    orientation_z = calculate_domain_z_score(orientation_vars)
    executive_z = calculate_domain_z_score(executive_vars)
    
    # 3. 计算全局认知z分数
    domain_scores = pd.DataFrame({
        'memory': memory_z,
        'orientation': orientation_z,
        'executive': executive_z
    })

    # 计算每个人缺失的比例
    missing_ratio = domain_scores.isna().mean(axis=1)

    # 如果缺失比例 > 0.3（即超过30%），就直接设为 NaN
    global_z_score = domain_scores.mean(axis=1, skipna=True)
    global_z_score[missing_ratio > 0.3] = np.nan
    
    # 使用全局z分数重新标准化
    global_mean = global_z_score.mean()
    global_std = global_z_score.std()
    
    if global_std > 0:
        global_z_score_final = (global_z_score - global_mean) / global_std
    else:
        global_z_score_final = global_z_score - global_mean
    
    # 4. 生成认知障碍标签（低于均值1.5个标准差）
    cognitive_impairment_label = (global_z_score_final < -1.5).astype(int)
    
    # 5. 输出统计信息
    print(f"\n计算结果:")
    print(f"- 有效样本数: {global_z_score_final.notna().sum()}")
    print(f"- 全局认知z分数均值: {global_z_score_final.mean():.3f}")
    print(f"- 全局认知z分数标准差: {global_z_score_final.std():.3f}")
    print(f"- 认知障碍阳性样本数: {cognitive_impairment_label.sum()}")
    print(f"- 认知障碍阳性比例: {cognitive_impairment_label.mean():.3f}")
    
    # 6. 返回结果
    result_df = pd.DataFrame({
        'global_cognitive_z_score': global_z_score_final,
        'cognitive_impairment_label': cognitive_impairment_label
    }, index=df_cognition.index)
    
    return result_df


def filter_samples(df, min_non_sparsity=0.7):
    """
    筛选样本：过滤掉稀疏度过高的样本和认知分数缺失的样本
    
    Args:
        df: 数据框
        min_non_sparsity: 最小非稀疏度阈值
        
    Returns:
        pd.DataFrame: 过滤后的数据框
    """
    original_count = len(df)
    
    # 条件1：global_cognitive_z_score不为NaN
    if 'global_cognitive_z_score' in df.columns:
        condition1 = df['global_cognitive_z_score'].notna()
        print(f"条件1 - global_cognitive_z_score不为NaN的样本数: {condition1.sum()}")
    else:
        condition1 = pd.Series([True] * len(df), index=df.index)
        print("警告: 未找到global_cognitive_z_score列，跳过此条件")
    
    # 条件2：稀疏度 < (1 - min_non_sparsity)
    sparsity = df.isna().mean(axis=1)
    condition2 = sparsity < (1 - min_non_sparsity)
    print(f"条件2 - 稀疏度<{1-min_non_sparsity:.1f}的样本数: {condition2.sum()}")
    
    # 应用筛选条件
    both_conditions = condition1 & condition2
    df_filtered = df[both_conditions].copy()
    
    print(f"\n样本筛选结果:")
    print(f"- 原始样本数: {original_count}")
    print(f"- 过滤后样本数: {len(df_filtered)}")
    print(f"- 保留比例: {len(df_filtered)/original_count:.3f}")
    
    # 显示筛选后数据的稀疏度统计
    if len(df_filtered) > 0:
        final_sparsity = df_filtered.isna().mean(axis=1)
        print(f"- 过滤后平均稀疏度: {final_sparsity.mean():.3f}")
        print(f"- 稀疏度范围: [{final_sparsity.min():.3f}, {final_sparsity.max():.3f}]")
    
    return df_filtered


def load_and_filter_data(raw_data_path="../data/raw/h_elsa_g3.dta", min_waves=8, min_non_sparsity=0.7):
    """
    完整的数据加载和筛选流程
    
    Args:
        raw_data_path: 原始数据文件路径
        min_waves: 最小波次数要求
        min_non_sparsity: 最小非稀疏度要求
        
    Returns:
        tuple: (过滤后的数据框, 认知变量数据框, 变量筛选过程信息)
    """
    print("=== 开始数据加载和筛选流程 ===")
    
    # 1. 加载原始数据
    df = load_raw_data(raw_data_path)
    
    # 2. 按section筛选变量
    print("\n=== 按section筛选变量 ===")
    selected_vars = filter_vars_by_sections(df)
    print(f"初步筛选后变量数: {len(selected_vars)}")
    
    # 3. 删除数据泄露变量
    print("\n=== 删除数据泄露变量 ===")
    selected_vars = remove_data_leakage_vars(selected_vars)
    var_names, var_count = count_unique_vars(selected_vars)
    print(f"删除数据泄露变量后变量种类数: {var_count}")
    
    # 4. 删除不相关变量
    print("\n=== 删除不相关变量 ===")
    selected_vars = remove_irrelevant_vars(selected_vars)
    var_names, var_count = count_unique_vars(selected_vars)
    print(f"删除不相关变量后变量种类数: {var_count}")
    
    # 5. 保留汇总得分，删除单个问题项
    print("\n=== 保留汇总得分 ===")
    selected_vars = keep_summary_scores_only(selected_vars)
    var_names, var_count = count_unique_vars(selected_vars)
    print(f"保留汇总得分后变量种类数: {var_count}")
    
    # 6. 按波次过滤变量
    print(f"\n=== 按波次过滤变量（最少{min_waves}个波次） ===")
    selected_vars, removed_vars = filter_vars_by_wave_count(selected_vars, min_waves)
    var_names, var_count = count_unique_vars(selected_vars)
    print(f"波次过滤后变量种类数: {var_count}")
    
    # 7. 处理特殊变量（如r3shlt）
    if "r2shlt" in selected_vars and "r3shlt" not in selected_vars:
        r2shlt_index = selected_vars.index("r2shlt")
        selected_vars.insert(r2shlt_index + 1, "r3shlta")
        print("添加了r3shlt变量")
    
    # 8. 计算认知障碍标签
    print("\n=== 计算认知障碍标签 ===")
    cols = list(df.columns)
    cognition_vars = []
    for pattern in ["tr20", "orient", "verbf"]:
        cognition_vars.extend([col for col in cols if pattern in col.lower()])
    
    if cognition_vars:
        df_cognition = df[cognition_vars]
        cognitive_results = calculate_cognitive_impairment_label(df_cognition)
    else:
        print("警告: 未找到认知相关变量")
        cognitive_results = pd.DataFrame(index=df.index)
    
    # 9. 构建最终数据框
    print("\n=== 构建最终数据框 ===")
    df_final = pd.concat([df[selected_vars], cognitive_results], axis=1)
    
    # 处理列名（如重命名r3shlta为r3shlt）
    if "r3shlta" in df_final.columns:
        df_final = df_final.rename(columns={"r3shlta": "r3shlt"})
    
    var_names, var_count = count_unique_vars(list(df_final.columns))
    print(f"最终数据框变量种类数: {var_count}")
    print(f"最终数据框形状: {df_final.shape}")
    
    # 10. 筛选样本
    print(f"\n=== 筛选样本（最小非稀疏度: {min_non_sparsity}） ===")
    df_filtered = filter_samples(df_final, min_non_sparsity)
    
    print("\n=== 数据加载和筛选完成 ===")
    
    return df_filtered, df_cognition if 'df_cognition' in locals() else None, {
        'selected_vars': selected_vars,
        'removed_vars': removed_vars if 'removed_vars' in locals() else [],
        'final_var_count': var_count
    }