def count_unique_vars(col_list):
    """
    统计变量种类数（去掉波次，保持顺序），返回变量名列表和数量
    """
    import re
    var_names = []
    for col in col_list:
        m = re.match(r'r\d+([a-zA-Z0-9_]+)', col)
        name = m.group(1) if m else col
        if name not in var_names:
            var_names.append(name)
    return var_names, len(var_names)

def remove_multiple_ranges(col_list, ranges):
    """
    批量删除多个区间，每个区间为(start_var, end_var)，返回新列表
    """
    to_remove = set()
    for start_var, end_var in ranges:
        try:
            start_idx = col_list.index(start_var)
            end_idx = col_list.index(end_var)
        except ValueError:
            raise ValueError(f"{start_var}或{end_var}不在col_list中")
        to_remove.update(col_list[start_idx:end_idx+1])
    return [col for col in col_list if col not in to_remove]

def remove_multiple_ranges_with_keep(col_list, remove_ranges, keep_ranges):
    """
    批量删除多个区间，每个区间为(start_var, end_var)；
    同时批量保留多个区间，每个区间为(start_var, end_var)
    """
    # 需要删除的变量集合
    to_remove = set()
    for start_var, end_var in remove_ranges:
        try:
            start_idx = col_list.index(start_var)
            end_idx = col_list.index(end_var)
        except ValueError:
            raise ValueError(f"{start_var}或{end_var}不在col_list中")
        to_remove.update(col_list[start_idx:end_idx+1])

    # 需要保留的变量集合
    to_keep = set()
    for start_var, end_var in keep_ranges:
        try:
            start_idx = col_list.index(start_var)
            end_idx = col_list.index(end_var)
        except ValueError:
            raise ValueError(f"{start_var}或{end_var}不在col_list中")
        to_keep.update(col_list[start_idx:end_idx+1])

    # 最终删除的是 to_remove 中但不在 to_keep 的变量
    final_remove = to_remove - to_keep
    return [col for col in col_list if col not in final_remove]

def count_wave_numbers(col_list):
    """
    统计列表中变量名中r后面数字出现的次数
    例如：r1casp19 中的1会被统计一次
    
    参数:
        col_list: 列名列表
    
    返回:
        dict: {数字: 出现次数}
    """
    import re
    from collections import Counter
    
    wave_numbers = []
    
    for col in col_list:
        # 匹配 r 后面跟数字的模式
        match = re.match(r'r(\d+)', col.lower())
        if match:
            wave_number = int(match.group(1))
            wave_numbers.append(wave_number)
    
    # 统计每个数字出现的次数
    return dict(Counter(wave_numbers))

