"""
配置管理模块
负责加载和管理项目配置文件
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    # 后处理配置
    config = _postprocess_config(config)
    
    return config


def _postprocess_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    后处理配置，自动设置一些参数
    
    Args:
        config: 原始配置字典
        
    Returns:
        处理后的配置字典
    """
    # 自动设置设备
    if config.get("train", {}).get("device") == "cuda":
        if torch.cuda.is_available():
            config["train"]["device"] = "cuda"
            print(f"使用CUDA设备: {torch.cuda.get_device_name(0)}")
        else:
            config["train"]["device"] = "cpu"
            print("CUDA不可用，使用CPU设备")
    
    # 创建输出目录
    output_dirs = [
        config.get("output", {}).get("model_save_dir", "outputs/models"),
        config.get("output", {}).get("results_save_dir", "outputs/results"),
        config.get("output", {}).get("log_dir", "outputs/logs")
    ]
    
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # 根据任务类型设置输出维度
    if config.get("model", {}).get("task_type") == "classification":
        # 二分类任务
        config["model"]["output_dim"] = 2
    elif config.get("model", {}).get("task_type") == "regression":
        # 回归任务
        config["model"]["output_dim"] = 1
    
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    保存配置文件
    
    Args:
        config: 配置字典
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
    
    print(f"配置已保存至: {save_path}")


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    更新配置
    
    Args:
        config: 原始配置字典
        updates: 更新的配置项
        
    Returns:
        更新后的配置字典
    """
    def _update_nested_dict(d: dict, u: dict) -> dict:
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = _update_nested_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    updated_config = _update_nested_dict(config.copy(), updates)
    return _postprocess_config(updated_config)


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    获取嵌套配置值
    
    Args:
        config: 配置字典
        key_path: 键路径，如 "model.hidden_dims"
        default: 默认值
        
    Returns:
        配置值
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def validate_config(config: Dict[str, Any]) -> bool:
    """
    验证配置文件的有效性
    
    Args:
        config: 配置字典
        
    Returns:
        配置是否有效
    """
    required_keys = [
        "data.path",
        "model.task_type", 
        "train.batch_size",
        "train.learning_rate",
        "train.num_epochs"
    ]
    
    for key_path in required_keys:
        if get_config_value(config, key_path) is None:
            print(f"缺少必需的配置项: {key_path}")
            return False
    
    # 验证任务类型
    task_type = get_config_value(config, "model.task_type")
    if task_type not in ["classification", "regression"]:
        print(f"不支持的任务类型: {task_type}")
        return False
    
    # 验证设备设置
    device = get_config_value(config, "train.device")
    if device not in ["cpu", "cuda"]:
        print(f"不支持的设备类型: {device}")
        return False
    
    print("配置验证通过")
    return True