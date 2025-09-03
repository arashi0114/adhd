"""
配置管理器
处理简化的YAML配置文件
"""
import yaml
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._setup_device()
        self._setup_output_dirs()
        
        print(f"配置加载完成: {self.config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _setup_device(self):
        """设置计算设备"""
        device_config = self.config.get('training', {}).get('deep_learning', {}).get('device', 'auto')
        
        if device_config == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = device_config
        
        # 更新配置中的设备设置
        self.config['training']['deep_learning']['device'] = device
        print(f"使用设备: {device}")
    
    def _setup_output_dirs(self):
        """设置输出目录"""
        output_config = self.config.get('output', {})
        base_dir = Path(output_config.get('base_dir', 'outputs'))
        
        # 如果使用时间戳
        if output_config.get('use_timestamp', True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = self.config.get('experiment', {}).get('name', 'experiment')
            base_dir = base_dir / f"{exp_name}_{timestamp}"
        
        # 创建子目录
        dirs = output_config.get('dirs', {})
        self.output_dirs = {
            'base': base_dir,
            'models': base_dir / dirs.get('models', 'models'),
            'results': base_dir / dirs.get('results', 'results'),
            'plots': base_dir / dirs.get('plots', 'plots'),
            'logs': base_dir / dirs.get('logs', 'logs')
        }
        
        # 创建目录
        for dir_path in self.output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"输出目录: {self.output_dirs['base']}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key_path: 配置键路径，用.分隔，如 'data.train_ratio'
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_enabled_models(self, task_type: str = None) -> Dict[str, Dict[str, Any]]:
        """
        获取启用的模型配置
        
        Args:
            task_type: 任务类型，如果指定则只返回支持该任务的模型
            
        Returns:
            启用的模型配置字典
        """
        enabled_models = {}
        
        # 深度学习模型
        dl_models = self.config.get('models', {}).get('deep_learning', {})
        for model_name, model_config in dl_models.items():
            if model_config.get('enabled', False):
                enabled_models[f"deep_learning_{model_name}"] = {
                    'category': 'deep_learning',
                    'name': model_name,
                    'config': model_config
                }
        
        # 传统ML模型
        ml_models = self.config.get('models', {}).get('traditional_ml', {})
        for model_name, model_config in ml_models.items():
            if model_config.get('enabled', False):
                enabled_models[f"traditional_ml_{model_name}"] = {
                    'category': 'traditional_ml',
                    'name': model_name,
                    'config': model_config
                }
        
        return enabled_models
    
    def get_enabled_tasks(self) -> List[str]:
        """获取启用的任务列表"""
        tasks = []
        task_config = self.config.get('tasks', {})
        
        if task_config.get('classification', {}).get('enabled', False):
            tasks.append('classification')
        
        if task_config.get('regression', {}).get('enabled', False):
            tasks.append('regression')
        
        return tasks
    
    def get_output_path(self, dir_type: str, filename: str = None) -> Path:
        """
        获取输出路径
        
        Args:
            dir_type: 目录类型 ('models', 'results', 'plots', 'logs')
            filename: 文件名（可选）
            
        Returns:
            完整路径
        """
        if dir_type not in self.output_dirs:
            raise ValueError(f"未知的目录类型: {dir_type}")
        
        path = self.output_dirs[dir_type]
        
        if filename:
            path = path / filename
        
        return path
    
    def get_data_config(self) -> Dict[str, Any]:
        """获取数据配置"""
        return self.config.get('data', {})
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """获取预处理配置"""
        return self.config.get('preprocessing', {})
    
    def get_training_config(self, model_category: str = 'deep_learning') -> Dict[str, Any]:
        """
        获取训练配置
        
        Args:
            model_category: 模型类别
            
        Returns:
            训练配置
        """
        training_config = self.config.get('training', {})
        
        if model_category in training_config:
            return training_config[model_category]
        else:
            # 返回深度学习配置作为默认
            return training_config.get('deep_learning', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """获取评估配置"""
        return self.config.get('evaluation', {})
    
    def get_advanced_analysis_config(self) -> Dict[str, Any]:
        """获取高级分析配置"""
        return self.config.get('advanced_analysis', {})
    
    def is_enabled(self, feature_path: str) -> bool:
        """
        检查功能是否启用
        
        Args:
            feature_path: 功能路径，如 'advanced_analysis.cross_validation.enabled'
            
        Returns:
            是否启用
        """
        return self.get(f"{feature_path}.enabled", False)
    
    def save_config_copy(self, suffix: str = "used") -> str:
        """
        保存配置文件副本
        
        Args:
            suffix: 文件名后缀
            
        Returns:
            保存的文件路径
        """
        config_copy_path = self.get_output_path('logs', f'config_{suffix}.yaml')
        
        with open(config_copy_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, 
                     allow_unicode=True, indent=2)
        
        return str(config_copy_path)
    
    def update_config(self, updates: Dict[str, Any]):
        """
        更新配置
        
        Args:
            updates: 更新的配置字典
        """
        def update_nested_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_nested_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = update_nested_dict(self.config, updates)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        enabled_models = list(self.get_enabled_models().keys())
        enabled_tasks = self.get_enabled_tasks()
        
        summary = {
            "config_file": str(self.config_path),
            "experiment_name": self.get('experiment.name'),
            "random_seed": self.get('random_seed'),
            "enabled_tasks": enabled_tasks,
            "enabled_models": enabled_models,
            "total_models": len(enabled_models),
            "device": self.get('training.deep_learning.device'),
            "output_base": str(self.output_dirs['base']),
            "data_dir": self.get('data.data_dir'),
            "preprocessing": {
                "imputation": self.is_enabled('preprocessing.multiple_imputation'),
                "outlier_detection": self.is_enabled('preprocessing.outlier_detection'),
                "scaling": self.is_enabled('preprocessing.scaling')
            }
        }
        
        return summary


def load_config(config_path: str = "config/config.yaml") -> ConfigManager:
    """
    便捷函数：加载配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置管理器实例
    """
    return ConfigManager(config_path)


if __name__ == "__main__":
    # 测试代码
    print("测试配置管理器...")
    
    try:
        # 加载配置
        config = load_config("config/config.yaml")
        
        # 测试基本功能
        print(f"随机种子: {config.get('random_seed')}")
        print(f"实验名称: {config.get('experiment.name')}")
        
        # 测试模型配置
        models = config.get_enabled_models()
        print(f"启用的模型数量: {len(models)}")
        print(f"模型列表: {list(models.keys())}")
        
        # 测试任务配置
        tasks = config.get_enabled_tasks()
        print(f"启用的任务: {tasks}")
        
        # 测试输出路径
        model_path = config.get_output_path('models', 'test_model.pth')
        print(f"模型保存路径示例: {model_path}")
        
        # 生成摘要
        summary = config.get_summary()
        print(f"\n配置摘要: {summary}")
        
        # 保存配置副本
        config_copy = config.save_config_copy('test')
        print(f"配置副本已保存: {config_copy}")
        
        print("\n✅ 配置管理器测试成功!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()