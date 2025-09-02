"""
工具函数模块
包含模型保存/加载和结果可视化功能
"""

import os
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import matplotlib.font_manager as fm


def save_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
               epoch: int, val_loss: float, val_score: float, filepath: str) -> None:
    """
    保存模型检查点
    
    Args:
        model: 要保存的模型
        optimizer: 优化器
        epoch: 当前轮数
        val_loss: 验证损失
        val_score: 验证得分
        filepath: 保存路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_score': val_score,
    }
    
    torch.save(checkpoint, filepath)


def load_model(filepath: str) -> Dict[str, Any]:
    """
    加载模型检查点
    
    Args:
        filepath: 模型文件路径
        
    Returns:
        检查点字典
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"模型文件不存在: {filepath}")
    
    return torch.load(filepath, map_location='cpu')


def plot_results(results: Dict[str, Dict[str, List[float]]]) -> None:
    """
    绘制训练结果对比图
    
    Args:
        results: 结果字典，格式为:
                {
                    'classification': {
                        'train_losses': [...],
                        'val_losses': [...],
                        'scores': [...]
                    },
                    'regression': {
                        'train_losses': [...],
                        'val_losses': [...], 
                        'scores': [...]
                    }
                }
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ADHD模型训练结果对比', fontsize=16)
    
    # 分类任务 - 损失
    if 'classification' in results:
        cls_data = results['classification']
        epochs = range(1, len(cls_data['train_losses']) + 1)
        
        axes[0, 0].plot(epochs, cls_data['train_losses'], 'b-', label='训练损失', linewidth=2)
        axes[0, 0].plot(epochs, cls_data['val_losses'], 'r-', label='验证损失', linewidth=2)
        axes[0, 0].set_title('分类任务 - 损失变化')
        axes[0, 0].set_xlabel('轮数')
        axes[0, 0].set_ylabel('损失值')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 分类任务 - 准确率
        axes[0, 1].plot(epochs, cls_data['scores'], 'g-', label='验证准确率', linewidth=2)
        axes[0, 1].set_title('分类任务 - 准确率变化')
        axes[0, 1].set_xlabel('轮数')
        axes[0, 1].set_ylabel('准确率')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 回归任务 - 损失
    if 'regression' in results:
        reg_data = results['regression']
        epochs = range(1, len(reg_data['train_losses']) + 1)
        
        axes[1, 0].plot(epochs, reg_data['train_losses'], 'b-', label='训练损失', linewidth=2)
        axes[1, 0].plot(epochs, reg_data['val_losses'], 'r-', label='验证损失', linewidth=2)
        axes[1, 0].set_title('回归任务 - 损失变化')
        axes[1, 0].set_xlabel('轮数')
        axes[1, 0].set_ylabel('MSE损失')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 回归任务 - R²分数
        axes[1, 1].plot(epochs, reg_data['scores'], 'purple', label='验证R²分数', linewidth=2)
        axes[1, 1].set_title('回归任务 - R²分数变化')
        axes[1, 1].set_xlabel('轮数')
        axes[1, 1].set_ylabel('R²分数')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("训练结果图表已保存至: results/training_results.png")