"""
模型训练模块
包含单任务训练函数和相关工具函数
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
import time
import copy
from pathlib import Path


class EarlyStopping:
    """早停机制类"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001, restore_best_weights: bool = True):
        """
        初始化早停机制
        
        Args:
            patience: 耐心值，连续多少个epoch没有改善就停止
            min_delta: 最小改善阈值
            restore_best_weights: 是否恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        检查是否需要早停
        
        Args:
            score: 当前验证得分
            model: 模型
            
        Returns:
            是否早停
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    self.restore_checkpoint(model)
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        
        return self.early_stop
    
    def save_checkpoint(self, model: nn.Module):
        """保存检查点"""
        if self.restore_best_weights:
            self.best_weights = copy.deepcopy(model.state_dict())
    
    def restore_checkpoint(self, model: nn.Module):
        """恢复检查点"""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def train_single_task(model: nn.Module,
                     train_loader: DataLoader,
                     val_loader: DataLoader,
                     criterion: nn.Module,
                     optimizer: torch.optim.Optimizer,
                     scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                     device: torch.device,
                     task_name: str,
                     num_epochs: int = 100,
                     patience: int = 15,
                     min_delta: float = 0.0001,
                     save_best_model: bool = True,
                     model_save_path: Optional[str] = None) -> Dict[str, List[float]]:
    """
    单任务训练函数
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 训练设备
        task_name: 任务名称
        num_epochs: 训练轮数
        patience: 早停耐心值
        min_delta: 早停最小改善阈值
        save_best_model: 是否保存最佳模型
        model_save_path: 模型保存路径
        
    Returns:
        训练历史字典
    """
    print(f"\n=== 开始训练 {task_name} ===")
    print(f"训练设备: {device}")
    print(f"训练样本数: {len(train_loader.dataset)}")
    print(f"验证样本数: {len(val_loader.dataset)}")
    print(f"最大轮数: {num_epochs}")
    print(f"早停耐心值: {patience}")
    
    # 确保模型在正确的设备上
    model = model.to(device)
    
    # 初始化早停机制
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    
    # 训练历史记录
    history = {
        'train_losses': [],
        'val_losses': [],
        'val_scores': [],
        'learning_rates': []
    }
    
    # 确定任务类型
    is_classification = isinstance(criterion, nn.CrossEntropyLoss)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            features, targets = features.to(device), targets.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(features)
            
            # 计算损失
            if is_classification:
                loss = criterion(outputs, targets)
            else:  # regression
                outputs = outputs.squeeze()
                loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # 计算平均训练损失
        avg_train_loss = train_loss / train_batches
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_score = 0.0
        val_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                
                outputs = model(features)
                
                # 计算损失
                if is_classification:
                    loss = criterion(outputs, targets)
                    # 计算准确率
                    predictions = torch.argmax(outputs, dim=1)
                    correct = (predictions == targets).sum().item()
                    batch_score = correct / targets.size(0)
                else:  # regression
                    outputs = outputs.squeeze()
                    loss = criterion(outputs, targets)
                    # 暂时使用负MSE作为分数（越大越好）
                    batch_score = -loss.item()
                
                val_loss += loss.item()
                val_score += batch_score
                val_batches += 1
                
                # 收集预测和真实值用于详细评估
                if is_classification:
                    all_predictions.extend(predictions.cpu().numpy())
                else:
                    all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # 计算平均验证损失和得分
        avg_val_loss = val_loss / val_batches
        avg_val_score = val_score / val_batches
        
        # 如果是回归任务，计算R²分数
        if not is_classification:
            from sklearn.metrics import r2_score
            avg_val_score = r2_score(all_targets, all_predictions)
        
        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        # 记录历史
        history['train_losses'].append(avg_train_loss)
        history['val_losses'].append(avg_val_loss)
        history['val_scores'].append(avg_val_score)
        history['learning_rates'].append(current_lr)
        
        # 计算epoch时间
        epoch_time = time.time() - epoch_start_time
        
        # 打印进度
        if is_classification:
            score_name = "Accuracy"
            score_format = f"{avg_val_score:.4f}"
        else:
            score_name = "R²"
            score_format = f"{avg_val_score:.4f}"
        
        print(f"Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s) - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val {score_name}: {score_format}, "
              f"LR: {current_lr:.2e}")
        
        # 早停检查
        if early_stopping(avg_val_score, model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    print(f"\n训练完成! 总用时: {total_time:.2f}秒")
    print(f"最佳验证{score_name}: {early_stopping.best_score:.4f}")
    
    # 保存最佳模型
    if save_best_model and model_save_path:
        Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_score': early_stopping.best_score,
            'history': history,
            'model_config': model.get_model_info() if hasattr(model, 'get_model_info') else {}
        }, model_save_path)
        print(f"最佳模型已保存至: {model_save_path}")
    
    return history


def load_checkpoint(model: nn.Module, 
                   optimizer: torch.optim.Optimizer, 
                   checkpoint_path: str) -> Tuple[nn.Module, torch.optim.Optimizer, dict]:
    """
    加载检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        checkpoint_path: 检查点路径
        
    Returns:
        加载后的模型、优化器和历史记录
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    history = checkpoint.get('history', {})
    
    print(f"检查点加载完成: {checkpoint_path}")
    print(f"最佳得分: {checkpoint.get('best_score', 'N/A')}")
    
    return model, optimizer, history


def create_optimizer(model: nn.Module, 
                    optimizer_type: str = "adam",
                    learning_rate: float = 0.001,
                    weight_decay: float = 0.0,
                    **kwargs) -> torch.optim.Optimizer:
    """
    创建优化器
    
    Args:
        model: 模型
        optimizer_type: 优化器类型
        learning_rate: 学习率
        weight_decay: 权重衰减
        **kwargs: 其他优化器参数
        
    Returns:
        优化器
    """
    if optimizer_type.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_type.lower() == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")


def create_scheduler(optimizer: torch.optim.Optimizer,
                    scheduler_type: str = "plateau",
                    **kwargs) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型
        **kwargs: 调度器参数
        
    Returns:
        学习率调度器
    """
    if scheduler_type.lower() == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, **kwargs)
    elif scheduler_type.lower() == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1, **kwargs)
    elif scheduler_type.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, **kwargs)
    elif scheduler_type.lower() == "none":
        return None
    else:
        raise ValueError(f"不支持的调度器类型: {scheduler_type}")