"""
深度学习模型训练模块
"""
import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, Any, List, Optional
from torch.utils.data import DataLoader


def train_deep_learning_model(model: nn.Module,
                             train_loader: DataLoader,
                             val_loader: DataLoader,
                             criterion: nn.Module,
                             optimizer: torch.optim.Optimizer,
                             device: torch.device,
                             num_epochs: int = 100,
                             patience: int = 15,
                             min_delta: float = 0.0001,
                             scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                             verbose: bool = True) -> Dict[str, List[float]]:
    """
    训练深度学习模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        num_epochs: 训练轮数
        patience: 早停耐心值
        min_delta: 最小改进阈值
        scheduler: 学习率调度器
        verbose: 是否打印训练信息
        
    Returns:
        训练历史字典
    """
    model.to(device)
    
    # 训练历史
    history = {
        'train_losses': [],
        'val_losses': [],
        'train_accuracies': [],
        'val_accuracies': [],
        'learning_rates': []
    }
    
    # 早停相关变量
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    if verbose:
        print(f"Training on {device} for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # 训练阶段
        train_loss, train_acc = _train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证阶段
        val_loss, val_acc = _validate_epoch(model, val_loader, criterion, device)
        
        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step(val_loss)
            
        # 记录历史
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['train_accuracies'].append(train_acc)
        history['val_accuracies'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        # 早停检查
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # 打印训练信息
        if verbose and (epoch + 1) % 10 == 0:
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                  f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, "
                  f"lr={current_lr:.6f}, time={epoch_time:.2f}s")
        
        # 早停
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    if verbose:
        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    return history


def _train_epoch(model: nn.Module,
                train_loader: DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device) -> tuple:
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        
        # 计算准确率（仅分类任务）
        if len(output.shape) > 1 and output.shape[1] > 1:  # 分类任务
            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()
        
        total_samples += data.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples if total_correct > 0 else 0.0
    
    return avg_loss, accuracy


def _validate_epoch(model: nn.Module,
                   val_loader: DataLoader,
                   criterion: nn.Module,
                   device: torch.device) -> tuple:
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 统计
            total_loss += loss.item()
            
            # 计算准确率（仅分类任务）
            if len(output.shape) > 1 and output.shape[1] > 1:  # 分类任务
                pred = output.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(target.view_as(pred)).sum().item()
            
            total_samples += data.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_samples if total_correct > 0 else 0.0
    
    return avg_loss, accuracy


def calculate_model_flops(model: nn.Module, input_shape: tuple) -> Optional[int]:
    """
    计算模型FLOPS (简化版本)
    
    Args:
        model: 模型
        input_shape: 输入形状
        
    Returns:
        FLOPS数量，如果计算失败返回None
    """
    try:
        # 简单估算：只计算Linear层的FLOPS
        total_flops = 0
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                total_flops += module.in_features * module.out_features
        
        return total_flops
    except:
        return None


if __name__ == "__main__":
    # 测试代码
    print("Testing deep learning training module...")
    
    # 创建简单测试模型
    from ..models.classification import ClassificationModel
    
    config = {
        'input_dim': 100,
        'hidden_dims': [64, 32],
        'output_dim': 2,
        'dropout': 0.3,
        'activation': 'relu'
    }
    
    model = ClassificationModel(config)
    device = torch.device('cpu')
    
    # 创建测试数据
    from torch.utils.data import TensorDataset
    
    X = torch.randn(200, 100)
    y = torch.randint(0, 2, (200,))
    
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset[:160], batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset[160:], batch_size=32, shuffle=False)
    
    # 训练参数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 测试训练
    history = train_deep_learning_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=10,
        patience=5,
        verbose=True
    )
    
    print(f"Training history: {len(history['train_losses'])} epochs")
    print("Deep learning training test completed!")