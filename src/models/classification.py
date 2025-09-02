"""
分类模型定义
用于ADHD认知障碍二分类任务
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any


class ClassificationModel(nn.Module):
    """
    用于ADHD分类任务的深度神经网络模型
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化分类模型
        
        Args:
            config: 模型配置字典，包含以下键：
                - input_dim: 输入特征维度
                - hidden_dims: 隐藏层维度列表
                - output_dim: 输出维度（类别数）
                - dropout: Dropout率
                - activation: 激活函数类型
        """
        super(ClassificationModel, self).__init__()
        
        self.input_dim = config.get('input_dim')
        self.hidden_dims = config.get('hidden_dims', [128, 64, 32])
        self.output_dim = config.get('output_dim', 2)
        self.dropout_rate = config.get('dropout', 0.3)
        self.activation_type = config.get('activation', 'relu')
        
        if self.input_dim is None:
            raise ValueError("input_dim必须在配置中指定")
        
        # 构建网络层
        self.layers = self._build_layers()
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # 选择激活函数
        self.activation = self._get_activation_function()
        
        # 初始化权重
        self.apply(self._init_weights)
        
        print(f"分类模型初始化完成:")
        print(f"  - 输入维度: {self.input_dim}")
        print(f"  - 隐藏层: {self.hidden_dims}")
        print(f"  - 输出维度: {self.output_dim}")
        print(f"  - Dropout: {self.dropout_rate}")
        print(f"  - 激活函数: {self.activation_type}")
        print(f"  - 总参数量: {self.count_parameters():,}")
    
    def _build_layers(self) -> nn.ModuleList:
        """
        构建网络层
        
        Returns:
            网络层列表
        """
        layers = nn.ModuleList()
        
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        
        # 隐藏层之间的连接
        for i in range(len(self.hidden_dims) - 1):
            layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]))
        
        # 最后一个隐藏层到输出层
        layers.append(nn.Linear(self.hidden_dims[-1], self.output_dim))
        
        return layers
    
    def _get_activation_function(self):
        """
        获取激活函数
        
        Returns:
            激活函数
        """
        activation_functions = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(negative_slope=0.01),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        
        if self.activation_type.lower() in activation_functions:
            return activation_functions[self.activation_type.lower()]
        else:
            print(f"警告: 未知的激活函数 '{self.activation_type}'，使用ReLU")
            return nn.ReLU()
    
    def _init_weights(self, module):
        """
        初始化网络权重
        
        Args:
            module: 网络模块
        """
        if isinstance(module, nn.Linear):
            # 使用Xavier初始化
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_dim]
            
        Returns:
            输出张量 [batch_size, output_dim]
        """
        # 通过所有隐藏层
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # 输出层（不使用激活函数和dropout）
        x = self.layers[-1](x)
        
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测概率
        
        Args:
            x: 输入张量
            
        Returns:
            预测概率张量
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            return probabilities
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测类别
        
        Args:
            x: 输入张量
            
        Returns:
            预测类别张量
        """
        probabilities = self.predict_proba(x)
        predictions = torch.argmax(probabilities, dim=1)
        return predictions
    
    def count_parameters(self) -> int:
        """
        计算模型参数总数
        
        Returns:
            参数总数
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'model_type': 'classification',
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation_type,
            'total_parameters': self.count_parameters(),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def freeze_layers(self, num_layers: int):
        """
        冻结前几层的参数
        
        Args:
            num_layers: 要冻结的层数
        """
        for i, layer in enumerate(self.layers):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
                print(f"已冻结第 {i+1} 层")
    
    def unfreeze_all_layers(self):
        """
        解冻所有层的参数
        """
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = True
        print("已解冻所有层")
    
    def get_layer_outputs(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        获取指定层的输出（用于可视化和分析）
        
        Args:
            x: 输入张量
            layer_idx: 层索引
            
        Returns:
            指定层的输出
        """
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                if i <= layer_idx:
                    if i < len(self.layers) - 1:  # 不是最后一层
                        x = layer(x)
                        x = self.activation(x)
                    else:  # 最后一层
                        x = layer(x)
                    
                    if i == layer_idx:
                        return x
        
        return x