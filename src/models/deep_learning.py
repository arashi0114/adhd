"""
深度学习模型库
实现适配表格数据的各种深度学习架构
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional


class TabularTransformer(nn.Module):
    """
    适配表格数据的Transformer模型
    将表格特征视为序列进行处理
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(TabularTransformer, self).__init__()
        
        self.input_dim = config.get('input_dim')
        self.d_model = config.get('d_model', 128)
        self.nhead = config.get('nhead', 8)
        self.num_layers = config.get('num_layers', 3)
        self.dim_feedforward = config.get('dim_feedforward', 512)
        self.dropout = config.get('dropout', 0.1)
        self.output_dim = config.get('output_dim', 1)
        self.task_type = config.get('task_type', 'classification')
        
        # 计算特征分组数量（每组作为一个token）
        self.group_size = max(1, self.input_dim // 16)  # 每16个特征一组
        self.num_groups = (self.input_dim + self.group_size - 1) // self.group_size
        
        # 输入投影层
        self.input_projection = nn.Linear(self.group_size, self.d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(self.d_model, dropout=self.dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # 输出层
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model * self.num_groups, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.output_dim)
        )
        
        print(f"TabularTransformer initialized:")
        print(f"  - Input dim: {self.input_dim}")
        print(f"  - Groups: {self.num_groups} (size: {self.group_size})")
        print(f"  - Model dim: {self.d_model}")
        print(f"  - Attention heads: {self.nhead}")
        print(f"  - Encoder layers: {self.num_layers}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # 将输入重塑为组
        # 如果最后一组不完整，用零填充
        if x.size(1) < self.num_groups * self.group_size:
            padding = self.num_groups * self.group_size - x.size(1)
            x = F.pad(x, (0, padding))
        
        x = x.view(batch_size, self.num_groups, self.group_size)
        
        # 输入投影
        x = self.input_projection(x)  # (batch, num_groups, d_model)
        
        # 转换为Transformer格式：(seq_len, batch, d_model)
        x = x.transpose(0, 1)
        
        # 添加位置编码
        x = self.pos_encoding(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 转换回：(batch, seq_len, d_model)
        x = x.transpose(0, 1)
        
        # 展平并分类
        x = x.contiguous().view(batch_size, -1)
        x = self.classifier(x)
        
        return x


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TabularResNet(nn.Module):
    """
    适配表格数据的ResNet模型
    使用残差连接和批归一化
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(TabularResNet, self).__init__()
        
        self.input_dim = config.get('input_dim')
        self.block_sizes = config.get('block_sizes', [64, 128, 256])
        self.num_blocks = config.get('num_blocks', [2, 2, 2])
        self.dropout = config.get('dropout', 0.3)
        self.output_dim = config.get('output_dim', 1)
        
        # 输入投影
        self.input_projection = nn.Linear(self.input_dim, self.block_sizes[0])
        self.bn_input = nn.BatchNorm1d(self.block_sizes[0])
        
        # 构建残差块
        self.blocks = nn.ModuleList()
        in_features = self.block_sizes[0]
        
        for i, (out_features, num_block) in enumerate(zip(self.block_sizes, self.num_blocks)):
            for j in range(num_block):
                block = ResidualBlock(
                    in_features,
                    out_features,
                    dropout=self.dropout,
                    downsample=(j == 0 and in_features != out_features)
                )
                self.blocks.append(block)
                in_features = out_features
        
        # 全局平均池化和分类器
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(self.block_sizes[-1], self.output_dim)
        self.dropout_final = nn.Dropout(self.dropout)
        
        print(f"TabularResNet initialized:")
        print(f"  - Input dim: {self.input_dim}")
        print(f"  - Block sizes: {self.block_sizes}")
        print(f"  - Num blocks: {self.num_blocks}")
        print(f"  - Total blocks: {len(self.blocks)}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入投影
        x = self.input_projection(x)
        x = self.bn_input(x)
        x = F.relu(x)
        
        # 为卷积操作添加维度
        x = x.unsqueeze(-1)  # (batch, features, 1)
        
        # 通过残差块
        for block in self.blocks:
            x = block(x)
        
        # 全局池化
        x = self.global_pool(x)
        x = x.squeeze(-1)
        
        # 分类
        x = self.dropout_final(x)
        x = self.classifier(x)
        
        return x


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3, downsample: bool = False):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_features, out_features, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.conv2 = nn.Conv1d(out_features, out_features, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        
        self.shortcut = nn.Sequential()
        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_features, out_features, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_features)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out


class TabularMobileNet(nn.Module):
    """
    适配表格数据的MobileNet模型
    使用深度可分离卷积的概念
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(TabularMobileNet, self).__init__()
        
        self.input_dim = config.get('input_dim')
        self.width_multiplier = config.get('width_multiplier', 1.0)
        self.depth_multiplier = config.get('depth_multiplier', 1.0)
        self.dropout = config.get('dropout', 0.2)
        self.output_dim = config.get('output_dim', 1)
        
        # 计算通道数
        def make_divisible(v, divisor=8):
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
        
        # MobileNet配置：[输出通道, 步长]
        cfg = [
            [32, 1],
            [64, 1],
            [128, 1], 
            [128, 1],
            [256, 1],
            [256, 1],
            [512, 1]
        ]
        
        # 调整通道数
        cfg = [[make_divisible(c * self.width_multiplier), s] for c, s in cfg]
        
        # 输入层
        self.input_projection = nn.Linear(self.input_dim, cfg[0][0])
        self.bn_input = nn.BatchNorm1d(cfg[0][0])
        
        # 构建深度可分离卷积层
        self.features = nn.ModuleList()
        in_channels = cfg[0][0]
        
        for out_channels, stride in cfg[1:]:
            self.features.append(
                DepthwiseSeparableConv1d(
                    in_channels, 
                    out_channels, 
                    stride=stride,
                    dropout=self.dropout
                )
            )
            in_channels = out_channels
        
        # 分类器
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(cfg[-1][0], self.output_dim)
        )
        
        print(f"TabularMobileNet initialized:")
        print(f"  - Input dim: {self.input_dim}")
        print(f"  - Width multiplier: {self.width_multiplier}")
        print(f"  - Depth multiplier: {self.depth_multiplier}")
        print(f"  - Feature layers: {len(self.features)}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入投影
        x = self.input_projection(x)
        x = self.bn_input(x)
        x = F.relu6(x)
        
        # 添加维度用于卷积
        x = x.unsqueeze(-1)  # (batch, features, 1)
        
        # 通过深度可分离卷积
        for layer in self.features:
            x = layer(x)
        
        # 全局池化和分类
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        
        return x


class DepthwiseSeparableConv1d(nn.Module):
    """深度可分离卷积1D"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.2):
        super(DepthwiseSeparableConv1d, self).__init__()
        
        # 深度卷积
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=1, 
                                  stride=stride, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        
        # 逐点卷积
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu6(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.relu6(x)
        x = self.dropout(x)
        
        return x


class TabularRNN(nn.Module):
    """
    适配表格数据的RNN模型
    将特征重塑为序列进行处理
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(TabularRNN, self).__init__()
        
        self.input_dim = config.get('input_dim')
        self.hidden_size = config.get('hidden_size', 128)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.3)
        self.bidirectional = config.get('bidirectional', True)
        self.cell_type = config.get('cell_type', 'LSTM')
        self.sequence_length = config.get('sequence_length', 16)
        self.output_dim = config.get('output_dim', 1)
        
        # 计算每个时间步的特征数
        self.features_per_step = max(1, self.input_dim // self.sequence_length)
        self.actual_sequence_length = (self.input_dim + self.features_per_step - 1) // self.features_per_step
        
        # 选择RNN类型
        if self.cell_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=self.features_per_step,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout if self.num_layers > 1 else 0,
                bidirectional=self.bidirectional,
                batch_first=True
            )
        elif self.cell_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=self.features_per_step,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout if self.num_layers > 1 else 0,
                bidirectional=self.bidirectional,
                batch_first=True
            )
        else:  # RNN
            self.rnn = nn.RNN(
                input_size=self.features_per_step,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout if self.num_layers > 1 else 0,
                bidirectional=self.bidirectional,
                batch_first=True
            )
        
        # 计算RNN输出维度
        rnn_output_size = self.hidden_size * (2 if self.bidirectional else 1)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(rnn_output_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, self.output_dim)
        )
        
        print(f"TabularRNN ({self.cell_type}) initialized:")
        print(f"  - Input dim: {self.input_dim}")
        print(f"  - Sequence length: {self.actual_sequence_length}")
        print(f"  - Features per step: {self.features_per_step}")
        print(f"  - Hidden size: {self.hidden_size}")
        print(f"  - Bidirectional: {self.bidirectional}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # 填充到合适的长度
        target_length = self.actual_sequence_length * self.features_per_step
        if x.size(1) < target_length:
            padding = target_length - x.size(1)
            x = F.pad(x, (0, padding))
        
        # 重塑为序列
        x = x.view(batch_size, self.actual_sequence_length, self.features_per_step)
        
        # RNN前向传播
        rnn_output, _ = self.rnn(x)
        
        # 使用最后一个时间步的输出（或双向的拼接）
        if self.bidirectional:
            # 取前向和后向的最后隐状态
            hidden = torch.cat([
                rnn_output[:, -1, :self.hidden_size],
                rnn_output[:, 0, self.hidden_size:]
            ], dim=1)
        else:
            hidden = rnn_output[:, -1, :]
        
        # 分类
        output = self.classifier(hidden)
        
        return output


class TabularAlexNet(nn.Module):
    """
    适配表格数据的AlexNet模型
    使用大型全连接层和Dropout
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(TabularAlexNet, self).__init__()
        
        self.input_dim = config.get('input_dim')
        self.dropout = config.get('dropout', 0.5)
        self.output_dim = config.get('output_dim', 1)
        
        # AlexNet风格的特征提取器
        self.features = nn.Sequential(
            # 第一层：大幅降维
            nn.Linear(self.input_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            
            # 第二层：继续降维
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            
            # 第三层：进一步压缩
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
        )
        
        # AlexNet风格的分类器
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            
            nn.Linear(256, self.output_dim),
        )
        
        # 权重初始化（AlexNet风格）
        self._initialize_weights()
        
        print(f"TabularAlexNet initialized:")
        print(f"  - Input dim: {self.input_dim}")
        print(f"  - Dropout: {self.dropout}")
        print(f"  - Architecture: {self.input_dim} -> 4096 -> 2048 -> 1024 -> 512 -> 256 -> {self.output_dim}")
    
    def _initialize_weights(self):
        """AlexNet风格的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_deep_learning_model(model_name: str, config: Dict[str, Any]) -> nn.Module:
    """
    创建深度学习模型
    
    Args:
        model_name: 模型名称
        config: 模型配置
        
    Returns:
        模型实例
    """
    model_classes = {
        'mlp': None,  # 使用现有的MLP模型
        'transformer': TabularTransformer,
        'resnet': TabularResNet,
        'alexnet': TabularAlexNet,
        'mobilenet': TabularMobileNet,
        'rnn': TabularRNN,
        'lstm': TabularRNN  # LSTM使用RNN类但设置cell_type='LSTM'
    }
    
    if model_name == 'mlp':
        # 使用现有的分类或回归模型
        from .classification import ClassificationModel
        from .regression import RegressionModel
        
        task_type = config.get('task_type', 'classification')
        if task_type == 'classification':
            return ClassificationModel(config)
        else:
            return RegressionModel(config)
    
    elif model_name in model_classes:
        model_class = model_classes[model_name]
        if model_name == 'lstm':
            config['cell_type'] = 'LSTM'
        return model_class(config)
    
    else:
        raise ValueError(f"Unknown deep learning model: {model_name}")


if __name__ == "__main__":
    # 测试代码
    test_config = {
        'input_dim': 1306,
        'output_dim': 2,
        'task_type': 'classification'
    }
    
    models = ['transformer', 'resnet', 'alexnet', 'mobilenet', 'rnn', 'lstm']
    
    for model_name in models:
        print(f"\n=== Testing {model_name} ===")
        model = create_deep_learning_model(model_name, test_config)
        
        # 测试前向传播
        x = torch.randn(32, 1306)
        with torch.no_grad():
            y = model(x)
            print(f"Output shape: {y.shape}")