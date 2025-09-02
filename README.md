# ADHD认知障碍预测项目

## 项目概述

本项目使用深度学习方法对ADHD相关的认知障碍进行预测，支持分类和回归两种任务类型。

## 项目结构

```
adhd/
├── main.py                 # 主程序入口
├── config/
│   └── default.yaml       # 配置文件
├── src/
│   ├── config.py          # 配置管理模块
│   ├── data/              # 数据处理模块
│   │   ├── load.py        # 数据加载和筛选
│   │   ├── preprocess.py  # 数据预处理
│   │   └── dataset.py     # PyTorch数据集定义
│   ├── models/            # 模型定义
│   │   ├── classification.py  # 分类模型
│   │   └── regression.py      # 回归模型
│   ├── training/          # 训练和评估
│   │   ├── train.py       # 训练函数
│   │   └── evaluate.py    # 评估函数
│   └── utils/
│       └── utils.py       # 工具函数
├── data/                  # 数据目录
├── outputs/               # 输出目录
│   ├── models/           # 保存的模型
│   ├── results/          # 评估结果
│   └── logs/             # 日志文件
└── requirements.txt      # 依赖库
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置说明

编辑 `config/default.yaml` 文件来配置项目参数：

- `data.path`: 数据文件路径
- `model.task_type`: 任务类型 ("classification" 或 "regression")
- `train.batch_size`: 批次大小
- `train.learning_rate`: 学习率
- `train.num_epochs`: 最大训练轮数

## 运行项目

```bash
python main.py
```

## 支持的功能

### 数据处理
- 自动加载和筛选ELSA数据
- 缺失值插补
- 数据标准化
- 异常值处理
- 特征选择

### 模型
- **分类模型**: 用于认知障碍二分类任务
- **回归模型**: 用于认知分数预测任务
- 可配置的网络架构
- 支持多种激活函数和正则化方法

### 训练
- 早停机制
- 学习率调度
- 模型检查点保存
- 训练过程可视化

### 评估
- 分类指标: 准确率、精确率、召回率、F1分数、ROC曲线
- 回归指标: MSE、RMSE、MAE、R²分数
- 混淆矩阵和残差分析
- 结果可视化和保存

## 输出文件

训练完成后，在 `outputs/` 目录下可以找到：
- `models/`: 训练好的模型文件
- `results/`: 评估结果和可视化图表
- `logs/`: 训练日志

## 注意事项

1. 确保数据文件路径正确
2. 根据可用GPU内存调整批次大小
3. 分类任务需要足够的正负样本
4. 回归任务的目标值应该经过适当的预处理