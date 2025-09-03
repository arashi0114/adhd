# ADHD认知障碍预测系统

基于ELSA数据的老年人群认知障碍早筛和发病时间预测研究项目。

## 🎯 项目概述

本项目实现了一个完整的机器学习实验系统，用于：

- **早筛任务**：二分类预测老年人是否会发生认知障碍
- **发病时间预测**：回归任务预测认知障碍发病时间
- **多模型比较**：批量训练深度学习和传统机器学习模型
- **完整评估**：提供详细的性能指标、可视化和统计分析

## 📊 数据规格

- **数据来源**：ELSA（English Longitudinal Study of Ageing）
- **数据位置**：`data/cleaned/` 目录下的CSV文件
- **数据规模**：3803样本 × 1316特征
- **标签格式**：
  - 分类：`racogimp_label` (-1=未发病，其他数字=发病波次)
  - 回归：`r{wave}cogimpt_label` (发病时间预测)

## 🏗️ 系统架构

```
├── main.py                      # 🚀 主入口文件
├── config/
│   └── config.yaml             # ⚙️ 统一配置文件
└── src/
    ├── config/                 # 配置管理
    │   ├── __init__.py
    │   └── config_manager.py
    ├── data/                   # 数据处理
    │   ├── __init__.py
    │   ├── loader.py          # CSV数据加载器
    │   ├── preprocessor.py    # 数据预处理器
    │   └── dataset.py         # PyTorch数据集
    ├── models/                # 模型定义
    │   ├── __init__.py
    │   ├── deep_learning.py   # 深度学习模型
    │   └── traditional_ml.py  # 传统ML模型
    ├── training/              # 模型训练
    │   ├── __init__.py
    │   └── train_dl.py        # 深度学习训练器
    └── evaluation/            # 模型评估
        ├── __init__.py
        └── evaluator.py       # 综合评估系统
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone <repository-url>
cd adhd

# 安装依赖
pip install -r requirements.txt

# 可选：安装SHAP用于模型解释
pip install shap
```

### 2. 数据准备

确保 `data/cleaned/` 目录包含CSV数据文件：
```
data/
└── cleaned/
    └── your_data.csv  # 系统自动选择最新文件
```

### 3. 运行实验

```bash
# 🎯 完整实验（推荐）
python main.py

# 🔧 自定义配置
python main.py --config config/config.yaml

# 📝 分步骤执行
python main.py --steps data training
python main.py --steps evaluation

# 🎲 指定任务
python main.py --tasks classification
python main.py --tasks regression
python main.py --tasks classification regression

# 🤖 指定模型
python main.py --models deep_learning_transformer traditional_ml_xgboost

# 🔀 组合使用
python main.py --steps training evaluation --tasks classification --models deep_learning_transformer
```

## 🤖 支持的模型

### 深度学习模型

| 模型 | 描述 | 适用场景 |
|------|------|----------|
| **MLP** | 多层感知机 | 基础深度学习baseline |
| **Transformer** | 注意力机制（适配表格数据） | 复杂特征交互建模 |
| **ResNet** | 残差网络（1D） | 深层网络训练 |
| **AlexNet** | 大型全连接网络 | 高容量模型 |
| **MobileNet** | 深度可分离卷积概念 | 轻量级高效模型 |
| **RNN/LSTM** | 循环神经网络 | 序列特征建模 |

### 传统机器学习模型

| 模型 | 描述 | 适用场景 |
|------|------|----------|
| **XGBoost** | 梯度提升树 | 表格数据金标准 |
| **Random Forest** | 随机森林 | 鲁棒性强，可解释 |
| **SVM** | 支持向量机 | 小样本高维数据 |
| **Linear Models** | 线性/逻辑回归 | 基础模型，高可解释性 |

## 📈 数据预处理

系统自动进行完整的数据预处理：

### 🔧 预处理步骤
1. **缺失值处理**：5重多重插补（IterativeImputer）
2. **异常值检测**：IQR方法识别和处理异常值
3. **数据标准化**：StandardScaler标准化数值特征
4. **数据划分**：训练(80%) / 验证(10%) / 测试(10%)

### ⚙️ 配置选项
```yaml
preprocessing:
  multiple_imputation:
    enabled: true
    n_imputations: 5
  outlier_detection:
    enabled: true  
    method: "iqr"
    threshold: 1.5
  scaling:
    enabled: true
    method: "standard"
```

## 📊 评估体系

### 分类任务指标
- **基础指标**：准确率、精确率、召回率、F1分数
- **概率指标**：ROC-AUC、PR-AUC
- **可视化**：混淆矩阵、ROC曲线、PR曲线

### 回归任务指标
- **误差指标**：MSE、RMSE、MAE
- **拟合指标**：R²分数、MAPE
- **可视化**：残差图、Q-Q图

### 性能指标
- **计算开销**：训练时间、推理时间、模型参数量
- **资源使用**：内存占用、FLOPS计算量

## 🎛️ 高级功能

### 1. 交叉验证
```bash
# 启用10折交叉验证
# 在config.yaml中设置
advanced_analysis:
  cross_validation:
    enabled: true
    folds: 10
```

### 2. 模型可解释性
```bash
# SHAP分析（需要安装shap）
advanced_analysis:
  interpretability:
    shap:
      enabled: true
      max_display: 20
```

### 3. 统计显著性测试
```bash
# 模型间比较的统计测试
advanced_analysis:
  statistical_tests:
    enabled: true
    confidence_level: 0.95
```

## 📁 输出结果

实验完成后，结果保存在时间戳命名的目录中：

```
outputs/ADHD_cognitive_prediction_20231201_143022/
├── models/                          # 🏛️ 训练好的模型
│   ├── deep_learning_transformer_classification.pth
│   ├── traditional_ml_xgboost_classification.joblib
│   └── ...
├── results/                         # 📊 实验结果
│   ├── classification_model_comparison.csv
│   ├── regression_model_comparison.csv
│   └── final_experiment_report.json
├── plots/                          # 📈 可视化图表
│   ├── transformer_confusion_matrix.png
│   ├── xgboost_roc_curve.png
│   ├── regression_residuals.png
│   └── ...
└── logs/                           # 📝 日志文件
    ├── data_preparation_report.json
    └── config_used.yaml
```

### 关键输出文件

| 文件 | 描述 |
|------|------|
| `final_experiment_report.json` | 📄 完整实验总结报告 |
| `{task}_model_comparison.csv` | 📋 模型性能对比表 |
| `{model}_*.png` | 🖼️ 各种可视化图表 |
| `{model}_{task}.pth/.joblib` | 💾 训练好的模型文件 |

## ⚙️ 配置说明

主配置文件 `config/config.yaml` 包含所有设置：

### 📋 主要配置项

```yaml
# 🎲 随机种子（确保可复现）
random_seed: 42

# 📊 数据设置
data:
  data_dir: "data/cleaned"
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1

# 🤖 启用的模型
models:
  deep_learning:
    transformer:
      enabled: true
    resnet:
      enabled: true
  traditional_ml:
    xgboost:
      enabled: true
    random_forest:
      enabled: true

# 🏋️ 训练设置
training:
  deep_learning:
    batch_size: 64
    num_epochs: 100
    learning_rate: 0.001
    device: "auto"  # 自动选择GPU/CPU
```

## 🛠️ 命令行参数

| 参数 | 描述 | 示例 |
|------|------|------|
| `--config` | 指定配置文件 | `--config my_config.yaml` |
| `--steps` | 执行的步骤 | `--steps data training evaluation` |
| `--tasks` | 运行的任务 | `--tasks classification regression` |
| `--models` | 训练的模型 | `--models deep_learning_transformer` |

## 🔬 使用案例

### 基础使用
```bash
# 🎯 运行所有配置的模型和任务
python main.py
```

### 快速验证
```bash
# 🚀 只训练几个核心模型
python main.py --models deep_learning_mlp traditional_ml_xgboost
```

### 深度分析
```bash
# 🔍 只做分类任务的深入分析
python main.py --tasks classification --steps evaluation
```

### 增量实验
```bash
# 📈 从训练步骤继续（跳过数据准备）
python main.py --steps training evaluation
```

## 🚨 常见问题

### Q: CUDA内存不足怎么办？
A: 减小配置中的batch_size，或设置device为"cpu"

### Q: 找不到数据文件？
A: 确保`data/cleaned/`目录存在且包含CSV文件

### Q: SHAP分析报错？
A: 安装shap库：`pip install shap`，或在配置中禁用SHAP

### Q: 实验中断了怎么恢复？
A: 使用`--steps training evaluation`从训练步骤继续

### Q: 如何只运行特定模型？
A: 使用`--models`参数指定，如`--models deep_learning_transformer traditional_ml_xgboost`

## 📋 系统要求

- **Python**: 3.8+
- **主要依赖**: PyTorch, scikit-learn, pandas, numpy, matplotlib
- **可选依赖**: SHAP (模型解释), seaborn (可视化增强)
- **硬件**: 支持CPU和GPU训练

## 🔄 版本信息

- **当前版本**: 2.0 (重构版)
- **更新日期**: 2024-12-01
- **兼容性**: 完全重写，不向后兼容1.x版本


