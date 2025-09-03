import torch
from torch.utils.data import DataLoader

# ==== 1. 导入项目内的模块 ====
from src.data.load import load_and_filter_data  # 使用完整的数据处理流程
from src.data.preprocess import preprocess_data
from src.data.dataset import CognitiveDataset
from src.models.classification import ClassificationModel
from src.models.regression import RegressionModel
from src.training.train import train_single_task
from src.training.evaluate import evaluate_single_task
from src.config import load_config


def main():
    # ==== 2. 加载配置 ==== 
    config = load_config("config/default.yaml")
    
    # ==== 3. 加载 & 筛选数据 ====
    df_filtered, df_cognition, filter_info = load_and_filter_data(
        raw_data_path=config["data"]["path"],
        min_waves=config["data"]["filters"]["min_waves"],
        min_non_sparsity=config["data"]["filters"]["min_non_sparsity"]
    )

    # ==== 4. 数据清洗 & 缺失处理 ====
    df_clean, process_info = preprocess_data(
        df_filtered,
        perform_imputation=config["preprocessing"]["perform_imputation"],
        perform_standardization=config["preprocessing"]["perform_standardization"],
        handle_outliers_flag=config["preprocessing"]["handle_outliers"],
        outlier_method=config["preprocessing"]["outlier_method"]
    )
    
    # ==== 5. 准备 Dataset & DataLoader ====
    task_type = config["model"]["task_type"]  # 获取任务类型（分类或回归）
    
    train_dataset = CognitiveDataset(df_clean, split="train", task_type=task_type)
    val_dataset = CognitiveDataset(df_clean, split="val", task_type=task_type)
    test_dataset = CognitiveDataset(df_clean, split="test", task_type=task_type)

    train_loader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["train"]["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["train"]["batch_size"])

    # ==== 6. 定义模型 ====
    # 设置输入维度
    config["model"]["input_dim"] = train_dataset.get_feature_dim()
    
    if task_type == "classification":
        model = ClassificationModel(config["model"])
    elif task_type == "regression":
        model = RegressionModel(config["model"])
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    device = torch.device(config["train"]["device"])
    model.to(device)

    # ==== 7. 训练模型 ====
    print(f"\n开始训练{task_type}模型...")
    
    # 创建优化器和调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config["train"]["early_stopping"]["patience"]//2
    )
    
    # 选择损失函数
    if task_type == "classification":
        criterion = torch.nn.CrossEntropyLoss()
        task_name = "Classification"
    else:  # regression
        criterion = torch.nn.MSELoss()
        task_name = "Regression"
    
    # 训练模型
    history = train_single_task(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        task_name=task_name,
        num_epochs=config["train"]["num_epochs"],
        patience=config["train"]["early_stopping"]["patience"],
        min_delta=config["train"]["early_stopping"]["min_delta"],
        save_best_model=True,
        model_save_path=f'{config["output"]["model_save_dir"]}/{task_name.lower()}_best_model.pth'
    )

    # ==== 8. 评估模型 ====
    print("\n开始评估模型...")
    evaluation_results = evaluate_single_task(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        task_name=task_name,
        save_results=True,
        results_dir=config["output"]["results_save_dir"]
    )
    
    print(f"\n=== {task_name}任务完成 ===")
    print(f"训练历史: {len(history['train_losses'])} 个epoch")
    print(f"模型已保存到: {config['output']['model_save_dir']}")
    print(f"结果已保存到: {config['output']['results_save_dir']}")


if __name__ == "__main__":
    main()
