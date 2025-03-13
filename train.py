from data_loader import get_data_loaders, FacialExpressionDataset
from model import train_model, FacialActionModelWithAuxLoss, evaluate_model
import torch
import os
from datetime import datetime

# 示例用法
if __name__ == "__main__":
    from data_loader import get_data_loaders
    
    # 创建实验日志目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = f'experiment_{timestamp}'
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"实验结果将保存在: {experiment_dir}")
    
    # 获取数据加载器
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        dataset_root="./datasets",
        batch_size=32
    )
    
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")
    print(f"类别数: {num_classes}")
    
    # 创建模型
    model = FacialActionModelWithAuxLoss(num_classes=num_classes, backbone="shufflenet")
    
    # 训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 使用更新后的训练函数，设置可视化参数
    trained_model = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=30, 
        learning_rate=0.001, 
        device=device,
        visualize_every=2,  # 每2个epoch可视化一次
        samples_to_visualize=5,  # 每次可视化5个样本
        log_dir=experiment_dir  # 保存在实验目录
    )
    
    # 评估模型
    print("\n开始评估最终模型...")
    metrics = evaluate_model(
        trained_model, 
        test_loader, 
        device=device,
        output_dir=os.path.join(experiment_dir, 'evaluation')
    )
    
    print("\n评估结果:")
    print(f"测试损失: {metrics['test_loss']:.4f}")
    print(f"整体准确率: {metrics['accuracy']:.4f}")
    print(f"平均F1分数: {metrics['f1_score']:.4f}")
    
    # # 也可以加载并评估最佳模型
    # best_model_path = os.path.join(experiment_dir, 'best_facial_action_model.pth')
    # if os.path.exists(best_model_path):
    #     print("\n加载并评估最佳模型...")
    #     best_model = FacialActionModelWithAuxLoss(num_classes=num_classes, backbone="resnet50")
    #     best_model.load_state_dict(torch.load(best_model_path))
        
    #     best_metrics = evaluate_model(
    #         best_model, 
    #         test_loader, 
    #         device=device,
    #         output_dir=os.path.join(experiment_dir, 'best_model_evaluation')
    #     )
        
    #     print("\n最佳模型评估结果:")
    #     print(f"测试损失: {best_metrics['test_loss']:.4f}")
    #     print(f"整体准确率: {best_metrics['accuracy']:.4f}")
    #     print(f"平均F1分数: {best_metrics['f1_score']:.4f}")