import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

class SpatialAttention(nn.Module):
    """空间注意力模块，关注图像中的重要区域"""
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, kernel_size=1)
        
    def forward(self, x):
        # 生成空间注意力图
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = torch.sigmoid(attention)
        
        # 应用注意力
        return x * attention

class FacialActionModel(nn.Module):
    """面部动作识别模型"""
    def __init__(self, num_classes=43, pretrained=True, backbone="resnet50"):
        super(FacialActionModel, self).__init__()
        
        # 选择骨干网络
        if backbone == "resnet50":
            base_model = models.resnet50(pretrained=pretrained)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])
            feature_dim = 2048
        elif backbone == "resnet18":
            base_model = models.resnet18(pretrained=pretrained)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])
            feature_dim = 512
        elif backbone == "efficientnet":
            base_model = models.efficientnet_b0(pretrained=pretrained)
            self.feature_extractor = base_model.features
            feature_dim = 1280
        else:
            raise ValueError(f"不支持的骨干网络: {backbone}")
        
        # 注意力模块
        self.attention = SpatialAttention(feature_dim)
        
        # 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 回归层
        self.fc1 = nn.Linear(feature_dim, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.output = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # 特征提取
        features = self.feature_extractor(x)
        
        # 应用注意力
        attended_features = self.attention(features)
        
        # 全局池化
        pooled = self.global_pool(attended_features).view(x.size(0), -1)
        
        # 回归预测
        x = self.fc1(pooled)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.output(x)
        
        # 使用sigmoid确保输出在[0,1]范围内，符合幅度的定义
        return torch.sigmoid(x)

class FacialActionModelWithAuxLoss(nn.Module):
    """带有辅助损失的面部动作识别模型，用于处理动作依赖关系"""
    def __init__(self, num_classes=43, pretrained=True, backbone="resnet50"):
        super(FacialActionModelWithAuxLoss, self).__init__()
        
        # 主模型
        self.main_model = FacialActionModel(num_classes, pretrained, backbone)
        
        # 动作分组（根据提供的ACTION_DEPENDENCIES分组）
        self.jaw_group = nn.Linear(256, 4)  # jawOpen, jawLeft, jawRight, jawForward
        self.tongue_group = nn.Linear(256, 12)  # 所有舌头相关动作
        self.mouth_group = nn.Linear(256, 23)  # 所有嘴相关动作
        self.cheek_group = nn.Linear(256, 4)  # 所有脸颊相关动作
        
    def forward(self, x):
        # 使用主模型提取特征
        features = self.main_model.feature_extractor(x)
        attended_features = self.main_model.attention(features)
        pooled = self.main_model.global_pool(attended_features).view(x.size(0), -1)
        
        # 共享特征层
        shared_features = self.main_model.fc1(pooled)
        shared_features = F.relu(shared_features)
        shared_features = self.main_model.dropout(shared_features)
        shared_features = self.main_model.fc2(shared_features)
        shared_features = F.relu(shared_features)
        
        # 主输出
        main_output = self.main_model.output(shared_features)
        main_output = torch.sigmoid(main_output)
        
        # 分组输出用于辅助损失
        jaw_output = torch.sigmoid(self.jaw_group(shared_features))
        tongue_output = torch.sigmoid(self.tongue_group(shared_features))
        mouth_output = torch.sigmoid(self.mouth_group(shared_features))
        cheek_output = torch.sigmoid(self.cheek_group(shared_features))
        
        if self.training:
            return main_output, jaw_output, tongue_output, mouth_output, cheek_output
        else:
            return main_output

def visualize_prediction(model, val_loader, device, epoch, output_dir, num_samples=3, threshold=0.5):
    """可视化模型在验证集上的预测结果"""
    from data_loader import decode_label
    
    model.eval()
    
    # 确保输出目录存在
    vis_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(vis_dir, exist_ok=True)
    
    with torch.no_grad():
        for images, labels, img_paths in val_loader:
            images = images.to(device)
            outputs = model(images)
            
            # 可视化前几个样本
            for i in range(min(num_samples, len(images))):
                image = images[i].cpu()
                label = labels[i]
                prediction = outputs[i].cpu()
                img_path = img_paths[i]
                
                # 反标准化图像以便可视化
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image = image * std + mean
                image = image.permute(1, 2, 0).numpy()
                image = np.clip(image, 0, 1)
                
                # 解码标签和预测
                true_labels = decode_label(label, threshold)
                pred_labels = decode_label(prediction, threshold)
                
                # 创建图像
                plt.figure(figsize=(12, 6))
                
                # 左侧显示图像
                plt.subplot(1, 2, 1)
                plt.imshow(image)
                plt.title(f"Epoch {epoch}: {os.path.basename(img_path)}")
                plt.axis('off')
                
                # 右侧显示标签和预测
                plt.subplot(1, 2, 2)
                plt.axis('off')
                
                # 构建文本
                info_text = "Ground Truth Labels:\n"
                if true_labels:
                    for name, value in true_labels:
                        info_text += f"- {name}: {value:.2f}\n"
                else:
                    info_text += "No active labels found.\n"
                
                info_text += "\nPredicted Labels:\n"
                if pred_labels:
                    for name, value in pred_labels:
                        info_text += f"- {name}: {value:.2f}\n"
                else:
                    info_text += "No predicted labels above threshold.\n"
                
                plt.text(0, 0.9, info_text, fontsize=10, verticalalignment='top', wrap=True)
                
                # 保存图像
                img_name = os.path.basename(img_path).split('.')[0]
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f'epoch_{epoch}_sample_{img_name}.png'))
                plt.close()
            
            # 只处理一个批次
            break

def plot_training_progress(train_losses, val_losses, metrics, output_dir):
    """绘制训练进度图"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()
    
    # 绘制评估指标
    if metrics:
        epochs = list(range(1, len(metrics['accuracy']) + 1))
        
        plt.figure(figsize=(15, 10))
        
        # 准确率
        plt.subplot(2, 2, 1)
        plt.plot(epochs, metrics['accuracy'], 'o-', label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy')
        plt.grid(True)
        
        # F1分数
        plt.subplot(2, 2, 2)
        plt.plot(epochs, metrics['f1_score'], 'o-', label='F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Model F1 Score')
        plt.grid(True)
        
        # 准确率和F1分数对比
        plt.subplot(2, 1, 2)
        plt.plot(epochs, metrics['accuracy'], 'o-', label='Accuracy')
        plt.plot(epochs, metrics['f1_score'], 's-', label='F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Accuracy vs F1 Score')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics.png'))
        plt.close()

def calculate_metrics(model, val_loader, device, threshold=0.5):
    """计算验证集上的性能指标"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            outputs = model(images)
            
            all_preds.append(outputs.cpu())
            all_labels.append(labels)
    
    # 合并所有批次的预测和标签
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 计算性能指标
    predictions = (all_preds > threshold).float()
    correct_predictions = (predictions == all_labels).float()
    accuracy = correct_predictions.mean().item()
    
    # 计算F1分数
    true_positives = (predictions * all_labels).sum(dim=0)
    predicted_positives = predictions.sum(dim=0)
    actual_positives = all_labels.sum(dim=0)
    
    precision = true_positives / (predicted_positives + 1e-8)
    recall = true_positives / (actual_positives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1.mean().item()
    }

# 训练和评估函数
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, device="cuda",
               visualize_every=5, samples_to_visualize=3, log_dir=None):
    """训练面部动作识别模型，并可视化训练进度"""
    model = model.to(device)
    
    # 创建日志目录
    if log_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = f'training_logs_{timestamp}'
    
    os.makedirs(log_dir, exist_ok=True)
    
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()  # 二元交叉熵损失适合多标签问题
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_val_loss = float('inf')
    
    # 用于跟踪的列表
    train_losses = []
    val_losses = []
    eval_metrics = {'accuracy': [], 'f1_score': []}
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            
            if isinstance(model, FacialActionModelWithAuxLoss):
                outputs, jaw_out, tongue_out, mouth_out, cheek_out = model(images)
                
                # 主损失
                loss = criterion(outputs, labels)
                
                # 辅助损失 (简化版，实际实现需要映射标签到各组)
                # 这里只是示意，需要根据实际的标签索引进行调整
                jaw_indices = [4, 5, 6, 41]  # 对应jaw相关动作在标签中的索引
                tongue_indices = list(range(29, 41))  # 舌头相关动作
                mouth_indices = list(range(7, 29)) + [42]  # 嘴相关动作
                cheek_indices = list(range(0, 4))  # 脸颊相关动作
                
                jaw_loss = criterion(jaw_out, labels[:, jaw_indices])
                tongue_loss = criterion(tongue_out, labels[:, tongue_indices])
                mouth_loss = criterion(mouth_out, labels[:, mouth_indices])
                cheek_loss = criterion(cheek_out, labels[:, cheek_indices])
                
                # 组合损失
                aux_weight = 0.2
                total_loss = loss + aux_weight * (jaw_loss + tongue_loss + mouth_loss + cheek_loss)
            else:
                outputs = model(images)
                total_loss = criterion(outputs, labels)
            
            # 反向传播和优化
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                if isinstance(model, FacialActionModelWithAuxLoss):
                    outputs = model(images)  # 在eval模式下只返回主输出
                else:
                    outputs = model(images)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # 计算评估指标
        metrics = calculate_metrics(model, val_loader, device)
        eval_metrics['accuracy'].append(metrics['accuracy'])
        eval_metrics['f1_score'].append(metrics['f1_score'])
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_facial_action_model.pth'))
        
        # 打印训练信息
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Accuracy: {metrics["accuracy"]:.4f}, F1: {metrics["f1_score"]:.4f}')
        
        # 可视化训练进度
        plot_training_progress(train_losses, val_losses, eval_metrics, log_dir)
        
        # 定期可视化预测结果
        if (epoch + 1) % visualize_every == 0 or epoch == 0 or epoch == num_epochs - 1:
            visualize_prediction(model, val_loader, device, epoch + 1, log_dir, num_samples=samples_to_visualize)
        
        # 每个epoch保存模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'metrics': metrics
        }, os.path.join(log_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    return model

def evaluate_model(model, test_loader, device="cuda", threshold=0.5, output_dir=None):
    """评估模型性能"""
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f'evaluation_results_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    model = model.to(device)
    model.eval()
    
    criterion = nn.BCELoss()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # 收集预测和标签用于计算指标
            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())
    
    test_loss /= len(test_loader)
    
    # 合并所有批次的预测和标签
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 计算多标签评估指标
    predictions = (all_preds > threshold).float()
    
    correct_predictions = (predictions == all_labels).float()
    accuracy = correct_predictions.mean().item()
    
    # 逐类别计算准确率
    per_class_accuracy = correct_predictions.mean(dim=0)
    
    # 计算F1分数
    true_positives = (predictions * all_labels).sum(dim=0)
    predicted_positives = predictions.sum(dim=0)
    actual_positives = all_labels.sum(dim=0)
    
    precision = true_positives / (predicted_positives + 1e-8)
    recall = true_positives / (actual_positives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Overall Accuracy: {accuracy:.4f}')
    print(f'Average F1 Score: {f1.mean().item():.4f}')
    
    # 可视化每个类别的性能
    from data_loader import TAGS_MAPPING
    
    # 绘制每个类别的F1分数
    plt.figure(figsize=(15, 10))
    plt.bar(range(len(f1)), f1.numpy())
    plt.title('F1 Score for Each Class')
    plt.xlabel('Class Index')
    plt.ylabel('F1 Score')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'per_class_f1.png'))
    plt.close()
    
    # 绘制混淆矩阵热图 (简化版 - 只显示预测正确和错误的比例)
    plt.figure(figsize=(15, 10))
    plt.imshow(correct_predictions.mean(dim=0).reshape(1, -1), cmap='Blues')
    plt.colorbar(label='Accuracy')
    plt.title('Per-Class Accuracy Heatmap')
    plt.xlabel('Class Index')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_accuracy.png'))
    plt.close()
    
    # 将结果保存到文件
    results = {
        'test_loss': test_loss,
        'accuracy': accuracy,
        'f1_score': f1.mean().item(),
        'per_class_accuracy': per_class_accuracy.tolist(),
        'per_class_f1': f1.tolist()
    }
    
    import json
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # 生成详细的评估报告
    with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w') as f:
        f.write(f'Test Loss: {test_loss:.4f}\n')
        f.write(f'Overall Accuracy: {accuracy:.4f}\n')
        f.write(f'Average F1 Score: {f1.mean().item():.4f}\n\n')
        
        f.write('Per-Class Performance:\n')
        f.write('-' * 80 + '\n')
        f.write(f"{'Class Name':<30} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}\n")
        f.write('-' * 80 + '\n')
        
        for i in range(len(f1)):
            class_idx = i + 1  # 因为TAGS_MAPPING从1开始
            if class_idx in TAGS_MAPPING:
                class_name = TAGS_MAPPING[class_idx]
                f.write(f"{class_name:<30} {per_class_accuracy[i]:.4f}{'':6} {precision[i]:.4f}{'':6} {recall[i]:.4f}{'':6} {f1[i]:.4f}\n")
    
    return results