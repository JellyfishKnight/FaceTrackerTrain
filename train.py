import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# 带有幅度标签的数据集类
class FacialExpressionMagnitudeDataset(Dataset):
    def __init__(self, data_dir, annotations_file, transform=None):
        """
        初始化数据集
        
        Args:
            data_dir (str): 图像数据目录
            annotations_file (str): 包含图像路径、动作类型和幅度标签的CSV文件
            transform: 图像变换
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # 读取标注文件
        # 假设CSV格式: image_path,action_type,magnitude
        # 例如: frames/video1/frame001.jpg,cheekPuffLeft,0.75
        self.annotations = pd.read_csv(annotations_file)
        
        # 所有可能的动作类型
        self.action_types = list(self.annotations['action_type'].unique())
        self.action_to_idx = {action: idx for idx, action in enumerate(self.action_types)}

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # 获取图像路径和标签
        img_path = os.path.join(self.data_dir, self.annotations.iloc[idx, 0])
        action_type = self.annotations.iloc[idx, 1]
        magnitude = self.annotations.iloc[idx, 2]
        
        # 读取图像
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # 确保图像尺寸为256×256
        if image.shape != (256, 256):
            image = cv2.resize(image, (256, 256))
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        else:
            # 默认变换：转换为张量并归一化
            image = torch.from_numpy(image).float() / 255.0
            image = image.unsqueeze(0)  # 添加通道维度
        
        # 创建标签向量
        # 两种方式：
        # 1. 单独的动作类型(分类)和幅度(回归)
        action_idx = self.action_to_idx[action_type]
        action_onehot = torch.zeros(len(self.action_types))
        action_onehot[action_idx] = 1.0
        
        # 2. 为每个可能的动作参数创建直接输出值
        # 创建45个参数的目标向量
        param_target = torch.zeros(45)
        
        # 根据动作类型和幅度设置相应参数
        # 这里需要一个映射表，将动作类型映射到对应的参数索引
        action_to_param_idx = {
            "cheekPuffLeft": 0,
            "cheekPuffRight": 1,
            # ... 其他映射
        }
        
        if action_type in action_to_param_idx:
            param_idx = action_to_param_idx[action_type]
            param_target[param_idx] = magnitude
        
        return {
            'image': image,
            'action_type': action_idx,
            'action_onehot': action_onehot,
            'magnitude': torch.tensor(magnitude, dtype=torch.float32),
            'param_target': param_target
        }

# 多任务模型：同时预测动作类型和幅度
class FacialExpressionMagnitudeModel(nn.Module):
    def __init__(self, num_action_types, num_params=45):
        super(FacialExpressionMagnitudeModel, self).__init__()
        
        # 共享的特征提取部分
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 特征扁平化后的尺寸
        self.feature_size = 256 * 16 * 16
        
        # 动作类型分类分支
        self.action_classifier = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_action_types)
        )
        
        # 幅度回归分支
        self.magnitude_regressor = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()  # 确保幅度在0-1范围内
        )
        
        # 参数直接预测分支
        self.param_regressor = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_params),
            nn.Sigmoid()  # 确保所有参数值在0-1范围内
        )
    
    def forward(self, x):
        # 提取特征
        features = self.features(x)
        features = torch.flatten(features, 1)
        
        # 预测动作类型
        action_logits = self.action_classifier(features)
        
        # 预测幅度
        magnitude = self.magnitude_regressor(features)
        
        # 预测所有参数
        params = self.param_regressor(features)
        
        return {
            'action_logits': action_logits,
            'magnitude': magnitude,
            'params': params
        }

# 改进版条件幅度预测模型
class ConditionalMagnitudeModel(nn.Module):
    def __init__(self, num_action_types, num_params=45):
        super(ConditionalMagnitudeModel, self).__init__()
        
        # 特征提取部分
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            self._make_residual_block(64, 64, 2),
            self._make_residual_block(64, 128, 2, stride=2),
            self._make_residual_block(128, 256, 2, stride=2),
            self._make_residual_block(256, 512, 2, stride=2)
        )
        
        # 全局池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 特征维度
        self.feature_dim = 512
        
        # 动作识别分支
        self.action_branch = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_action_types)
        )
        
        # 幅度预测分支 - 为每种动作类型创建单独的回归器
        self.magnitude_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feature_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
                nn.Sigmoid()
            ) for _ in range(num_action_types)
        ])
        
        # 参数直接预测分支
        self.param_regressor = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_params),
            nn.Sigmoid()
        )
    
    def _make_residual_block(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        # 这里简化了ResNet中的残差块实现
        # 实际使用时应该使用proper residual blocks
        
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                                   stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 特征提取
        x = self.features(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        
        # 动作类型预测
        action_logits = self.action_branch(features)
        action_probs = torch.softmax(action_logits, dim=1)
        
        # 为每种动作类型预测幅度
        all_magnitudes = torch.zeros(x.size(0), len(self.magnitude_branches)).to(x.device)
        for i, branch in enumerate(self.magnitude_branches):
            magnitude = branch(features)
            all_magnitudes[:, i] = magnitude.squeeze()
        
        # 创建最终参数输出
        # 方法1: 直接从参数回归器预测
        direct_params = self.param_regressor(features)
        
        # 方法2: 根据动作类型和幅度计算参数（条件预测）
        # 这里使用action_probs作为权重，与对应的幅度相乘
        # 这种方法需要预定义的动作类型到参数的映射矩阵
        
        return {
            'action_logits': action_logits,
            'action_probs': action_probs,
            'magnitudes': all_magnitudes,
            'params': direct_params
        }

# 定义多任务损失函数
class MultiTaskLoss(nn.Module):
    def __init__(self, action_weight=1.0, magnitude_weight=1.0, param_weight=1.0):
        super(MultiTaskLoss, self).__init__()
        self.action_weight = action_weight
        self.magnitude_weight = magnitude_weight
        self.param_weight = param_weight
        
        self.action_criterion = nn.CrossEntropyLoss()
        self.magnitude_criterion = nn.MSELoss()
        self.param_criterion = nn.MSELoss()
    
    def forward(self, predictions, targets):
        action_loss = self.action_criterion(
            predictions['action_logits'], 
            targets['action_type']
        )
        
        magnitude_loss = self.magnitude_criterion(
            predictions['magnitude'].squeeze(), 
            targets['magnitude']
        )
        
        param_loss = self.param_criterion(
            predictions['params'], 
            targets['param_target']
        )
        
        # 总损失是加权和
        total_loss = (
            self.action_weight * action_loss + 
            self.magnitude_weight * magnitude_loss + 
            self.param_weight * param_loss
        )
        
        return {
            'total_loss': total_loss,
            'action_loss': action_loss,
            'magnitude_loss': magnitude_loss,
            'param_loss': param_loss
        }

# 训练函数
def train_magnitude_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30):
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    
    history = {
        'train_total_loss': [], 'train_action_loss': [], 
        'train_magnitude_loss': [], 'train_param_loss': [],
        'val_total_loss': [], 'val_action_loss': [],
        'val_magnitude_loss': [], 'val_param_loss': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 训练阶段
        model.train()
        running_losses = {'total': 0.0, 'action': 0.0, 'magnitude': 0.0, 'param': 0.0}
        
        for batch in tqdm(train_loader):
            inputs = batch['image'].to(device)
            targets = {
                'action_type': batch['action_type'].to(device),
                'magnitude': batch['magnitude'].to(device),
                'param_target': batch['param_target'].to(device)
            }
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            losses = criterion(outputs, targets)
            total_loss = losses['total_loss']
            
            # 反向传播和优化
            total_loss.backward()
            optimizer.step()
            
            # 累加损失
            batch_size = inputs.size(0)
            running_losses['total'] += losses['total_loss'].item() * batch_size
            running_losses['action'] += losses['action_loss'].item() * batch_size
            running_losses['magnitude'] += losses['magnitude_loss'].item() * batch_size
            running_losses['param'] += losses['param_loss'].item() * batch_size
        
        # 计算平均损失
        dataset_size = len(train_loader.dataset)
        epoch_losses = {k: v / dataset_size for k, v in running_losses.items()}
        
        # 记录训练损失
        history['train_total_loss'].append(epoch_losses['total'])
        history['train_action_loss'].append(epoch_losses['action'])
        history['train_magnitude_loss'].append(epoch_losses['magnitude'])
        history['train_param_loss'].append(epoch_losses['param'])
        
        # 验证阶段
        model.eval()
        running_losses = {'total': 0.0, 'action': 0.0, 'magnitude': 0.0, 'param': 0.0}
        
        with torch.no_grad():
            for batch in tqdm(val_loader):
                inputs = batch['image'].to(device)
                targets = {
                    'action_type': batch['action_type'].to(device),
                    'magnitude': batch['magnitude'].to(device),
                    'param_target': batch['param_target'].to(device)
                }
                
                # 前向传播
                outputs = model(inputs)
                
                # 计算损失
                losses = criterion(outputs, targets)
                
                # 累加损失
                batch_size = inputs.size(0)
                running_losses['total'] += losses['total_loss'].item() * batch_size
                running_losses['action'] += losses['action_loss'].item() * batch_size
                running_losses['magnitude'] += losses['magnitude_loss'].item() * batch_size
                running_losses['param'] += losses['param_loss'].item() * batch_size
        
        # 计算平均损失
        dataset_size = len(val_loader.dataset)
        epoch_losses = {k: v / dataset_size for k, v in running_losses.items()}
        
        # 记录验证损失
        history['val_total_loss'].append(epoch_losses['total'])
        history['val_action_loss'].append(epoch_losses['action'])
        history['val_magnitude_loss'].append(epoch_losses['magnitude'])
        history['val_param_loss'].append(epoch_losses['param'])
        
        print(f"Train Total Loss: {history['train_total_loss'][-1]:.4f}")
        print(f"Val Total Loss: {history['val_total_loss'][-1]:.4f}")
        
        # 如果是最佳模型，保存权重
        if epoch_losses['total'] < best_loss:
            best_loss = epoch_losses['total']
            best_model_wts = model.state_dict().copy()
            
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_wts,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_facial_expression_magnitude_model.pth')
        
        # 学习率调整
        scheduler.step(epoch_losses['total'])
        
        print()
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    
    return model, history

# 数据集准备示例（如何创建带有幅度标注的数据集）
def create_magnitude_dataset_example():
    """
    示例函数：如何从视频中提取帧并创建带有幅度标注的数据集
    """
    # 数据结构
    data = []
    
    # 假设的视频目录
    videos_dir = './videos'
    frames_dir = './frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    # 视频和动作映射
    video_actions = {
        'cheekpuffs.mp4': [
            {'start_frame': 10, 'end_frame': 30, 'action': 'cheekPuffLeft', 'max_magnitude_frame': 20},
            {'start_frame': 40, 'end_frame': 60, 'action': 'cheekPuffRight', 'max_magnitude_frame': 50},
            # ...
        ],
        # 其他视频...
    }
    
    # 处理每个视频
    for video_name, actions in video_actions.items():
        video_path = os.path.join(videos_dir, video_name)
        video_basename = os.path.splitext(video_name)[0]
        video_frames_dir = os.path.join(frames_dir, video_basename)
        os.makedirs(video_frames_dir, exist_ok=True)
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 提取帧
        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            
            # 保存帧
            frame_path = os.path.join(video_frames_dir, f'frame_{frame_idx:04d}.jpg')
            cv2.imwrite(frame_path, frame)
            
            # 检查这一帧是否在任何动作的时间范围内
            for action_info in actions:
                if action_info['start_frame'] <= frame_idx <= action_info['end_frame']:
                    action_name = action_info['action']
                    
                    # 计算幅度 - 基于与最大幅度帧的距离
                    max_frame = action_info['max_magnitude_frame']
                    start_frame = action_info['start_frame']
                    end_frame = action_info['end_frame']
                    
                    # 简单的幅度计算 - 从动作开始到最大幅度点线性增加，然后线性减少
                    if frame_idx <= max_frame:
                        # 上升阶段
                        magnitude = (frame_idx - start_frame) / (max_frame - start_frame)
                    else:
                        # 下降阶段
                        magnitude = 1.0 - (frame_idx - max_frame) / (end_frame - max_frame)
                    
                    # 确保幅度在[0,1]范围内
                    magnitude = max(0.0, min(1.0, magnitude))
                    
                    # 添加到数据列表
                    rel_path = os.path.join(video_basename, f'frame_{frame_idx:04d}.jpg')
                    data.append([rel_path, action_name, magnitude])
        
        cap.release()
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(data, columns=['image_path', 'action_type', 'magnitude'])
    df.to_csv('facial_expression_magnitudes.csv', index=False)
    
    print(f"创建了包含 {len(df)} 个带幅度标注的样本")
    return df

# 主函数示例
def main_with_magnitude():
    # 数据变换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 创建数据集
    data_dir = './facial_expression_data'  # 替换为实际数据目录
    annotations_file = 'facial_expression_magnitudes.csv'  # 带有幅度标注的CSV文件
    
    # 首次运行时，创建带幅度标注的数据集
    # create_magnitude_dataset_example()
    
    # 加载数据集
    dataset = FacialExpressionMagnitudeDataset(data_dir, annotations_file, transform)
    
    # 获取动作类型数量
    num_action_types = len(dataset.action_types)
    
    # 分割数据集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size + test_size])
    val_dataset, test_dataset = torch.utils.data.random_split(
        val_test_dataset, [val_size, test_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 创建模型
    # model = FacialExpressionMagnitudeModel(num_action_types).to(device)
    model = ConditionalMagnitudeModel(num_action_types).to(device)
    
    # 定义损失函数
    criterion = MultiTaskLoss(action_weight=1.0, magnitude_weight=1.0, param_weight=1.0)
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    
    # 训练模型
    model, history = train_magnitude_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30)
    
    # 保存模型
    torch.save(model, 'facial_expression_magnitude_model.pth')
    
    print("训练完成!")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main_with_magnitude()