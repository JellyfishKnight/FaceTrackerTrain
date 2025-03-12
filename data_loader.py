import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class FacialExpressionDataset(Dataset):
    """
    数据集类，用于加载面部表情数据集
    数据集结构假设为:
    - datasets/
        - images/
            - [subfolder]/
                - [subfolder]_[video_name]_[frame_id].jpg
        - labels/
            - [subfolder]/
                - [subfolder]_[video_name]_[frame_id].txt
    """
    def __init__(self, dataset_root="./datasets", split="train", transform=None, test_ratio=0.2, val_ratio=0.1, seed=42):
        """
        初始化数据集
        
        Args:
            dataset_root: 数据集根目录
            split: 分割模式 - 'train', 'val', 或 'test'
            transform: 数据增强转换
            test_ratio: 测试集比例
            val_ratio: 验证集比例
            seed: 随机种子，用于数据集划分
        """
        self.dataset_root = dataset_root
        self.images_dir = os.path.join(dataset_root, "images")
        self.labels_dir = os.path.join(dataset_root, "labels")
        self.transform = transform
        self.split = split
        
        # 获取所有图像文件路径
        self.image_paths = []
        self.label_paths = []
        
        # 获取所有子文件夹
        subfolders = [f for f in os.listdir(self.images_dir) 
                      if os.path.isdir(os.path.join(self.images_dir, f))]
        
        # 遍历每个子文件夹收集文件
        for subfolder in subfolders:
            subfolder_img_dir = os.path.join(self.images_dir, subfolder)
            subfolder_label_dir = os.path.join(self.labels_dir, subfolder)
            
            # 获取所有图像文件
            for img_file in os.listdir(subfolder_img_dir):
                if img_file.endswith('.jpg'):
                    img_path = os.path.join(subfolder_img_dir, img_file)
                    
                    # 构造对应的标签文件路径
                    label_file = img_file.replace('.jpg', '.txt')
                    label_path = os.path.join(subfolder_label_dir, label_file)
                    
                    # 确保标签文件存在
                    if os.path.exists(label_path):
                        self.image_paths.append(img_path)
                        self.label_paths.append(label_path)
        
        # 设置随机种子
        np.random.seed(seed)
        
        # 数据集划分
        n_samples = len(self.image_paths)
        indices = np.random.permutation(n_samples)
        
        test_size = int(test_ratio * n_samples)
        val_size = int(val_ratio * n_samples)
        train_size = n_samples - test_size - val_size
        
        # 根据split选择适当的索引
        if split == 'train':
            self.indices = indices[:train_size]
        elif split == 'val':
            self.indices = indices[train_size:train_size + val_size]
        elif split == 'test':
            self.indices = indices[train_size + val_size:]
        else:
            raise ValueError(f"未知的数据集划分: {split}")
        
        # 获取类别数(标签的维度)
        if len(self.label_paths) > 0:
            # 读取第一个标签文件来确定标签的维度
            with open(self.label_paths[0], 'r') as f:
                first_label = f.read().strip().split()
                self.num_classes = len(first_label)
                print(f"标签维度为 {self.num_classes}")
        else:
            self.num_classes = 43  # 根据原脚本中TAGS的数量减1(不包括neutural)
            print("警告: 未找到标签文件，默认标签维度为 43")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.indices)
    
    def __getitem__(self, idx):
        """获取单个数据样本"""
        # 获取实际索引
        real_idx = self.indices[idx]
        
        # 加载图像
        img_path = self.image_paths[real_idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB
        
        # 加载标签
        label_path = self.label_paths[real_idx]
        with open(label_path, 'r') as f:
            label_str = f.read().strip()
            label = np.array([float(x) for x in label_str.split()], dtype=np.float32)
        
        # 应用数据增强
        if self.transform:
            # 将numpy数组转换为PIL图像
            image = Image.fromarray(image)
            image = self.transform(image)
        else:
            # 默认转换为PyTorch张量
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        # 转换标签为张量
        label = torch.from_numpy(label).float()
        
        # 返回图像路径以便调试
        return image, label, img_path

def get_data_loaders(dataset_root="./datasets", batch_size=32, num_workers=4):
    """
    创建训练集、验证集和测试集的DataLoader
    
    Args:
        dataset_root: 数据集根目录
        batch_size: 批量大小
        num_workers: 数据加载的工作线程数
        
    Returns:
        train_loader, val_loader, test_loader: 三个数据加载器
    """
    # 定义数据转换
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = FacialExpressionDataset(
        dataset_root=dataset_root,
        split='train',
        transform=train_transform
    )
    
    val_dataset = FacialExpressionDataset(
        dataset_root=dataset_root,
        split='val',
        transform=val_test_transform
    )
    
    test_dataset = FacialExpressionDataset(
        dataset_root=dataset_root,
        split='test',
        transform=val_test_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.num_classes

# 标签的映射，与process_datasets.py中的TAGS一致
TAGS_MAPPING = {
    1: "cheekPuffLeft",
    2: "cheekPuffRight",
    3: "cheekSuckLeft",
    4: "cheekSuckRight",
    5: "jawForward",
    6: "jawLeft",
    7: "jawRight",
    8: "noseSneerLeft",
    9: "noseSneerRight",
    10: "mouthPressLeft",
    11: "mouthPressRight",
    12: "lipPucker",
    13: "lipFunnel",
    14: "mouthLeft",
    15: "mouthRight",
    16: "lipSuckLower",
    17: "lipSuckUpper",
    18: "mouthSmileLeft",
    19: "mouthSmileRight",
    20: "mouthDimpleLeft",
    21: "mouthDimpleRight",
    22: "mouthFrownLeft",
    23: "mouthFrownRight",
    24: "mouthStretchLeft",
    25: "mouthStretchRight",
    26: "mouthUpperUpLeft",
    27: "mouthUpperUpRight",
    28: "mouthLowerDownLeft",
    29: "mouthLowerDownRight",
    30: "tongueOut",
    31: "tongueLeft",
    32: "tongueRight",
    33: "tongueUp",
    34: "tongueDown",
    35: "tongueFunnel",
    36: "tongueTwistLeft",
    37: "tongueTwistRight",
    38: "tongueFlat",
    39: "tongueSkinny",
    40: "tongueBendDown",
    41: "tongueCurlUp",
    42: "jawOpen",
    43: "mouthClose",
}

def decode_predictions(predictions, threshold=0.5):
    """
    将模型预测解码为人类可读的标签
    
    Args:
        predictions: 模型预测张量 [batch_size, num_classes]
        threshold: 置信度阈值
        
    Returns:
        预测的面部表情列表
    """
    results = []
    
    # 如果是单个样本的预测
    if len(predictions.shape) == 1:
        predictions = predictions.unsqueeze(0)
    
    for pred in predictions:
        # 找出超过阈值的所有类别
        active_indices = torch.where(pred > threshold)[0]
        active_labels = []
        
        for idx in active_indices:
            # 索引需要加1，因为TAGS_MAPPING从1开始
            tag_idx = idx.item() + 1
            if tag_idx in TAGS_MAPPING:
                label_name = TAGS_MAPPING[tag_idx]
                confidence = pred[idx].item()
                active_labels.append((label_name, confidence))
        
        results.append(active_labels)
    
    return results

def decode_label(label, threshold=0.5):
    """
    解码单个标签为人类可读的标签列表
    
    Args:
        label: 标签张量 [num_classes]
        threshold: 置信度阈值
        
    Returns:
        活跃的面部表情列表 [(name, value), ...]
    """
    active_labels = []
    
    # 找出超过阈值的所有类别
    if isinstance(label, torch.Tensor):
        active_indices = torch.where(label > threshold)[0]
        for idx in active_indices:
            # 索引需要加1，因为TAGS_MAPPING从1开始
            tag_idx = idx.item() + 1
            if tag_idx in TAGS_MAPPING:
                label_name = TAGS_MAPPING[tag_idx]
                confidence = label[idx].item()
                active_labels.append((label_name, confidence))
    else:  # numpy数组
        active_indices = np.where(label > threshold)[0]
        for idx in active_indices:
            tag_idx = idx + 1
            if tag_idx in TAGS_MAPPING:
                label_name = TAGS_MAPPING[tag_idx]
                confidence = label[idx]
                active_labels.append((label_name, confidence))
    
    return active_labels

def visualize_sample(image, label, title=None, prediction=None, threshold=0.5):
    """
    可视化单个样本及其标签
    
    Args:
        image: 图像张量 [C, H, W]
        label: 标签张量 [num_classes]
        title: 可选标题
        prediction: 可选的预测张量 [num_classes]
        threshold: 置信度阈值
    """
    # 转换图像为numpy数组以便显示
    if isinstance(image, torch.Tensor):
        # 如果图像已经标准化，需要反标准化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        
        image = image.permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)
    
    # 解码实际标签
    active_labels = decode_label(label, threshold)
    
    # 创建图像
    plt.figure(figsize=(12, 6))
    
    # 左侧显示图像
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    if title:
        plt.title(title)
    else:
        plt.title("Sample Image")
    plt.axis('off')
    
    # 右侧显示标签信息
    plt.subplot(1, 2, 2)
    plt.axis('off')
    
    # 构建标签文本
    info_text = "Ground Truth Labels:\n"
    if active_labels:
        for name, value in active_labels:
            info_text += f"- {name}: {value:.2f}\n"
    else:
        info_text += "No active labels found.\n"
    
    # 如果有预测值，也显示预测信息
    if prediction is not None:
        pred_labels = decode_predictions(prediction, threshold)[0]
        info_text += "\nPredicted Labels:\n"
        if pred_labels:
            for name, value in pred_labels:
                info_text += f"- {name}: {value:.2f}\n"
        else:
            info_text += "No predicted labels above threshold.\n"
    
    plt.text(0, 0.9, info_text, fontsize=10, verticalalignment='top', wrap=True)
    plt.tight_layout()
    plt.show()

def visualize_batch_samples(loader, num_samples=3, threshold=0.3):
    """
    可视化数据加载器中的多个样本
    
    Args:
        loader: DataLoader实例
        num_samples: 要显示的样本数量
        threshold: 标签置信度阈值
    """
    # 获取一个batch
    for images, labels, paths in loader:
        for i in range(min(num_samples, len(images))):
            # 提取文件名作为标题
            filename = os.path.basename(paths[i])
            visualize_sample(images[i], labels[i], title=filename, threshold=threshold)
            
            # 是否继续显示下一个样本
            if i < min(num_samples, len(images)) - 1:
                if input("继续查看下一个样本? (y/n): ").lower() != 'y':
                    return
        break

# 示例用法
if __name__ == "__main__":
    # 测试数据加载
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        dataset_root="./datasets",
        batch_size=8
    )
    
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")
    print(f"类别数: {num_classes}")
    
    # 调整阈值以突出显示更多活跃标签
    threshold = 0.3
    print(f"使用标签置信度阈值: {threshold}")
    
    print("\n可视化训练集样本:")
    visualize_batch_samples(train_loader, num_samples=10, threshold=threshold)