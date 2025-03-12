import cv2
import numpy as np
import os
import csv

# 视频和数据集目录路径
ROOT_VIDEO_DIR = "./videos"  # 包含多个子文件夹的视频根目录
DATASET_ROOT = "./datasets"  # 数据集根目录
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")  # 图片目录
LABELS_DIR = os.path.join(DATASET_ROOT, "labels")  # 标签目录

# 确保目录存在
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

# 标签以及在输出数组里面的顺序， neutural不在输出数据集中，neutural代表当前没有任何状态以及动作
TAGS = {
    "neutural": 0,
    "cheekPuffLeft": 1,
    "cheekPuffRight": 2,
    "cheekSuckLeft": 3,
    "cheekSuckRight": 4,
    "jawForward": 5,
    "jawLeft": 6,
    "jawRight": 7,
    "noseSneerLeft": 8,
    "noseSneerRight": 9,
    "mouthPressLeft": 10,
    "mouthPressRight": 11,
    "lipPucker": 12,
    "lipFunnel": 13,
    "mouthLeft": 14,
    "mouthRight": 15,
    "lipSuckLower": 16,
    "lipSuckUpper": 17,
    "mouthSmileLeft": 18,
    "mouthSmileRight": 19,
    "mouthDimpleLeft": 20,
    "mouthDimpleRight": 21,
    "mouthFrownLeft": 22,
    "mouthFrownRight": 23,
    "mouthStretchLeft": 24,
    "mouthStretchRight": 25,
    "mouthUpperUpLeft": 26,
    "mouthUpperUpRight": 27,
    "mouthLowerDownLeft": 28,
    "mouthLowerDownRight": 29,
    "tongueOut": 30,
    "tongueLeft": 31,
    "tongueRight": 32,
    "tongueUp": 33,
    "tongueDown": 34,
    "tongueFunnel": 35,
    "tongueTwistLeft": 36,
    "tongueTwistRight": 37,
    "tongueFlat": 38,
    "tongueSkinny": 39,
    "tongueBendDown": 40,
    "tongueCurlUp": 41,
    "jawOpen": 42,
    "mouthClose": 43,
}

# 动作依赖关系（有些动作可能依赖于其他动作）
ACTION_DEPENDENCIES = {
    "tongueOut": ["jawOpen"],
    "tongueLeft": ["tongueOut"],
    "tongueRight": ["tongueOut"],
    "tongueUp": ["tongueOut"],
    "tongueDown": ["tongueOut"],
    "tongueFunnel": ["tongueOut"],
    "tongueTwistLeft": ["tongueOut"],
    "tongueTwistRight": ["tongueOut"],
    "tongueFlat": ["tongueOut"],
    "tongueSkinny": ["tongueOut"],
    "tongueBendDown": ["tongueOut"],
    "tongueCurlUp": ["tongueOut"],
}

# 动作时间戳（单位：毫秒），包括最大幅度时间点
# 第一个时间为动作起始时间
# 第二个时间为动作最大幅度时间，当动作为neutural的时候，动作最大幅度时间为0。当动作不为neutural且最大幅度时间为0的时候，说明当前动作幅度在线性下降结束
# 第三个时间为动作结束时间
# 第四个时间为动作名称列表
# 注意：如果动作被另外一个动作依赖，那么当另外一个动作开始的时候，当前动作的幅度为最大幅度，即1
# 动作幅度计算方法：最大峰值幅度为1，最小为0，当动作为neutural的时候，所有动作幅度为0。
#                 当动作开始时，动作幅度根据时间信息线性增加，当达到最大幅度时间点时，动作幅度为1
#                 当动作结束时，动作幅度根据时间信息线性减少，当达到结束时间点时，动作幅度为0
TIMESTAMP_DICT = {
    "cheekpuffs.mp4": [(0 ,0, 800, ["neutural"]), (800, 1300, 1800, ["cheekPuffLeft"]), (1800 ,0, 2400, ["neutural"]), (2400, 2900, 3500, ["cheekPuffRight"]), (3500 ,0, 4150, ["neutural"]), (4150, 4700, 5300, ["cheekPuffRight", "cheekPuffLeft"]), (5300, 0, 6200, ["neutural"])],
    "jawopen+suck.mp4": [(0 ,0, 500, ["neutural"]), (500, 1100, 1800, ["jawOpen"]), (1800 ,0, 2400, ["neutural"]), (2400, 3050, 3700, ["cheekSuckLeft", "cheekSuckRight"]), (3700 ,0, 4700, ["neutural"])],
    "jawf-l-r.mp4": [(0 ,0, 1000, ["neutural"]), (1000, 1550, 2000, ["jawForward"]), (2000 ,0, 2650, ["neutural"]), (2650, 3300, 3700, ["jawLeft"]), (3700 ,0, 4450, ["neutural"]), (4450, 5150, 5700, ["jawRight"]), (5700 ,0, 6600, ["neutural"])],
    "sneer+press.mp4": [(0 ,0, 1700, ["neutural"]), (1700, 2300, 2900, ["noseSneerLeft", "noseSneerRight"]), (2900 ,0, 3100, ["neutural"]), (3100, 4250, 4900, ["mouthPressLeft", "mouthPressRight"]), (4900 ,0, 6000, ["neutural"])],
    "pucker-funnel.mp4": [(0 ,0, 1200, ["neutural"]), (1200, 1950, 2600, ["lipPucker"]), (2600 ,0, 3000, ["neutural"]), (3000, 3600, 4300, ["lipFunnel"]), (4300 ,0, 5350, ["neutural"])],
    "mouth-lr.mp4": [(0 ,0, 1700, ["neutural"]), (1700, 2100, 2800, ["mouthLeft"]), (2800 ,0, 3200, ["neutural"]), (3200, 4000, 4600, ["mouthRight"]), (4600 ,0, 5900, ["neutural"])],
    "lipsuck.mp4": [(0 ,0, 700, ["neutural"]), (700, 1200, 1900, ["lipSuckLower"]), (1900 ,0, 2400, ["neutural"]), (2400, 3100, 3600, ["lipSuckUpper"]), (3600 ,0, 4200, ["neutural"]), (4200, 5000, 5700, ["lipSuckUpper", "lipSuckLower"]), (5700 ,0, 6600, ["neutural"])],
    # "lipshrug.mp4": [(0 ,0, 900, ["neutural"]), (900, 1350, 1900, ["lipShrug"]), (1900 ,0, 3100, ["neutural"])],
    "apesmiles.mp4": [(0 ,0, 900, ["neutural"]), (900, 1450, 2200, ["mouthClose"]), (2200 ,0, 2800, ["neutural"]), (2800, 3450, 4000, ["mouthSmileLeft", "mouthDimpleLeft"]), (4000 ,0, 4500, ["neutural"]), (4500, 5180, 5900, ["mouthSmileRight", "mouthDimpleRight"]), (5900 ,0, 6700, ["neutural"]), (6700, 7500, 8200, ["mouthSmileLeft", "mouthSmileRight"]), (8200 ,0, 9200, ["neutural"])],
    "frowns.mp4": [(0 ,0, 700, ["neutural"]), (700, 1200, 1400, ["mouthFrownLeft", "mouthStretchLeft"]), (1400 ,0, 1900, ["neutural"]), (1900, 2600, 2900, ["mouthFrownRight", "mouthStretchRight"]), (2900 ,0, 3800, ["neutural"]), (3800, 4300, 4900, ["mouthFrownLeft", "mouthFrownRight", "mouthStretchLeft", "mouthStretchRight"]), (4900 ,0, 5900, ["neutural"])],
    "upperuplowerdown.mp4": [(0 ,0, 900, ["neutural"]), (900, 1400, 2100, ["mouthUpperUpLeft"]), (2100 ,0, 2200, ["neutural"]), (2200, 3300, 3800, ["mouthUpperUpRight"]), (3800 ,0, 4400, ["neutural"]), (4400, 4900, 5300, ["mouthLowerDownLeft"]), (5300 ,0, 5900, ["neutural"]), (5900, 6600, 7200, ["mouthLowerDownRight"]), (7200, 0, 7600, ["neutural"])],
    "tonguemain.mp4": [(0 ,0, 1200, ["neutural"]), (1200, 1800, 2200, ["jawOpen"]), (2200, 2700, 3150, ["tongueOut"]), (3150, 3700, 4200, ["tongueUp"]), (4200, 0, 4900, ["tongueOut"]), (4900, 5250, 6000, ["tongueDown"]), (6000, 0, 6500, ["tongueOut"]), (6500, 7200, 8000, ["tongueLeft"]), (8000, 0, 8500, ["tongueOut"]), (8500, 9100, 9650, ["tongueRight"]), (9650 ,0, 10800, ["tongueOut"]), (10800 ,0, 11600, ["jawOpen"]), (11600 ,0, 12600, ["neutural"])],
    "tonguefunnel.mp4": [(0 ,0, 1600, ["neutural"]), (1600, 2100, 2500, ["jawOpen"]), (2500, 3000, 3200, ["tongueOut"]), (3200, 4000, 4800, ["tongueFunnel"]), (4800, 0, 5900, ["tongueOut"]), (5900 ,0, 6850, ["jawOpen"]), (6850 ,0, 7800, ["neutural"])],
    "tonguetwist.mp4": [(0 ,0, 600, ["neutural"]), (600, 1000, 1500, ["jawOpen"]), (1500, 1800, 2300, ["tongueOut"]), (2300, 3000, 3600, ["tongueTwistLeft"]), (3600, 0, 4000, ["tongueOut"]), (4000, 4700, 5300, ["tongueTwistRight"]), (5300 ,0, 6100, ["tongueOut"]), (6100 ,0, 7500, ["jawOpen"]), (7500 ,0, 8200, ["neutural"])],
    "tongueflatsquish.mp4": [(0 ,0, 550, ["neutural"]), (550, 1000, 1200, ["jawOpen"]), (1200, 1700, 1800, ["tongueOut"]), (1800, 2500, 3200, ["tongueFlat"]), (3200, 0, 4300, ["tongueOut"]), (4300, 5050, 5700, ["tongueSkinny"]), (5700 ,0, 7000, ["tongueOut"]), (7000 ,0, 8000, ["jawOpen"]), (8000 ,0, 9300, ["neutural"])],
    "tonguecurlbend.mp4": [(0 ,0, 450, ["neutural"]), (450, 950, 1300, ["jawOpen"]), (1300, 1800, 2500, ["tongueBendDown"]), (2500 ,0, 3000, ["jawOpen"]), (3000, 3750, 4500, ["tongueCurlUp"]), (4500 ,0, 5800, ["jawOpen"]), (5800 ,0, 7200, ["neutural"])],
    "face-pog.mp4": [(0 ,0, 750, ["neutural"]), (750, 1350, 2000, ["jawOpen", "lipFunnel"]), (2000 ,0, 3000, ["neutural"])],
    "jawopen+smile-frown.mp4": [(0 ,0, 800, ["neutural"]), (800 ,1400, 1500, ["jawOpen"]), (1500 ,1900, 2500, ["jawOpen", "mouthSmileLeft", "mouthSmileRight"]), (2500 ,0, 3000, ["jawOpen"]), (3000 ,3650, 4600, ["jawOpen", "mouthFrownLeft", "mouthFrownRight"]), (4600 ,0, 5300, ["jawOpen"]), (5300 ,0, 6100, ["neutural"])],
}

def ensure_dependencies(actions):
    """确保所有依赖的动作都包含在动作列表中"""
    if isinstance(actions, str):
        # 处理单个字符串的情况
        actions = [actions]
    
    complete_actions = list(actions)  # 创建副本，避免修改原始列表
    
    for action in actions:
        if action in ACTION_DEPENDENCIES:
            for dependency in ACTION_DEPENDENCIES[action]:
                if dependency not in complete_actions:
                    complete_actions.append(dependency)
    
    return complete_actions

def create_label_vector(actions, amplitudes, tags_dict):
    """创建43维标签向量，每个动作对应一个幅度值"""
    # 创建一个全零向量（不包括neutural）
    label_vector = [0.0] * (len(tags_dict) - 1)  # 减1是因为不包括neutural
    
    for action, amplitude in zip(actions, amplitudes):
        if action != "neutural":  # neutural不在输出中
            # 因为neutural的索引是0，所以其他动作的索引需要减1
            label_vector[tags_dict[action] - 1] = amplitude
    
    return label_vector

def find_active_actions_at_time(timestamps, frame_time_ms):
    """在给定时间点找出所有活跃的动作及其幅度"""
    active_actions = []
    active_amplitudes = []
    
    for start_time, peak_time, end_time, action_names in timestamps:
        if start_time <= frame_time_ms < end_time:
            # 当前时间段内的动作是活跃的
            for action in action_names:
                if action == "neutural":
                    continue  # 跳过neutural状态
                
                # 计算当前动作的幅度
                if peak_time > 0:  # 有指定峰值时间
                    if frame_time_ms <= peak_time:
                        # 从起始到峰值，线性增加
                        amplitude = min(1.0, max(0.0, (frame_time_ms - start_time) / (peak_time - start_time + 1e-6)))
                    else:
                        # 从峰值到结束，线性减少
                        amplitude = min(1.0, max(0.0, 1.0 - (frame_time_ms - peak_time) / (end_time - peak_time + 1e-6)))
                else :
                    if action != "neutural":
                        # 动作没有峰值时间，线性下降
                        amplitude = min(1.0, max(0.0, (frame_time_ms - start_time) / (end_time - start_time + 1e-6)))
                    else:
                        amplitude = 0.0
                
                active_actions.append(action)
                active_amplitudes.append(amplitude)
    
    return active_actions, active_amplitudes

def extract_frames_and_save_dataset(video_path, timestamps, output_img_dir, output_label_dir, subfolder, video_name):
    """处理视频并提取帧和标签"""
    # 确保输出目录存在
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return []
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"处理视频: {video_path}, FPS: {fps}, 总帧数: {total_frames}")
    
    # 遍历视频的每一帧
    for frame_id in range(total_frames):
        # 读取当前帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            print(f"读取帧 {frame_id} 失败")
            continue
        
        # 计算当前帧对应的时间（毫秒）
        frame_time_ms = (frame_id * 1000) / fps
        
        # 查找当前帧时间点的所有活跃动作及其幅度
        base_actions, base_amplitudes = find_active_actions_at_time(timestamps, frame_time_ms)
        
        if not base_actions or "neutural" in base_actions:
            # 如果没有活跃动作或只有neutural，创建一个全零向量
            label_vector = [0.0] * (len(TAGS) - 1)
        else:
            # 处理动作依赖关系
            actions_with_deps = []
            amplitudes_with_deps = []
            
            # 为每个基础动作添加其依赖
            for action, amplitude in zip(base_actions, base_amplitudes):
                actions_with_deps.append(action)
                amplitudes_with_deps.append(amplitude)
                
                # 检查是否有其他动作依赖于当前动作
                for possible_dependent in base_actions:
                    if possible_dependent in ACTION_DEPENDENCIES and action in ACTION_DEPENDENCIES[possible_dependent]:
                        # 当前动作是另一个动作的依赖，确保其幅度为1
                        index = actions_with_deps.index(action)
                        amplitudes_with_deps[index] = 1.0
            
            # 确保添加所有依赖动作
            for action in base_actions:
                if action in ACTION_DEPENDENCIES:
                    for dependency in ACTION_DEPENDENCIES[action]:
                        if dependency not in actions_with_deps:
                            # 添加依赖动作，幅度设为1
                            actions_with_deps.append(dependency)
                            amplitudes_with_deps.append(1.0)
            
            # 创建标签向量
            label_vector = create_label_vector(actions_with_deps, amplitudes_with_deps, TAGS)
        
        # 保存图片
        img_filename = f"{subfolder}_{video_name[:-4]}_{frame_id:06d}.jpg"
        img_path = os.path.join(output_img_dir, img_filename)
        cv2.imwrite(img_path, frame)
        
        # 保存标签文件 (与图像同名但扩展名为.txt)
        label_filename = f"{subfolder}_{video_name[:-4]}_{frame_id:06d}.txt"
        label_path = os.path.join(output_label_dir, label_filename)
        
        # 将标签向量写入文本文件
        with open(label_path, 'w') as f:
            f.write(' '.join([f"{value:.6f}" for value in label_vector]))
    
    cap.release()
    print(f"视频 {video_path} 处理完成")

def process_all_videos():
    """处理所有子文件夹中的所有视频"""
    # 获取videos目录下的所有子文件夹
    subfolders = [f for f in os.listdir(ROOT_VIDEO_DIR) 
                  if os.path.isdir(os.path.join(ROOT_VIDEO_DIR, f))]
    
    for subfolder in subfolders:
        print(f"处理子文件夹: {subfolder}")
        
        # 为每个子文件夹创建对应的输出目录
        subfolder_images_dir = os.path.join(IMAGES_DIR, subfolder)
        subfolder_labels_dir = os.path.join(LABELS_DIR, subfolder)
        
        os.makedirs(subfolder_images_dir, exist_ok=True)
        os.makedirs(subfolder_labels_dir, exist_ok=True)
        
        # 遍历时间戳字典中的所有视频
        for video_name in TIMESTAMP_DICT.keys():
            video_path = os.path.join(ROOT_VIDEO_DIR, subfolder, video_name)
            
            if os.path.exists(video_path):
                timestamps = TIMESTAMP_DICT[video_name]
                extract_frames_and_save_dataset(
                    video_path, 
                    timestamps, 
                    subfolder_images_dir, 
                    subfolder_labels_dir,
                    subfolder,
                    video_name
                )
            else:
                print(f"视频文件不存在: {video_path}")

if __name__ == "__main__":
    process_all_videos()
    print(f"所有视频处理完成")
    print(f"图像保存在: {IMAGES_DIR}")
    print(f"标签保存在: {LABELS_DIR}")