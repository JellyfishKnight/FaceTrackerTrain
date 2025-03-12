import cv2
import numpy as np
import os
import csv

# 视频目录路径（请修改为你的实际路径）
VIDEO_DIR = "./videos"
DATASET_DIR = "./dataset"  # 存储图片和标签的目录
os.makedirs(DATASET_DIR, exist_ok=True)

# 定义动作依赖关系，键是动作名称，值是该动作依赖的其他动作
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
    "lipShrug": 19,
    "mouthClose": 20,
    "mouthSmileLeft": 21,
    "mouthDimpleLeft": 22,
    "mouthSmileRight": 23,
    "mouthDimpleRight": 24,
    "mouthFrownLeft": 25,
    "mouthStretchLeft": 26,
    "mouthFrownRight": 27,
    "mouthStretchRight": 28,
    "mouthUpperUpLeft": 29,
    "mouthUpperUpRight": 30,
    "mouthLowerDownLeft": 31,
    "mouthLowerDownRight": 32,
    "tongueOut": 33,
    "tongueLeft": 34,
    "tongueRight": 35,
    "tongueUp": 36,
    "tongueDown": 37,
    "tongueFunnel": 38,
    "tongueTwistLeft": 39,
    "tongueTwistRight": 40,
    "tongueFlat": 41,
    "tongueSkinny": 42,
    "tongueBendDown": 43,
    "tongueCurlUp": 44,
    "jawOpen": 45,
}


ACTION_DEPENDENCIES = {
    "tongueLeft": ["tongueOut"],
    "tongueRight": ["tongueOut"],
    "tongueUp": ["tongueOut"],
    "tongueDown": ["tongueOut"],
    "tongueFunnel": ["tongueOut"],
    "tongueTwistLeft": ["tongueOut"],
    "tongueTwistRight": ["tongueOut"],
    "tongueFlat": ["tongueOut"],
    "tongueSkinny": ["tongueOut"],
    "tongueBendDown": ["jawOpen"],
    "tongueCurlUp": ["jawOpen"]
}

# 动作时间戳（单位：毫秒），包括最大幅度时间点
TIMESTAMP_DICT = {
    "cheekpuffs.mp4": [(0 ,0, 800, ["neutural"]), (800, 1300, 1800, ["cheekPuffLeft"]), (1800 ,0, 2400, ["neutural"]), (2400, 2900, 3500, ["cheekPuffRight"]), (3500 ,0, 4150, ["neutural"]), (4150, 4700, 5300, ["cheekPuffRight", "cheekPuffLeft"]), (5300, 0, 6200, ["neutural"])],
    "jawopen+suck.mp4": [(0 ,0, 500, ["neutural"]), (500, 1100, 1800, ["jawOpen"]), (1800 ,0, 2400, ["neutural"]), (2400, 3050, 3700, ["cheekSuckLeft", "cheekSuckRight"]), (3700 ,0, 4700, ["neutural"])],
    "jawf-l-r.mp4": [(0 ,0, 1000, ["neutural"]), (1000, 1550, 2000, ["jawForward"]), (2000 ,0, 2650, ["neutural"]), (2650, 3300, 3700, ["jawLeft"]), (3700 ,0, 4450, ["neutural"]), (4450, 5150, 5700, ["jawRight"]), (5700 ,0, 6600, ["neutural"])],
    "sneer+press.mp4": [(0 ,0, 1700, ["neutural"]), (1700, 2300, 2900, ["noseSneerLeft", "noseSneerRight"]), (2900 ,0, 3100, ["neutural"]), (3100, 4250, 4900, ["mouthPressLeft", "mouthPressRight"]), (4900 ,0, 6000, ["neutural"])],
    "pucker-funnel.mp4": [(0 ,0, 1200, ["neutural"]), (1200, 1950, 2600, ["lipPucker"]), (2600 ,0, 3000, ["neutural"]), (3000, 3600, 4300, ["lipFunnel"]), (4300 ,0, 5350, ["neutural"])],
    "mouth-lr.mp4": [(0 ,0, 1700, ["neutural"]), (1700, 2100, 2800, ["mouthLeft"]), (2800 ,0, 3200, ["neutural"]), (3200, 4000, 4600, ["mouthRight"]), (4600 ,0, 5900, ["neutural"])],
    "lipsuck.mp4": [(0 ,0, 700, ["neutural"]), (700, 1200, 1900, ["lipSuckLower"]), (1900 ,0, 2400, ["neutural"]), (2400, 3100, 3600, ["lipSuckUpper"]), (3600 ,0, 4200, ["neutural"]), (4200, 5000, 5700, ["lipSuckUpper", "lipSuckLower"]), (5700 ,0, 6600, ["neutural"])],
    "lipshrug.mp4": [(0 ,0, 900, ["neutural"]), (900, 1350, 1900, ["lipShrug"]), (1900 ,0, 3100, ["neutural"])],
    "apesmiles.mp4": [(0 ,0, 900, ["neutural"]), (900, 1450, 2200, ["mouthClose"]), (2200 ,0, 2800, ["neutural"]), (2800, 3450, 4000, ["mouthSmileLeft", "mouthDimpleLeft"]), (4000 ,0, 4500, ["neutural"]), (4500, 5180, 5900, ["mouthSmileRight", "mouthDimpleRight"]), (5900 ,0, 6700, ["neutural"]), (6700, 7500, 8200, ["mouthSmileLeft", "mouthSmileRight"]), (8200 ,0, 9200, ["neutural"])],
    "frowns.mp4": [(0 ,0, 700, ["neutural"]), (700, 1200, 1400, ["mouthFrownLeft", "mouthStretchLeft"]), (1400 ,0, 1900, ["neutural"]), (1900, 2600, 2900, ["mouthFrownRight", "mouthStretchRight"]), (2900 ,0, 3800, ["neutural"]), (3800, 4300, 4900, ["mouthFrownLeft", "mouthFrownRight", "mouthStretchLeft", "mouthStretchRight"]), (4900 ,0, 5900, ["neutural"])],
    "upperuplowerdown.mp4": [(0 ,0, 900, ["neutural"]), (900, 1400, 2100, ["mouthUpperUpLeft"]), (2100 ,0, 2200, ["neutural"]), (2200, 3300, 3800, ["mouthUpperUpRight"]), (3800 ,0, 4400, ["neutural"]), (4400, 4900, 5300, ["mouthLowerDownLeft"]), (5300 ,0, 5900, ["neutural"]), (5900, 6600, 7200, ["mouthLowerDownRight"]), (7200, 0, 7600, ["neutural"])],
    "tonguemain.mp4": [(0 ,0, 1200, ["neutural"]), (1200, 1800, 2200, ["jawOpen"]), (2200, 2700, 3150, ["tongueOut", "jawOpen"]), (3150, 3700, 4200, ["tongueUp", "tongueOut", "jawOpen"]), (4200, 0, 4900, ["tongueOut", "jawOpen"]), (4900, 5250, 6000, ["tongueDown", "tongueOut", "jawOpen"]), (6000, 0, 6500, ["tongueOut", "jawOpen"]), (6500, 7200, 8000, ["tongueLeft", "tongueOut", "jawOpen"]), (8000, 0, 8500, ["tongueOut", "jawOpen"]), (8500, 9100, 9650, ["tongueRight", "tongueOut", "jawOpen"]), (9650 ,0, 10800, ["tongueOut", "jawOpen"]), (10800 ,0, 11600, ["jawOpen"]), (11600 ,0, 12600, ["neutural"])],
    "tonguefunnel.mp4": [(0 ,0, 1600, ["neutural"]), (1600, 2100, 2500, ["jawOpen"]), (2500, 3000, 3200, ["tongueOut", "jawOpen"]), (3200, 4000, 4800, ["tongueFunnel", "tongueOut", "jawOpen"]), (4800, 0, 5900, ["tongueOut", "jawOpen"]), (5900 ,0, 6850, ["jawOpen"]), (6850 ,0, 7800, ["neutural"])],
    "tonguetwist.mp4": [(0 ,0, 600, ["neutural"]), (600, 1000, 1500, ["jawOpen"]), (1500, 1800, 2300, ["tongueOut", "jawOpen"]), (2300, 3000, 3600, ["tongueTwistLeft", "tongueOut", "jawOpen"]), (3600, 0, 4000, ["tongueOut", "jawOpen"]), (4000, 4700, 5300, ["tongueTwistRight", "tongueOut", "jawOpen"]), (5300 ,0, 6100, ["tongueOut", "jawOpen"]), (6100 ,0, 7500, ["jawOpen"]), (7500 ,0, 8200, ["neutural"])],
    "tongueflatsquish.mp4": [(0 ,0, 550, ["neutural"]), (550, 1000, 1200, ["jawOpen"]), (1200, 1700, 1800, ["tongueOut", "jawOpen"]), (1800, 2500, 3200, ["tongueFlat", "tongueOut", "jawOpen"]), (3200, 0, 4300, ["tongueOut", "jawOpen"]), (4300, 5050, 5700, ["tongueSkinny", "tongueOut", "jawOpen"]), (5700 ,0, 7000, ["tongueOut", "jawOpen"]), (7000 ,0, 8000, ["jawOpen"]), (8000 ,0, 9300, ["neutural"])],
    "tonguecurlbend.mp4": [(0 ,0, 450, ["neutural"]), (450, 950, 1300, ["jawOpen"]), (1300, 1800, 2500, ["tongueBendDown", "jawOpen"]), (2500 ,0, 3000, ["jawOpen"]), (3000, 3750, 4500, ["tongueCurlUp", "jawOpen"]), (4500 ,0, 5800, ["jawOpen"]), (5800 ,0, 7200, ["neutural"])],
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

def extract_amplitude_and_save_dataset(video_path, timestamps, video_name):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    dataset = []  # 存储数据集 (image_path, labels, amplitudes)
    
    for start_time, peak_time, end_time, action_names in timestamps:
        start_frame = int(start_time * fps / 1000)
        peak_frame = int(peak_time * fps / 1000) if peak_time > 0 else start_frame
        end_frame = int(end_time * fps / 1000)
        
        # 确保所有依赖动作都包含在列表中
        complete_actions = ensure_dependencies(action_names)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        
        for frame_id in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append((frame_id, frame, gray_frame))
        
        for frame_id, frame, gray_frame in frames:
            # 计算归一化幅度
            if peak_time > 0:  # 只有当指定了峰值时间时才计算幅度
                if frame_id <= peak_frame:
                    normalized_amplitude = (frame_id - start_frame) / (peak_frame - start_frame + 1e-6)
                else:
                    normalized_amplitude = 1 - (frame_id - peak_frame) / (end_frame - peak_frame + 1e-6)
            else:
                normalized_amplitude = 0  # 中性状态或没有指定峰值时幅度为0
            
            # 对每个动作分别记录幅度
            action_amplitudes = {}
            for action in complete_actions:
                action_amplitudes[action] = normalized_amplitude
            
            img_filename = f"{video_name}_{frame_id}.jpg"
            img_path = os.path.join(DATASET_DIR, img_filename)
            cv2.imwrite(img_path, frame)
            
            # 将每个动作和对应的幅度分开保存
            for action in complete_actions:
                dataset.append((img_filename, action, action_amplitudes[action]))
    
    cap.release()
    return dataset

# 处理所有视频并保存数据集
all_dataset = []
for video_name, timestamps in TIMESTAMP_DICT.items():
    video_path = os.path.join(VIDEO_DIR, video_name)
    if os.path.exists(video_path):
        print(f"处理视频: {video_name}")
        dataset = extract_amplitude_and_save_dataset(video_path, timestamps, video_name)
        all_dataset.extend(dataset)

# 保存数据集为 CSV 文件
csv_path = os.path.join(DATASET_DIR, "dataset_labels.csv")
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image", "label", "amplitude"])
    writer.writerows(all_dataset)

print(f"数据集已保存到 {DATASET_DIR}，标签文件: {csv_path}")