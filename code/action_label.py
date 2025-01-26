import os
import numpy as np

def load_label(label_file):
    """
    读取 label.dat 文件为 int64 类型的数组。
    
    参数:
        label_file (str): label.dat 文件的路径
    
    返回:
        np.ndarray: 标签数组
    """
    try:
        label = np.memmap(label_file, dtype='int64', mode='r')
        print(f"成功加载标签文件: {label_file}, 标签长度: {label.shape[0]}")
        return label
    except FileNotFoundError:
        print(f"错误: 标签文件未找到: {label_file}")
        return None
    except Exception as e:
        print(f"读取标签文件时出错: {e}")
        return None

def load_data(data_file):
    """
    使用 memmap 读取数据文件并重塑为 (T, 16, 288) 的形状。
    
    参数:
        data_file (str): 数据文件的路径
    
    返回:
        np.ndarray: 重塑后的数据数组
    """
    try:
        data = np.memmap(data_file, dtype='float32', mode='r').reshape(-1, 16, 288)
        print(f"成功加载数据文件: {data_file}, 形状: {data.shape}")
        return data
    except Exception as e:
        print(f"加载数据文件 {data_file} 时出错: {e}")
        return None

def map_labels_to_actions(label, action_list):
    """
    将标签数组映射到动作名称，并记录每个动作对应的帧索引。
    
    参数:
        label (np.ndarray): 标签数组
        action_list (list): 按索引排列的动作名称列表
    
    返回:
        dict: 动作名称到帧索引的映射字典
    """
    label_to_action = {}
    for idx, action in enumerate(action_list):
        if action == "NA":
            continue  # 忽略 'NA' 标签
        if idx == 0:
            continue  # 忽略 'No action'
        frame_indices = np.where(label == idx)[0]
        if frame_indices.size > 0:
            label_to_action[action] = frame_indices
            print(f"动作 '{action}' 对应的帧数量: {frame_indices.size}")
    return label_to_action

def extract_labeled_data(data, label_to_action):
    """
    根据动作名称映射提取对应的数据帧。
    
    参数:
        data (np.ndarray): 数据数组，形状为 (T, 16, 288)
        label_to_action (dict): 动作名称到帧索引的映射字典
    
    返回:
        dict: 动作名称到对应数据帧的映射字典
    """
    action_data = {}
    for action, frames in label_to_action.items():
        action_data[action] = data[frames]
        print(f"提取动作 '{action}' 的数据形状: {action_data[action].shape}")
    return action_data

def main():
    # 定义预定义的动作列表，索引对应标签值
    # 0 表示 'No action'
    action_list = [
        "No action",  # 0
        "Walk to kitchen",  # 1
        "Take plate out from cabinet",  # 2
        "Take food out from fridge",  # 3
        "Take cup out of cabinet",  # 4
        "Take drink out of fridge",  # 5
        "Pour the drink into the cup",  # 6
        "Walk to table and eat the food",  # 7
        "Place the items on the sink",  # 8
        "Walk over to the couch",  # 9
        "Read magazine for 30 seconds",  # 10
        "Walk to the bathroom",  # 11
        "Shower simulation",  # 12
        "Brush teeth",  # 13
        "Wash face",  # 14
        "Simulate using toilet",  # 15
        "Walk to kitchen sink",  # 16
        "Simulate washing dishes",  # 17
        "Washing hands",  # 18
        "Walk to the bedroom",  # 19
        "Hang towel in cabinet",  # 20
        "Remove shoes",  # 21
        "Lie down on the bed",  # 22
        "Get up and take on shoes",  # 23
        "Watch the video for 30 seconds",  # 24
        "Enter shower stall",  # 25
        "Get another towel",  # 26
        "Fold and place it in the cabinet",  # 27
        "Get up and stay seated",  # 28
        "Remove the device and wear your shoes",  # 29
        "NA",  # 30
        "NA",  # 31
        "NA"   # 32
    ]
    
    # 定义所有会话的列表
    sessions = [
        "6e5iYM_ADL_1"
        # 在这里添加更多的 session 名称
    ]
    
    # 遍历每个会话
    for session in sessions:
        print(f"\n===== 处理会话: {session} =====")
        
        # 定义文件路径
        data_file = f"D:/OneDrive/桌面/code/ADL_localization/data/all_activities/{session}_data.dat"
        label_file = f"D:/OneDrive/桌面/code/ADL_localization/data/all_activities/{session}_label.dat"
        mask_file = f"D:/OneDrive/桌面/code/ADL_localization/data/all_activities/{session}_mask_mannual.dat"
        
        # 读取数据文件
        data = load_data(data_file)
        if data is None:
            print(f"跳过会话 {session} 由于数据文件加载失败。")
            continue
        
        # 读取标签文件
        label = load_label(label_file)
        if label is None:
            print(f"跳过会话 {session} 由于标签文件加载失败。")
            continue
        
        # 打印标签数组的基本信息
        print(f"标签数组长度: {label.shape[0]}")
        
        # 调整打印选项以显示整个数组（警告：如果数组很大，可能会导致命令行响应缓慢）
        # np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        
        # # 显示部分标签数据作为示例
        # print("标签数组示例 (前100帧):")
        # print(label[:100])
        # print("...")
        
        # 映射标签到动作名称
        label_to_action = map_labels_to_actions(label, action_list)
        
        # 提取有标签的数据
        action_data = extract_labeled_data(data, label_to_action)
        
        # 示例：访问特定动作的数据
        # 例如，访问 "Walk to kitchen" 的数据帧
        action_label = "Get up and stay seated"
        if action_label in action_data:
            print(f"\n===== 动作 '{action_label}' 的数据 =====")
            print(f"动作 '{action_label}' 的数据形状: {action_data[action_label].shape}")
            print(f"动作 '{action_label}' 的数据示例 (第1帧):")
            print(action_data[action_label][0])  # 打印第1帧的数据
        else:
            print(f"动作 '{action_label}' 没有对应的数据帧。")
        
        print(f"===== 完成会话: {session} =====\n")

if __name__ == "__main__":
    main()
