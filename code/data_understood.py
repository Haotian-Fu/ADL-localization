import os
import numpy as np
from datetime import datetime, timezone, timedelta

def compute_range_data_from_memmap(distance_dict, sensor_ids, data_start_time, start_time, end_time, target_fps=120):
    """
    从读取的 distance 数据字典中，对每个 sensor 的距离数据进行加权平均，并根据 start_time 和 end_time 切片，
    得到一个新的字典 range_data，其键为 sensor id，值为 shape (T,) 的加权平均距离。

    加权平均公式：
      weighted_avg = dot(s, d) / sum(s)
    其中：
      s 是每一帧的距离数据 (188,)
      d 是距离向量 [0.05, 0.10, ..., 9.40]

    参数：
      distance_dict: dict, key 为 sensor id, value 为 (T, 188) 的 NumPy 数组
      sensor_ids: list of str, 传感器 id 列表
      data_start_time: datetime, 数据的起始时间
      start_time: datetime, 需要切片的开始时间
      end_time: datetime, 需要切片的结束时间
      target_fps: int, 数据的帧率

    返回：
      range_data: dict, key 为 sensor id, value 为 (selected_T,) 的加权平均距离
    """
    range_data = {}
    # 构造距离向量：第 i 个 bin 对应 i*0.05 米，从 0.05 到 9.40 米
    distance_vector = np.arange(1, 189) * 0.05  # shape: (188,)

    # 计算需要切片的帧索引
    delta_start = (start_time - data_start_time).total_seconds()
    delta_end = (end_time - data_start_time).total_seconds()
    frame_start = int(delta_start * target_fps)
    frame_end = int(delta_end * target_fps)

    # 获取数据总帧数
    T_total = distance_dict[sensor_ids[0]].shape[0]

    # 确保帧索引在合理范围内
    frame_start = max(0, frame_start)
    frame_end = min(T_total - 1, frame_end)

    print(f"Slicing data from frame {frame_start} to frame {frame_end} based on start_time and end_time.")

    for sensor in sensor_ids:
        data = distance_dict[sensor][frame_start:frame_end + 1, :]  # shape: (selected_T, 188)
        # 计算加权和和权重和
        weighted_sum = np.dot(data, distance_vector)  # shape: (selected_T,)
        weights_sum = np.sum(data, axis=1)           # shape: (selected_T,)
        # 计算加权平均，避免除以零
        weighted_avg = np.where(weights_sum != 0, weighted_sum / weights_sum, 0)
        range_data[sensor] = weighted_avg  # shape: (selected_T,)
        print(f"Computed range_data for sensor {sensor} with shape: {weighted_avg.shape}")

    return range_data

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
        if action in ["No action", "NA"]:
            continue  # 忽略 'No action' 和 'NA' 标签
        frame_indices = np.where(label == idx)[0]
        if frame_indices.size > 0:
            label_to_action[action] = frame_indices
            print(f"动作 '{action}' 对应的帧数量: {frame_indices.size}")
    return label_to_action

def create_distance_dict(data, sensor_ids, num_bins=188):
    """
    从数据数组中创建 distance_dict，适配 compute_range_data_from_memmap 函数的需求。

    参数:
        data (np.ndarray): 数据数组，形状为 (T, 16, 288)
        sensor_ids (list of str): 传感器 ID 列表
        num_bins (int): 每个传感器的距离 bin 数量

    返回:
        dict: distance_dict，key 为 sensor id，value 为 (T, num_bins) 的 NumPy 数组
    """
    distance_dict = {}
    for i, sensor_id in enumerate(sensor_ids):
        # 假设每个传感器的距离数据存储在 data[:, i, :188]
        distance_data = data[:, i, :num_bins]  # shape: (T, 188)
        distance_dict[sensor_id] = distance_data
        print(f"创建 distance_dict for sensor {sensor_id} with shape: {distance_data.shape}")
    return distance_dict

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

    # 传感器 ID 列表，假设为 '1' 到 '16'
    sensor_ids = [str(i) for i in range(1, 17)]  # ['1', '2', ..., '16']

    # 遍历每个会话
    for session in sessions:
        print(f"\n===== 处理会话: {session} =====")

        # 定义文件路径
        # data_file = f"D:/OneDrive/桌面/code/ADL_localization/data/all_activities/{session}_data.dat"
        # label_file = f"D:/OneDrive/桌面/code/ADL_localization/data/all_activities/{session}_label.dat"
        # mask_file = f"D:/OneDrive/桌面/code/ADL_localization/data/all_activities/{session}_mask_mannual.dat"
        data_file = f"data/new_dataset/bedroom_data/{session}_data.dat"
        label_file = f"data/new_dataset/bedroom_data/{session}_label.dat"
        mask_file = f"data/new_dataset/bedroom_data/{session}_mask_mannual.dat"

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

        # 显示部分标签数据作为示例
        print("标签数组示例 (前100帧):")
        print(label[:100])
        print("...")

        # 映射标签到动作名称
        label_to_action = map_labels_to_actions(label, action_list)

        # 检查是否有有效的动作
        if not label_to_action:
            print(f"会话 {session} 中没有有效的动作标签。")
            continue

        # 创建 distance_dict
        distance_dict = create_distance_dict(data, sensor_ids, num_bins=188)

        # 示例：选择一个动作并提取对应的 range_data
        # 你可以根据需要循环处理所有动作或特定动作
        # 这里以 "Walk to kitchen" 为例
        action_label = "Walk to kitchen"
        if action_label in label_to_action:
            frames = label_to_action[action_label]
            print(f"动作 '{action_label}' 对应的帧索引数量: {frames.size}")

            # 计算动作的开始时间和结束时间
            # 假设数据的起始时间为某个已知时间，例如：
            data_start_time = datetime(2023, 6, 29, 16, 54, 23, tzinfo=timezone.utc)
            # 获取动作的最早和最晚帧
            frame_start = frames.min()
            frame_end = frames.max()
            action_start_time = data_start_time + timedelta(seconds=frame_start / 120)
            action_end_time = data_start_time + timedelta(seconds=frame_end / 120)
            print(f"动作 '{action_label}' 开始时间: {action_start_time}")
            print(f"动作 '{action_label}' 结束时间: {action_end_time}")

            # 调用 compute_range_data_from_memmap
            range_data = compute_range_data_from_memmap(
                distance_dict=distance_dict,
                sensor_ids=sensor_ids,
                data_start_time=data_start_time,
                start_time=action_start_time,
                end_time=action_end_time,
                target_fps=120
            )

            # 打印 range_data 内容
            print("\n----- range_data 内容 -----")
            for sensor_id, distances in range_data.items():
                print(f"传感器 {sensor_id} 的加权平均距离，形状: {distances.shape}")
                print(f"传感器 {sensor_id} 的前5帧距离数据:")
                print(distances[:5])
                print("-" * 50)
            print("----- end of range_data -----\n")
        else:
            print(f"动作标签 '{action_label}' 不存在于标签字典中。")

        print(f"===== 完成会话: {session} =====\n")

if __name__ == "__main__":
    main()
