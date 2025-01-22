import os
import numpy as np

# 假设 sessions 是一个包含会话ID的列表，例如：
sessions = ["6e5iYM_ADL_1"]  # 根据实际情况修改

# 循环处理每个 session
for session in sessions:
    # 构造数据、标签和 mask 文件的路径
    data_file = os.path.join("data", "all_activities", f"{session}_data.dat")
    label_file = os.path.join("data", "all_activities", f"{session}_label.dat")
    mask_file = os.path.join("data", "all_activities", f"{session}_mask_mannual.dat")
    
    # 使用 memmap 读取数据：
    # 这里假设数据以 float32 格式存储，总数据形状为 (T * 16 * 288)，
    # 重塑后得到 (T, 16, 288)
    try:
        data = np.memmap(data_file, dtype='float32', mode='r')
        T = data.size // (16 * 288)
        data = data.reshape((T, 16, 288))
    except Exception as e:
        print(f"Error reading data from {data_file}: {e}")
        continue

    # 读取标签数据（假设存储为 int64）
    try:
        label = np.memmap(label_file, dtype='int64', mode='r')
    except Exception as e:
        print(f"Error reading label from {label_file}: {e}")
        continue

    # 读取 mask 数据（假设 mask 数据为 float32，且总元素数为 T * 16）
    try:
        mask = np.memmap(mask_file, dtype='float32', mode='r')
        mask = mask.reshape((-1, 16))
    except Exception as e:
        print(f"Error reading mask from {mask_file}: {e}")
        continue

    # 打印读取的数据形状
    print(f"Session: {session}")
    print("Data shape:", data.shape)   # 应该为 (T, 16, 288)
    print("Label shape:", label.shape)
    print("Mask shape:", mask.shape)

    # 下面可以进行后续的处理，例如：
    # - 对 data 进行预处理/特征提取
    # - 利用 label 和 mask 进行监督/无监督学习
    # - 或将数据送入后续的雷达定位、动作识别等模块
