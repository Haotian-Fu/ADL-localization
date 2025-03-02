import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from collections import deque
from tqdm import tqdm
import localization as lx  # Ensure this module is correctly implemented and accessible
from scipy.interpolate import interp1d
from datetime import timedelta, timezone, datetime
from helper import load_txt_to_datetime, datetime_from_str, color_scale, seg_index, load_segment_file_to_datetime
from config import nodes_info, nodes_info_pi4, room_info
import imageio
import matplotlib.animation as animation
import random

import warnings
warnings.filterwarnings("ignore", message="set_ticklabels() should only be used with a fixed number of ticks")

# Ensure Matplotlib can find FFmpeg
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'  # Update this path if necessary

rooms = {
    "kitchen": (0, 1.5, 4.563, 6),
    "livingroom": (0, 8.5, 0, 4.563),
    "bathroom": (1.5, 4.5, 4.563, 6.6),
    "bedroom": (4.5, 8.5, 4.563, 7.5),
    "Exit": (0, 0.5, 0, 0.5),
    "Couch": (7, 7.5, 1, 3)
}

loc_nod = {
    '1':   [0.289, 2.832, 1.410],
    '2':   [0.630,   3.141,  1.439],
    '3':   [0.340, 3.742, 0.871],
    '4':   [2.104, 4.534, 1.480],
    '5':   [1.906, 5.004, 1.557],
    '6':   [4.346, 4.273, 0.894],
    '7':   [1.5, 5.127, 1.232],
    '8':   [3.764, 5.058, 0.912],
    '9':   [2.008, 6.6, 1.352],
    '10':  [4.5, 5.153, 1.444],
    '11':  [7.733, 7.5, 1.643],
    '12':  [7.956, 5.028, 0.881],
    '13':  [5.365, 0.787, 0.892],
    '14':  [5.687, 0.749, 0.885],
    '15':  [7.745,   1.412,  0.901],
    '16':  [8.572,   3.100,  1.405]
}

# 判断被观察者所处房间的函数（这里以矩形边界为例）
def get_room_by_rect(x, y, rooms):
    """
    根据矩形区域判断点 (x,y) 所处房间
    参数：
       x, y: 被判断的坐标
       rooms: dict, key 为房间名，value 为 (xmin, xmax, ymin, ymax)
    返回：
       房间名称，若没有匹配返回 "Unknown"
    """
    for room_name, (xmin, xmax, ymin, ymax) in rooms.items():
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return room_name
    return "Unknown"

def lse_localization_3d(range_data, nodes_anc, loc_nod, offset=0.0, start=0, end=20000):
    """
    使用 3D LSE 定位。
    参数：
      range_data: 字典，key 为节点编号，value 为 (T,) 的距离数据数组
      nodes_anc: 需要参与定位的节点编号列表（字符串），
      loc_nod: 字典，key 为节点编号，value 为对应锚点的 3D 坐标 [x, y, z]
      offset: 测距数据的偏置（默认为 0.0）
      save_path_loc: 保存定位结果的文件名
    返回：
      loc_rdm_pred: NumPy 数组，形状为 (T, 3)，每一行为 [x, y, z] 的预测坐标
    """
    # T = range_data[nodes_anc[0]].shape[0]
    loc_rdm_pred = []
    print("Starting 3D LSE Localization using nodes:", nodes_anc)

    for t in tqdm(range(start, end), desc="3D Localizing"):
        P = lx.Project(mode='3D', solver='LSE')
        # Add anchors
        for nod in nodes_anc:
            P.add_anchor(nod, loc_nod[nod])
        # Add target
        target, _ = P.add_target()
        # Add measurements
        for nod in nodes_anc:
            measure_val = range_data[nod][t] + offset
            target.add_measure(nod, measure_val)
        # Solve
        P.solve()
        # Extract (x,y,z)
        loc_current = np.array([target.loc.x, target.loc.y, target.loc.z])
        loc_rdm_pred.append(loc_current)

    loc_rdm_pred = np.array(loc_rdm_pred)
    print("3D Localization Completed. Results shape:", loc_rdm_pred.shape)
    return loc_rdm_pred


def save_animation_mp4_with_room(loc_rdm_pred, mp4_save_path, frame_start, frame_end, ffmpeg_path, rooms, true_room="Living Room", result_txt="room_results.txt"):
    mpl.rcParams['animation.ffmpeg_path'] = ffmpeg_path

    # 动态计算房屋尺寸范围
    x_min = min(room[0] for room in rooms.values())
    x_max = max(room[1] for room in rooms.values())
    y_min = min(room[2] for room in rooms.values())
    y_max = max(room[3] for room in rooms.values())

    # 适当增加边距，避免轨迹贴近边界
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05

    predicted_rooms = []
    trajectory = []
    
    # fig, ax = plt.subplots(figsize=(10, 8))

    # def draw_room_layout(ax):
    #     """ 绘制房间结构 """
    #     for room_name, (xmin, xmax, ymin, ymax) in rooms.items():
    #         rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
    #                                  fill=False, edgecolor='black', linewidth=2)
    #         ax.add_patch(rect)
    #         ax.text((xmin + xmax) / 2, (ymin + ymax) / 2, room_name, 
    #                 fontsize=12, ha='center', va='center')

    # def init():
    #     ax.clear()
    #     draw_room_layout(ax)
    #     ax.set_xlim(x_min - x_margin, x_max + x_margin)
    #     ax.set_ylim(y_max + y_margin, y_min - y_margin)
    #     ax.set_xlabel("x (m)")
    #     ax.set_ylabel("y (m)")
    #     ax.set_title("Localization Prediction (2D Projection)")
    #     return []

    # def update(frame):
    #     ax.clear()
    #     draw_room_layout(ax)
    #     ax.set_xlim(x_min - x_margin, x_max + x_margin)
    #     ax.set_ylim(y_max + y_margin, y_min - y_margin)
    #     ax.set_xlabel("x (m)")
    #     ax.set_ylabel("y (m)")

    #     # 获取当前预测坐标
    #     current_point = loc_rdm_pred[frame, :2]
    #     pred_room = get_room_by_rect(current_point[0], current_point[1], rooms)
    #     predicted_rooms.append(pred_room)
    #     trajectory.append(current_point)

    #     # 根据房间判断结果调整背景颜色
    #     if pred_room == true_room:
    #         ax.set_facecolor("#ccffcc")  # 正确 (绿色)
    #     elif pred_room == "Unknown":
    #         ax.set_facecolor("#dddddd")  # 未知 (灰色)
    #     else:
    #         ax.set_facecolor("#ffcccc")  # 错误 (红色)

    #     ax.set_title(f"Frame {frame}\nPredicted Room: {pred_room} (True: {true_room})")

    #     # 轨迹绘制
    #     start_frame = max(0, frame - 20)
    #     traj = loc_rdm_pred[start_frame:frame+1, :2]
    #     ax.plot(traj[:, 0], traj[:, 1], linestyle='-', color='blue', alpha=0.7)
    #     ax.scatter(traj[:, 0], traj[:, 1], marker='x', color='blue', s=50, alpha=0.7)
    #     ax.scatter(current_point[0], current_point[1], marker='o', color='red', s=100, zorder=10)

    #     return []
    
    # frames_range = range(frame_start, frame_end + 1)
    # anim = animation.FuncAnimation(fig, update, frames=frames_range, init_func=init, blit=False)
    
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=10, metadata=dict(artist='Your Name'), bitrate=1800)
    # anim.save(mp4_save_path, writer=writer)
    # plt.close(fig)
    
    # 对定位结果进行房间判断，并保存到 txt 文件
    predicted_rooms = []
    trajectory = []

    # 遍历所有预测的坐标 (loc_rdm_pred)
    for idx in range(loc_rdm_pred.shape[0]):
        # 获取 (x,y) 坐标
        current_point = loc_rdm_pred[idx, :2]
        # 根据房间边界判断当前点所在房间
        pred_room = get_room_by_rect(current_point[0], current_point[1], rooms)
        predicted_rooms.append(pred_room)
        trajectory.append(current_point)

    # 保存结果到文本文件
    result_txt = "room_results.txt"
    with open(result_txt, "w", encoding="utf-8") as f:
        f.write("frame_idx, x_coordinate, y_coordinate, predicted_room\n")
        for idx, (point, room) in enumerate(zip(trajectory, predicted_rooms)):
            f.write(f"{idx}, {point[0]}, {point[1]}, {room}\n")

    print(f"Room judgment results saved to: {result_txt}")

    # # 保存房间判断结果
    # with open(result_txt, "w", encoding="utf-8") as f:
    #     f.write("frame_idx, x_coordinate, y_coordinate, predicted_room\n")
    #     for t, room in enumerate(predicted_rooms):
    #         f.write(f"{t}, {trajectory[0]}, {trajectory[1]}, {room}\n")

    # print(f"Localization MP4 animation with room info saved to: {mp4_save_path}")
    # print(f"Room judgment results saved to: {result_txt}")


# 读取之前存储的 node2 的距离数据
distance_node7 = np.load("data/dist/SB-94975U/SB-94975U_node7_dist.npy")
distance_node8 = np.load("data/dist/SB-94975U/SB-94975U_node8_dist.npy")
distance_node9 = np.load("data/dist/SB-94975U/SB-94975U_node9_dist.npy")

# 构造距离数据字典，键为节点编号，值为对应的距离数据数组（假设所有节点的数据长度一致）
range_data = {
    "7": distance_node7,
    "8": distance_node8,
    "9": distance_node9
}

# 定义参与定位的节点列表
nodes_anc = ["7", "8", "9"]

# 调用3D LSE定位函数进行定位
loc_rdm_pred = lse_localization_3d(range_data, nodes_anc, loc_nod, offset=0.0, start=6000, end=12000)

# 这里默认真值为 "bedroom"
gif_save_path = 'localization_animation_room.gif'
mp4_save_path = 'localization_animation_room.mp4'
ffmpeg_path = r'C:\ffmpeg\bin\ffmpeg.exe'  # 修改为实际路径

# 注意：动画部分仍绘制 (x,y) 投影，同时在标题中显示房间判断结果，并设置背景颜色
# save_animation_gif_with_room(loc_rdm_pred, gif_save_path, frame_start=0, frame_end=281, rooms=rooms, true_room="living", result_txt="room_results_anim.txt")
save_animation_mp4_with_room(
    loc_rdm_pred, 
    mp4_save_path=mp4_save_path, 
    frame_start=0, 
    # frame_end=label_to_action[action_label].shape[0]-1, 
    frame_end=5999,
    ffmpeg_path=ffmpeg_path,
    rooms=rooms, 
    true_room="Bathroom",
    result_txt="room_results_anim.txt"
)
