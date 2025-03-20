import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import localization as lx  # 请确保该模块已正确实现并在环境中可用
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib as mpl

# 房间边界信息
rooms = {
    "kitchen": (0, 1.5, 4.563, 6),
    "livingroom": (0, 8.5, 0, 4.563),
    "bathroom": (1.5, 4.5, 4.563, 6.6),
    "bedroom": (4.5, 8.5, 4.563, 7.5),
    "Exit": (0, 0.5, 0, 0.5),
    "Couch": (7, 7.5, 1, 3)
}

# 锚点位置
loc_nod = {
    '1':   [0.289, 2.832, 1.410],
    '2':   [0.630, 3.141, 1.439],
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
    '15':  [7.745, 1.412, 0.901],
    '16':  [8.572, 3.100, 1.405]
}

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

def save_animation_mp4_with_room(loc_rdm_pred, mp4_save_path, frame_start, frame_end, ffmpeg_path, rooms, true_room="Living Room", result_txt="room_results.txt"):
    mpl.rcParams['animation.ffmpeg_path'] = ffmpeg_path

    # 动态计算房屋尺寸范围（针对主图绘制）
    x_min = min(room[0] for room in rooms.values())
    x_max = max(room[1] for room in rooms.values())
    y_min = min(room[2] for room in rooms.values())
    y_max = max(room[3] for room in rooms.values())

    # 适当增加边距，避免轨迹贴近边界
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05

    predicted_rooms = []

    # 创建两列子图：左侧用于绘制定位图，右侧用于显示实时文本
    fig, (ax_main, ax_text) = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [3, 1]})

    def draw_room_layout(ax):
        """ 绘制房间结构 """
        for room_name, (xmin, xmax, ymin, ymax) in rooms.items():
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                     fill=False, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text((xmin + xmax) / 2, (ymin + ymax) / 2, room_name, 
                    fontsize=12, ha='center', va='center')

    def init():
        ax_main.clear()
        draw_room_layout(ax_main)
        ax_main.set_xlim(x_min - x_margin, x_max + x_margin)
        ax_main.set_ylim(y_max + y_margin, y_min - y_margin)
        ax_main.set_xlabel("x (m)")
        ax_main.set_ylabel("y (m)")
        ax_main.set_title("Localization Prediction (2D Projection)")
        
        ax_text.clear()
        ax_text.axis('off')
        return []

    def update(frame):
        # 如果任意节点的测距 < 0，说明该帧此节点无效
        if any(range_data[nod][frame] < 0 for nod in nodes_anc):
            return []
        # 更新主图
        ax_main.clear()
        draw_room_layout(ax_main)
        ax_main.set_xlim(x_min - x_margin, x_max + x_margin)
        ax_main.set_ylim(y_max + y_margin, y_min - y_margin)
        ax_main.set_xlabel("x (m)")
        ax_main.set_ylabel("y (m)")

        current_point = loc_rdm_pred[frame, :2]
        pred_room = get_room_by_rect(current_point[0], current_point[1], rooms)
        predicted_rooms.append(pred_room)

        # 根据房间判断结果调整背景颜色
        if pred_room == true_room:
            ax_main.set_facecolor("#ccffcc")  # 正确 (绿色)
        elif pred_room == "Unknown":
            ax_main.set_facecolor("#dddddd")  # 未知 (灰色)
        else:
            ax_main.set_facecolor("#ffcccc")  # 错误 (红色)

        ax_main.set_title(f"Frame {frame}\nPredicted Room: {pred_room} (True: {true_room})")

        # 绘制轨迹：展示最近20帧的定位结果
        start_frame = max(0, frame - 20)
        traj = loc_rdm_pred[start_frame:frame+1, :2]
        ax_main.plot(traj[:, 0], traj[:, 1], linestyle='-', color='blue', alpha=0.7)
        ax_main.scatter(traj[:, 0], traj[:, 1], marker='x', color='blue', s=50, alpha=0.7)
        ax_main.scatter(current_point[0], current_point[1], marker='o', color='red', s=100, zorder=10)

        # 更新右侧文本区域，不覆盖主图内容
        ax_text.clear()
        ax_text.axis('off')
        text_str = f"Predicted (x,y): ({current_point[0]:.2f}, {current_point[1]:.2f})\n"
        # 依次打印每个 sensor 节点的测距值
        for nod in nodes_anc:
            sensor_distance = range_data[nod][frame]
            text_str += f"Node {nod} distance: {sensor_distance:.2f}\n"
        ax_text.text(0.02, 0.98, text_str, transform=ax_text.transAxes, fontsize=10, 
                     verticalalignment='top',
                     bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))
        return []

    frames_range = range(frame_start, frame_end + 1)
    anim = animation.FuncAnimation(fig, update, frames=frames_range, init_func=init, blit=False)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Your Name'), bitrate=1800)
    anim.save(mp4_save_path, writer=writer)
    plt.close(fig)

    # 保存房间判断结果
    with open(result_txt, "w", encoding="utf-8") as f:
        f.write("frame_idx, predicted_room\n")
        for t, room in enumerate(predicted_rooms):
            f.write(f"{t}, {room}\n")

    print(f"Localization MP4 animation with room info saved to: {mp4_save_path}")
    print(f"Room judgment results saved to: {result_txt}")

# ------------------------------
# 定义参与定位的节点列表
# nodes_anc = ["2", "6", "7", "8", "9", "10", "11", "12", "16"]
nodes_anc = ["7", "8", "9"]

data_session = "SB-94975U"

# 加载我们自己保存的距离数据（npy 文件路径根据实际情况修改）
distance_nodes = {}
for node in nodes_anc:
    distance_node = np.load(f"data/dist/{data_session}/{data_session}_node{node}_dist.npy")
    distance_nodes[node] = distance_node
# distance_node7 = np.load("data/dist/SB-94975U/SB-94975U_node7_dist.npy")
# distance_node8 = np.load("data/dist/SB-94975U/SB-94975U_node8_dist.npy")
# distance_node9 = np.load("data/dist/SB-94975U/SB-94975U_node9_dist.npy")
# distance_node10 = np.load("data/dist/SB-94975U/SB-94975U_node10_dist.npy")
# distance_node11 = np.load("data/dist/SB-94975U/SB-94975U_node11_dist.npy")
# distance_node12 = np.load("data/dist/SB-94975U/SB-94975U_node12_dist.npy")
# distance_node2 = np.load("data/dist/SB-94975U/SB-94975U_node2_dist.npy")
# distance_node6 = np.load("data/dist/SB-94975U/SB-94975U_node6_dist.npy")
# distance_node16 = np.load("data/dist/SB-94975U/SB-94975U_node16_dist.npy")

# 构造距离数据字典，key 为节点编号（字符串），value 为对应的距离数据数组
range_data = {
    "7": distance_nodes["7"],
    "8": distance_nodes["8"],
    "9": distance_nodes["9"]
    # "10": distance_nodes["10"],
    # "11": distance_nodes["11"],
    # "12": distance_nodes["12"],
    # "2": distance_nodes["2"],
    # "6": distance_nodes["6"],
    # "16": distance_nodes["16"]
}

# 假定已定义起始索引 start 与终止索引 right
# start = 6384   # 示例起始索引
# right = 10610  # 示例终止索引
# start = 0   # 示例起始索引
# right = 10000  # 示例终止索引
intensity_threshold = 0.0025  # 只考虑强度大于该阈值的传感器数据

# 选取切片数据，并以最小长度确定迭代帧数
T = min([range_data[nod][:].shape[0] for nod in nodes_anc])
print(f"Performing localization on {T} frames")

loc_rdm_pred = []
valid_frames = []
for t in tqdm(range(T), desc="2D Localizing"):
    # 如果任意节点的测距 < 0，说明该帧此节点无效
    if any(range_data[nod][t] < 0 for nod in nodes_anc):
        continue
    
    # 创建定位工程，选择2D LSE 定位
    P = lx.Project(mode='2D', solver='LSE')
    # 添加参与定位的锚点
    for nod in nodes_anc:
        P.add_anchor(nod, loc_nod[nod])
    # 添加待定位目标
    target, _ = P.add_target()
    # 给每个锚点添加对应的测距值（这里取切片数据）
    for nod in nodes_anc:
        measure_val = range_data[nod][t]
        target.add_measure(nod, measure_val)
    # 求解定位问题
    P.solve()
    # 提取定位结果（2D 定位时 z 通常为 0）
    loc_rdm_pred.append([target.loc.x, target.loc.y, target.loc.z])
    valid_frames.append(t)  # 记录有效帧索引

loc_rdm_pred = np.array(loc_rdm_pred)

# 保存 MP4 动画及房间判断结果
frame_start = 0
frame_end = len(valid_frames) - 1  # **改为有效帧数**
true_room = "bathroom"  # 示例真值房间名称
mp4_save_path = f"{data_session}_localization_animation_{true_room}.mp4"
ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"  # 请根据实际情况修改 FFmpeg 路径

save_animation_mp4_with_room(loc_rdm_pred, mp4_save_path, frame_start, frame_end, ffmpeg_path, rooms, true_room, result_txt="room_results.txt")
