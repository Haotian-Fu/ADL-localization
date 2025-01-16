import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from tqdm import tqdm
import localization as lx  # 确保此模块已正确实现并可访问
from datetime import datetime, timezone

# 确保Matplotlib可以找到FFmpeg（如果需要）
import matplotlib as mpl
# mpl.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'  # Windows示例，如果不需要FFmpeg，可以注释掉

def loc_nod_init():
    # 初始化锚点节点的位置
    loc_nod = {}
    nodes = ['2', '15', '16']
    loc_nod['2'] = [0, 0, 0]
    loc_nod['15'] = [8.5, 3, 0]
    loc_nod['16'] = [8.5, -1.5, 0]

    print("Initialized Anchor Node Locations:")
    for node_id, coords in loc_nod.items():
        print(f"  Node {node_id}: {coords}")

    return loc_nod

def parse_timestamp(ts_str):
    # 将时间戳字符串解析为datetime对象
    try:
        datetime_obj = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        try:
            datetime_obj = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
        except ValueError as e:
            raise ValueError(f"Timestamp format not recognized: '{ts_str}'") from e
    return datetime_obj

def load_all_doppler_data(data_dir, node_sensor_map):
    # 加载所有节点的Doppler衍生的Range-Doppler数据和时间戳
    range_data = {}
    timestamps_dict = {}
    frame_counts = {}

    for node_id, sensor_id in node_sensor_map.items():
        doppler_1d_path = os.path.join(data_dir, f'doppler_1d_from_UWB_{sensor_id}.npy')
        timestamps_path = os.path.join(data_dir, f'{sensor_id}_timestamp.txt')

        if not os.path.exists(doppler_1d_path):
            raise FileNotFoundError(f"Doppler 1D data file not found for Sensor ID {sensor_id}: {doppler_1d_path}")

        doppler_1d = np.load(doppler_1d_path)

        if not os.path.exists(timestamps_path):
            raise FileNotFoundError(f"Timestamps file not found for Sensor ID {sensor_id}: {timestamps_path}")

        with open(timestamps_path, 'r') as f:
            timestamps_str = f.readlines()
        timestamps_node = np.array([parse_timestamp(ts.strip()) for ts in timestamps_str])

        num_frames, num_bins = doppler_1d.shape
        if len(timestamps_node) != num_frames:
            print(f"Warning: Number of timestamps ({len(timestamps_node)}) does not match number of frames ({num_frames}) for sensor {sensor_id}.")
            min_frames = min(len(timestamps_node), num_frames)
            doppler_1d = doppler_1d[:min_frames, :]
            timestamps_node = timestamps_node[:min_frames]
            frame_counts[node_id] = min_frames
        else:
            frame_counts[node_id] = num_frames

        timestamps_dict[node_id] = timestamps_node

        range_bin_indices = np.argmax(doppler_1d, axis=1)
        bin_size = 0.05
        ranges = range_bin_indices * bin_size

        range_data[node_id] = ranges

        print(f"Loaded and processed Doppler data for Node {node_id} (Sensor ID {sensor_id}): {ranges.shape}")

    min_frames = min(frame_counts.values())
    print(f"Minimum number of frames across all sensors: {min_frames}")

    reference_timestamps = timestamps_dict[list(node_sensor_map.keys())[0]][:min_frames]
    start_time = reference_timestamps[0]
    timestamps = np.array([(ts - start_time).total_seconds() for ts in reference_timestamps])

    for node_id in range_data:
        range_data[node_id] = range_data[node_id][:min_frames]

    return range_data, timestamps

def main():
    # 配置
    data_dir = r"D:\OneDrive\桌面\code\ADL_localization\data\SB-94975U"  # 修改为你的数据路径

    # 节点与传感器ID映射
    node_sensor_map = {
        '2': 'e4-5f-01-78-c9-40',
        '15': 'e4-5f-01-8a-78-6f',
        '16': 'e4-5f-01-51-a7-97'
    }

    # 初始化锚点节点位置
    loc_nod = loc_nod_init()

    # 加载所有节点的Doppler数据和时间戳
    range_data, timestamps = load_all_doppler_data(data_dir, node_sensor_map)

    # 定义锚点节点
    nodes_anc = list(node_sensor_map.keys())

    # 总帧数
    N_total = len(timestamps)
    print(f"Total number of synchronized frames: {N_total}")

    # ================== 引入帧跳过以降低移动频率 ==================
    frame_skip = 3  # 选择每隔3帧处理一次，根据需要调整此值
    frames_to_process = np.arange(0, N_total, frame_skip)
    N = len(frames_to_process)
    print(f"Frame skip set to {frame_skip}. Processing {N} frames out of {N_total}.")

    # 切片数据以包含选择的帧
    range_data = {k: v[frames_to_process] for k, v in range_data.items()}
    timestamps = timestamps[frames_to_process]
    print(f"Processing and animating {N} frames.")

    # 初始化存储定位结果的列表
    loc_rdm_pred = []

    # 执行定位
    for ix in tqdm(range(N), desc="Localizing"):
        P = lx.Project(mode='2D', solver='LSE')

        # 将锚点节点添加到项目中
        for nod in nodes_anc:
            P.add_anchor(nod, loc_nod[nod])

        # 添加目标
        t_rdm, _ = P.add_target()

        # 将测距数据添加到目标
        for nod in nodes_anc:
            dist_rdm = range_data[nod][ix] + 0.35  # 根据需要应用偏移量
            t_rdm.add_measure(nod, dist_rdm)

        # 解决定位问题
        P.solve()

        # 提取目标位置
        loc_rdm_ = np.asarray([t_rdm.loc.x, t_rdm.loc.y, t_rdm.loc.z])

        # 将估计的位置添加到列表中
        loc_rdm_pred.append(loc_rdm_)

    # 转换为NumPy数组并提取X和Y坐标
    loc_rdm_pred = np.asarray(loc_rdm_pred)
    loc_rdm_pred = loc_rdm_pred[:, :2]  # 仅保留X和Y坐标

    # ==================== 定义轨迹平滑函数（可选） ====================
    def smooth_trajectory(x, y, window_size=5):
        """ 使用简单移动平均平滑轨迹 """
        window = np.ones(window_size) / window_size
        x_smooth = np.convolve(x, window, mode='same')
        y_smooth = np.convolve(y, window, mode='same')
        return x_smooth, y_smooth

    # 平滑轨迹（可选）
    x_data, y_data = smooth_trajectory(loc_rdm_pred[:, 0], loc_rdm_pred[:, 1], window_size=5)

    # ==================== 动画可视化定位结果 ====================

    # 定义总动画时长
    total_animation_duration_ms = 10000  # 30秒，单位毫秒

    # 计算帧间隔，以确保所有帧在30秒内展示完毕
    if N == 0:
        raise ValueError("No data frames loaded.")
    interval = total_animation_duration_ms / N  # 单位：毫秒
    print(f"Total animation duration: {total_animation_duration_ms} ms")
    print(f"Animation interval between frames: {interval:.3f} ms")

    # 设置图形和坐标轴
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(np.min(x_data) - 1, np.max(x_data) + 1)
    ax.set_ylim(np.min(y_data) - 1, np.max(y_data) + 1)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title("Localization Results")
    ax.grid(True)

    # 绘制锚点节点位置
    anchor_x = [loc_nod[nod][0] for nod in nodes_anc]
    anchor_y = [loc_nod[nod][1] for nod in nodes_anc]
    ax.scatter(anchor_x, anchor_y, marker='^', color='blue', label='Anchor Nodes', zorder=3)

    # 初始化轨迹线条
    trajectory_line, = ax.plot([], [], color='green', linewidth=2, label='Trajectory', zorder=2)

    # 初始化目标位置散点
    target_scatter, = ax.plot([], [], marker='o', color='red', label='Estimated Target Position', zorder=4)

    # 初始化时间文本
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    # 使用deque存储最近5秒的轨迹位置
    trajectory_window = 5.0  # 秒
    trajectory_frames = int(trajectory_window * 1000 / interval)  # 根据帧间隔计算帧数
    trajectory_deque = deque(maxlen=trajectory_frames)
    print(f"Trajectory window: {trajectory_window} seconds ({trajectory_frames} frames)")

    # 初始化存储轨迹坐标的列表
    trajectory_coords = []

    # 定义更新函数
    def update(frame):
        # 更新目标位置
        target_scatter.set_data([x_data[frame]], [y_data[frame]])

        # 更新轨迹deque
        trajectory_deque.append((x_data[frame], y_data[frame]))
        trajectory_points = np.array(trajectory_deque)
        trajectory_line.set_data(trajectory_points[:, 0], trajectory_points[:, 1])

        # 添加当前坐标到轨迹列表
        trajectory_coords.append((x_data[frame], y_data[frame]))

        # 更新时间文本
        time_text.set_text(f"Time: {timestamps[frame]:.3f} s")

        return target_scatter, trajectory_line, time_text

    # 创建动画
    ani = FuncAnimation(fig, update, frames=N, interval=interval, blit=True)

    # 添加图例
    ax.legend()

    # 定义保存动画和轨迹坐标的函数
    def save_animation_and_trajectory(ani, trajectory_coords):
        # 保存动画为GIF文件
        gif_filename = 'localization_animation.gif'
        writer = mpl.animation.PillowWriter(fps=int(1000 / interval))
        ani.save(gif_filename, writer=writer)
        print(f"Animation saved as '{gif_filename}'.")

        # 保存轨迹坐标到文本文件
        print("\nTrajectory Coordinates:")
        with open("trajectory.txt", "w") as traj_file:
            traj_file.write("Trajectory Coordinates (X, Y):\n")
            for coord in trajectory_coords:
                coord_str = f"({coord[0]:.3f}, {coord[1]:.3f})"
                # print(coord_str)
                traj_file.write(f"{coord[0]:.3f}, {coord[1]:.3f}\n")
        print("Trajectory coordinates saved to 'trajectory.txt'.")

    # 连接动画关闭事件以保存GIF和轨迹
    def on_close(event):
        save_animation_and_trajectory(ani, trajectory_coords)

    fig.canvas.mpl_connect('close_event', on_close)

    # 显示动画
    plt.show()

if __name__ == "__main__":
    main()
