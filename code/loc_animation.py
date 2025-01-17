import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
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

# Ensure Matplotlib can find FFmpeg
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'  # Update this path if necessary

def load_data(session_path, node_idx, pi_version=4):
    """
    load complex data and datetime from raw data files
    """
    if pi_version == 3:
        node_id = nodes_info[node_idx]["id"]
    else:
        node_id = nodes_info_pi4[node_idx]["id"]
    print(f"Loading data from {node_id}")
    
    complex_file = os.path.join(session_path, node_id + '_complex.npy')
    ts_uwb_file = os.path.join(session_path, node_id + '_timestamp.txt')

    data_complex = np.load(complex_file)
    dt = load_txt_to_datetime("", ts_uwb_file)    # load timestamp of each data frame
    return data_complex, dt

def resample(data_complex, dt, start_time, end_time, target_fps=120):
    """
    resample data_complex to target_fps frames per second. Using timestamp of each frame to interpolate the real and imaginary parts of the complex data.

    We may use sequence number to interpolate in future.

    data_complex: 2-d array, shape (time, range_bin_num), the complex data of the radar.
    dt: 1-d array, shape (time, ), the timestamp of each frame.
    start_time: datetime, the start time of the data.
    end_time: datetime, the end time of the data.
    target_fps: int, the target frames per second.

    return: 2-d array, shape (time, range_bin_num), the resampled complex data; time_target: 1-d array, shape (time, ), the timestamp of each frame.
    """
    time_original = np.array([(t - dt[0]).total_seconds() for t in dt])
    time_target = np.arange((start_time - dt[0]).total_seconds(), (end_time - dt[0]).total_seconds(), 1/target_fps)
    
    # interpolate real and imaginary parts separately
    real_interp = interp1d(time_original, data_complex.real, axis=0, kind='linear', fill_value='extrapolate')
    imag_interp = interp1d(time_original, data_complex.imag, axis=0, kind='linear', fill_value='extrapolate')
    
    data_complex_resampled = real_interp(time_target) + 1j * imag_interp(time_target)
    return data_complex_resampled, time_target - (start_time - dt[0]).total_seconds()

def load_align_resample(session_path, nodes, target_fps=120,
                        start_time=datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc), end_time=datetime(2030, 1, 1, 0, 0, 0, tzinfo=timezone.utc)):
    """
    load data, align multiple nodes frame by frame and resample to 120 frames per second.

    session_path: str, the path of the session.
    nodes: list, the list of node indices.
    start_time: datetime, the start time of the data.
    end_time: datetime, the end time of the data. Limit the time range of the data to align.

    return: data_complex_resampled: dict, the resampled complex data of each node; 
        time_target: 1-d array, the timestamp of each frame; 
        start_time: datetime, the start time of the data; 
        end_time: datetime, the end time of the data.
    """
    data_complex_multi = {}
    dt_multi = {}
    # load data
    data_complex_resampled = {}
    for node_idx in nodes:
        print(f"Loading data from node {node_idx}...")
        data_complex, dt = load_data(session_path, node_idx)
        data_complex_multi[node_idx] = data_complex
        dt_multi[node_idx] = dt
        if start_time < dt[0]:
            print(f"Start time {start_time} is earlier than the start time of the data {dt[0]}.")
        if end_time > dt[-1]:
            print(f"End time {end_time} is later than the end time of the data {dt[-1]}.")
        start_time = max(start_time, dt[0])
        end_time = min(end_time, dt[-1])
        

    # resample data to 120 frames per second and align among multiple nodes
    for node_idx, data_complex in data_complex_multi.items():
        print(f"Resampling data from node {node_idx}...")
        data_complex_resampled[node_idx], time_target = resample(data_complex, dt_multi[node_idx], start_time, end_time, target_fps=target_fps)

    return data_complex_resampled, time_target, start_time, end_time
    # need to align frames from different nodes. do simple down/up sampling - done
    # need to generate labels for each frame based on the segmentations - done
    # then do training and testing. save history and model
    
def compute_distance_for_one_frame(range_doppler=np.random.rand(32, 188)):
    """
    get the peak columns in one range doppler frame, return the mean value of the peak columns, which is the velocity distribution.
    Assumption here is, during the time of the frame, the object is moving within 20cm, so we use left 2 columns and right 2 columns of the peak column to calculate the sum.

    range_doppler_window: 2-d array, shape (32, 188), the range doppler window of one frame.
    return: 1-d array, shape (32, ), the sum of the peak columns in one range doppler frame.

    """
    doppler_bin_num, range_bin_num = range_doppler.shape
    # range_doppler[14:18, :] = np.zeros((4, range_bin_num))
    
    # d-fft: x is distance, y is velocity, value is intensity.
    # the peak value is the intensity at the particular velocity and distance.
    # find the peak value in 2-d d_fft and its index (v, d) in d_fft, then use the left 2 columns and right columns, 5 columns in total to calculate the sum along x-axis
    # the index is the velocity and distance
    # the mean is the intensity among the 5 columns
    peak = np.max(range_doppler)
    v, d = np.where(range_doppler == peak)
    v, d = v[0], d[0]
    up = v-2 if v-2 >= 0 else 0
    bottom = v+3 if v+3 < doppler_bin_num else doppler_bin_num-1
    return np.sum(range_doppler[up:bottom, :], axis=0)/(bottom - up), v, d, peak

def visualize_velocity_distance_samples(session_path, seg_file, node_idx, pi_version=3, doppler_bin_num=32, step=5, discard_bins=[14, 15, 16, 17]):
# compute and visualize the velocity and distance distribution over time
    # os.chdir("/home/mengjingliu/ADL_Detection")

    data_complex, dt = load_data(session_path, node_idx, pi_version=pi_version)
    segs, acts = seg_index(dt, seg_file)
    velocities = []
    distances = []
    indices = []
    first_act, last_act = 12, 20
    for i in range(first_act, last_act):
        start, end = segs[i]
        # print(acts[i])
        velocity, distance, index = compute_features_over_time(data_complex[start:end, :], doppler_bin_num=doppler_bin_num, step=step, range_bin_num=188, Discard_bins=discard_bins)
        velocities.append(velocity)
        distances.append(distance)
        indices.append(index)

    print("velocity")
    # velocities is a list of 2-d array, with differnet shapes. normalize the values to 0~1
    min_v = min([np.min(velocity) for velocity in velocities])
    max_v = max([np.max(velocity) for velocity in velocities])
    i = first_act
    for v in velocities:
        # print(acts[i])
        i += 1
        v = (v - min_v) / (max_v - min_v)   # normalize the values to 0~1
        sns.heatmap(v.T, cmap='viridis', cbar=False, xticklabels=False, yticklabels=False)
        plt.title(acts[i])
        plt.show()
        
    # distance is a list of 2-d array, with differnet shapes. normalize the values to 0~1
    print("distance")
    min_d = min([np.min(distance) for velocity in distances])
    max_d = max([np.max(distance) for velocity in distances])
    i = 14
    for d in distances:
        # print(acts[i])
        i += 1
        d = (d - min_d) / (max_d - min_d)   # normalize the values to 0~1
        sns.heatmap(d, cmap='viridis', cbar=False, xticklabels=False, yticklabels=False)
        plt.title(acts[i])
        plt.show()

def compute_features_over_time(data_complex, doppler_bin_num=32, step=5, range_bin_num=188, Discard_bins=[14, 15, 16, 17]):
    """
    For each data window (2-d, (time, range_bin_num)), compute the range doppler using sliding window and FFT. window siez is 32, step is 5. 
        -> Get all range doppler frames (3-d, (time, doppler_bin_num, range_bin_num)).
    For each range doppler frame (2-d, (doppler_bin_num, range_bin_num)), compute the velocity distribution (1-d, (doppler_bin_num, )).
    Concatenate the velocity distribution of each frame to get the velocity distribution over time (2-d, (time, doppler_bin_num)).
    
    input: data_complex: 2-d array, shape (time, range_bin_num), the complex data of the radar.
    output: 
        velocity: 2-d array, shape ((time-32)//step, doppler_bin_num), the velocity distribution over time.
        distance: 2-d array, shape ((time-32)//step, range_bin_num), the distance distribution over time.
        indices: 1-d array, shape ((time-32)//step, ), the indices of the data frames.
    """
    frame_num = (data_complex.shape[0] - doppler_bin_num) // step + 1
    range_doppler = np.zeros((frame_num, doppler_bin_num, range_bin_num))
    for i in range(0, data_complex.shape[0] - doppler_bin_num, step):
        range_doppler[i//step] = np.abs(np.fft.fft(data_complex[i:i+doppler_bin_num, :], axis=0))  # FFT
        range_doppler[i//step] = np.fft.fftshift(range_doppler[i//step], axes=0)  # shifts
        range_doppler[i//step, Discard_bins, :] = np.zeros((4, range_bin_num))

    distance = np.array([compute_distance_for_one_frame(range_doppler[i, :, :])[0] for i in range(frame_num)])
    indices = np.arange(0, data_complex.shape[0] - doppler_bin_num, step) + (doppler_bin_num // 2)

    return distance, indices

def visualize_velocity_distance_samples(session_path, seg_file, node_idx, pi_version=3, doppler_bin_num=32, step=5, discard_bins=[14, 15, 16, 17]):
# compute and visualize the velocity and distance distribution over time
    # os.chdir("/home/mengjingliu/ADL_Detection")

    data_complex, dt = load_data(session_path, node_idx, pi_version=pi_version)
    segs, acts = seg_index(dt, seg_file)
    distances = []
    indices = []
    first_act, last_act = 12, 20
    for i in range(first_act, last_act):
        start, end = segs[i]
        print(acts[i])
        distance, index = compute_features_over_time(data_complex[start:end, :], doppler_bin_num=doppler_bin_num, step=step, range_bin_num=188, Discard_bins=discard_bins)
        distances.append(distance)
        indices.append(index)
        
    # distance is a list of 2-d array, with differnet shapes. normalize the values to 0~1
    print("distance")
    min_d = min([np.min(distance) for velocity in distances])
    max_d = max([np.max(distance) for velocity in distances])
    i = 14
    for d in distances:
        # print(acts[i])
        i += 1
        d = (d - min_d) / (max_d - min_d)   # normalize the values to 0~1
        sns.heatmap(d, cmap='viridis', cbar=False, xticklabels=False, yticklabels=False)
        plt.title(acts[i])
        plt.show()

def loc_nod_init():
    # Initialize the locations of anchor nodes.
    loc_nod = {}
    nodes = ['2', '15', '16']
    # x_delta = 1.38  # Adjusted value based on sensor placement
    # y_delta = 1.6   # Adjusted value based on sensor placement

    # loc_nod['1'] = [0, 0, 0]
    # loc_nod['2'] = [-1 * x_delta, 1 * y_delta, 0]
    # loc_nod['3'] = [-1 * x_delta, 2 * y_delta, 0]
    # loc_nod['4'] = [-1 * x_delta, 3 * y_delta, 0]
    # loc_nod['5'] = [0, 4 * y_delta, 0]
    # loc_nod['7'] = [1 * x_delta, 2 * y_delta, 0]
    # loc_nod['8'] = [1 * x_delta, 1 * y_delta, 0]
    loc_nod['2'] = [0, 0, 0]
    # loc_nod['8'] = [0, 0, 0]
    # loc_nod['7'] = [3, 0.5, 0]
    # loc_nod['9'] = [2, 2.4, 0]
    loc_nod['15'] = [8.5, 3, 0]
    loc_nod['16'] = [8.5, -1.5, 0]

    print("Initialized Anchor Node Locations:")
    for node_id, coords in loc_nod.items():
        print(f"  Node {node_id}: {coords}")

    return loc_nod

def parse_timestamp(ts_str):
    # Parse a timestamp string into a datetime object.
    try:
        datetime_obj = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        try:
            datetime_obj = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
        except ValueError as e:
            raise ValueError(f"Timestamp format not recognized: '{ts_str}'") from e
    return datetime_obj

def perform_localization(nodes_anc, loc_nod, list_dist_ske_, list_dist_rdm_, offset=0.35):
    """
    使用至少 3 个传感器节点的数据进行定位，分别对基于 ske 的距离和 rdm 的距离执行 LSE。

    参数:
    --------
    nodes_anc : list
        锚点节点的列表，例如 ['2', '15', '16']。
    loc_nod : dict
        每个锚点节点的坐标映射，如:
            {
                '2': [x2, y2, 0],
                '15': [x15, y15, 0],
                '16': [x16, y16, 0]
            }
    list_dist_ske_ : dict
        键为节点ID (与 nodes_anc 对应)，值为该节点的 ske 距离列表/数组 (长度 N)。
        e.g. list_dist_ske_['2'][ix] => 第 ix 帧，节点 '2' 的 ske 距离。
    list_dist_rdm_ : dict
        键为节点ID，值为该节点的 rdm 距离列表/数组 (长度 N)。
    offset : float
        对 rdm 距离添加的偏移量(米)，例如 0.35 米。

    返回:
    --------
    loc_ske_pred : ndarray, shape (N, 2)
        基于 ske 距离做 LSE 得到的 (x, y) 坐标。
    loc_rdm_pred : ndarray, shape (N, 2)
        基于 rdm 距离做 LSE 得到的 (x, y) 坐标。
    """

    # 假设所有节点的距离列表长度相同
    # N 为帧数
    N = len(list_dist_ske_[nodes_anc[0]])

    loc_ske_pred = []
    loc_rdm_pred = []

    print("Starting Localization with anchor nodes:", nodes_anc)
    for ix in tqdm(range(N), desc="Localizing"):
        # 1) 初始化 LSE 工程
        P = lx.Project(mode='2D', solver='LSE')

        # 2) 添加锚点节点
        for nod in nodes_anc:
            # loc_nod[nod] 是 [x, y, 0]
            P.add_anchor(nod, loc_nod[nod])

        # 3) 添加两个 target，分别用于 ske 和 rdm
        t_ske, _ = P.add_target()
        t_rdm, _ = P.add_target()

        # 4) 对每个节点，添加测距数据
        for nod in nodes_anc:
            dist_ske = list_dist_ske_[nod][ix]         # ske 距离
            dist_rdm = list_dist_rdm_[nod][ix] + offset  # rdm 距离 + offset

            t_ske.add_measure(nod, dist_ske)
            t_rdm.add_measure(nod, dist_rdm)

        # 5) 求解
        P.solve()

        # 6) 提取目标位置 (X, Y, Z)
        loc_ske_ = np.asarray([t_ske.loc.x, t_ske.loc.y, t_ske.loc.z])
        loc_rdm_ = np.asarray([t_rdm.loc.x, t_rdm.loc.y, t_rdm.loc.z])

        # 仅取前 2 维 (x, y)
        loc_ske_pred.append(loc_ske_[:2])
        loc_rdm_pred.append(loc_rdm_[:2])

    loc_ske_pred = np.asarray(loc_ske_pred)  # shape (N, 2)
    loc_rdm_pred = np.asarray(loc_rdm_pred)  # shape (N, 2)

    print("Localization Completed. LSE results for ske and rdm generated.")
    return loc_ske_pred, loc_rdm_pred

def visualize_localization(loc_rdm_pred, timestamps, loc_nod, nodes_anc, first_act, last_act):
    """
    可视化定位结果，包括目标位置和轨迹。

    Parameters:
        loc_rdm_pred (numpy.ndarray): 预测的目标位置数组（X, Y），形状 (N, 2)。
        timestamps (numpy.ndarray): 同步的时间戳数组 (N,)。
        loc_nod (dict): 锚点节点的位置，形如 { '2':[x2,y2,0], '15':[x15,y15,0], '16':[x16,y16,0] }。
        nodes_anc (list): 锚点节点ID列表，如 ['2','15','16']。
        max_time (float): 动画中显示的最大时间（秒）。
    """
    # data_complex, dt = load_data(session_path, node_idx, pi_version=4)
    # segs, acts = seg_index(dt, seg_file)

    # loc_rdm_pred = loc_rdm_pred[5699:27391]
    # timestamps = timestamps[5699:27391]
    N = len(timestamps)
    # print(timestamps)

    x_data = loc_rdm_pred[:, 0]
    y_data = loc_rdm_pred[:, 1]

    total_animation_duration_ms = 30000  # 30秒
    interval = total_animation_duration_ms / N  # 每帧的显示间隔（毫秒）

    print(f"Total animation duration: {total_animation_duration_ms} ms")
    print(f"Animation interval between frames: {interval:.3f} ms")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(np.min(x_data) - 1, np.max(x_data) + 1)
    ax.set_ylim(np.min(y_data) - 1, np.max(y_data) + 1)
    ax.set_xlabel("X (meters)", fontsize=14)
    ax.set_ylabel("Y (meters)", fontsize=14)
    ax.set_title(f"Localization Results Over Selected Actions", fontsize=16)
    ax.grid(True)

    # 绘制锚点节点位置
    anchor_x = [loc_nod[nod][0] for nod in nodes_anc]
    anchor_y = [loc_nod[nod][1] for nod in nodes_anc]
    ax.scatter(anchor_x, anchor_y, marker='^', color='blue', label='Anchor Nodes', zorder=3, s=100)

    trajectory_scatter = ax.scatter([], [], marker='x', color='green', label='Trajectory Points', zorder=2, s=50)
    trajectory_line, = ax.plot([], [], color='green', linewidth=2, label='Trajectory Path', zorder=1)
    target_scatter, = ax.plot([], [], marker='x', color='red', label='Estimated Target Position', zorder=4, markersize=12)

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    trajectory_window = 5.0  # seconds
    trajectory_frames = int(trajectory_window / (interval / 1000))
    trajectory_deque = deque(maxlen=trajectory_frames)

    print(f"Trajectory window: {trajectory_window} seconds ({trajectory_frames} frames)")

    trajectory_coords = []

    def update(frame):
        target_scatter.set_data([x_data[frame]], [y_data[frame]])

        trajectory_deque.append((x_data[frame], y_data[frame]))
        trajectory_points = np.array(trajectory_deque)

        trajectory_line.set_data(trajectory_points[:, 0], trajectory_points[:, 1])
        trajectory_scatter.set_offsets(trajectory_points)

        trajectory_coords.append((x_data[frame], y_data[frame]))
        time_text.set_text(f"Time: {timestamps[frame]:.3f} s")

        return target_scatter, trajectory_scatter, trajectory_line, time_text

    ani = FuncAnimation(fig, update, frames=N, interval=interval, blit=True)

    ax.legend(loc='upper right', fontsize=12)

    # ============== 在这里先保存 MP4 ==============
    from matplotlib.animation import FFMpegWriter
    fps = int(1000.0 / interval) if interval > 0 else 20
    ffmpeg_writer = FFMpegWriter(fps=fps, codec="libx264")

    print("Saving animation to 'localization_animation.mp4' before displaying...")
    ani.save("localization_animation.mp4", writer=ffmpeg_writer)
    print("Animation saved as 'localization_animation.mp4'.")

    # ============== 在窗口关闭时再保存轨迹坐标(可选) ==============
    def on_close(event):
        print("\nTrajectory Coordinates:")
        with open("trajectory.txt", "w") as traj_file:
            traj_file.write("Trajectory Coordinates (X, Y):\n")
            for coord in trajectory_coords:
                coord_str = f"({coord[0]:.3f}, {coord[1]:.3f})"
                print(coord_str)
                traj_file.write(f"{coord[0]:.3f}, {coord[1]:.3f}\n")
        print("Trajectory coordinates saved to 'trajectory.txt'.")

    fig.canvas.mpl_connect('close_event', on_close)

    # 最后再显示动画
    plt.show()

def main():
    """
    修改后的 main 函数：对每一个 node_idx (3 个) 计算 distance，并在 LSE 中用 3 个 distance 推算目标坐标。
    """

    # ========== 1) 基础配置 ==========
    session_path = r"D:\OneDrive\桌面\code\ADL_localization\data\SB-94975U-2"
    seg_file = r"D:\OneDrive\桌面\code\ADL_localization\data\SB-94975U-2\segment\2024-12-03-22-36-00_SB-94975U-2_shifted.txt"

    # 要使用的 3 个节点
    nodes_anc = ['2', '15', '16']
    loc_nod = {
        '2':   [0.0,   0.0,  0],
        '15':  [8.5,   3.0,  0],
        '16':  [8.5,  -1.5,  0]
    }

    # 若需自动转换 node_id -> node_idx
    # 假设: '2' -> 2, '15' -> 15, '16' -> 16, or custom mapping
    # 仅示例: 强转 float(node_idx) 用于 load_data
    # 也可在 config 中写死映射
    # Example:
    node_map = {'2':2, '15':15, '16':16}

    # ========== 2) 计算距离, 并存储到 range_data = { node_id: [dist1, dist2, ...] } ==========
    data_complex, dt, start_time, end_time = load_align_resample(session_path, nodes_anc, target_fps=100, start_time=datetime(2024, 11, 22, 17, 43, 16, tzinfo=timezone.utc), end_time=datetime(2024, 11, 22, 17, 44, 4, tzinfo=timezone.utc))
    bin_size = 0.05  # 每个 bin 代表的米数 (示例)
    for node_id in nodes_anc:
        # (a) 加载数据
        node_idx = float(node_map[node_id])  # 根据实际映射

        # 准备拼接所有段的距离
        single_dist_list = []
        # compute_features_over_time => (distance_2d, indices)
        distance_2d, index_arr = compute_features_over_time(
            data_complex[node_idx],
            doppler_bin_num=32,
            step=5,
            range_bin_num=188,
            Discard_bins=[14, 15, 16, 17]
        )
        print(f'distance_2d is {distance_2d}\n')
        # distance_2d shape: (frame_num, range_bin_num)
        # 需对每行聚合成单值:
        bin_size = 0.05  # 每个 bin 对应的实际距离步长
        for row in distance_2d:
            # row: shape=(range_bin_num,) 表示该帧多bin的强度分布
            # idxs: [0, 1, 2, ..., range_bin_num-1]
            idxs = np.arange(len(row))

            # 计算加权索引 (类似 center of mass)
            # sum(row[k]*k) / sum(row[k]) => 获得索引位置(浮点)，再乘 bin_size 得距离
            denominator = np.sum(row)
            if denominator == 0:
                # 如果整行和为 0（极端情况），可做特殊处理
                distance_1d = 0.0
            else:
                weighted_idx = np.sum(row * idxs) / denominator
                distance_1d = weighted_idx * bin_size

            single_dist_list.append(distance_1d)

        # (d) 存入 range_data[node_id]
        range_data[node_id] = np.array(single_dist_list)

    # 检查 3 个节点距离数组长度
    lengths = [len(range_data[n]) for n in nodes_anc]
    print("Range data lengths:", {n: len(range_data[n]) for n in nodes_anc})
    min_len = min(lengths)

    # 截断或对齐
    for n in nodes_anc:
        range_data[n] = range_data[n][:min_len]

    N = min_len  # 总帧数

    # ========== 3) LSE 定位 ==========

    loc_rdm_pred = []
    offset = 0.35  # rdm 偏移
    print("Starting Localization with anchor nodes:", nodes_anc)
    for ix in tqdm(range(N), desc="Localizing"):
        # (a) 初始化 LSE 工程
        P = lx.Project(mode='2D', solver='LSE')

        # (b) 添加锚点节点
        for nod in nodes_anc:
            P.add_anchor(nod, loc_nod[nod])

        # (c) 添加目标
        t_rdm, _ = P.add_target()

        # (d) 添加测距数据
        for nod in nodes_anc:
            dist_val = range_data[nod][ix] + offset
            t_rdm.add_measure(nod, dist_val)

        # (e) 求解
        P.solve()

        # (f) 提取 (x,y)
        loc_rdm_ = np.array([t_rdm.loc.x, t_rdm.loc.y, t_rdm.loc.z])
        loc_rdm_pred.append(loc_rdm_[:2])

    loc_rdm_pred = np.array(loc_rdm_pred)  # shape (N, 2)

    print("Localization Completed. Results shape:", loc_rdm_pred.shape)

    # ========== 4) 构造时间戳并可视化 ==========
    sampling_interval = 1/100
    timestamps = np.array([i * sampling_interval for i in range(N)])

    # 假设只可视化 [6, 11) s
    first_act_vis = 6.0
    last_act_vis = 11.0

    visualize_localization(loc_rdm_pred, timestamps, loc_nod, nodes_anc, first_act_vis, last_act_vis)

    # 若需要保存结果
    np.save(os.path.join(session_path, "loc_rdm_pred.npy"), loc_rdm_pred)
    print("loc_rdm_pred saved to loc_rdm_pred.npy")

def create_localization_animation_single_sc(loc_rdm_pred, timestamps, loc_nod, nodes_anc,
                                           sc_idx=0,
                                           total_animation_duration_ms=30000,
                                           trajectory_window=5.0):
    """
    可视化定位结果（仅某个子载波）的动画，包括目标位置和轨迹。
    
    Parameters:
        loc_rdm_pred (numpy.ndarray): 预测的目标位置数组，形状 (N, 188, 2)。
        timestamps (numpy.ndarray): 时间戳数组，形状 (N, )。
        loc_nod (dict): 锚点节点位置，如 { '2':[x2,y2,0], '15':[x15,y15,0], '16':[x16,y16,0] }。
        nodes_anc (list): 锚点节点ID列表，如 ['2','15','16']。
        sc_idx (int): 选择的子载波索引。
        total_animation_duration_ms (float): 动画总时长（毫秒）。
        trajectory_window (float): 轨迹显示的时间窗口（秒）。
    """
    # 1) 只取某个子载波 sc_idx 的 (x,y) 轨迹
    N = loc_rdm_pred.shape[0]
    x_data = loc_rdm_pred[:, sc_idx, 0]
    y_data = loc_rdm_pred[:, sc_idx, 1]

    # 2) 根据 N 和 total_animation_duration_ms 计算每帧的间隔
    interval = total_animation_duration_ms / N  # ms
    fps = int(1000.0 / interval) if interval > 0 else 20

    # 3) 轨迹窗口对应的帧数
    trajectory_frames = int(trajectory_window / (interval / 1000))

    # 4) 准备绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(np.min(x_data) - 1, np.max(x_data) + 1)
    ax.set_ylim(np.min(y_data) - 1, np.max(y_data) + 1)
    ax.set_xlabel("X (m)", fontsize=14)
    ax.set_ylabel("Y (m)", fontsize=14)
    ax.set_title(f"Localization Results (Subcarrier {sc_idx})", fontsize=16)
    ax.grid(True)

    # 5) 绘制锚点
    anchor_x = [loc_nod[nod][0] for nod in nodes_anc]
    anchor_y = [loc_nod[nod][1] for nod in nodes_anc]
    ax.scatter(anchor_x, anchor_y, marker='^', color='blue',
               label='Anchor Nodes', zorder=3, s=100)

    # 6) 用于动态更新的元素
    trajectory_scatter = ax.scatter([], [], marker='.', color='green', label='Trajectory Points', zorder=2, s=30)
    trajectory_line, = ax.plot([], [], color='green', linewidth=2, label='Trajectory Path', zorder=1)
    target_scatter, = ax.plot([], [], marker='x', color='red', label='Estimated Target Position', zorder=4, markersize=12)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.5))

    ax.legend(loc='upper right', fontsize=12)

    # 7) 用 deque 维护滑动窗口内的轨迹点
    trajectory_deque = deque(maxlen=trajectory_frames)
    trajectory_coords = []  # 记录所有点，最后可选导出

    def update(frame):
        # 当前帧的目标位置
        tx, ty = x_data[frame], y_data[frame]
        target_scatter.set_data([tx], [ty])

        # 更新轨迹窗口
        trajectory_deque.append((tx, ty))
        trajectory_points = np.array(trajectory_deque)

        # 更新散点和线
        trajectory_line.set_data(trajectory_points[:, 0], trajectory_points[:, 1])
        trajectory_scatter.set_offsets(trajectory_points)

        # 时间戳文字
        t_str = f"Time: {timestamps[frame]:.3f} s"
        time_text.set_text(t_str)

        # 也可以把本帧轨迹坐标存到列表
        trajectory_coords.append((tx, ty))

        return target_scatter, trajectory_scatter, trajectory_line, time_text

    ani = FuncAnimation(fig, update, frames=N, interval=interval, blit=True)

    # 8) 保存 MP4
    from matplotlib.animation import FFMpegWriter
    ffmpeg_writer = FFMpegWriter(fps=fps, codec="libx264")
    output_mp4 = f"localization_sc{sc_idx}.mp4"
    ani.save(output_mp4, writer=ffmpeg_writer)
    print(f"Animation saved as '{output_mp4}'.")

    # 9) 在关闭窗口时可选保存轨迹
    def on_close(event):
        with open(f"trajectory_sc{sc_idx}.txt", "w", encoding="utf-8") as traj_file:
            traj_file.write("Trajectory Coordinates (X, Y):\n")
            for coord in trajectory_coords:
                traj_file.write(f"{coord[0]:.3f}, {coord[1]:.3f}\n")
        print(f"Trajectory coordinates saved to 'trajectory_sc{sc_idx}.txt'.")

    fig.canvas.mpl_connect('close_event', on_close)

    # 10) 显示动画
    plt.show()
    
# --------------------------
# 需要你根据实际情况导入或定义以下函数和对象：
# def load_align_resample(session_path, nodes_list, target_fps, start_time, end_time): ...
# def compute_features_over_time(data_complex_node, doppler_bin_num, step, range_bin_num, Discard_bins): ...
# lx.Project: 定位库对象，需支持 .add_anchor(), .add_target(), .solve(), 以及 target.loc 属性
# --------------------------

# 示例：计算距离数据，并返回字典 range_data
def compute_range_data(session_path, nodes_anc, start_time, end_time, target_fps=100):
    data_complex, dt, st, et = load_align_resample(session_path, list(map(float, nodes_anc)),
                                                     target_fps=target_fps,
                                                     start_time=start_time,
                                                     end_time=end_time)
    bin_size = 0.05  # 每个 bin 表示的米数
    # 生成距离向量：第 i 个 bin 对应 i*0.05 米，从 0.05 到 9.4 米
    distance = np.arange(1, 189) * 0.05

    range_data = {}
    for node_id in nodes_anc:
        node_idx = float(node_id)
        distance_2d, index_arr = compute_features_over_time(
            data_complex[node_idx][:, :],
            doppler_bin_num=32,
            step=5,
            range_bin_num=188,
            Discard_bins=[14, 15, 16, 17]
        )
        # 计算加权平均：每一帧计算 weighted_avg = dot(s, d) / sum(s)
        weighted_sum = np.dot(distance_2d, distance)           # shape: (T,)
        weights_sum = np.sum(distance_2d, axis=1)                # shape: (T,)
        weighted_avg = np.where(weights_sum != 0, weighted_sum / weights_sum, 0)
        range_data[node_id] = weighted_avg   # (T,) 数组

        print(f"Computed range_data for node {node_id} with shape {weighted_avg.shape}")
    return range_data

def save_range_data_txt(range_data, save_path):
    with open(save_path, "w", encoding="utf-8") as fd:
        fd.write("node_id, frame_idx, distance(m)\n")
        for node_id in range_data:
            distances = range_data[node_id]
            T = distances.shape[0]
            for t in range(T):
                fd.write(f"{node_id}, {t}, {distances[t]:.4f}\n")
    print(f"Distance results saved to: {save_path}")

# 使用 LSE 定位，返回预测坐标 loc_rdm_pred (T,2) 并保存到 txt 文件
def lse_localization(range_data, nodes_anc, loc_nod, offset=0.0, save_path_loc="loc_results.txt"):
    T = range_data[nodes_anc[0]].shape[0]
    f_loc = open(save_path_loc, "w", encoding="utf-8")
    f_loc.write("frame_idx, x(m), y(m)\n")
    
    loc_rdm_pred = []
    print("Starting LSE Localization using 3 sensors:", nodes_anc)
    for t in tqdm(range(T), desc="Localizing"):
        # 初始化 LSE 工程，假定 lx.Project 已定义且支持如下接口
        P = lx.Project(mode='2D', solver='LSE')
        for nod in nodes_anc:
            P.add_anchor(nod, loc_nod[nod])
        target, _ = P.add_target()
        for nod in nodes_anc:
            measure_val = range_data[nod][t] + offset
            target.add_measure(nod, measure_val)
        P.solve()
        loc_current = np.array([target.loc.x, target.loc.y, target.loc.z])
        result = loc_current[:2]
        loc_rdm_pred.append(result)
        f_loc.write(f"{t}, {result[0]:.4f}, {result[1]:.4f}\n")
    f_loc.close()
    loc_rdm_pred = np.array(loc_rdm_pred)
    print("Localization Completed. Results shape:", loc_rdm_pred.shape)
    print(f"Localization results saved to: {save_path_loc}")
    return loc_rdm_pred

# --------------------------
# 3D LSE 定位函数
def lse_localization_3d(range_data, nodes_anc, loc_nod, offset=0.0, save_path_loc="loc_results_3d.txt"):
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
    T = range_data[nodes_anc[0]].shape[0]
    f_loc = open(save_path_loc, "w", encoding="utf-8")
    f_loc.write("frame_idx, x(m), y(m), z(m)\n")
    
    loc_rdm_pred = []
    print("Starting 3D LSE Localization using nodes:", nodes_anc)
    for t in tqdm(range(T), desc="3D Localizing"):
        # 初始化 3D LSE 定位工程
        P = lx.Project(mode='3D', solver='LSE')
        # 添加锚点，传入 3D 坐标
        for nod in nodes_anc:
            P.add_anchor(nod, loc_nod[nod])
        # 添加目标：目标节点需要 3D 坐标
        target, _ = P.add_target()
        # 输入测距数据（测距数据仍然为标量）
        for nod in nodes_anc:
            measure_val = range_data[nod][t] + offset
            target.add_measure(nod, measure_val)
        # 求解 3D 定位问题
        P.solve()
        # 提取目标的 3D 坐标（假设 target.loc 返回有 x, y, z 属性）
        loc_current = np.array([target.loc.x, target.loc.y, target.loc.z])
        # 如果定位结果中 z 维数不合理，可直接设置 z = 1.7；例如：
        # loc_current[2] = 1.7
        loc_rdm_pred.append(loc_current)
        f_loc.write(f"{t}, {loc_current[0]:.4f}, {loc_current[1]:.4f}, {loc_current[2]:.4f}\n")
    f_loc.close()
    loc_rdm_pred = np.array(loc_rdm_pred)
    print("3D Localization Completed. Results shape:", loc_rdm_pred.shape)
    print(f"3D Localization results saved to: {save_path_loc}")
    return loc_rdm_pred

# 以下函数与之前的2D版本类似，此处不作修改（仅注意 loc_rdm_pred 改为 2D 数据），
# 3D 动画的可视化与 2D 动画类似，这里仍对 (x,y) 进行动画展示，
# 若需要 3D 动画，请采用 mpl_toolkits.mplot3d 进行绘图。

def save_animation_gif(loc_rdm_pred, gif_save_path, frame_start, frame_end):
    frames = []
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x_min, x_max = np.min(loc_rdm_pred[:,0]), np.max(loc_rdm_pred[:,0])
    y_min, y_max = np.min(loc_rdm_pred[:,1]), np.max(loc_rdm_pred[:,1])
    x_margin = (x_max - x_min) * 0.1 if x_max != x_min else 1
    y_margin = (y_max - y_min) * 0.1 if y_max != y_min else 1

    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_max + y_margin, y_min - y_margin)  # 大值在上，小值在下
    yticks = ax.get_yticks()
    new_labels = np.sort(yticks)
    ax.set_yticklabels(new_labels)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Localization Prediction (2D Projection)")
    
    for t in range(frame_start, frame_end + 1):
        ax.cla()
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_max + y_margin, y_min - y_margin)
        yticks = ax.get_yticks()
        new_labels = np.sort(yticks)
        ax.set_yticklabels(new_labels)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"Frame {t}")
        start_frame = t - 20 if t >= 20 else 0
        traj = loc_rdm_pred[start_frame:t+1, :2]  # 仅取 x,y 显示
        ax.plot(traj[:, 0], traj[:, 1], linestyle='-', color='blue', alpha=0.7)
        ax.scatter(traj[:, 0], traj[:, 1], marker='x', color='blue', s=50, alpha=0.7)
        ax.scatter(loc_rdm_pred[t, 0], loc_rdm_pred[t, 1], marker='o', color='red', s=100, zorder=10)
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
    
    plt.close(fig)
    imageio.mimsave(gif_save_path, frames, fps=10)
    print(f"Localization GIF animation saved to: {gif_save_path}")

def save_animation_mp4(loc_rdm_pred, mp4_save_path, frame_start, frame_end, ffmpeg_path):
    mpl.rcParams['animation.ffmpeg_path'] = ffmpeg_path
    x_min, x_max = np.min(loc_rdm_pred[:,0]), np.max(loc_rdm_pred[:,0])
    y_min, y_max = np.min(loc_rdm_pred[:,1]), np.max(loc_rdm_pred[:,1])
    x_margin = (x_max - x_min) * 0.1 if x_max != x_min else 1
    y_margin = (y_max - y_min) * 0.1 if y_max != y_min else 1
    
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    
    def init():
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_max + y_margin, y_min - y_margin)
        yticks = ax.get_yticks()
        new_labels = np.sort(yticks)
        ax.set_yticklabels(new_labels)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Localization Prediction (2D Projection)")
        return []
    
    def update(frame):
        t = frame
        ax.cla()
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_max + y_margin, y_min - y_margin)
        yticks = ax.get_yticks()
        new_labels = np.sort(yticks)
        ax.set_yticklabels(new_labels)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"Frame {t}")
        start_frame = t - 20 if t >= 20 else 0
        traj = loc_rdm_pred[start_frame:t+1, :2]
        ax.plot(traj[:, 0], traj[:, 1], linestyle='-', color='blue', alpha=0.7)
        ax.scatter(traj[:, 0], traj[:, 1], marker='x', color='blue', s=50, alpha=0.7)
        ax.scatter(loc_rdm_pred[t, 0], loc_rdm_pred[t, 1], marker='o', color='red', s=100, zorder=10)
        return []
    
    frames_range = range(frame_start, frame_end + 1)
    anim = animation.FuncAnimation(fig, update, frames=frames_range, init_func=init, blit=False)
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Your Name'), bitrate=1800)
    anim.save(mp4_save_path, writer=writer)
    plt.close(fig)
    print(f"Localization MP4 animation saved to: {mp4_save_path}")

def plot_node_distance(range_data, node_id):
    """
    绘制指定节点的距离数据折线图。
    参数：
      range_data: 字典，key 为节点编号，value 为形状 (T,) 的距离数据数组。
      node_id: 要绘制数据的节点编号（字符串）。
    """
    if node_id not in range_data:
        print(f"节点 {node_id} 的数据不存在！")
        return
    distances = range_data[node_id]
    frames = np.arange(len(distances))
    plt.figure(figsize=(10, 5))
    plt.plot(frames, distances, marker='o', linestyle='-', color='b', label=f'Distance of node {node_id}')
    plt.xlabel("Frame Index")
    plt.ylabel("Distance (m)")
    plt.title(f"Distance Data for Node {node_id}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --------------------------
# 主函数
def main():
    # 配置各项路径与参数
    session_path = r'D:\OneDrive\桌面\code\ADL_localization\data\SB-50274X'
    seg_file = r'D:\OneDrive\桌面\code\ADL_localization\data\SB-50274X\segment\2024-10-27-18-13-39_SB-50274X.txt'
    nodes_anc = ['2', '15', '16']
    loc_nod = {
        '2':   [0.630,   3.141,  1.439],
        '16':  [8.572,   3.100,  1.405],
        '15':  [7.745,   1.412,  0.901]
    }
    node_map = {'2':2, '15':15, '16':16, '13':13, '6':6, '14':14}

    # 设置采样时间
    start_time = datetime(2024, 10, 27, 22, 18, 7, tzinfo=timezone.utc)
    end_time = datetime(2024, 10, 27, 22, 18, 19, tzinfo=timezone.utc)
    
    # 1) 计算距离数据
    range_data = compute_range_data(session_path, nodes_anc, start_time, end_time, target_fps=120)
    T = range_data[nodes_anc[0]].shape[0]
    save_path_distance = "distance_results.txt"
    save_range_data_txt(range_data, save_path_distance)
    
    # 2) 3D LSE 定位，获得预测的 (x, y, z) 坐标
    loc_results_file = "loc_results_3d.txt"
    loc_rdm_pred = lse_localization_3d(range_data, nodes_anc, loc_nod, offset=0.0, save_path_loc=loc_results_file)
    
    print("3D Localization Completed. Results shape:", loc_rdm_pred.shape)
    print(f"3D Localization results saved to: {loc_results_file}")
    
    # # 3) 绘制指定节点的距离数据折线图，例如绘制节点 '16'（请确认该节点的距离数据在 range_data 中存在）
    # plot_node_distance(range_data, '16')
    
    # 4) 生成动画：保存 GIF 和 MP4（帧范围示例：150 到 200）
    gif_save_path = 'localization_animation_21516.gif'
    mp4_save_path = 'localization_animation_21516.mp4'
    ffmpeg_path = r'C:\ffmpeg\bin\ffmpeg.exe'  # 修改为你实际的 FFmpeg 路径
    
    # 注意：动画部分仅绘制 (x,y) 坐标的投影
    save_animation_gif(loc_rdm_pred, gif_save_path, frame_start=100, frame_end=200)
    save_animation_mp4(loc_rdm_pred, mp4_save_path, frame_start=100, frame_end=200, ffmpeg_path=ffmpeg_path)

if __name__ == "__main__":
    main()