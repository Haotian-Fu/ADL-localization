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

def load_data(session_path, node_idx, pi_version=3):
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

# --------------------------
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

# --------------------------
# 在动画中增加房间判断及背景颜色设置，同时记录每帧的房间结果
def save_animation_gif_with_room(loc_rdm_pred, gif_save_path, frame_start, frame_end, rooms, true_room="living", result_txt="room_results.txt"):
    frames = []
    # 用于存储每一帧的房间判断结果
    predicted_rooms = []
    
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x_min, x_max = np.min(loc_rdm_pred[:,0]), np.max(loc_rdm_pred[:,0])
    y_min, y_max = np.min(loc_rdm_pred[:,1]), np.max(loc_rdm_pred[:,1])
    x_margin = (x_max - x_min) * 0.1 if x_max != x_min else 1
    y_margin = (y_max - y_min) * 0.1 if y_max != y_min else 1

    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    ax.set_ylim(y_max + y_margin, y_min - y_margin)  # 注意：大值在上，小值在下
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
        
        # 获取当前预测坐标
        current_point = loc_rdm_pred[t, :2]
        # 判断房间
        pred_room = get_room_by_rect(current_point[0], current_point[1], rooms)
        predicted_rooms.append(pred_room)
        # 判断是否正确（假设正确房间为 true_room，默认为 "bedroom"）
        if pred_room == true_room:
            # 正确，背景设为浅绿色
            ax.set_facecolor("#ccffcc")
        else:
            # 错误，背景设为浅红色
            ax.set_facecolor("#ffcccc")
        
        ax.set_title(f"Frame {t}\nPredicted Room: {pred_room} (True: {true_room})")
        start_frame = t - 20 if t >= 20 else 0
        traj = loc_rdm_pred[start_frame:t+1, :2]  # 仅取 x,y
        ax.plot(traj[:, 0], traj[:, 1], linestyle='-', color='blue', alpha=0.7)
        ax.scatter(traj[:, 0], traj[:, 1], marker='x', color='blue', s=50, alpha=0.7)
        ax.scatter(current_point[0], current_point[1], marker='o', color='red', s=100, zorder=10)
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
    
    # 保存房间判断结果到 txt 文件
    with open(result_txt, "w", encoding="utf-8") as f:
        f.write("frame_idx, predicted_room\n")
        for idx, room in enumerate(predicted_rooms, start=frame_start):
            f.write(f"{idx}, {room}\n")
    print(f"Room judgment results saved to: {result_txt}")
    
    plt.close(fig)
    imageio.mimsave(gif_save_path, frames, fps=10)
    print(f"Localization GIF animation with room info saved to: {gif_save_path}")

def save_animation_mp4_with_room(loc_rdm_pred, mp4_save_path, frame_start, frame_end, ffmpeg_path, rooms, true_room="living", result_txt="room_results.txt"):
    mpl.rcParams['animation.ffmpeg_path'] = ffmpeg_path

    # 动态计算房屋尺寸范围
    x_min = min(room[0] for room in rooms.values())
    x_max = max(room[1] for room in rooms.values())
    y_min = min(room[2] for room in rooms.values())
    y_max = max(room[3] for room in rooms.values())

    # 给可视化增加边界，避免轨迹接近边缘
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05

    predicted_rooms = []
    
    fig, ax = plt.subplots(figsize=(10, 8))

    def draw_room_layout(ax):
        """ 手动绘制房间结构 """
        for room_name, (xmin, xmax, ymin, ymax) in rooms.items():
            ax.add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                           fill=False, edgecolor='black', linewidth=2))
            ax.text((xmin + xmax) / 2, (ymin + ymax) / 2, room_name, 
                    fontsize=12, ha='center', va='center')

    def init():
        ax.clear()
        draw_room_layout(ax)
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_max + y_margin, y_min - y_margin)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Localization Prediction (2D Projection)")
        return []

    def update(frame):
        ax.clear()
        draw_room_layout(ax)
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_max + y_margin, y_min - y_margin)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        # 获取当前预测坐标
        current_point = loc_rdm_pred[frame, :2]
        pred_room = get_room_by_rect(current_point[0], current_point[1], rooms)
        predicted_rooms.append(pred_room)

        # 设置背景色
        ax.set_facecolor("#ccffcc" if pred_room == true_room else "#ffcccc")

        ax.set_title(f"Frame {frame}\nPredicted Room: {pred_room} (True: {true_room})")

        # 轨迹绘制
        start_frame = max(0, frame - 20)
        traj = loc_rdm_pred[start_frame:frame+1, :2]
        ax.plot(traj[:, 0], traj[:, 1], linestyle='-', color='blue', alpha=0.7)
        ax.scatter(traj[:, 0], traj[:, 1], marker='x', color='blue', s=50, alpha=0.7)
        ax.scatter(current_point[0], current_point[1], marker='o', color='red', s=100, zorder=10)

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
    plt.figure(figsize=(20, 5))
    plt.plot(frames, distances, marker='o', linestyle='-', color='b', label=f'Distance of node {node_id}')
    plt.xlabel("Frame Index")
    plt.ylabel("Distance (m)")
    plt.title(f"Distance Data for Node {node_id}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def ransac_filter(distances, num_iterations=100, threshold=0.5):
    """
    基于RANSAC的简单滤波函数，对一维距离数据进行鲁棒估计，剔除多径等异常值。
    参数：
      distances: 1D NumPy 数组，每个元素为单帧的测距值
      num_iterations: 迭代次数（默认 100）
      threshold: 误差阈值（单位与距离数据一致），小于该值视为inlier（默认 0.5）
    返回：
      filtered: 筛选后的一维 NumPy 数组，其中异常值以估计模型（例如inlier的中值）替换
    """
    best_inlier_count = 0
    best_model = None
    N = len(distances)
    for i in range(num_iterations):
        # 随机选取一个数据点作为候选模型
        idx = random.randint(0, N - 1)
        candidate = distances[idx]
        # 计算所有数据与候选模型之间的误差
        errors = np.abs(distances - candidate)
        inliers = errors < threshold
        count = np.sum(inliers)
        # 如果当前候选模型有更多 inliers，则记录下中值作为模型
        if count > best_inlier_count:
            best_inlier_count = count
            best_model = np.median(distances[inliers])
    # 对于每个数据点，如果与 best_model 的误差大于 threshold，则用 best_model 替代
    filtered = np.where(np.abs(distances - best_model) < threshold, distances, best_model)
    return filtered

def preprocess_range_data(range_data, threshold=0.5, num_iterations=100):
    """
    对字典中每个节点的距离数据使用 RANSAC 进行预处理。
    参数：
      range_data: dict, key 为节点编号，value 为一维 NumPy 数组（距离数据）
      threshold, num_iterations: 传递给 ransac_filter 的参数
    返回：
      new_range_data: 字典，与 range_data 键一致，但每个值为滤波后的数据
    """
    new_range_data = {}
    for node_id, distances in range_data.items():
        filtered = ransac_filter(distances, num_iterations=num_iterations, threshold=threshold)
        new_range_data[node_id] = filtered
        print(f"Node {node_id}: Filtered {len(distances)} measurements (threshold={threshold}).")
    return new_range_data

# --------------------------
# 主函数
def main():
    session_path = r'D:\OneDrive\桌面\code\ADL_localization\data\6e5iYM_ADL_1'
    seg_file = r'D:\OneDrive\桌面\code\ADL_localization\data\6e5iYM_ADL_1\segment\2023-06-29-16-54-23_6e5iYM_ADL_1_shifted.txt'
    nodes_anc = ['2', '15', '16']
    loc_nod = {
        '2':   [0.630,   3.141,  1.439],
        '16':  [8.572,   3.100,  1.405],
        '15':  [7.745,   1.412,  0.901],
        '7':   [1.5, 5.127, 1.232],
        '8':   [3.764, 5.058, 0.912],
        '9':   [2.008, 6.6, 1.352],
        '10':  [4.5, 5.153, 1.444],
        '11':  [7.733, 7.5, 1.643],
        '12':  [7.956, 5.028, 0.881]
    }
    node_map = {'2':2, '15':15, '16':16, '13':13, '6':6, '14':14, '7':7, '8':8, '9':9, '10':10, '11':11, '12':12}

    start_time = datetime(2023, 6, 29, 20, 51, 42, tzinfo=timezone.utc)
    end_time = datetime(2023, 6, 29, 20, 52, 19, tzinfo=timezone.utc)
    
    # 1) 计算距离数据
    range_data = compute_range_data(session_path, nodes_anc, start_time, end_time, target_fps=120)
    # 重要：在定位前对距离数据进行预处理，剔除多径异常数据
    # range_data = preprocess_range_data(range_data, threshold=0.5, num_iterations=100)
    T = range_data[nodes_anc[0]].shape[0]
    save_path_distance = "distance_results.txt"
    save_range_data_txt(range_data, save_path_distance)
    
    # 2) 3D LSE 定位，获得预测的 (x, y, z) 坐标
    loc_results_file = "loc_results_3d.txt"
    loc_rdm_pred = lse_localization_3d(range_data, nodes_anc, loc_nod, offset=0.0, save_path_loc=loc_results_file)
    
    print("3D Localization Completed. Results shape:", loc_rdm_pred.shape)
    print(f"3D Localization results saved to: {loc_results_file}")
    
    # # 3) 绘制指定节点的距离数据折线图，例如绘制节点 '16'
    # plot_node_distance(range_data, '2')
    # plot_node_distance(range_data, '15')
    # plot_node_distance(range_data, '16')
    
    # 4) 对预测的 (x,y) 坐标进行房间判断，并保存房间判断结果到一个 txt 文件。
    # 定义房间边界（这里以矩形为例），单位与预测坐标相同
    # 例如：假设 true_room 为 bedroom，且房间边界设置如下：
    # (x_min, x_max, y_min, y_max)
    # 房屋房间定义 (x_min, x_max, y_min, y_max)
    rooms = {
        "Kitchen": (0, 1.5, 4.563, 6),
        "Living Room": (0, 8.5, 0, 4.563),
        "Bathroom": (1.5, 4.5, 4.563, 6.6),
        "Bedroom": (4.5, 8.5, 4.563, 7.5),
        "Exit": (0, 0.5, 0, 0.5),
        "Couch": (7, 7.5, 1, 3)
    }
    # 4.1 对每一帧判断房间，并存入列表，同时统计判断正确的帧数
    predicted_rooms = []
    correct_count = 0
    true_room = "Living Room"  # 真值房间
    for t in range(T):
        x, y = loc_rdm_pred[t, :2]
        room = get_room_by_rect(x, y, rooms)
        predicted_rooms.append(room)
        if room == true_room:
            correct_count += 1

    # 计算准确率
    accuracy = correct_count / T

    # 将房间判断结果保存到 txt 文件
    room_result_file = "room_results.txt"
    with open(room_result_file, "w", encoding="utf-8") as f:
        f.write("frame_idx, predicted_room\n")
        for idx, room in enumerate(predicted_rooms):
            f.write(f"{idx}, {room}\n")
        # 在最后一行追加准确率信息
        f.write(f"Accuracy: {accuracy*100:.2f}% ({correct_count} correct frames out of {T})\n")
    print(f"Room judgment results saved to: {room_result_file}")
    
    # 5) 生成动画：在每一帧上显示房间判断，并根据判断结果设置背景颜色（正确：浅绿色，错误：浅红色）
    # 这里默认真值为 "bedroom"
    gif_save_path = 'localization_animation_room.gif'
    mp4_save_path = 'localization_animation_room.mp4'
    ffmpeg_path = r'C:\ffmpeg\bin\ffmpeg.exe'  # 修改为实际路径
    
    # 注意：动画部分仍绘制 (x,y) 投影，同时在标题中显示房间判断结果，并设置背景颜色
    save_animation_gif_with_room(loc_rdm_pred, gif_save_path, frame_start=0, frame_end=281, rooms=rooms, true_room="living", result_txt="room_results_anim.txt")
    save_animation_mp4_with_room(
        loc_rdm_pred, 
        mp4_save_path="localization_with_rooms.mp4", 
        frame_start=0, 
        frame_end=281, 
        ffmpeg_path=r"C:\ffmpeg\bin\ffmpeg.exe",
        rooms=rooms, 
        true_room="living",
        result_txt="room_results_anim.txt"
    )

    print(f"Accuracy: {accuracy*100:.2f}% ({correct_count} correct frames out of {T})")
    
if __name__ == "__main__":
    main()