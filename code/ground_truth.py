import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import localization as lx  # 确保这个模块已实现并可用

# ---------------- 参数配置 ---------------
nodes_anc = ["7", "8", "9"]
data_session = "SB-94975U"

KINECT_X, KINECT_Y, KINECT_Z = 3.591, 5.036, 0.95  # Kinect 摄像头坐标
intensity_threshold = 0.0025                      # 测距有效阈值

# 滑动窗口size
FPS_PER_SECOND = 20

# 锚点坐标 (假设只需 7,8,9 三个节点)
loc_nod = {
    "7": [1.5,   5.127, 1.232],
    "8": [3.764, 5.058, 0.912],
    "9": [2.008, 6.6,   1.352]
}

# ---------- 1) 加载节点测距数据 ----------
range_data = {}
for nod in nodes_anc:
    file_path = f"data/dist/{data_session}/{data_session}_node{nod}_dist.npy"
    dist_array = np.load(file_path)
    range_data[nod] = dist_array
    print(f"Loaded distance for node {nod}, shape={dist_array.shape}")

# 找到最小长度，保证后续遍历安全
T = min(arr.shape[0] for arr in range_data.values())
print(f"Total frames = {T} (based on min length among sensors 7,8,9).")

# ---------- 2) 加载真实距离数据 (每秒一个距离) 并存为 real_distance_array ----------
# 文件格式：每行 "YYYY-MM-DD HH:MM:SS  real_dist"
manual_dist_file = "data\dist\SB-94975U\ske_manual_dist_interpolated.txt"
lines = []
with open(manual_dist_file, "r") as fmd:
    for ln in fmd:
        ln = ln.strip()
        if ln:
            lines.append(ln)

time_dist_pairs = []
for line in lines:
    # 假设如 "2024-11-22 17:47:34 1.6"
    # 以 rsplit 分割
    try:
        time_str, dist_str = line.rsplit(" ", 1)
        dist_val = float(dist_str)
        time_dist_pairs.append(dist_val) # 若需要时间可再解析 time_str
    except ValueError:
        print(f"Skipping invalid line: {line}")
        continue

real_distance_array = np.array(time_dist_pairs)  # shape = S, S秒
S = len(real_distance_array)
print(f"Loaded {S} lines of manual distance. (1 second => 1 line)")

# 若所有帧都有效，则理论帧数为 S * 5; 但有无效帧时少于此数

# ---------- 3) 遍历帧进行定位，只保留有效帧 ----------
loc_rdm_pred = []
valid_frames = []

for t in tqdm(range(T), desc="2D Localizing"):
    # # 如果任意节点的测距 < 0，说明该帧此节点无效
    # if any(range_data[nod][t] < 0 for nod in nodes_anc):
    #     continue
    
    invalid_sensor_count = sum(range_data[nod][t] < 0 for nod in nodes_anc)
    if invalid_sensor_count > 1:
        continue
    
    # 构建 LSE 定位工程
    P = lx.Project(mode='2D', solver='LSE')
    # 添加锚点
    for nod in nodes_anc:
        P.add_anchor(nod, loc_nod[nod])
    # 添加目标
    target, _ = P.add_target()
    # 加入测距
    for nod in nodes_anc:
        measure_val = range_data[nod][t]
        target.add_measure(nod, measure_val)
    # 求解
    P.solve()
    
    # 得到 (x, y, z)
    x_est, y_est, z_est = target.loc.x, target.loc.y, target.loc.z
    loc_rdm_pred.append((x_est, y_est, z_est))
    valid_frames.append(t)

loc_rdm_pred = np.array(loc_rdm_pred)
N = loc_rdm_pred.shape[0]
print(f"Number of valid frames: {N}")

# ---------- (D) 计算误差并输出到dist_comp.txt ----------
# 注：如果 valid_frames[i] // FPS_PER_SECOND >= S，则说明真实距离不足以覆盖到该帧所对应的秒数，可跳过。
estimated_dists = []
errors_abs = []
errors = []
rel_errors = []   # relative error = abs_diff / true_dist

output_file = "code/dist_comp_2dist.txt"
with open(output_file, "w") as fout:
    fout.write("est_dist(m)\ttrue_dist(m)\tsquared_error(m^2)\n")
    
    for i in range(N):
        frame_index = valid_frames[i]
        
        # Kinect距离
        x_est, y_est, z_est = loc_rdm_pred[i]
        dx = x_est - KINECT_X
        dy = y_est - KINECT_Y
        dz = z_est - KINECT_Z
        est_dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        estimated_dists.append(est_dist)
        
        # 映射到真实距离数组：second_index
        second_index = frame_index // FPS_PER_SECOND
        # if second_index >= S:
        #     # 如果超出真实数据范围，可选择跳过
        #     continue
        
        true_dist = real_distance_array[second_index]
        
        diff_abs = np.abs(true_dist - est_dist)
        diff = true_dist - est_dist
        errors_abs.append(diff_abs)
        errors.append(diff)
        
        if true_dist <= 0.0:
            # 若真实距离<=0, 处理方式可以是跳过 or 用eps
            relative_error = 1
        else:
            relative_error = diff / true_dist * 100
            
        rel_errors.append(relative_error)
        
        # 写入文件 dist_comp.txt, 每行: est_dist, true_dist, diff_sq
        fout.write(f"{est_dist:.3f}\t{true_dist:.3f}\t{diff_abs:.5f}\t{relative_error:.3f}%\n")

print(f"Distance comparison saved to {output_file}.")
errors_abs = np.array(errors_abs)
errors = np.array(errors)
rel_errors = np.array(rel_errors)

# ---------- 5) 可视化误差平方 ----------
plt.figure(figsize=(10, 4))
plt.plot(errors_abs, 'r-', label="|EstimatedDist - RealDist|")
plt.title("Absolute Error of Kinect Distance Over Valid Frames - 2 dist")
plt.xlabel("Index among computed valid frames")
plt.ylabel("Absolute Error (m)")
plt.legend()
plt.grid(True)
plt.show()

# (B) 绘制相对误差
plt.figure(figsize=(10, 4))
plt.plot(rel_errors, 'b-', label="Relative Error")
plt.title("Relative Error of Kinect Distance - 2 dist")
plt.xlabel("Index among valid frames")
plt.ylabel("Relative Error (dimensionless)")
plt.grid(True)
plt.legend()
plt.show()

# 打印统计信息
mse_err = np.mean(errors)
max_err = np.max(errors)
min_err = np.min(errors)
print(f"MSE = {mse_err:.5f}, Max Error = {max_err:.5f}, Min Error = {min_err:.5f}")

mse_abs = np.mean(errors_abs)
max_abs = np.max(errors_abs)
min_abs = np.min(errors_abs)
print(f"MSE = {mse_abs:.5f}, Max Error^2 = {max_abs:.5f}, Min Error^2 = {min_abs:.5f}")

mean_rel = np.mean(rel_errors)
max_rel = np.max(rel_errors)
min_rel = np.min(rel_errors)
print(f"Rel Error: mean={mean_rel:.4f}, max={max_rel:.4f}, min={min_rel:.4f}")
