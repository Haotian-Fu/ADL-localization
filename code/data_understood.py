import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import numpy as np

# Ensure Matplotlib can find FFmpeg
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'  # Update this path if necessary

def visualize_uwb_heatmap(real_data, imag_data, timestamps, save_path=None):
    """
    可视化 UWB 实部和虚部数据的热图。
    
    参数:
        real_data (list of np.ndarray): 实部数据列表，每个元素为一帧数据的实部。
        imag_data (list of np.ndarray): 虚部数据列表，每个元素为一帧数据的虚部。
        timestamps (list of float): 每帧数据的时间戳（秒）。
        save_path (str, optional): 保存热图的路径。如果为 None，则显示图像。
    """
    # 转换为二维数组（距离 x 时间）
    real_matrix = np.array(real_data).T  # 形状: (num_distances, num_times)
    imag_matrix = np.array(imag_data).T

    # 计算幅度
    magnitude = np.sqrt(real_matrix**2 + imag_matrix**2)

    plt.figure(figsize=(15, 8))
    sns.heatmap(magnitude, cmap='viridis', cbar=True)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Dist. Bin', fontsize=14)
    plt.title('UWB Signal Magnitude Heat Map', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"UWB 幅度热图已保存到 {save_path}")
    else:
        plt.show()
    plt.close()

def visualize_uwb_animation(real_data, imag_data, timestamps, save_path=None):
    """
    创建 UWB 数据随时间变化的动画。
    
    参数:
        real_data (list of np.ndarray): 实部数据列表，每个元素为一帧数据的实部。
        imag_data (list of np.ndarray): 虚部数据列表，每个元素为一帧数据的虚部。
        timestamps (list of float): 每帧数据的时间戳（秒）。
        save_path (str, optional): 保存动画的路径（如 'uwb_animation.mp4'）。如果为 None，则显示动画。
    """
    magnitude = [np.sqrt(r**2 + i**2) for r, i in zip(real_data, imag_data)]
    magnitude = np.array(magnitude).T  # 形状: (num_distances, num_times)

    fig, ax = plt.subplots(figsize=(15, 8))
    cax = ax.imshow(magnitude, cmap='viridis', aspect='auto',
                    extent=[timestamps[8000], timestamps[10000], 0, magnitude.shape[0]],
                    origin='lower')
    fig.colorbar(cax, ax=ax, label='Magnitude')

    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Dist. Bin', fontsize=14)
    ax.set_title('UWB Signal Magnitude Animation', fontsize=16)

    def update(frame):
        current_magnitude = magnitude[:, :frame+1]
        ax.imshow(current_magnitude, cmap='viridis', aspect='auto',
                  extent=[timestamps[0], timestamps[frame], 0, magnitude.shape[0]],
                  origin='lower')
        return cax,

    ani = FuncAnimation(fig, update, frames=magnitude.shape[1], interval=50, blit=False)

    if save_path:
        ani.save(save_path, writer='ffmpeg', dpi=300)
        print(f"UWB 动画已保存到 {save_path}")
    else:
        plt.show()
    plt.close()

def main():
    complex_file = 'data/SB-94975U/e4-5f-01-51-a7-97_complex.npy'
    timestamp_file = 'data/SB-94975U/e4-5f-01-51-a7-97_timestamp.txt'
    
    # 加载复数数据
    complex_data = np.load(complex_file)
    print(f"加载复数数据形状: {complex_data.shape}")
    # 分离实部和虚部
    real_data = np.real(complex_data)
    imag_data = np.imag(complex_data)
    
    with open(timestamp_file, 'r') as f:
        timestamps = f.read().splitlines()

    # 可视化热图
    visualize_uwb_heatmap(real_data, imag_data, timestamps, save_path='uwb_magnitude_heatmap.png')

    # 可视化动画
    visualize_uwb_animation(real_data, imag_data, timestamps, save_path='uwb_magnitude_animation.mp4')

if __name__ == "__main__":
    main()
    