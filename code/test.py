import numpy as np

# ------------------------------
# 加载我们自己保存的距离数据（npy 文件路径根据实际情况修改）
distance_node7 = np.load("data/dist/SB-94975U/SB-94975U_node7_dist.npy")
distance_node8 = np.load("data/dist/SB-94975U/SB-94975U_node8_dist.npy")
distance_node9 = np.load("data/dist/SB-94975U/SB-94975U_node9_dist.npy")

print(np.min(distance_node9[6000:12000]))
print(np.max(distance_node9[6000:12000]))