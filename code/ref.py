"""
compute doppler_original (2-d) and doppler_gt (1-d) from baseband data. No alignment with video data.
save frames_common, doppler_original, doppler_gt.
"""
import os
# import subprocess
# import random
# import math
# import time
import numpy as np
# import shutil
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn import svm
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
import cv2

from helper import first_peak
import matplotlib
from scipy.signal import find_peaks

path = "test_data\dc-a6-32-f7-bd-39"
complex_file = "test_data\dc-a6-32-f7-bd-39_complex.npy"
timestamp_file = "test_data\dc-a6-32-f7-bd-39_timestamp.txt"

# if not os.path.exists(complex_file):
#     raise FileNotFoundError(f"{complex_file} does not exist.")
# if not os.path.exists(timestamp_file):
#     raise FileNotFoundError(f"{timestamp_file} does not exist.")

data_complex = np.load(complex_file)

# Read timestamps as strings
with open(timestamp_file, 'r') as f:
    ts_uwb = f.read().splitlines()

# Split into real and imaginary parts
data_real = np.real(data_complex)
data_imag = np.imag(data_complex)

# range_ = np.array([30, 39])
# np.save(os.path.join(path, "range.npy"), range_)
# exit(0)
doppler = []
doppler_bin_num = 32
DISCARD_BINS = [15, 16, 17]
# DISCARD_BINS = [16]
fps_uwb = 180
start_ind = 6000
stop_ind = 10000

# # load raw data
# imag_file = os.path.join(path, 'frame_buff_imag.txt')
# real_file = os.path.join(path, 'frame_buff_real.txt')
# ts_uwb_file = os.path.join(path, 'timestamp.txt')
# ts_uwb = np.loadtxt(ts_uwb_file)  # load timestamps

# data_imag = np.loadtxt(imag_file)      # load imaginary part
# data_real = np.loadtxt(real_file)      # load real part
number_of_frames = min(len(data_imag), len(data_real))
print("imaginary file: {} lines, real file: {} lines".format(len(data_imag), len(data_real)))
data_imag = data_imag[0:number_of_frames]
data_real = data_real[0:number_of_frames]

data_complex = data_real + 1j * data_imag  # compute complex number
data_complex = data_complex[start_ind:stop_ind, :]

# compute range profile
range_profile = np.abs(data_complex)

mean_dis = np.mean(range_profile, axis=0)
range_profile = range_profile - mean_dis
# hp = sns.heatmap(range_profile[:, :70])
# plt.axvline(x=left, color='r')
# plt.axvline(x=right, color='r')
# plt.ylabel("time (1/{} second)".format(fps_uwb))
# plt.xlabel("range bin")
# plt.title("range profile")
# plt.savefig(os.path.join(path, "range_profile.png"))
# plt.show()

std_dis = np.std(range_profile, axis=0)		# ignore first and last 10 seconds

# if os.path.exists(os.path.join(path, "range.npy")):
# 	# 有些数据有人在附近动，影响了first peak的选择。手动看range profile然后选了存下来
# 	range_ = np.load(os.path.join(path, "range.npy"))
# 	left = range_[0]
# 	right = range_[1]
# else:
# 	left, right = first_peak(std_dis)
# 	print("left: {}, right: {}.".format(left, right))
left = 1
right = 10

# plt.plot(np.arange(0, std_dis.shape[0], 1), std_dis)
# plt.axvline(x=left, color='r')
# plt.axvline(x=right, color='r')
# # plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
# plt.ylabel("standard deviation")
# plt.xlabel("range bin")
# plt.title("first peak is {}~{}".format(left, right))
# plt.savefig(os.path.join(path, "std_of_range_profile.png"))
# plt.show()

mean_dis = np.mean(range_profile, axis=0)
range_profile = range_profile - mean_dis
# hp = sns.heatmap(range_profile[:3000, :70])
# plt.axvline(x=left, color='r')
# plt.axvline(x=right, color='r')
# plt.title("range profile")
# plt.ylabel("time (1/{} second)".format(fps_uwb))
# plt.xlabel("range bin")
# plt.title("range profile")
# hp.figure.savefig(os.path.join(path, "range_profile_sample.png"))
# plt.show()

# exit(0)

doppler_from_UWB = []
doppler_1d = []
for i in range(doppler_bin_num, data_complex.shape[0], 2):
	d_fft = np.abs(np.fft.fft(data_complex[i - doppler_bin_num:i, :], axis=0))  # FFT
	d_fft = np.fft.fftshift(d_fft, axes=0)  # shift
	doppler_from_UWB.append(d_fft)

	fft_gt = np.copy(d_fft[:, left:right])
	# fft_gt[DISCARD_BINS, :] = np.zeros((len(DISCARD_BINS), right - left))
	# sum over range
	fft_gt = np.sum(fft_gt, axis=1)
	doppler_1d.append(fft_gt)


# doppler_1d = np.array(doppler_1d)
# np.save(os.path.join(path, 'doppler_1d_from_UWB.npy'), doppler_1d)
# sns.heatmap(np.array(doppler_1d))
# plt.show()

# np.save(os.path.join(path, 'doppler_1d_from_UWB.npy'), doppler_1d)
np.save(path+"_doppler_1d_from_UWB.npy", doppler_1d)
doppler_from_UWB = np.array(doppler_from_UWB)
# np.save(os.path.join(path, 'doppler_original_from_UWB.npy'), doppler_from_UWB)
np.save(path+"_doppler_original_from_UWB.npy", doppler_from_UWB)

plt.figure(figsize=(9, 6))
doppler_sample = np.array(doppler_1d)[:3000, :]
doppler_sample[:, DISCARD_BINS] = np.zeros((doppler_sample.shape[0], len(DISCARD_BINS)))
cmap = sns.color_palette("Blues", as_cmap=True)
doppler_sample = np.transpose(np.delete(doppler_sample, [15, 16, 17], axis=1))
# print(f"min: {np.min(doppler_sample)}, max: {np.max(doppler_sample)}")
doppler_sample = (doppler_sample - np.min(doppler_sample)) / (np.max(doppler_sample) - np.min(doppler_sample))
hp = sns.heatmap(doppler_sample, vmin=0, vmax=1, cmap=cmap)
plt.title("Real Doppler", fontsize=45)
plt.xticks([0, 900, 1800], ["0", "10", "20"], fontsize=45)
plt.yticks([0, 14, 28], ["1", "0", "-1"], fontsize=45)
plt.xlabel("time (s)", fontsize=45)
plt.ylabel("Doppler", fontsize=45)

cbar_kws = {'ticks': [0, 1]}  # Define the desired tick locations
cbar = plt.gca().collections[0].colorbar
cbar.set_ticks(cbar_kws['ticks'])
cbar.ax.tick_params(labelsize=45)  # Adjust the size as needed


# hp.figure.savefig(os.path.join(path, "doppler_real.png"))
hp.figure.savefig(path+"_doppler_real.png")
plt.tight_layout()
plt.savefig("dop_v4_r.pdf")
plt.show()
# np.save(os.path.join(path, 'doppler_1d_sample.npy'), np.array(doppler_1d)[:3000, :])
np.save(path+"doppler_1d_sample.npy", np.array(doppler_1d)[:3000, :])

# synthetic doppler
plt.figure(figsize=(9, 6))
# doppler = np.load(os.path.join(path, "output/rgb/synth_doppler.npy"))
# frames = np.load(os.path.join(path, "frames.npy"))
# frames_common = np.load(os.path.join(path, "frames_common.npy"))
# indices = np.isin(frames, frames_common)
# doppler = doppler[indices, :]
# doppler_sample = doppler[300:1000, :]
# # print(f"min: {np.min(doppler_sample)}, max: {np.max(doppler_sample)}")
# doppler_sample = (doppler_sample - np.min(doppler_sample)) / (np.max(doppler_sample) - np.min(doppler_sample))
# cmap = sns.color_palette("Blues", as_cmap=True)
# doppler_sample = np.transpose(np.delete(doppler_sample, [14, 15, 16, 17], axis=1))
# hp = sns.heatmap(doppler_sample, vmin=0, vmax=1, cmap=cmap)
# plt.title("Principle Synthetic Doppler", fontsize=45)
# plt.xticks([0, 300, 600], ["0", "10", "20"], fontsize=45)
# plt.yticks([0, 14, 27], ["1", "0", "-1"], fontsize=45)
# plt.xlabel("time (s)", fontsize=45)
# plt.ylabel("Doppler", fontsize=45)
# cbar_kws = {'ticks': [0, 1]}  # Define the desired tick locations
# cbar = plt.gca().collections[0].colorbar
# cbar.set_ticks(cbar_kws['ticks'])
# cbar.ax.tick_params(labelsize=45)  # Adjust the size as needed
# plt.tight_layout()
# # hp.figure.savefig(os.path.join(path, "doppler_syn.png"))
# plt.savefig("dop_v4_s.pdf")
# plt.show()