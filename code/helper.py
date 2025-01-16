import os.path

import numpy as np
# from scipy.signal import find_peaks
import datetime
from datetime import timedelta, timezone
# from scipy.signal import butter, lfilter
import imutils
from matplotlib import cm
import cv2
import matplotlib.pyplot as plt


def plot_loss(path, filename, training_loss, test_loss, test_accuracy, test_interval=10):
	plt.plot(training_loss, label='Training Loss')
	plt.plot(np.arange(0, len(test_loss), 1) * test_interval, test_loss, label='Val Loss')
	plt.plot(np.arange(0, len(test_accuracy), 1) * test_interval, test_accuracy, label='Val Accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Loss & Accuracy')
	
	plt.legend()
	plt.grid()
	plt.title("loss & accuracy")
	plt.savefig(os.path.join(path, filename))
	plt.cla()
	# plt.show()


def color_scale(img, norm, text=None):
    if len(img.shape) == 2:
        img = cm.magma(norm(img), bytes=True)
    img = imutils.resize(img, height=300)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    if text is not None:
        img = cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def apply_lowpass_filter(data, normal_cutoff, order=5):
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # b, a = butter_lowpass(cutoff_freq, fs, order=order)
    filtered_data = lfilter(b, a, data)
    return filtered_data


def first_peak(std_dis):
    height = max(np.percentile(std_dis, 50), np.max(std_dis) / 3)
    peaks, _ = find_peaks(std_dis, height=height)
    peaks_, _ = find_peaks(-std_dis)
    
    first_peak = peaks[0]
    second_peak = peaks[1]
    left = 0
    right = second_peak
    for pp in peaks_:
        if pp < first_peak:
            left = max(left, pp)
        else:
            break
    
    print("first peak: {}, left: {}, right: {}".format(first_peak, left, right))
    
    return left, right


def datetime_from_str(str, tzinfo=None):
    # timezone: default is system local time
    if tzinfo is None:
        try:
            r = datetime.datetime.strptime(str.strip('\n'), '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=timezone.utc)
        except Exception as e:
            try:
                r = datetime.datetime.strptime(str.strip('\n'), '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
            except Exception as e:
                try:
                    r = datetime.datetime.strptime(str.strip('\n'), '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=timezone.utc)
                except Exception as e:
                    try:
                        r = datetime.datetime.strptime(str.strip('\n'), '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
                    except Exception as e:
                        try:
                            r = datetime.datetime.strptime(str.strip('\n'), '%Y-%m-%dT%H:%M:%S.%fZZ').replace(tzinfo=timezone.utc)
                        except Exception as e:
                            print("exception: ", str)
                            return ""
    else:
        try:
            r = datetime.datetime.strptime(str.strip('\n'), '%Y-%m-%d %H:%M:%S.%f')
        except Exception as e:
            try:
                r = datetime.datetime.strptime(str.strip('\n'), '%Y-%m-%d %H:%M:%S')
            except Exception as e:
                try:
                    r = datetime.datetime.strptime(str.strip('\n'), '%Y-%m-%dT%H:%M:%S.%fZ')
                except Exception as e:
                    try:
                        r = datetime.datetime.strptime(str.strip('\n'), '%Y-%m-%dT%H:%M:%SZ')
                    except Exception as e:
                        try:
                            r = datetime.datetime.strptime(str.strip('\n'), '%Y-%m-%dT%H:%M:%S.%fZZ')
                        except Exception as e:
                            print("exception: ", str)
                            return ""
        r = r.astimezone(tzinfo)
    return r


def load_txt_to_datetime(path, filename):
    """
    This is to load timestamp of UWB data exported from InfluxDB. the timezone is UTC time.
    """
    # load txt file which has a str of datetime in each line. convert str to datetime, return the list of all lines.
    with open(os.path.join(path, filename), 'r') as f:
        lines = f.readlines()
        return [datetime_from_str(line,  tzinfo=None) for line in lines]     # indicate the timezone is UTC time


def load_segment_file_to_datetime(seg_file):
    """
    load App format segment file from android app, return activity list and datetime list [start, stop].
    example: Walk to kitchen - start: 2024-10-15 17:27:20.015,stop: 2024-10-15 17:27:33.322
    
    Args:
        seg_file: segment file, absolute/relative path

    Returns: datetime list [start, stop], UTC time; activity list

    """
    dt = []
    acts = []
    with open(seg_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            act = line.strip('\n').split(' - ')[0]
            acts.append(act)
            try:
                ss = line.strip('\n').split(' - ')[1].split(',')
                if len(ss) == 3:    # activity restarts, drop the first segment
                    dt.append([datetime_from_str(ss[1][9:], tzinfo=timezone.utc), datetime_from_str(ss[2][6:], tzinfo=timezone.utc)])
                if len(ss) == 2:
                    dt.append([datetime_from_str(ss[0][7:], tzinfo=timezone.utc), datetime_from_str(ss[1][6:], tzinfo=timezone.utc)])   # convert timezone to UTC time. default is the local time of the OS
            except Exception as e:
                print("not formative line in segment file: ", line)
        return dt, acts


def dt_delta(dt1, dt2):
    return (dt1 - dt2).seconds * 1e6 + (dt1 - dt2).microseconds


def seg_index(dt, seg_file, start_seg=0, stop_seg=None):
    """
    load segment file (format example: Walk to kitchen - start: 2024-10-15 17:27:20.015,stop: 2024-10-15 17:27:33.322) and 
    UWB frame timestamp file, find the frame indices of each segment in UWB data.
    """
    seg, acts = load_segment_file_to_datetime(seg_file)     # load segmentation from android app
    for s, act in zip(seg, acts):
        print(f"activity {act}, start: {s[0]}, stop: {s[1]}")

    i = 0   # current segment index
    start_ind = 0
    stop_ind = len(dt)
    indices = []
    acts_found = []
    for j in range(1, len(dt)):    # find the start and stop index for each segment
        t = dt[j].replace(tzinfo=timezone.utc)
        while i < len(seg):
            try:
                start, stop = seg[i][0].replace(tzinfo=timezone.utc), seg[i][1].replace(tzinfo=timezone.utc)
            except Exception as e:
                print("seg:", seg[i], "i:", i)
            if t <= start:
                start_ind = j
                break
            if stop >= t > start and start_ind == 0:
                start_ind = j
                break
            if stop >= t > start and start_ind != 0:
                stop_ind = j
                break
            if t > stop and stop_ind != len(dt):
                # print("activity {}, start: {}, stop: {}, index: {} ~ {}".format(acts[i], str(start), str(stop), start_ind,
                #                                                                 stop_ind))
                acts_found.append(acts[i])
                indices.append([start_ind, stop_ind])
                i += 1
                start_ind = 0
                stop_ind = len(dt)
                break
            if t > stop and stop_ind == len(dt):
                print(f"activity {i}, {acts[i]} is lost")
                i += 1
                start_ind = 0
                stop_ind = len(dt)
            
    return indices, acts_found


def find_index(start, stop, dt):
    """
    Args:
        start: start datetime of an activity
        stop: stop datetime of an activity
        dt: datetime lists of UWB data

    Returns: index

    """
    start_ind = None
    stop_ind = None
    for j in range(len(dt)):  # find the start and stop index for each segment
        t = dt[j]
        if t < start:
            continue
        if start <= t < stop and start_ind is None:
            start_ind = j
            continue
        if t >= stop and j == 0:
            print("error.")
            return None, None
        if t >= stop and stop_ind is None:
            stop_ind = j
            return start_ind, stop_ind
    return start_ind, j

    