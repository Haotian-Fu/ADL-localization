import numpy as np

sessions = ['6e5iYM_ADL_1']

for session in sessions:
    data_file = f"D:\OneDrive\桌面\code\ADL_localization\data\Preprocessed_dataset_doppler_fps_20/{session}_data.dat"
    label_file = f"D:\OneDrive\桌面\code\ADL_localization\data\Preprocessed_dataset_doppler_fps_20/{session}_label.dat"
    mask_file = f"D:\OneDrive\桌面\code\ADL_localization\data\Preprocessed_dataset_doppler_fps_20/{session}_mask_mannual.dat"
    data = np.memmap(data_file, dtype='float32', mode='r').reshape(-1, 16, 288)
    label = np.memmap(label_file, dtype='int64', mode='r')
    mask = np.memmap(mask_file, dtype='float32', mode='r').reshape(-1, 16)
    
    print(f'data is {data}\n')
    print(f'label is {label}\n')
    print(f'mask is {mask}\n')