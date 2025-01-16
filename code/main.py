import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from helper import first_peak
from scipy.signal import find_peaks

def load_sensor_data(path, sensor_id):
    """
    Load data for a specified sensor.

    Parameters:
        path (str): The directory where the data is stored.
        sensor_id (str): The ID of the sensor (file name prefix).

    Returns:
        data_complex (numpy.ndarray): The raw complex data.
        timestamps (list of str): The timestamps as strings.
    """
    complex_file = os.path.join(path, f"{sensor_id}_complex.npy")
    timestamp_file = os.path.join(path, f"{sensor_id}_timestamp.txt")
    
    if not os.path.exists(complex_file):
        raise FileNotFoundError(f"{complex_file} does not exist.")
    if not os.path.exists(timestamp_file):
        raise FileNotFoundError(f"{timestamp_file} does not exist.")
    
    data_complex = np.load(complex_file)
    
    # Read timestamps as strings
    with open(timestamp_file, 'r') as f:
        timestamps = f.read().splitlines()
    
    return data_complex, timestamps

def compute_range_profile(data_complex):
    """
    Compute the range profile.

    Parameters:
        data_complex (numpy.ndarray): The raw complex data.

    Returns:
        range_profile (numpy.ndarray): The range profile data.
    """
    range_profile = np.abs(data_complex)
    mean_dis = np.mean(range_profile, axis=0)
    range_profile -= mean_dis
    return range_profile

def visualize_range_profile(range_profile, path, sensor_id):
    """
    Visualize the range profile.

    Parameters:
        range_profile (numpy.ndarray): The range profile data.
        path (str): The directory where images will be saved.
        sensor_id (str): The ID of the sensor (used in image file name).
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(range_profile, cmap='viridis', cbar=True)
    plt.title(f"Range Profile - Sensor {sensor_id}")
    plt.xlabel("Range Bin")
    plt.ylabel("Time Frame")
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"range_profile_{sensor_id}.png"))
    plt.close()  # Close the figure to save resources

def visualize_std_dis(std_dis, path, sensor_id):
    """
    Save a plot of the standard deviation across range bins.

    Parameters:
        std_dis (numpy.ndarray): Standard deviation data.
        path (str): Directory to save the plot.
        sensor_id (str): Sensor ID for naming.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(std_dis)
    plt.title(f"Standard Deviation of Range Bins - Sensor {sensor_id}")
    plt.xlabel("Range Bin")
    plt.ylabel("Standard Deviation")
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"std_range_profile_plot_{sensor_id}.png"))
    plt.close()

def visualize_std(range_profile, path, sensor_id):
    """
    Visualize the standard deviation for each range bin.

    Parameters:
        range_profile (numpy.ndarray): The range profile data.
        path (str): The directory where images will be saved.
        sensor_id (str): The ID of the sensor (used in image file name).
    """
    std_dis = np.std(range_profile, axis=0)
    
    # Save the std_dis plot
    visualize_std_dis(std_dis, path, sensor_id)
    
    # Optionally, also plot and save the std_dis figure
    plt.figure(figsize=(12, 6))
    plt.plot(std_dis)
    plt.title(f"Standard Deviation of Range Bins - Sensor {sensor_id}")
    plt.xlabel("Range Bin")
    plt.ylabel("Standard Deviation")
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"std_range_profile_{sensor_id}.png"))
    plt.close()  # Close the figure to save resources

def select_top_n_bins(std_dis, n=2):
    """
    Select top N range bins with the highest standard deviation.

    Parameters:
        std_dis (numpy.ndarray): Standard deviation of range bins.
        n (int): Number of bins to select.

    Returns:
        selected_bins (list): Indices of selected range bins.
    """
    selected_bins = np.argsort(std_dis)[-n:]
    return sorted(selected_bins)

def determine_range_bins(range_profile, path, sensor_id, method='first_peak', top_n=2):
    """
    Determine the range bins for Doppler calculation using the first_peak function
    or load from a pre-saved range.npy file.

    Parameters:
        range_profile (numpy.ndarray): The range profile data.
        path (str): The directory where range.npy is saved or will be saved.
        sensor_id (str): The ID of the sensor.
        method (str): Method to determine range bins ('first_peak' or 'top_n').
        top_n (int): Number of top bins to select if using 'top_n'.

    Returns:
        left (int): Left boundary of range bins.
        right (int): Right boundary of range bins.
    """
    std_dis = np.std(range_profile, axis=0)  # Compute standard deviation across time frames

    # Save the std_dis plot for inspection
    visualize_std_dis(std_dis, path, sensor_id)

    range_file = os.path.join(path, "range.npy")
    if os.path.exists(range_file):
        # If range.npy exists, load it
        range_ = np.load(range_file)
        left, right = range_[0], range_[1]
        print(f"Manual selection - left: {left}, right: {right}")
    else:
        try:
            if method == 'first_peak':
                # Try to determine using first_peak function
                left, right = first_peak(std_dis, distance=5, prominence=0.05)
            elif method == 'top_n':
                # Select top N bins with highest std
                top_bins = select_top_n_bins(std_dis, n=top_n)
                left, right = top_bins[0], top_bins[-1]
            else:
                raise ValueError("Unknown method for determining range bins.")
            
            print(f"Auto selection - left: {left}, right: {right}")
            # Save the determined range for future use
            np.save(range_file, np.array([left, right]))
        except ValueError as ve:
            print(f"Auto selection failed: {ve}")
            if method == 'first_peak':
                # Fallback to top_n method
                top_bins = select_top_n_bins(std_dis, n=2)
                left, right = top_bins[0], top_bins[-1]
                print(f"Using top_n method - left: {left}, right: {right}")
                np.save(range_file, np.array([left, right]))
            else:
                # Set default or manual range bins
                left, right = 30, 50  # Example default values, adjust as needed
                print(f"Using default range bins - left: {left}, right: {right}")
                np.save(range_file, np.array([left, right]))
    
    return left, right

def compute_doppler(range_profile, doppler_bin_num, discard_bins, left, right):
    """
    Compute the Doppler data from the range profile.

    Parameters:
        range_profile (numpy.ndarray): The range profile data.
        doppler_bin_num (int): Number of bins for Doppler FFT.
        discard_bins (list): List of Doppler bins to discard.
        left (int): Left boundary of range bins.
        right (int): Right boundary of range bins.

    Returns:
        doppler_original (numpy.ndarray): 2D Doppler data.
        doppler_gt (numpy.ndarray): 1D Doppler data.
    """
    doppler_original = []
    doppler_gt = []
    for i in range(doppler_bin_num, range_profile.shape[0], 2):
        # Extract window for FFT
        window = range_profile[i - doppler_bin_num:i, :]
        # Perform FFT
        d_fft = np.abs(np.fft.fft(window, axis=0))
        d_fft = np.fft.fftshift(d_fft, axes=0)  # Shift zero frequency component to center
        doppler_original.append(d_fft)
        
        # Extract significant range bins and sum over them
        fft_gt = np.sum(d_fft[:, left:right], axis=1)
        doppler_gt.append(fft_gt)
    
    doppler_original = np.array(doppler_original)
    doppler_gt = np.array(doppler_gt)
    
    return doppler_original, doppler_gt

def visualize_doppler(doppler_gt, path, sensor_id, discard_bins, x_ticks, y_ticks, title, filename):
    """
    Visualize the Doppler data.

    Parameters:
        doppler_gt (numpy.ndarray): 1D Doppler data.
        path (str): The directory where images will be saved.
        sensor_id (str): The ID of the sensor (used in image file name).
        discard_bins (list): List of Doppler bins to discard.
        x_ticks (dict): Dictionary with 'positions' and 'labels' for x-axis ticks.
        y_ticks (dict): Dictionary with 'positions' and 'labels' for y-axis ticks.
        title (str): Title of the plot.
        filename (str): Filename for saving the plot.
    """
    plt.figure(figsize=(9, 6))
    doppler_sample = doppler_gt[:3000, :]  # Sample the first 3000 frames
    doppler_sample[:, discard_bins] = 0  # Discard specified Doppler bins
    doppler_sample = np.transpose(np.delete(doppler_sample, discard_bins, axis=1))  # Remove discarded bins and transpose
    
    # Normalize the Doppler data
    doppler_min = np.min(doppler_sample)
    doppler_max = np.max(doppler_sample)
    if doppler_max - doppler_min == 0:
        doppler_normalized = np.zeros_like(doppler_sample)
    else:
        doppler_normalized = (doppler_sample - doppler_min) / (doppler_max - doppler_min)
    
    cmap = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(doppler_normalized, vmin=0, vmax=1, cmap=cmap)
    plt.title(title, fontsize=45)
    plt.xticks(ticks=x_ticks['positions'], labels=x_ticks['labels'], fontsize=45)
    plt.yticks(ticks=y_ticks['positions'], labels=y_ticks['labels'], fontsize=45)
    plt.xlabel("Time (s)", fontsize=45)
    plt.ylabel("Doppler", fontsize=45)
    
    # Configure color bar
    cbar = plt.gca().collections[0].colorbar
    cbar.set_ticks([0, 1])
    cbar.ax.tick_params(labelsize=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(path, filename))
    plt.close()  # Close the figure to save resources

def main():
    # Configuration
    path = "data"  # Modify this to your data path
    doppler_bin_num = 32
    DISCARD_BINS = [15, 16, 17]  # Doppler bins to discard
    fps_uwb = 180
    start_ind = 6000
    stop_ind = 10000

    # Retrieve all sensor IDs (assuming file names are formatted as <sensor_id>_complex.npy and <sensor_id>_timestamp.txt)
    all_files = os.listdir(path)
    sensor_ids = set()
    for file in all_files:
        if file.endswith("_complex.npy"):
            sensor_id = file.replace("_complex.npy", "")
            sensor_ids.add(sensor_id)

    print(f"Found {len(sensor_ids)} sensors: {sensor_ids}")

    for sensor_id in sensor_ids:
        print(f"\nProcessing sensor: {sensor_id}")
        try:
            # Step 1: Load data
            data_complex, timestamps = load_sensor_data(path, sensor_id)
            print(f"Data shape: {data_complex.shape}")
            print(f"Number of timestamps: {len(timestamps)}")

            # Step 2: Compute range profile
            range_profile = compute_range_profile(data_complex[start_ind:stop_ind, :])

            # Step 3: Visualize range profile
            visualize_range_profile(range_profile, path, sensor_id)

            # Step 4: Visualize standard deviation
            visualize_std(range_profile, path, sensor_id)

            # Step 5: Determine range bins for Doppler calculation
            left, right = determine_range_bins(range_profile, path, sensor_id, method='first_peak', top_n=2)

            # Step 6: Compute Doppler data
            doppler_original, doppler_gt = compute_doppler(range_profile, doppler_bin_num, DISCARD_BINS, left, right)

            # Step 7: Save Doppler data
            np.save(os.path.join(path, f'doppler_1d_from_UWB_{sensor_id}.npy'), doppler_gt)
            np.save(os.path.join(path, f'doppler_original_from_UWB_{sensor_id}.npy'), doppler_original)

            # Step 8: Visualize Doppler data
            x_ticks = {'positions': [0, 900, 1800], 'labels': ["0", "10", "20"]}
            y_ticks = {'positions': [0, 14, 28], 'labels': ["1", "0", "-1"]}
            visualize_doppler(
                doppler_gt,
                path,
                sensor_id,
                DISCARD_BINS,
                x_ticks,
                y_ticks,
                title="Real Doppler",
                filename=f"doppler_real_{sensor_id}.png"
            )

            # Optional: Save Doppler samples
            np.save(os.path.join(path, f'doppler_1d_sample_{sensor_id}.npy'), doppler_gt[:3000, :])

        except Exception as e:
            print(f"Error processing sensor {sensor_id}: {e}")

if __name__ == "__main__":
    main()
