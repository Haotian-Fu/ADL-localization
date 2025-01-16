import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def load_sensor_data(path, sensor_id):
    """
    Load data for a specified sensor.

    Parameters:
        path (str): The directory where the data is stored.
        sensor_id (str): The ID of the sensor (file name prefix).

    Returns:
        data_complex (numpy.ndarray): The raw complex data.
        timestamps (list of str): The timestamps.
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

def visualize_std(range_profile, path, sensor_id):
    """
    Visualize the standard deviation for each range bin.

    Parameters:
        range_profile (numpy.ndarray): The range profile data.
        path (str): The directory where images will be saved.
        sensor_id (str): The ID of the sensor (used in image file name).
    """
    std_dis = np.std(range_profile, axis=0)
    
    plt.figure(figsize=(12, 6))
    plt.plot(std_dis)
    plt.title(f"Standard Deviation of Range Bins - Sensor {sensor_id}")
    plt.xlabel("Range Bin")
    plt.ylabel("Standard Deviation")
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"std_range_profile_{sensor_id}.png"))
    plt.close()  # Close the figure to save resources

def main():
    # Configuration
    path = "../data"  # Modify this to your data path
    
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
            range_profile = compute_range_profile(data_complex)
            
            # Step 3: Visualize range profile
            visualize_range_profile(range_profile, path, sensor_id)
            
            # Step 4: Visualize standard deviation
            visualize_std(range_profile, path, sensor_id)
        
        except Exception as e:
            print(f"Error processing sensor {sensor_id}: {e}")

if __name__ == "__main__":
    main()
