import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import localization as lx  # Ensure this module is correctly implemented and accessible

def loc_nod_init():
    """
    Initialize the locations of anchor nodes.

    Returns:
        loc_nod (dict): Dictionary mapping node numbers to their (x, y, z) coordinates.
    """
    loc_nod = {}
    nodes = ['7', '8', '9']
    # x_delta = 1.38  # Adjusted value based on sensor placement
    # y_delta = 1.6   # Adjusted value based on sensor placement

    # loc_nod['1'] = [0, 0, 0]
    # loc_nod['2'] = [-1 * x_delta, 1 * y_delta, 0]
    # loc_nod['3'] = [-1 * x_delta, 2 * y_delta, 0]
    # loc_nod['4'] = [-1 * x_delta, 3 * y_delta, 0]
    # loc_nod['5'] = [0, 4 * y_delta, 0]
    # loc_nod['7'] = [1 * x_delta, 2 * y_delta, 0]
    # loc_nod['8'] = [1 * x_delta, 1 * y_delta, 0]
    loc_nod['8'] = [0, 0, 0]
    loc_nod['7'] = [3, 0.5, 0]
    loc_nod['9'] = [2, 2.4, 0]

    print("Initialized Anchor Node Locations:")
    for node_id, coords in loc_nod.items():
        print(f"  Node {node_id}: {coords}")

    return loc_nod

def load_all_doppler_data(data_dir, node_sensor_map):
    """
    Load Doppler-derived Range-Doppler data for all nodes based on the node-sensor mapping.

    Parameters:
        data_dir (str): Path to the directory containing Doppler data files.
        node_sensor_map (dict): Mapping from node numbers to sensor IDs.

    Returns:
        range_data (dict): Dictionary mapping node numbers to their range measurements.
    """
    range_data = {}

    for node_id, sensor_id in node_sensor_map.items():
        # Define paths to Doppler data files
        doppler_1d_path = os.path.join(data_dir, f'doppler_1d_from_UWB_{sensor_id}.npy')

        # Check if Doppler data file exists
        if not os.path.exists(doppler_1d_path):
            raise FileNotFoundError(f"Doppler 1D data file not found for Sensor ID {sensor_id}: {doppler_1d_path}")

        # Load Doppler data
        doppler_1d = np.load(doppler_1d_path)  # Shape: (num_frames, num_bins)

        # Process Doppler data to extract range measurements
        # Assume the maximum value in each frame corresponds to the target
        num_frames, num_bins = doppler_1d.shape
        range_bin_indices = np.argmax(doppler_1d, axis=1)
        bin_size = 0.05  # Example: each bin represents 0.05 meters
        ranges = range_bin_indices * bin_size  # Convert bin indices to range measurements

        # Store in dictionary
        range_data[node_id] = ranges  # Shape: (num_frames,)

        print(f"Loaded and processed Doppler data for Node {node_id} (Sensor ID {sensor_id}): {ranges.shape}")

    return range_data

def main():
    """
    Main function to perform localization and visualization using the localization module.
    """
    # Configuration
    data_dir = "D:\OneDrive\桌面\code\ADL_localization\data\SB-94975U"  # Modify this to your data path

    # Node-Sensor ID Mapping
    node_sensor_map = {
        # '1': 'dc-a6-32-f7-bd-39',
        # '2': 'e4-5f-01-8a-45-b5',
        # '3': 'e4-5f-01-8a-46-81',
        # '4': 'e4-5f-01-8a-78-6f',
        # '5': 'e4-5f-01-8a-df-32',
        # '7': 'e4-5f-01-8a-e1-d4',
        # '8': 'e4-5f-01-8a-e3-09'
        '7': 'e4-5f-01-88-59-21',
        '8': 'e4-5f-01-88-5b-ff',
        '9': 'e4-5f-01-8b-16-92'
    }

    # Initialize anchor node locations
    loc_nod = loc_nod_init()

    # Load Doppler data for all nodes using the node-sensor mapping
    range_data = load_all_doppler_data(data_dir, node_sensor_map)

    # Define anchor nodes
    nodes_anc = list(node_sensor_map.keys())

    # Number of frames
    N = len(next(iter(range_data.values())))

    # Initialize list to store localization results
    loc_rdm_pred = []

    # Perform localization using the 'localization' module
    for ix in tqdm(range(N), desc="Localizing"):
        P = lx.Project(mode='2D', solver='LSE')

        # Add anchor nodes to the project
        for nod in nodes_anc:
            P.add_anchor(nod, loc_nod[nod])

        # Add target
        t_rdm, _ = P.add_target()

        # Add range measurements to the target
        for nod in nodes_anc:
            dist_rdm = range_data[nod][ix] + 0.35  # Apply offset if necessary
            t_rdm.add_measure(nod, dist_rdm)

        # Solve the localization problem
        P.solve()

        # Extract target location
        loc_rdm_ = np.asarray([t_rdm.loc.x, t_rdm.loc.y, t_rdm.loc.z])

        # Append to the list of predicted locations
        loc_rdm_pred.append(loc_rdm_)

    # Convert to NumPy array and extract X and Y coordinates
    loc_rdm_pred = np.asarray(loc_rdm_pred)
    loc_rdm_pred = loc_rdm_pred[:, :2]  # Only X and Y coordinates

    # ==================== Visualization of Localization Results =====================
    plt.figure(figsize=(10, 8))
    plt.scatter(loc_rdm_pred[:, 0], loc_rdm_pred[:, 1], marker='o', color='red', label='Estimated Target Positions')
    plt.title("Localization Results")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
