import numpy as np
import matplotlib.pyplot as plt
import mat73
import os


def pixelwise_interpolation(sinogram, x_grid, y_grid, array_pos, c=1500, fs=4*7.5e6):
    """
    Convert sinogram data into pixel-interpolated maps using time-based interpolation.
    
    Parameters:
      sinogram : numpy.ndarray
          Array of shape (num_detectors, num_samples) representing the signal data.
      x_grid : numpy.ndarray
          2D array of shape (H, W) containing the depth (axial) coordinate for each grid point.
      y_grid : numpy.ndarray
          2D array of shape (H, W) containing the lateral coordinate for each grid point.
      array_pos : numpy.ndarray
          Array of shape (2, num_detectors) where the first row contains the depth positions 
          and the second row contains the lateral positions of each detector element.
      c : float, optional
          Speed of sound (or propagation speed). Default is 1500.
      fs : float, optional
          Sampling frequency. Default is 4*7.813e6.
    
    Returns:
      interp_vals : numpy.ndarray
          A numpy array of shape (num_detectors, H, W) where each slice along the first dimension
          is the interpolated map for a given detector.
    """
    # Initialize. 
    num_detectors, num_samples = sinogram.shape
    H, W = x_grid.shape
    
    # Precompute the time vector for the sinogram samples.
    t_vector = np.linspace(0, (num_samples - 1) / fs, num_samples)
    
    # Initialize the output array.
    interp_vals = np.empty((num_detectors, H, W), dtype=sinogram.dtype)
    
    # For each detector, compute distances from its position to each grid point,
    # then compute the time-of-flight and interpolate from the sinogram row.
    for i in range(num_detectors):
        # Get detector position (depth, lateral).
        depth_det = array_pos[0, i]
        lateral_det = array_pos[1, i]
        
        # Compute the Euclidean distance from the detector to each grid point.
        # x_grid and y_grid have shape (H, W).
        distances = np.sqrt((x_grid - depth_det)**2 + (y_grid - lateral_det)**2)
        
        # Compute time-of-flight (in seconds) for each grid point.
        tof = distances / c
        
        # Interpolate. 
        interp_vals[i,:,:] = np.interp(tof.flatten(), t_vector, sinogram[i, :], left=0.0, right=0.0).reshape(H, W)
        
        # # Visualize. 
        # plt.imshow(tof, cmap='jet', aspect='auto')
        # plt.colorbar()
        # plt.show()
        # print('hello_world')
    
    return interp_vals


def extract_maps(root_path, save_path, phase): 
    """
    Compute the pixel-interpolated maps of simulated data prior to training (all angles). 
    """
    # Get path. 
    data_path = os.path.join(root_path, phase)

    # Directory of augmented tissues. 
    tissue_folders = sorted(os.listdir(data_path), key=int)

    # Get grid for pixelwise interpolation. 
    xgrid_path = R"grid\imgrid_x.mat"
    ygrid_path = R"grid\imgrid_y.mat"
    xgrid_dict = mat73.loadmat(xgrid_path)
    ygrid_dict = mat73.loadmat(ygrid_path)
    xgrid = xgrid_dict['imgrid_x']  # depth
    ygrid = ygrid_dict['imgrid_y']  # lateral 
    
    # Get array positions for pixelwise interpolation. 
    arraypos_path = R"grid\arraypos.mat"
    arraypos_dict = mat73.loadmat(arraypos_path)
    arraypos = arraypos_dict['element_pos']  # 0 at center 

    # Iterate through data & extract interpolated maps. 
    for tissue in tissue_folders:
        # Path to simulation data. 
        tissue_path = os.path.join(data_path, tissue)

        # Create path to save current tissue. 
        maps_path = os.path.join(save_path, phase, tissue)
        if not os.path.exists(maps_path):
            os.makedirs(maps_path)

        # Path to current sensor data.  
        sensor_data_path = os.path.join(tissue_path, f"sensor_data.mat")

        # Load sensor data. 
        dict = mat73.loadmat(sensor_data_path)
        sensor_data = dict['sensor_data']  # based on MatLab & kWave

        # Compute pixel-interpolated maps. 
        # import time
        # start = time.time()
        interp_maps = pixelwise_interpolation(sensor_data, xgrid, ygrid, arraypos)
        # end = time.time()

        # Save. 
        map_path = os.path.join(maps_path, f"interp_map.npy")
        np.save(map_path, interp_maps)


if __name__ == '__main__':
    
    root_path = R"data/"
    save_path = R"pix_map/"
    phase = "test"
    extract_maps(root_path, save_path, phase)