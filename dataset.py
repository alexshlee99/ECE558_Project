import os
import mat73
from torch.utils.data import Dataset
from utils import random_mask, uniform_mask
import numpy as np

class RetinaSimDataset(Dataset):
    def __init__(self, sim_dir, phase, transform=None):
        """
        Dataset for MoDL training. 
        Based on simulation of retina dataset. 
        """
        # Initialize. 
        self.phase = phase
        self.transform = transform
        self.samples = []
        
        # Get sensor data to corresponding phase (train/val/test). 
        sim_path = os.path.join(sim_dir, self.phase)

        # Directory of augmented tissues. 
        tissue_folders = sorted(os.listdir(sim_path), key=int)  # assume same for both directories 

        # Iterate through each tissue. 
        for tissue in tissue_folders:
            # Path to sim data of current tissue.
            tissue_sim_path = os.path.join(sim_path, tissue)

            # Iterate through each angle & store sensor data. 
            if os.path.isdir(tissue_sim_path): 
                
                # Path to corresponding pixel-interpolated map. 
                sensor_data_path = os.path.join(tissue_sim_path, "sensor_data.mat")

                # Ground truth. 
                gt_path = os.path.join(tissue_sim_path, "init_p0.mat")  # same for all views 

                # Store. 
                sample = (sensor_data_path, gt_path)
                self.samples.append(sample)
            else: 
                print('Issue with data loading...')
                break
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        # Unpack variables. 
        sensor_data_path, gt_path = self.samples[index]

        # Load sensor data. 
        dict = mat73.loadmat(sensor_data_path)
        sensor_data = dict['sensor_data']   # based on MatLab & kWave

        # Load random mask. 
        if self.phase == 'train': 
            # mask = uniform_mask(128, 128)
            mask = random_mask(128, 64)
        else: 
            mask = uniform_mask(128, 16)
        
        # Load ground truth.
        gt_dict = mat73.loadmat(gt_path)
        gt_img = gt_dict['gt'] 
        
        return sensor_data, mask, gt_img


class RetinaSimDataset_DAS(Dataset):
    def __init__(self, sim_dir, map_dir, phase, transform=None):
        """
        Dataset for MoDL training. 
        Based on simulation of retina dataset. 
        """
        # Initialize. 
        self.phase = phase
        self.transform = transform
        self.samples = []
        
        # Get sensor data to corresponding phase (train/val/test). 
        sim_path = os.path.join(sim_dir, self.phase)
        map_path = os.path.join(map_dir, self.phase)

        # Directory of augmented tissues. 
        tissue_folders = sorted(os.listdir(sim_path), key=int)  # assume same for both directories 

        # Iterate through each tissue. 
        for tissue in tissue_folders:
            # Path to sim data of current tissue.
            tissue_sim_path = os.path.join(sim_path, tissue)
            tissue_map_path = os.path.join(map_path, tissue)

            # Iterate through each angle & store sensor data. 
            if os.path.isdir(tissue_sim_path): 
                
                # Path to corresponding pixel-interpolated map. 
                pix_map_path = os.path.join(tissue_map_path, "interp_map.npy")                

                # Ground truth. 
                gt_path = os.path.join(tissue_sim_path, "init_p0.mat")  # same for all views 

                # Store. 
                sample = (pix_map_path, gt_path)
                self.samples.append(sample)
            else: 
                print('Issue with data loading...')
                break
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        # Unpack variables. 
        pix_map_path, gt_path = self.samples[index]

        # Load sensor data. 
        pix_map = np.load(pix_map_path)

        # Load random mask. 
        if self.phase == 'train': 
            # mask = uniform_mask(128, 128)
            mask = random_mask(128, 64)
        else: 
            mask = uniform_mask(128, 16)
        
        # Load ground truth.
        gt_dict = mat73.loadmat(gt_path)
        gt_img = gt_dict['gt'] 
        
        return pix_map, mask, gt_img