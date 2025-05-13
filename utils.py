import numpy as np
import torch
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
#  Mask generator for a 128-element array
# -----------------------------------------------------------------------------
def uniform_mask(nb_detectors, keep):
    """
    Returns a float mask of shape [num_detectors]
    with num_active equally-spaced ones.
    """
    mask = torch.zeros(nb_detectors)
   
    # Implemented this way due to k-Wave ordering (column-wise). 
    if keep == 64: 
        pick_idx = [127,123,120,116,112,108,104,100,96,92,88,84,80,76,72,68,64,60,56,52,48,44,40,36,32,28,24,20,16,12,8,5,2,6,9,13,17,21,25,29,33,37,41,45,49,53,57,61,65,69,73,77,81,85,89,93,97,101,105,109,113,117,121,124]
    elif keep == 32: 
        pick_idx = [127,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,2,9,17,25,33,41,49,57,65,73,81,89,97,105,113,121]
    else: 
        pick_idx = [127,112,96,80,64,48,32,16,2,17,33,49,65,81,97,113]

    pick_idx_arr = torch.as_tensor(pick_idx, dtype=torch.long) - 1
    mask[pick_idx_arr] = 1.0

    return mask[:, None]


def random_mask(nb_detectors, keep):          # keep = number of elements
    idx = torch.randperm(nb_detectors)[:keep]
    mask = torch.zeros(nb_detectors)          # 0 = missing
    mask[idx] = 1.0                           # 1 = acquired
    return mask[:, None]                # broadcast to (B,T,D)


def abs_normalizer(stack):
    """
    Convert stack of images into absolute, min-max normalized counterparts. 
    Implemented for Torch tensors. 
    """
    # Absolute.  
    abs_stack = torch.abs(stack)
    
    # Find min, max values. 
    if stack.dim() == 4: 
        min_vals = abs_stack.amin(dim=(1, 2, 3), keepdim=True)  # shape: [B, 1, 1, 1]
        max_vals = abs_stack.amax(dim=(1, 2, 3), keepdim=True)  # shape: [B, 1, 1, 1]
    else: 
        min_vals = abs_stack.amin(dim=(1, 2), keepdim=True)  # shape: [B, 1, 1]
        max_vals = abs_stack.amax(dim=(1, 2), keepdim=True)  # shape: [B, 1, 1]

    # Min-max normalization. 
    abs_norm_stack = (abs_stack - min_vals) / (max_vals - min_vals)
    return abs_norm_stack

def normalizer(stack):
    """
    Convert stack of images into absolute, min-max normalized counterparts. 
    Implemented for Torch tensors. 
    """    
    # Find min, max values. 
    if stack.dim() == 4: 
        min_vals = stack.amin(dim=(1, 2, 3), keepdim=True)  # shape: [B, 1, 1, 1]
        max_vals = stack.amax(dim=(1, 2, 3), keepdim=True)  # shape: [B, 1, 1, 1]
    else: 
        min_vals = stack.amin(dim=(1, 2), keepdim=True)  # shape: [B, 1, 1]
        max_vals = stack.amax(dim=(1, 2), keepdim=True)  # shape: [B, 1, 1]
    
    # Min-max normalization. 
    norm_stack = (stack - min_vals) / (max_vals - min_vals)
    return norm_stack