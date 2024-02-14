import numpy as np
import torch
from typing import Dict

def brats_segmentation_regions() -> Dict[str, int]:
    regions = {
        "NCR": 1,   # Non-Enhancing Tumor Core
        "Edema": 2,  # Edema
        "Non_Tumor": 3,  # Non-Tumor region
        "ET": 4   # Enhancing Tumor
    }
    return regions

def process_seg_data(seg_data: np.ndarray):
    if not (seg_data is None):
        output = []
        regions = brats_segmentation_regions()
        # For each region, make a given mask
        for k in regions.keys():
            reg = regions[k]
            proc_seg = torch.tensor((seg_data == reg), dtype=torch.float32)
            output.append(proc_seg)

        return torch.cat(output, dim=0)