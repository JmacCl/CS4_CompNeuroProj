from typing import List
from scipy.stats import zscore

import numpy as np
import torch

# get stuff to work, then experiment with more pre-processing
# time investment should be directed towards simplicty

def brain_data_preprocessing(data: np.ndarray, clip: List[int] = None):
    """
    This function will do a number of things for BraTS 2020 data in order to
    process it for deep learning algorithms.
    1. Normalize data by getting z-scores of data
    2. Clip the images to a certain range to remove outlier, default is [-5, 5]
    3. Rescale to data with range of [0, 1], setting non brain regions to 0
    :param clip:
    :param data: a ndarray of brain data
    :return: a processed nd array of brain data
    """
    if clip is None:
        clip = [-5, 5]
    pass

    # First lets calculate z-score scores of axis
    norm_data = zscore(data, axis=None, nan_policy="omit")

    pnorm_data = np.nan_to_num(norm_data, 0)

    # Second resize values between clip
    clipped_data = np.clip(pnorm_data, clip[0], clip[1])

    min_clip = np.min(clipped_data)
    max_clip = np.max(clipped_data)
    # Third and finally, rescale image to 0, 1 and set all non brain voxels to 0
    denominator = max_clip - min_clip
    if denominator == 0:
        return None
    else:
        final_data = (clipped_data - np.min(clipped_data)) / denominator
        final_data[(data < -5) | (data > 5)] = 0
    return torch.tensor(final_data, dtype=torch.float32)



def brain_data_augmentation():
    pass