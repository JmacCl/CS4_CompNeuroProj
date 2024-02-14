from typing import List

import numpy as np
import os
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import zscore

from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanIoU

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


# Assuming you have a trained UNet model
TRAIN_PATH = "/src/data/raw/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData"
model_path = '/src/experiments/saved_model/TestUnet.keras'
model = load_model(model_path)

# Assuming you have validation data (val_data) and corresponding ground truth labels (val_labels)
# Replace this with your actual validation data loading logic

# Preprocess the validation data if needed
train_string = "BraTS20_Validation_" + "001"

# Identify stings
t1_string = os.path.join(TRAIN_PATH, train_string, train_string + "_t1.nii")
t1ce_string = os.path.join(TRAIN_PATH, train_string, train_string + "_t1ce.nii")
t2_string = os.path.join(TRAIN_PATH, train_string, train_string + "_t2.nii")
flair_string = os.path.join(TRAIN_PATH, train_string, train_string + "_flair.nii")

# Get data
data_input_t1: np.ndarray = nib.load(t1_string).get_fdata()
data_input_t1ce: np.ndarray = nib.load(t1ce_string).get_fdata()
data_input_t2: np.ndarray = nib.load(t2_string).get_fdata()
data_input_flair: np.ndarray = nib.load(flair_string).get_fdata()
# seg: np.ndarray = nib.load(os.path.join(TRAIN_PATH, train_string, train_string + "_seg.nii")).get_fdata()

# loop over batches
step = 75
test = data_input_t1[:, :, step]
print(data_input_t1.shape)
# Get data
processed_t1 = brain_data_preprocessing(data_input_t1[:, :, step])
processed_t1ce = brain_data_preprocessing(data_input_t1ce[:, :, step])
processed_t2 = brain_data_preprocessing(data_input_t2[:, :, step])
processed_flair = brain_data_preprocessing(data_input_flair[:, :, step])
print(processed_t1.shape)

# train on batch
main_data = tf.stack([
    processed_t1,
    processed_t1ce,
    processed_t2,
    processed_flair],
    axis=-1)
print(main_data.shape)
dp = np.expand_dims(main_data, axis=0)

d = tf.convert_to_tensor(dp, dtype=tf.float32)
print(d.shape)
# print(model.summary())

# Perform inference on the validation data
predictions = model.predict(d)

num_classes = predictions.shape[-1]
cmap = ListedColormap(['#000000', '#FF0000', '#00FF00', '#000F00'])  # Replace with your desired colors

# Create a color-coded segmentation mask
segmentation_mask = np.argmax(predictions, axis=-1)

# Visualize the segmentation mask
plt.imshow(segmentation_mask[0], cmap=cmap, vmin=0, vmax=num_classes - 1)
plt.colorbar()
plt.show()

test = data_input_t1[:, :, step]
## Ignore no info

# print(new_shape.shape)
plt.imshow(test, )
plt.imshow(test, cmap="viridis")
#
# figure = plt.figure(test)
# ax = figure.add_subplot()
# ax.set_title("Tumor Segmentations")
# ax.legend(loc='best')
plt.show()

# Visualize all classes
num_classes = predictions.shape[-1]
fig, axs = plt.subplots(1, num_classes + 1, figsize=(15, 5))

# Visualize each class
for class_idx in range(num_classes):
    axs[class_idx].imshow(predictions[0, :, :, class_idx], cmap='viridis')
    axs[class_idx].set_title(f'Class {class_idx}')

# Visualize the sum of probabilities (optional)
sum_probabilities = np.sum(predictions, axis=-1)
axs[-1].imshow(sum_probabilities[0], cmap='viridis')
axs[-1].set_title('Sum of Probabilities')
#
plt.show()
#
# figure = plt.figure(test)
# ax = figure.add_subplot()
# ax.set_title("Tumor Segmentations")
# ax.legend(loc='best')
# plt.show()
# # Assuming predictions are one-hot encoded, convert them to binary masks
# binary_predictions = np.argmax(predictions, axis=-1)
#
# # Evaluate the results using Mean Intersection over Union (IoU) or other metrics
# iou_metric = MeanIoU(num_classes=4)  # num_classes should be adjusted based on your task
# # iou_metric.update_state(val_labels, binary_predictions)
# # iou = iou_metric.result().numpy()
#
# print(f'Mean IoU on Validation Set: {iou}')