import sys
from pathlib import Path

import torch
import pickle
import matplotlib.pyplot as plt
import nibabel as nib
from torch.utils.data import TensorDataset, DataLoader

from src.experiments.configs.config import BraTS2020Configuration
from src.experiments.training.loss_functions import DiceLoss
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import torch.nn.functional as F

import os

pickle_file = "C:\\Users\\james\\Uni_Projects\\CS4_CompNeuroProj\\src\\experiments\\saved_model\\Graph\\train_loss.pkl"
exp_loc = r"C:\Users\james\Uni_Projects\CS4_CompNeuroProj\src\data\processed\standard_experiment\inputs\epoch_121.pt"
label_loc = r"C:\Users\james\Uni_Projects\CS4_CompNeuroProj\src\data\processed\standard_experiment\targets\epoch_121.pt"


def view_elem(tensor):
    reshaped_tensor = tensor.view(tensor.size(0), -1)

    # Find unique elements along each channel
    unique_elements_per_channel = [torch.unique(reshaped_tensor[i]) for i in range(reshaped_tensor.size(0))]

    # Print unique elements for each channel
    for channel, unique_elements in enumerate(unique_elements_per_channel):
        print(f"Channel {channel}: {unique_elements}")

    # If you want to find unique elements across all channels
    unique_elements_all_channels = torch.unique(tensor.view(tensor.size(0), -1))
    print(f"Unique elements across all channels: {unique_elements_all_channels}")

#

def prepare_plot(origImage, origMask, predMask):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage, cmap="gray")
    ax[1].imshow(origMask, cmap="gray")
    ax[2].imshow(predMask, cmap="gray")
    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")
    # set the layout of the figure and display it
    figure.tight_layout()
    figure.show()

# def examine_binary_softmax_predictions(original, predictions):
#     # Derive the argmax values for the original
#     argmax_indices = torch.argmax(original, dim=0)
#     # Derive the indices positions of where the values are 1
#     tumor_indices = torch.nonzero(argmax_indices==1, as_tuple=False)
#     preds = predictions[tumor_indices]
#     # Get the channels values for the softmax predicitons
#     pred_amount = len(tumor_indices)
#     (x, y) = 0, 0
#
#     for i in range(pred_amount):
#         index = tumor_indices[i]
#         x += predictions[index[0]]/pred_amount
#         y += tumor_indices[i][1]/pred_amount
#
#     diff = x - y
#
#     coeff = pred_amount / (128*128)
#
#     final = coeff * diff
#
#     print(x)
#     print(y)





def convert_mask(mask, model:bool, threshold=0.6):
    if model:
        mask = torch.softmax(mask, dim=0)
    argmax_indices = torch.argmax(mask, dim=0)
    segmentation_np = argmax_indices.cpu().numpy()

    return segmentation_np

def image_reveal(img, batch, dim):
    # convert image
    img_batch = img[batch]
    img_batch = img_batch.squeeze(0)
    np_ver = img_batch.numpy()

    np_ver = np.transpose(np_ver, (1, 2, 0))
    return np_ver[:, :, dim]

def make_predictions(model, batch, mask, b):
    # set model to evaluation mode
    model.eval()
    # turn off gradient tracking
    batch_selection = batch[b].unsqueeze(0)
    with torch.no_grad():
        predMask = model(batch_selection)
        origMask = mask[b]
        predMask = predMask.squeeze(0)

        return image_reveal(batch, b, 0), convert_mask(origMask, False), convert_mask(predMask, True)

def plot_model_loss(loss_values, label, y_label):
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, label=label)
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.title(label)
    plt.legend()
    plt.show()

def graphing(config: dict):
    experiment_location = config["experiment_location"]
    data_location = config["data_input_loc"]
    loss_loc = config["loss_loc"]

    model_location = config["model_name"]
    input_ins_loc = config["exp_loc"]
    target_ins_loc = config["label_loc"]
    # Get Unet model config.
    prev_Unet = torch.load(os.path.join(experiment_location, model_location))
    prev_Unet.eval()
    batch_data = torch.load(os.path.join(data_location, input_ins_loc))  # .permute(0, 2, 1).unsqueeze(0)
    label_data = torch.load(os.path.join(data_location, target_ins_loc))  # .permute(0, 2, 1).unsqueeze(0)
    batch, orig_mask, pred_mask = make_predictions(prev_Unet, batch_data, label_data, 65)
    #derive the modal loss from a source
    if config["model_loss"]:
        with open(os.path.join(experiment_location, loss_loc), 'rb') as file:
            loaded_data = pickle.load(file)
        plot_model_loss(loaded_data, "Training Loss", "Loss")

    if config["iou"]:
        with open(os.path.join(experiment_location, "Graph", "iou.pkl"), 'rb') as file:
            loaded_data = pickle.load(file)
        plot_model_loss(loaded_data, "IoU Change", "IoU")

    if config["accuracy"]:
        with open(os.path.join(experiment_location, "Graph", "accuracy.pkl"), 'rb') as file:
            loaded_data = pickle.load(file)
        plot_model_loss(loaded_data, "Accuracy", "Accuracy")

    if config["hd"]:
        with open(os.path.join(experiment_location, "Graph", "hausdorff.pkl"), 'rb') as file:
            loaded_data = pickle.load(file)
        plot_model_loss(loaded_data, "Hausdorff", "Hausdorff")

    if config["segmentation_res"]:
        prepare_plot(batch, orig_mask, pred_mask)


if __name__ == "__main__":
    print(sys.argv)
    cwd = os.getcwd()
    config_path = os.path.join(cwd, sys.argv[1])
    config = BraTS2020Configuration(config_path)
    graphing(config.graphing)

