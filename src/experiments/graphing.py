import sys
from pathlib import Path
from typing import List, Dict

import torch
import pickle
import random
import matplotlib.pyplot as plt
import nibabel as nib
from torch.utils.data import TensorDataset, DataLoader

from src.experiments.configs.config import BraTS2020Configuration
from src.experiments.utility_functions.data_access import *
from src.experiments.training_utils.loss_functions import DiceLoss
import matplotlib.pyplot as plt
import numpy as np
import torch

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

def prepare_plot(origImage):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    # plot the original image, its mask, and the predicted mask
    ax.imshow(origImage, cmap="gray")
    # set the titles of the subplots
    ax.set_title("Image")
    # set the layout of the figure and display it
    figure.tight_layout()
    figure.show()

def plot_experiment_images(image, seg, model_seg):
    figure, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Image")

    ax[1].imshow(seg, cmap="gray")
    ax[1].set_title("Original Segmentation")

    ax[2].imshow(model_seg, cmap="gray")
    ax[2].set_title("Model Segmentation")

    # set the titles of the subplots
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

def plot_learning_metrics(metrics: Dict):
    """
    Given a specific learning metric, plot and examine results in
    comparison with the epoch
    :param metrics:
    :param label:
    :param y_label:
    :return:
    """
    # Derive the title from the input metric
    title: str = list(metrics.keys())[0]
    # derive the pickle files from the metrics
    recordings = metrics[title]["recordings"]

    for val_keys, rec_path in recordings.items():
        with open(rec_path, "rb") as file:
            recordings = pickle.load(file)
            print(recordings)
            print(val_keys)
            print(len(recordings))
            print("\n")
        plt.plot(recordings, label=val_keys)
    y_label = metrics[title]["y_label"]
    # For each set of recordings, plot the lists
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()

def process_image_graph(data_tensor, dim):
    """
    Take a given image tensor and process it so that it can be graphed
    :param image_location:
    :return:
    """

    # Process it into a numpy array and return
    img_batch = data_tensor.squeeze(0)
    np_ver = img_batch.numpy()

    np_ver = np.transpose(np_ver, (1, 2, 0))
    return np_ver[:, :, dim-1]

def process_mask_graph(mask,  model:bool):
    """
    Take a given segmentation torch tensor and process it so it can be graphed
    :param mask:
    :param model:
    :return:
    """
    if model:
        mask = torch.softmax(mask, dim=0)
    argmax_indices = torch.argmax(mask, dim=0)
    segmentation_np = argmax_indices.cpu().numpy()

    return segmentation_np

# def convert_numpy(tensor, if_img):
#     """
#     Given a torch tensor, convert it to a numpy array
#     :param tensor:
#     :return:
#     """
#     img_batch = tensor.squeeze(0)
#     np_ver = img_batch.cpu().numpy()
#     if if_img:
#         np_ver = np.transpose(np_ver, (1, 2, 0))
#     return np_ver

def random_instance():
    pass

def set_up_model_image(path, mri_volumes, batch, seg):
    """
    Derive data from either a image path or seg path and format it
    into the desired form, via the mri specifications and the specified batch
    :param path:
    :param mri_volumes:
    :param batch:
    :param seg:
    :return:
    """

    data = torch.load(path)
    if seg is False:
        return data[batch, mri_volumes, :, :]
    else:
        return data[batch, :, :, :]

def graph_experiment_output(experiment: dict):
    """
    Given the defined format for the output of an experiment,
    create graphs of the image, true segmentation masks and output segmentation mask
    given the model
    :param experiment:
    :return:
    """
    experiment_name = list(experiment.keys())[0]
    data_information = experiment[experiment_name]["data_info"]
    mri_volumes = list(data_information["selected_mri"].values())
    selected_mri = data_information["mri_volume"]

    # Source the experiment files
    img_path = experiment[experiment_name]["input_image_path"]
    seg_path = experiment[experiment_name]["input_seg_path"]
    img_files, seg_files = source_instances(img_path, seg_path)

    # Derive the torch image and segmentation
    if data_information["batch"] is not None and data_information["instance"] is not None:
        batch = data_information["batch"]
        instance = data_information["instance"]
    else:
        batch, instance = random_instance()

    test_img = set_up_model_image(os.path.join(img_path, img_files[instance]),
                                  batch=batch,
                                  mri_volumes=mri_volumes,
                                  seg=False)

    test_seg = set_up_model_image(os.path.join(seg_path, seg_files[instance]),
                                  batch=batch, mri_volumes=mri_volumes, seg=True)


    # Source the model and get prediction results
    model = torch.load(os.path.join(experiment[experiment_name]["directory_path"],
                                    experiment_name,
                                    experiment[experiment_name]["model_name"]))

    model.eval()
    with torch.no_grad():
        output_seg = model(test_img.unsqueeze(0))

    # Reformat the data, so it is ready for graphing
    img_graph = process_image_graph(test_img, selected_mri)
    seg_graph = process_mask_graph(test_seg, model=False)
    result_graph = process_mask_graph(output_seg.squeeze(0), model=True)

    # Graph results
    plot_experiment_images(img_graph, seg_graph, result_graph)




def graphing(config: dict):
    # Plot single images

    # Plot experiment results
    experiments = config["experiments"]
    for exp in experiments:
        graph_experiment_output(exp)

    # Plot Learning metrics or Evaluation Results

    learning_metrics = config["learning_metrics"]
    for plots in learning_metrics:
        plot_learning_metrics(plots)



if __name__ == "__main__":
    print(sys.argv)
    cwd = os.getcwd()
    config_path = os.path.join(cwd, sys.argv[1])
    config = BraTS2020Configuration(config_path)
    graphing(config.graphing)



#
"""
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

"""
