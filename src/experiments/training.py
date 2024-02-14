# USAGE
# python train.py
# import the necessary packages
from src.experiments.graphing import make_predictions, image_reveal, prepare_plot
from src.experiments.training.loss_functions import DiceLoss
from src.experiments.configs.config import BraTS2020Configuration
# from src.experiments.training.training_configs import *
from src.experiments.utils import convert_num_to_string, convert_batch_train
from src.uneeded.utils.brain_data_preprocessing import brain_data_preprocessing
from src.models.UNet.unet2d import UNet
from src.data.data_processors.data_processors import BratsDataProcessor
from src.experiments.training.learning_metrics import *

from torch.nn import MultiLabelSoftMarginLoss, CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from imutils import paths

import torch
import time
import os
import nibabel as nib
import numpy as np
import pickle
import sys
import random
import configparser


# def remove_useless(input, labels):
#     """
#     This function will output the data files where there is no useless information
#     :param input:
#     :param labels:
#     :return:
#     """
#     new_input = []
#     new_target = []
#     for idx, label in enumerate(labels):
#         conv_label = convert_mask(label, False)
#         if not torch.all(conv_label == 0):
#             new_input.append(input[idx])
#             new_target.append(labels[idx])
#     return new_input, new_target



def map_and_plot(x, model_output, batch):
    # Ensure the tensor is in the range [0, 1] for proper visualization
    x_output = torch.clamp(x, 0, 1)
    model_output = torch.clamp(model_output, 0, 1)

    if batch == 5:

        # Select batch index and channel for visualization
        batch_index = 0
        channel_index = 0

        # Extract the image from the tensor
        image_x = x_output[batch_index, channel_index].numpy()
        print(image_x.shape)
        image_array = model_output[batch_index, channel_index].numpy()

        # Display the image using matplotlib
        plt.imshow(image_x, cmap='gray')  # Choose a colormap if needed
        plt.axis('off')  # Turn off axis labels
        plt.show()

        # Display the image using matplotlib
        plt.imshow(image_array, cmap='viridis')  # Choose a colormap if needed
        plt.axis('off')  # Turn off axis labels
        plt.show()

def update_metrics(loss, true_seg, pred_seg, results:dict, funcs:dict):
    for key in results.keys():
        if key == "train_loss":
            results[key] += loss
        else:
            results[key] += funcs[key](true_seg, pred_seg)

def process_metrics(results, current, batch_count):
    for key in current.keys():
        old = results[key]
        old.append(current[key] / batch_count)
        results[key] = old
        current[key] = 0


def train_model(data, target, model, loss_func, opt):
    x = data
    y = target
    # y = y.to(torch.long)
    pred = model(x)
    loss = loss_func(pred, y)
    # first, zero out any previously accumulated gradients, then
    # perform backpropagation, and then update model parameters
    opt.zero_grad()
    loss.backward()
    opt.step()
    # add the loss to the total training loss so far
    return loss.item(), pred

def training(config: BraTS2020Configuration):

    training_config = config.training
    data_config = config.data_creation
    create_data = training_config["special_operations"]["create_data"]

    loader = BratsDataProcessor(config.data_creation, create_data=create_data)
    classes = training_config["classes"]
    channels = len(data_config["modals"])

    unet = UNet(in_channels=channels, classes=classes)
    # initialize loss function and optimizer
    loss_func = DiceLoss(n_classes=classes)
    lr = training_config["ilr"]
    opt = Adam(unet.parameters(), lr=lr)

    # Define the BraTS data loader and get data
    train = loader.get_input_data()
    labels = loader.get_label_data()
    # calculate steps per epoch for training and test set
    learning_metrics = {"train_loss": [], "accuracy": [], "hausdorff": [], "iou": []}
    current_metrics = {"train_loss": 0, "accuracy": 0, "hausdorff": 0, "iou": 0}
    lm_funcs = {"accuracy": pixel_accuracy, "hausdorff": hausdorff_distance, "iou": intersection_over_union}
    loss_vals = []

    # Get experiment name
    exp_name = training_config["experiment"]
    save_path = training_config["save_path"]
    # Save the experiment deep learning model
    model_output = os.path.join(save_path, exp_name)
    if not os.path.exists(model_output):
        os.makedirs(model_output)

    print("[INFO] training the network...")
    epoch = training_config["epoch"]

    # Generate a list of indices excluding the specified index
    indices_to_sample_from = [i for i in range(len(train)) if i != 121]

    # Sample indices without replacement
    epoch_indices = random.sample(indices_to_sample_from, epoch)

    training_epoch = [train[i] for i in epoch_indices]
    label_epoch = [labels[i] for i in epoch_indices]
    unet.train()

    for e in range(len(training_epoch)):
        print("At epoch", e)

        batch = training_config["batch"]
        batch_indices = random.sample(range(128), batch)
        train_data = training_epoch[e]
        label_data = label_epoch[e]

        train_samples = [train_data[b] for b in batch_indices]
        label_data = [label_data[b] for b in batch_indices]

        # train_samples, label_data = remove_useless(train_samples, label_data)

        total_train_loss = 0

        for b in range(len(train_samples)):
            # perform a forward pass and calculate the training loss
            x = train_samples[b].unsqueeze(0)
            y = label_data[b].unsqueeze(0)

            loss, pred = train_model(x, y, model=unet,
                               loss_func=loss_func, opt=opt)

            # if e == 1:
            #     X, Y, P = convert_batch_train(x, 0), convert_mask(y.squeeze(0), False), convert_mask(pred.squeeze(0), True)
            #     prepare_plot(X, Y, P)
            update_metrics(loss, y, pred, current_metrics, lm_funcs)

            total_train_loss += loss

        epoch_train_loss = total_train_loss/len(train_samples)
        process_metrics(learning_metrics, current_metrics, len(train_samples))
        loss_vals.append(epoch_train_loss)
        torch.save({
            'model_state_dict': unet.state_dict(),
            'optim_state_dict': opt.state_dict(),
            'epoch': e,
        }, os.path.join(model_output, "model.pth"))
        # print("Epoch completed and model successfully saved!")
        #

    # Save the experiment deep learning model
    model_output = os.path.join(save_path, exp_name)
    if not os.path.exists(model_output):
        os.makedirs(model_output)
    torch.save(unet, os.path.join(model_output, "model.pth"))

    # Save the loss and other defined metrics
    loss_location = os.path.join(save_path, exp_name, "Graph")
    if not os.path.exists(loss_location):
        os.makedirs(loss_location)

    for key in learning_metrics.keys():
        pickle_save = key + '.pkl'
        with open(os.path.join(loss_location, pickle_save), 'wb') as file:
            pickle.dump(learning_metrics[key], file)

    # Save configuration format
    experiment_config = exp_name + "_" + "configuration.ini"
    with open(os.path.join(model_output, experiment_config), 'w') as configfile:
        config.write(configfile)


if __name__ == "__main__":
    config = BraTS2020Configuration(sys.argv[1])
    training(config)

