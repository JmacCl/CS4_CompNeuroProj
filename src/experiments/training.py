import os
import pickle
import sys
from pathlib import Path

import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from torchvision.transforms import transforms, RandomHorizontalFlip, RandomRotation, RandomVerticalFlip

from kornia.losses.focal import FocalLoss

from src.experiments.configs.config import BraTS2020Configuration
from src.experiments.datasets.BraTS2020 import BraTS2020Data
from src.experiments.training_utils.learning_metrics import *
from src.experiments.utility_functions.data_access import derive_loader
from src.models.UNet.unet2d import UNet

from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler


def loss_function(loss_config: dict, classes):
    loss_keys = list(loss_config.keys())

    final_loss = 0

    for k in loss_keys:
        if k == "dice":
            final_loss += loss_config[k]["coefficient"]*DiceLoss(n_classes=classes)
        elif k == "focal":
            alpha = loss_config[k]["alpha"]
            gamma = loss_config[k]["gamma"]
            final_loss += loss_config[k]["coefficient"]*FocalLoss(alpha=alpha, gamma=gamma)

    return final_loss

def update_metrics(loss, true_seg, pred_seg, results: dict, funcs: dict):
    for key in results.keys():
        if key == "loss":
            results[key] += loss
        else:
            results[key] += funcs[key](true_seg, pred_seg)

def process_augmentations(augmentations):
    """
    Given the specifications for augmentations, process them
    :param augmentations:
    :return:
    """
    return_list = []
    aug_keys = augmentations.keys()
    for k in aug_keys:
        if k == "vertical_flipping":
            return_list.append(RandomVerticalFlip(augmentations[k]))
        elif k == "horizontal_flipping":
            return_list.append(RandomHorizontalFlip(augmentations[k]))
        elif k == "rotation":
            return_list.append(RandomRotation(degrees=augmentations[k]["degree"]),
                                                p=augmentations[k]["probability"])

    return transforms.Compose([return_list])

def save_learning_metrics(save_path, exp_name, recordings, option):
    """
    given the save paths and experiment name, save the learning metric
    graphs
    :param save_path:
    :param exp_name:
    :param recordings:
    :param option:
    :return:
    """
    loss_location = os.path.join(save_path, exp_name, option, "Graph")
    if not os.path.exists(loss_location):
        os.makedirs(loss_location)
    for key in recordings.keys():
        pickle_save = key + '.pkl'
        with open(os.path.join(loss_location, pickle_save), 'wb') as file:
            pickle.dump(recordings[key], file)


def process_metrics(results, current):
    for key in current.keys():
        old = results[key]
        old.append(current[key])
        results[key] = old
        current[key] = 0


def configure_learning_recordings(training_config: dict):
    recordings = {"loss": []}
    defined_metrics = training_config["learning_metrics"]
    for met in defined_metrics:
        recordings[met] = []

    return recordings


def set_up_functions(training_config: dict):
    return_dic = {}
    for key in training_config:
        if key == "accuracy":
            return_dic[key] = pixel_accuracy
        elif key == "hausdorff":
            return_dic[key] = hausdorff_distance
        elif key == "IoU":
            return_dic[key] = intersection_over_union
    return return_dic


def set_up_recordings(defined_metrics: list):
    current_recordings = {"loss": []}
    for met in defined_metrics:
        current_recordings[met] = []
    return current_recordings


def set_up_current_recordings(defined_metrics: list):
    current_recordings = {"loss": 0}
    for met in defined_metrics:
        current_recordings[met] = 0
    return current_recordings

def train_model(loader, model, loss_func,
                opt, learning_metrics, learning_functions,
                clip_value, batch, device):
    model.train()
    for i, data in enumerate(loader):
        if i >= batch:
            break
        # perform a forward pass and calculate the training loss
        x = data["image"].to(device)
        y = data["segmentation"].to(device)
        opt.zero_grad()

        pred = model(x)
        loss = loss_func(pred, y)
        # first, zero out any previously accumulated gradients, then
        # perform backpropagation, and then update model parameters
        loss.backward()
        clip_grad_norm_(model.parameters(), clip_value)
        opt.step()
        # add the loss to the total training loss so far
        update_metrics(loss.item(), y, pred, learning_metrics, learning_functions)


def update_validation_results(validation_results, current, batch_count):
    for key in current.keys():
        old = current[key]
        validation_results[key] += old / batch_count
        current[key] = 0


def early_stopping(validation_loss, stagnation, best_valid):
    """
    Given the current stagnation score, determine fi the validation is poor
    enough to warrant stopping the training process
    :param validation_loss:
    :param stagnation:
    :param best_valid:
    :return:
    """
    if validation_loss < best_valid:
        best_valid = validation_loss
        stagnation = 0
    else:
        stagnation += 1

    return best_valid, stagnation


def validate_model(data_directory, model, loss_func,
                   validation_recordings, learning_functions,
                   batch, mri_vols, device):
    """
    Validate the model with information from the data configuration and the model
    :param batch:
    :param data_config:
    :param model:
    :param loss_func:
    :param validation_recordings:
    :param learning_functions:
    :return:
    """
    batch_results = set_up_current_recordings(learning_functions.keys())
    model.eval()
    loader = derive_loader(data_directory=data_directory, purpose="validation",
                           mri_vols=mri_vols, transforms=None, batch=batch)
    with torch.no_grad():
        for i, data in enumerate(loader):

            # perform a forward pass and calculate the  loss
            x = data["image"].to(device)
            y = data["segmentation"].to(device)
            pred = model(x)
            loss = loss_func(pred, y)
            update_metrics(loss.item(), y, pred, batch_results, learning_functions)
    process_metrics(validation_recordings, batch_results)

def training(config: BraTS2020Configuration):
    # Set up Model parameters
    training_config = config.training
    data_config = config.data_creation
    classes = data_config["classes"]
    selected_mri_vols = training_config["selected_mri"]
    channels = len(selected_mri_vols)

    model_config = training_config["model"]
    model = UNet(in_channels=channels, classes=classes,
                 layers=model_config["layers"],
                 dropout_p=model_config["dropout_rate"])

    # If GPu is specified
    gpu = training_config["GPU"]
    if gpu:
        model.to(device="cuda" if torch.cuda.is_available() else "cpu")

    # Set up main training hyperparameters
    loss_func = loss_function(training_config["loss"], classes=classes)
    lr = training_config["ilr"]
    weight_decay = training_config["weight_decay"]
    opt = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    clip_value = training_config["gradient_clipping"]

    # Set up patience and best validation loss
    best_valid = 0
    stagnation = 0
    patience = training_config["patience"]

    # calculate steps per epoch for training and test set
    learning_metrics = training_config["learning_metrics"]
    lm_funcs = set_up_functions(learning_metrics)
    training_recordings = set_up_recordings(learning_metrics)
    current_metrics = set_up_current_recordings(learning_metrics)
    validation_recordings = set_up_recordings(learning_metrics)

    # Get experiment name
    exp_name = training_config["experiment"]
    save_path = training_config["save_path"]

    # Save the experiment deep learning model
    model_output = os.path.join(save_path, exp_name)
    if not os.path.exists(model_output):
        os.makedirs(model_output)

    print("[INFO] training the network...")
    epoch = training_config["epoch"]
    batch = training_config["batch"]
    verbose_mode = training_config["verbose_mode"]
    data_dir = Path(training_config["data_directory"])
    augmentations = process_augmentations(training_config["augmentations"])
    # Get training

    for e in range(epoch):
        if verbose_mode:
            print(e+1)
        # Load the train loader
        train_loader = derive_loader(data_directory=data_dir, purpose="training",
                                     mri_vols=selected_mri_vols, transforms=augmentations,
                                     batch=batch)

        # Train the model and update the metric recordings
        train_model(train_loader, model, loss_func=loss_func,
                    opt=opt, learning_metrics=current_metrics,
                    learning_functions=lm_funcs, clip_value=clip_value,
                    batch=batch, device=gpu)

        validate_model(data_config, model, loss_func=loss_func,
                       validation_recordings=validation_recordings,
                       learning_functions=lm_funcs, batch=batch,
                       mri_vols=selected_mri_vols, device=gpu)

        process_metrics(training_recordings, current_metrics)

        # see if early stopping is required

        torch.save({
            'model_state_dict': model.state_dict(),
            'optim_state_dict': opt.state_dict(),
            'epoch': e,
        }, os.path.join(model_output, "model.pth"))

        save_learning_metrics(save_path, exp_name, training_recordings, "training")
        save_learning_metrics(save_path, exp_name, validation_recordings, "validation")

        # Determine if early stopping is necessary
        best_valid, stagnation = early_stopping(validation_loss=validation_recordings["loss"][e],
                                                best_valid=best_valid, stagnation=stagnation)
        if stagnation >= patience:
            break

    # Save the experiment deep learning model
    model_output = os.path.join(save_path, exp_name)
    if not os.path.exists(model_output):
        os.makedirs(model_output)
    torch.save(model, os.path.join(model_output, "model.pth"))

    # Save the loss and other defined metrics for training and validation
    save_learning_metrics(save_path, exp_name, training_recordings, "training")
    save_learning_metrics(save_path, exp_name, validation_recordings, "validation")


def save_config(config, training_config):
    """
    Save the configuration format for the training process
    :return:
    """
    # Get experiment name
    exp_name = training_config["experiment"]
    save_path = training_config["save_path"]
    model_output = os.path.join(save_path, exp_name)

    final_config = {}

    with open(config, 'r') as file:
        loaded_data = yaml.safe_load_all(file)

    # Specify the new file path
    experiment_config = exp_name + "_" + "configuration.yaml"

    # Save the loaded data to the new YAML file
    with open(os.path.join(model_output, experiment_config), 'w') as file:
        yaml.dump(loaded_data, file)


if __name__ == "__main__":
    config = BraTS2020Configuration(sys.argv[1])
    training(config)
    # save_config(config, training_config=config.training)

# TODO save configuration does not work