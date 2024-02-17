import os
import pickle
import sys

import torch
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from src.experiments.configs.config import BraTS2020Configuration
from src.experiments.datasets.BraTS2020 import BraTS2020Data
from src.experiments.training.learning_metrics import *
from src.models.UNet.unet2d import UNet


def update_metrics(loss, true_seg, pred_seg, results: dict, funcs: dict):
    for key in results.keys():
        if key == "loss":
            results[key] += loss
        else:
            results[key] += funcs[key](true_seg, pred_seg)


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


def process_metrics(results, current, batch_count):
    for key in current.keys():
        old = results[key]
        old.append(current[key] / batch_count)
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

def create_data_path(path, dataset_name, data_nature, purpose: str):
    """
    This function will create the path that can be used to access
    either the training or validation data required for the experiment depending
    on the purpose variable
    :param purpose: either variable for training, validation, or testing
    :param path: path to the directory for all processed data
    :param dataset_name: name of the dataset
    :param data_nature: either an augmented or non-augmented source
    :return: completed data path
    """
    img_path = os.path.join(path, dataset_name, data_nature, "inputs", purpose)
    seg_path = os.path.join(path, dataset_name, data_nature, "targets", purpose)

    return img_path, seg_path


def train_model(loader, model, loss_func,
                opt, learning_metrics, learning_functions,
                clip_value):
    model.train()
    for i, data in enumerate(loader):
        # perform a forward pass and calculate the training loss
        x = data["image"]
        y = data["segmentation"]
        pred = model(x)
        loss = loss_func(pred, y)
        # first, zero out any previously accumulated gradients, then
        # perform backpropagation, and then update model parameters
        opt.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), clip_value)
        opt.step()
        # add the loss to the total training loss so far
        update_metrics(loss.item(), y, pred, learning_metrics, learning_functions)

def update_validation_results(validaiton_results, current, batch_count):
    for key in current.keys():
        old = current[key]
        validaiton_results[key] += old / batch_count
        current[key] = 0

def early_stopping(validation_loss, stagnation, best_valid):
    """
    Given the current stagnation score, determine fi the validation is poor
    enough to warrant stopping the training process
    :param patience:
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

def validate_model(data_config, model, loss_func,
                   validation_recordings, learning_functions,
                   batch, augmentation):
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
    validation_results = set_up_current_recordings(learning_functions.keys())
    epoch_train_path, epoch_seg_path = create_data_source_path(data_config, "validation", augmentation)

    validation_imgs, validation_segs = source_instances(epoch_train_path, epoch_seg_path)
    model.eval()
    for v in range(len(validation_imgs)):
        img_val = validation_imgs[v]
        seg_val = validation_segs[v]

        # Load the train loader
        loader = derive_loader(os.path.join(epoch_train_path, img_val),
                               os.path.join(epoch_seg_path, seg_val), batch=1)
        with torch.no_grad():
            for i, data in enumerate(loader):
                # perform a forward pass and calculate the training loss
                x = data["image"]
                y = data["segmentation"]
                pred = model(x)
                loss = loss_func(pred, y)
                update_metrics(loss.item(), y, pred, batch_results, learning_functions)
            # add the loss to the total training loss so far
            update_validation_results(validation_results, batch_results, batch)
    process_metrics(validation_recordings, validation_results, batch_count=len(validation_imgs))

def create_data_source_path(data_config, purpose, data_nature):
    data_path = data_config["input_data_path"]
    dataset_name = data_config["data_name"]
    img_path, seg_path = create_data_path(data_path, dataset_name, data_nature, purpose)

    return img_path, seg_path

def source_instances(img_path, seg_path):
    """
    Given the path specifications in the data_config, derive a list of all the
    possible epochs for the training data or validation instances for validation
    :param purpose: either training, validation or testing
    :param data_config: configuration that holds all the data creation sources
    :return:
    """
    image_epochs = [file for file in sorted(os.listdir(img_path)) if file.endswith('.pt')]
    segmentation_epochs = [file for file in sorted(os.listdir(seg_path)) if file.endswith('.pt')]
    return image_epochs, segmentation_epochs


def derive_loader(img_path, seg_path, batch):
    """
    Given a specific division, be it training, validation or testing
    (specified by option), load the batches of that dataset for the given
    epoch
    :param img_path:
    :param seg_path:
    :param batch:
    :return:
    """
    dataset = BraTS2020Data(img_path=img_path, seg_path=seg_path)
    loader = DataLoader(dataset, shuffle=True, batch_size=1)
    return loader


def training(config: BraTS2020Configuration):

    # Set up Model parameters
    training_config = config.training
    data_config = config.data_creation
    classes = data_config["classes"]
    channels = len(data_config["modals"])
    unet = UNet(in_channels=channels, classes=classes)

    # Set up main training hyperparameters
    loss_func = DiceLoss(n_classes=classes)
    lr = training_config["ilr"]
    weight_decay = training_config["weight_decay"]
    opt = Adam(unet.parameters(), lr=lr, weight_decay=weight_decay)
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
    # Get training
    data_nature = training_config["data_nature"]
    epoch_train_path, epoch_seg_path = create_data_source_path(data_config, "training", data_nature)
    training_epochs, segmentation_epochs = source_instances(epoch_train_path, epoch_seg_path)

    for e in range(epoch):
        img_epoch = training_epochs[e]
        seg_epoch = segmentation_epochs[e]

        # Load the train loader
        train_loader = derive_loader(os.path.join(epoch_train_path, img_epoch),
                                     os.path.join(epoch_seg_path, seg_epoch),
                                     batch=batch)

        # Train the model and update the metric recordings
        train_model(train_loader, unet, loss_func=loss_func,
                    opt=opt, learning_metrics=current_metrics,
                    learning_functions=lm_funcs, clip_value=clip_value)

        validate_model(data_config, unet, loss_func=loss_func,
                       validation_recordings=validation_recordings,
                       learning_functions=lm_funcs, batch=batch, augmentation=data_nature)

        process_metrics(training_recordings, current_metrics, batch_count=batch)

        # see if early stopping is required



        torch.save({
            'model_state_dict': unet.state_dict(),
            'optim_state_dict': opt.state_dict(),
            'epoch': e,
        }, os.path.join(model_output, "model.pth"))

        # Determine if early stopping is necessary
        best_valid, stagnation = early_stopping(validation_loss=validation_recordings[e],
                                                best_valid=best_valid, stagnation=stagnation)
        if stagnation >= patience:
            break
    # print("Epoch completed and model successfully saved!")
        #

    # Save the experiment deep learning model
    model_output = os.path.join(save_path, exp_name)
    if not os.path.exists(model_output):
        os.makedirs(model_output)
    torch.save(unet, os.path.join(model_output, "model.pth"))

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
    save_config(config, training_config=config.training)
