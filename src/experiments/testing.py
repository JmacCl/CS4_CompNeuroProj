import os.path
import pickle
import sys

import torch
from torch.utils.data import DataLoader

from src.experiments.configs.config import BraTS2020Configuration
from src.experiments.datasets.BraTS2020 import BraTS2020Data
from src.experiments.training import create_data_source_path, source_instances, create_data_path
from src.experiments.testing_utils.metrics import *

def derive_loader(img_path, seg_path):
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

def update_test_scores(seg, pred, metric_function, metric_scores):
    """
    Update the test scores with the defined metric functions
    :param seg:
    :param pred:
    :param metric_function:
    :param metric_scores:
    :return:
    """
    for keys in metric_function:
        metric_scores[keys] += metric_function[keys](seg, pred)


def process_batch_scores(current_scores, final_scores, batch):
    """
    After each batch, update the final scores by dividing each records by batch
    :param current_scores:
    :param final_scores:
    :param batch:
    :return:
    """
    for key in current_scores:
        final_scores[key] += current_scores[key]/batch
        current_scores[key] = 0


def finalize_test_scores(final_scores, test_size):
    for key in final_scores:
        final_scores[key] = final_scores[key]/test_size


def test_model(model, loader, batch, metric_score, metric_function):
    """
    Test the model with the specified testing metrics with the given batch
    :param model:
    :param loader:
    :param batch:
    :param metric_score:
    :return:
    """
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= batch:
                break
            # perform a forward pass and calculate the  loss
            x = data["image"]
            y = data["segmentation"]
            pred = model(x)
            update_test_scores(y, pred, metric_function=metric_function,
                               metric_scores=metric_score)


def save_test_results(final_scores, test_sample, experiment_path, experiment_name):
    """
    Save the current test scores at the specified location
    :param final_scores:
    :param test_sample:
    :param experiment_path:
    :param experiment_name:
    :return:
    """
    save_results = {}
    for key in final_scores.keys():
        item = {"score": final_scores[key]/test_sample, "test_index": test_sample}
        save_results[key] = item
    with open(os.path.join(experiment_path, experiment_name + "_results.pkl", "rb")) as file:
        pickle.dump(save_results, file)

def testing(testing_config: dict):
    """
    Will test the model with the given defined metrics and save the results
    as a yaml
    :param testing_config:
    :return:
    """

    # Load the model
    experiment_location = testing_config["experiment_location"]
    experiment_name = testing_config["experiment_name"]
    model_name = testing_config["model_name"]
    experiment_path = os.path.join(experiment_location, experiment_name)
    model_location = os.path.join(experiment_path, model_name)
    model = torch.load(model_location)

    # Obtain the testing files
    data_location = testing_config["data_location"]
    data_name = testing_config["data_name"]
    epoch_train_path, epoch_seg_path = create_data_path(data_location, dataset_name=data_name,
                                                        data_nature="original", purpose="testing")
    testing_images, testing_masks = source_instances(epoch_train_path, epoch_seg_path)

    # Derive metrics
    metric_functions = derive_metrics(testing_config["metrics"])
    metric_score = set_up_testing_scores(metric_dict=metric_functions)
    final_scores = set_up_testing_scores(metric_dict=metric_functions)
    batch = testing_config["batch"]
    test_size = len(testing_images)

    for i in range(test_size):

        test_img = testing_images[i]
        test_mask = testing_images[i]
        loader = derive_loader(os.path.join(epoch_train_path, test_img),
                               os.path.join(epoch_seg_path, test_mask))
        test_model(model=model, loader=loader,
                   batch=batch, metric_score=metric_score,
                   metric_function=metric_functions)
        process_batch_scores(metric_score, final_scores, batch)

        save_test_results(final_scores, test_sample=i,
                          experiment_path=experiment_path,
                          experiment_name=experiment_name)

    finalize_test_scores(final_scores, test_size)






if __name__ == "__main__":
    config = BraTS2020Configuration(sys.argv[1])
    testing(config.testing)
