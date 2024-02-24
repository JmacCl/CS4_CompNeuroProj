import os.path
import pickle
import sys

import torch
from torch.utils.data import DataLoader

from src.experiments.configs.config import BraTS2020Configuration
from src.experiments.datasets.BraTS2020 import BraTS2020Data
from src.experiments.testing_utils.metrics import *
from src.experiments.utility_functions.data_access import *
from src.experiments.utility_functions.data_processing import process_metrics



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


def test_model(model, loader, metric_score, metric_function):
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
        record = 0
        for i, data in enumerate(loader):
            # perform a forward pass and calculate the  loss
            x = data["image"]
            y = data["segmentation"]
            pred = model(x)
            update_test_scores(y, pred, metric_function=metric_function,
                               metric_scores=metric_score)
            record = i
    finalize_test_scores(metric_score, record)


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
    save_path = os.path.join(experiment_path, experiment_name + "_results.pkl")

    with open(save_path, "wb") as file:
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
    gpu = testing_config["GPU"]
    if gpu:
        model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Obtain the testing files
    data_location = testing_config["data_location"]

    # Derive metrics
    metric_functions = derive_metrics(testing_config["metrics"])
    metric_score = set_up_testing_scores(metric_dict=metric_functions)
    batch = testing_config["batch"]
    mri_vols = testing_config["selected_mri"]
    test_loader = derive_loader(data_directory=data_location, purpose="testing", mri_vols=mri_vols, batch=batch, transforms=None)
    test_model(model=model, loader=test_loader, metric_score=metric_score, metric_function=metric_functions)

    # save_test_results(final_scores, test_sample=i+1,
    #                     experiment_path=experiment_path,
    #                     experiment_name=experiment_name)



    print(metric_score)






if __name__ == "__main__":
    config = BraTS2020Configuration(sys.argv[1])
    testing(config.testing)
