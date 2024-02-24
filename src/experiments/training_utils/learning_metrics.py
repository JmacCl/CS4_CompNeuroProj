import torch

import numpy as np
import numba
from src.experiments.utils import convert_mask
from src.experiments.training_utils.loss_functions import *
from hausdorff import hausdorff_distance as hd


def pixel_accuracy(original, predictions):
    """
    Calculate pixel accuracy for binary segmentation.
    Args:
    - original: Ground truth binary mask (tensor).
    - predictions: Predicted binary mask (tensor).
    Returns:
    - accuracy: Pixel accuracy.
    """
    # Convert the binary masks to 0s and 1s
    original_binary = convert_mask(original.squeeze(0), model=False)
    predictions_binary = convert_mask(predictions.squeeze(0), model=True)

    # Calculate the number of correctly classified pixels
    correct_pixels = torch.sum(original_binary == predictions_binary).item()

    # Calculate total number of pixels
    total_pixels = original_binary.numel()

    # Calculate pixel accuracy
    accuracy = correct_pixels / total_pixels

    return accuracy


def intersection_over_union(original, predictions):
    """
    Calculate Intersection over Union (IoU) for binary segmentation.

    Args:
    - original: Ground truth binary mask (tensor).
    - predictions: Model predictions (tensor).
    - model: Whether the predictions are logits and need to be processed.
    - threshold: Threshold for converting logits to probabilities.

    Returns:
    - iou: Intersection over Union.
    """

    # Convert the binary masks to 0s and 1s
    original_binary = convert_mask(original.squeeze(0), model=False)

    # Process model predictions
    predictions_binary = convert_mask(predictions.squeeze(0), model=True)

    # Calculate the intersection and union
    intersection = torch.sum(original_binary * predictions_binary).item()
    union = torch.sum((original_binary + predictions_binary) > 0).item()

    # Calculate IoU
    iou = intersection / union if union > 0 else 0.0

    return iou


# def odice_loss(score, target):
#     # Convert the binary masks to 0s and 1s
#     original_binary = convert_mask(target.squeeze(0), model=False)
#
#     # Process model predictions
#     score = convert_mask(score.squeeze(0), model=True)
#     target = original_binary.float()
#     smooth = 1e-5
#     intersect = torch.sum(score * target)
#     y_sum = torch.sum(target * target)
#     z_sum = torch.sum(score * score)
#     loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
#     return loss.item()

def dice(original, predictions):
    """
    Implementation for dice coefficient/ f1 score for evaluation metric
    :param original: original mask
    :param predictions: predictions from the model
    :return:
    """
    # Convert the binary masks to 0s and 1s
    original_binary = convert_mask(original.squeeze(0), model=False)

    # Process model predictions
    predictions_binary = convert_mask(predictions.squeeze(0), model=True)


    # Derive the given metrics
    intersection = torch.sum(original_binary * predictions_binary).item()
    union = len(torch.unique(torch.cat((original_binary, predictions_binary))))
    numerator = 2 * intersection
    denominator = union + intersection

    return numerator/denominator




def hausdorff_distance(original, predictions, distance):
    """
    Calculate Hausdorff Distance for binary segmentation.

    Args:
    - original: Ground truth binary mask (tensor).
    - predictions: Model predictions (tensor).
    - model: Whether the predictions are logits and need to be processed.
    - threshold: Threshold for converting logits to probabilities.

    Returns:
    - distance: Hausdorff Distance.
    """
    # Convert the binary masks to 0s and 1s
    original_binary = convert_mask(original.squeeze(0), model=False)
    original_binary = original_binary.cpu()

    # Process model predictions
    predictions_binary = convert_mask(predictions.squeeze(0), model=True)
    predictions_binary = predictions_binary.cpu()

    # convert torch arrays into np arrays
    distance = hd(original_binary.numpy(), predictions_binary.numpy(), distance=distance)
    return distance
