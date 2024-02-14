import torch

from scipy.spatial.distance import directed_hausdorff
from src.experiments.utils import convert_mask


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


def hausdorff_distance(original, predictions, model=False, threshold=0.6):
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

    # Process model predictions
    predictions_binary = convert_mask(predictions.squeeze(0), model=True)
    # Get the coordinates of the foreground pixels
    original_coords = torch.nonzero(original_binary).float()
    predictions_coords = torch.nonzero(predictions_binary).float()
    # Calculate Hausdorff Distance
    distance_original_to_pred = directed_hausdorff(original_coords.numpy(), predictions_coords.numpy())[0]
    distance_pred_to_original = directed_hausdorff(predictions_coords.numpy(), original_coords.numpy())[0]
    # Use the maximum distance as the final result
    distance = max(distance_original_to_pred, distance_pred_to_original)
    return distance