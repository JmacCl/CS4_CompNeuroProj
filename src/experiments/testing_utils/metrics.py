from src.experiments.training_utils.learning_metrics import hausdorff_distance, dice


def derive_metrics(metrics):
    return_metrics = {}

    for met in metrics:
        if met == "hausdorff":
            return_metrics[met] = lambda x, y: hausdorff_distance(x, y, distance="euclidean")
        elif met == "dice":
            return_metrics["dice"] = dice

    return return_metrics


def set_up_testing_scores(metric_dict: dict):
    return_dict = {}
    for keys in metric_dict.keys():
        return_dict[keys] = 0
    return return_dict