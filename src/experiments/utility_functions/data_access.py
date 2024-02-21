import os


def source_instances(img_path, seg_path):
    """
    Given the path specifications in the data_config, derive a list of all the
    possible epochs for the  training data or validation instances for validation
    :param seg_path:
    :param img_path:
    :return:
    """
    image_epochs = [file for file in sorted(os.listdir(img_path)) if file.endswith('.pt')]
    segmentation_epochs = [file for file in sorted(os.listdir(seg_path)) if file.endswith('.pt')]
    return image_epochs, segmentation_epochs