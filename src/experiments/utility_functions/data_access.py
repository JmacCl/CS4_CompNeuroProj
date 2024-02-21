import os

from torch.utils.data import DataLoader

from src.data.data_processors.brats_data import BraTS2020Data


def derive_loader(data_directory, purpose, mri_vols, transforms, batch):
    """
    Given a specific division, be it training, validation or testing
    (specified by option), load the batches of that dataset for the given
    epoch
    :param device: specified device, either cpu or gpu
    :param mri_vols: selected mri volumes for examination
    :param img_path: path to image data
    :param seg_path: path to segmentation data
    :return:
    """
    dataset = BraTS2020Data(directory=data_directory, mri_vols=mri_vols, transform=transforms,
                            sample_per_instance=128, purpose=purpose)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch)
    return loader

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