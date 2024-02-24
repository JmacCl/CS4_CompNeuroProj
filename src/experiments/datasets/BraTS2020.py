import torch
import math
import os

import random
from torch.utils.data import Dataset
from torchvision import transforms







class BraTS2020Data(Dataset):


    def __init__(self, directory, purpose, mri_vols: dict, sample_per_instance,
                 transform=None):
        """
        This dataset works by loading input and target directories of size N, such that each
        instance has B samples.
        :param directory: directory of the data that needs to be analyzed
        :param purpose: either the training, validation or testing
        :param mri_vols: the selected mri volumes to load and examine
        :param sample_per_instance: how many samples there are per training instance
        """
        # Load your data from the .pt files
        image_path, seg_path = self.__create_data_path(directory, purpose)
        image_instances, seg_instances = self.__source_instances(img_path=image_path,
                                                                 seg_path=seg_path)
        self.image_dir = image_path
        self.seg_path = seg_path
        self.images = sorted(image_instances, key=lambda x: int(x.split('_')[1].split(".")[0]))
        self.segmentations = sorted(seg_instances, key=lambda x: int(x.split('_')[1].split(".")[0]))
        self.mri_vols = self.__select_mri_vols(mri_vols)
        self.instances_per_dir = len(self.images)
        self.sample_per_instance = sample_per_instance
        if transform:
            self.transform = self.Augmentation(transform)
        else:
            self.transform = False

    def __len__(self):
        return self.sample_per_instance * self.instances_per_dir

    def __getitem__(self, index):
        """
        Given the index, derive its position from the possible data instances and batches.
        Then retreive and possibly augment it.
        :param index: data sample index
        :return: dictionary of image segmentation pair
        """
        # Assuming your data is a list of pairs (image, segmentation)
        DI_index = self.__derive_data_instance_index(index=index)
        sample_index = self.__derive_sample_index(index=index)

        image_instance = torch.load(os.path.join(self.image_dir, self.images[DI_index]))
        segmentation_instance = torch.load(os.path.join(self.seg_path, self.segmentations[DI_index]))

        volumes = self.mri_vols
        image = image_instance[sample_index, volumes, :, :]
        segmentation = segmentation_instance[sample_index]

        if self.transform:
            image, segmentation = self.transform([image, segmentation])

        return {'image': image, 'segmentation': segmentation}

    def __select_mri_vols(self, mri_vols: dict):
        """
        Given a dictionary of mri volume string keys and their respective positions in the data,
        process the data and return the requested mri volumes
        :param mri_vols:
        :return:
        """
        return_indices = []
        for keys in mri_vols:
            return_indices.append(keys)
        return return_indices

    def __derive_data_instance_index(self, index):
        """
        Given an index for a specific sample, determine what data instance in
        the main directory the sample index belongs to
        :return: Index for each data instance in directory
        """
        return math.ceil(index/self.sample_per_instance) - 1

    def __derive_sample_index(self, index):
        """
        Given a sample index, for the calculated data instance index, determine where
        in the instance it will be in
        :return: sample in possible samples
        """
        return index % self.sample_per_instance

    def __source_instances(self, img_path, seg_path):
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

    def __create_data_path(self, data_set_directory, purpose: str):
        """
        This function will create the path that can be used to access
        either the training or validation data required for the experiment depending
        on the purpose variable
        :param purpose: either variable for training, validation, or testing
        :param path: path to the directory for all processed data
        :param dataset_name: name of the dataset
        :return: completed data path
        """
        img_path = os.path.join(data_set_directory, "inputs", purpose)
        seg_path = os.path.join(data_set_directory, "targets", purpose)

        return img_path, seg_path

    class Augmentation(torch.nn.Module):
        def __init__(self, transforms):
            """
            Define the given augmentations to apply to a dataset pair
            :param transforms:
            """
            super().__init__()
            self.transforms = transforms

        def __call__(self, data):
            aug = random.choice(self.transforms.transforms[0])
            return [aug(d) for d in data]
