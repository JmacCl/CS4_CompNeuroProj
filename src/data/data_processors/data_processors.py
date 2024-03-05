import glob
import os.path
import pathlib
import numpy as np
import torch
import nibabel as nib
import random

from math import floor
from configparser import ConfigParser
from typing import List, Tuple, Dict
from pathlib import Path

import yaml

from src.data.data_processors.brats_data import BraTS2020Data
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from matplotlib import pyplot as plt
from scipy.stats import zscore

# This is the built-in path format for the raw_files
TRAINING_PATH = r"BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"


"""
Segmentation data is annotated as follows
    - 0: Unlabelled volume
    - 1: Necrotic Core and Non enhacning tumor core
    - 2: Peritumoral Edema
    - 3: NA
    - 4: GD Enhacning Tumor
"""

def map_and_plot(x, batch):
    # Ensure the tensor is in the range [0, 1] for proper visualization

    if batch == 5:

        # Select batch index and channel for visualization
        batch_index = 0
        channel_index = 0
        # Extract the image from the tensor
        image_x = x
        print(image_x.shape)

        # Display the image using matplotlib
        plt.imshow(image_x, cmap='gray')  # Choose a colormap if needed
        plt.axis('off')  # Turn off axis labels
        plt.show()


class BratsDataProcessor:

    def __init__(self, config: dict):
        # Set values
        self.raw_source = Path(config["raw_data_path"])
        self.processed_images_home = Path(config["processed_data_path"])
        self.classes = config["classes"]
        self.selected_mri_volumes: List[str] = config["modals"]
        self.w_bindings: List[int, int] = config["bindings"]["width"]
        self.h_bindings: List[int, int] = config["bindings"]["height"]
        self.scans_bindings: List[int, int] = config["bindings"]["pictures"]
        self.data_name: str = config["data_name"]
        self.data_size = self.__determine_data_size()
        self.data_split = self.__determine_split(config["data_split"])

        # Start process of creating data
        self.__process_data()

        # Save configuration file format
        self.__save_config(config)

# TODO: Add more info to the save configuration file to specify how many files there are in each folder



    def __save_config(self, config):
        """
        Given the dictionary that holds all the configurations for how to process the
        data, save it with the processed data
        :param config: Dictionary representing data creation configuration
        :return: create path and dump config and return nothing
        """

        # Get name of experiment
        save_path = os.path.join(self.processed_images_home, self.data_name)

        yaml_name = "data_config.yaml"

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        with open(os.path.join(save_path, yaml_name), "rb") as file:
            yaml.dump(config, file)

    # def __create_necessary_folders(self):
    #     """
    #     Create necessary folders to hold processed data
    #     :return:
    #     """
    #     input_path = os.path.join(self.processed_data, self.data_name, "inputs")
    #     target_path = os.path.join(self.target_masks, self.data_name, "targets")
    #
    #     if not os.path.exists(input_path):
    #         os.makedirs(input_path)
    #     if not os.path.exists(target_path):
    #         os.makedirs(target_path)

    def __determine_data_size(self):
        raw_data_path = os.path.join(self.raw_source, "BraTS2020_TrainingData", "MICCAI_BraTS2020_TrainingData")
        length = len([entry for entry in os.listdir(raw_data_path) if os.path.isdir(os.path.join(raw_data_path, entry))])
        return length

    def __determine_split(self, splits):
        data_size = self.data_size
        return_dict = {}

        train_index = floor(data_size * splits["train"])
        return_dict["training"] = train_index
        val_index = floor(data_size * splits["validation"]) + train_index
        return_dict["validation"] = val_index
        return_dict["testing"] = data_size

        return return_dict

    # def __process_augmentations(self, augmentations: List):
    #     """
    #     Process the dictionaries of augmentation specifications and return
    #     as a list of dictionaries of the different data augmentation functions
    #     :param augmentations:
    #     :return:
    #     """
    #     return_dict = {}
    #     for augs in augmentations:
    #         for key, vals in augs.items():
    #             if key == "flipping":
    #                 for techniques in vals:
    #                     if techniques == "horizontal":
    #                         return_dict["horizontal_flip"] = v2.RandomHorizontalFlip(1)
    #                     elif techniques == "vertical":
    #                         return_dict["vertical_flip"] = v2.RandomVerticalFlip(1)
    #             elif key == "rotations":
    #                 for rot in vals:
    #                     return_dict[key + str(rot)] = v2.RandomRotation(degrees=rot)
    #             elif key == "special":
    #                 for techniques in vals:
    #                     if techniques == "mix_up":
    #                         return_dict["mix_up"] = v2.MixUp(num_classes=self.classes)
    #     return return_dict
    def __get_raw_data_files(self) -> Tuple[Dict, List]:
        """
        This function will get the raw input and segmented data from the
        specified directories and retrieve their filename values
        :return: A tuple with a dictionary of each volume with the segmented values
        """
        volume_holder = {}
        # For each directory, get input data
        for vols in self.selected_mri_volumes:
            vol_type = "*" + vols + ".nii"
            raw = self.raw_source
            vol_path = os.path.join(raw, TRAINING_PATH, "*", vol_type)
            vol_list = sorted(glob.glob(vol_path))
            volume_holder[vols] = vol_list

        # Get labelled raw data
        seg = "*" + "seg" + ".nii"
        seg_path = os.path.join(self.raw_source, TRAINING_PATH, "*", seg)
        seg_list = sorted(glob.glob(seg_path))

        return volume_holder, seg_list

    def __collect_data(self, i, volume_holder, seg_list):
        """

        This function will use the configuration to obtain and process each of data instances
        into a Dataloader torch object. The data will be shuffled in order to keep the
        data collection process random

        :param i: The given epoch index to process
        :param volume_holder: Data path of each specified volume
        :param seg_list: Data path of each specified segmentation
        :return: Dataloader object
        """
      # Iterate over all data files

        input_raw_data = []
        # For each defined volume type, add it to the list and process
        for vol in volume_holder.keys():
            data_path = Path(volume_holder[vol][i])
            vol_data = self.__brain_data_preprocessing(data_path)
            input_raw_data.append(vol_data)
        final_input = torch.stack(input_raw_data, dim=1)

        # Process segmented_data
        seg_data_path = Path(seg_list[i])
        target, (vals, counts) = self.__determine_seg(seg_data_path)

        return final_input, target

    def __process_data(self):
        """
        Given the configuration, this function will do the following:
            1. Get the processed data from
        :return:
        """
        volume_holder, seg_list = self.__get_raw_data_files()
        # For each epoch, extract the given information and save them
        # Iterate over all data files
        indices = list(range(len(list(volume_holder.values())[0])))
        random.shuffle(indices)
        count = 1
        for i in indices:
            image, seg = self.__collect_data(i, volume_holder, seg_list)
            self.__save_instances(image, seg,  index=count)
            # for keys, augs in self.data_augmentation.items():
            #     if keys == "mix_up":
            #         dl_img, dl_seg = self.__augment_by_loader(image, seg, function=augs)
            #         self.__save_instances(dl_img, dl_seg, keys, index=count)
            #     else:
            #         self.__save_instances(augs(image), augs(seg), keys, index=count)
            count += 1

    def __determine_split_name(self, count):
        data_split = self.data_split
        if count < data_split["training"]:
            return "training"
        elif data_split["training"] < count and count < data_split["validation"]:
            return "validation"
        else:
            return "testing"

    # def __augment_by_loader(self, image, seg, function):
    #     dataset = BraTS2020Data(inputs=image, segmentations=seg)
    #     loader = DataLoader(dataset, batch_size=len(image), collate_fn=function, shuffle=True)
    #     images = []
    #     segs = []
    #     for idx, batch in enumerate(loader):
    #         for input, target in batch:
    #             images.append(input)
    #             segs.append(target)
    #
    #     return torch.stack(images, dim=0), torch.stack(segs, dim=0)


    def __save_instances(self, image, seg, index):
        """
        Given a Dataloader that represents a data epoch, and save the files to the defined file
        :param loader:
        :return:
        """
        current_split = self.__determine_split_name(index)

        # image data
        input_path = os.path.join(self.processed_images_home, self.data_name,
                                  "inputs", current_split)
        if not os.path.exists(input_path):
            os.makedirs(input_path)
        torch.save(image, os.path.join(input_path, "instance_" + str(index) + ".pt" ))

        # Targets
        target_path = os.path.join(self.processed_images_home, self.data_name,
                                   "targets", current_split)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        torch.save(seg, os.path.join(target_path, "instance_" + str(index) + ".pt" ))


    def __brain_data_preprocessing(self, input_path: Path, clip: List[int] = None):
        """
        This function will do a number of things for BraTS 2020 data in order to
        process it for deep learning algorithms.
        1. Normalize data by getting z-scores of data
        2. Clip the images to a certain range to remove outlier, default is [-5, 5]
        3. Rescale to data with range of [0, 1], setting non brain regions to 0
        :param clip:
        :param data: a ndarray of brain data
        :return: a processed nd array of brain data
        """
        data: np.ndarray = nib.load(input_path).get_fdata()
        b = 50
        # map_and_plot(data[:, :, b], 5)

        if clip is None:
            clip = [-5, 5]
        pass

        # First lets calculate z-score scores of axis
        norm_data = zscore(data, axis=None, nan_policy="omit")
        # map_and_plot(norm_data[:, :, b], 5)

        pnorm_data = np.nan_to_num(norm_data, 0)

        # map_and_plot(pnorm_data[:, :, b], 5)


        # Second resize values between clip
        clipped_data = np.clip(pnorm_data, clip[0], clip[1])
        # map_and_plot(clipped_data[:, :, b], 5)

        min_clip = np.min(clipped_data)
        max_clip = np.max(clipped_data)

        # Third and finally, rescale image to 0, 1 and set all non brain voxels to 0
        denominator = max_clip - min_clip

        final_data = (clipped_data - np.min(clipped_data)) / denominator
        # map_and_plot(final_data[:, :, b], 5)

        # map_and_plot(final_data[:, :, b], 5)


        # Bind the data in the given dimensions
        final_data = self.__apply_bindings(final_data)
        # map_and_plot(final_data[:, :, 50], 5)

        return torch.tensor(np.transpose(final_data, (2, 0, 1)), dtype=torch.float32)

    def __determine_seg(self, seg_data):
        """
        Given config input, determine what segmentation procedure to do
        :return:
        """
        classes = self.classes
        if classes == 2:
            return self.__seg_two_split(seg_data)
        elif classes == 4:
            return self.__seg_four_split(seg_data)


    def __seg_four_split(self, seg_path):
        """
        Process a given segmented file in the standard way, outlining each component of
        the tumor
        :param seg_path:
        :return:
        """

        seg_raw_data: np.ndarray = nib.load(seg_path).get_fdata()
        seg_raw_data = seg_raw_data.astype(np.uint8)

        # Reassign values for order to make more sense
        seg_raw_data[seg_raw_data == 4] = 3

        # Apply bindings
        zoomed_seg = self.__apply_bindings(seg_raw_data)

        # Retrieve useful values and counts for later processing
        val, counts = np.unique(zoomed_seg, return_counts=True)

        # Convert to tensor and outline labels
        torch_seg = torch.tensor(zoomed_seg, dtype=torch.int64).permute(2, 0, 1)
        final_data = torch.nn.functional.one_hot(torch_seg, num_classes=4).permute(0, 3, 1, 2)
        return final_data, (val, counts)

    def __seg_two_split(self, seg_path):
        seg_raw_data: np.ndarray = nib.load(seg_path).get_fdata()
        seg_raw_data = seg_raw_data.astype(np.uint8)

        # Reassign values for order to make more sense
        seg_raw_data[seg_raw_data == 2] = 1
        seg_raw_data[seg_raw_data == 4] = 1

        # Apply bindings
        zoomed_seg = self.__apply_bindings(seg_raw_data)

        # Retrieve useful values and counts for later processing
        val, counts = np.unique(zoomed_seg, return_counts=True)

        # Convert to tensor and outline labels
        torch_seg = torch.tensor(zoomed_seg, dtype=torch.int64).permute(2, 0, 1)
        final_data = torch.nn.functional.one_hot(torch_seg, num_classes=self.classes).permute(0, 3, 1, 2)
        return final_data, (val, counts)


    def __apply_bindings(self, data):
        """
        This function will apply the defined bindings as specified in the ini file,
        :param data: either a input data or target instance
        :return:
        """
        process_data = data[self.h_bindings[0]:self.h_bindings[1],
                     self.w_bindings[0]: self.w_bindings[1],
                     self.scans_bindings[0]:self.scans_bindings[1]]
        return process_data



    def __load_data(self):
        """
        This function will retreive the given data
        :return:
        """
        input_data_loc = Path(os.path.join(self.processed_images_home, self.data_name, "inputs"))
        label_data_loc = Path(os.path.join(self.processed_images_home, self.data_name, "targets"))

        input_data_files = [file for file in os.listdir(input_data_loc) if file.endswith(".pt")]
        label_data_files = [file for file in os.listdir(label_data_loc) if file.endswith(".pt")]

        input_data = [torch.load(os.path.join(input_data_loc, file)) for file in input_data_files]
        label_data = [torch.load(os.path.join(label_data_loc, file)) for file in label_data_files]

        return input_data, label_data

    def get_input_data(self):
        return self.input

    def get_label_data(self):
        return self.labels



def get_int_list(config: ConfigParser, section, value):
    result: str = config.get(section, value)
    return [int(num) for num in result.split(",")]


def get_str_list(config: ConfigParser, section, value):
    result: str = config.get(section, value)
    return [strings for strings in result.split(",")]
