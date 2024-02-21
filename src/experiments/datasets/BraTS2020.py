import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms


class BraTS2020Data(Dataset):

    def __init__(self, img_path, seg_path, mri_vols: dict, device):
        # Load your data from the .pt files
        self.image_data = torch.load(img_path)
        self.segmentation_data = torch.load(seg_path)
        self.mri_vols = self.__select_mri_vols(mri_vols)
        self.device = device

    def __len__(self):
        return len(self.image_data)

    def __select_mri_vols(self, mri_vols: dict):
        """
        Given a dictionary of mri volume string keys and their respective positions in the data,
        process the data and return the requested mri volumes
        :param mri_vols:
        :return:
        """
        return_indices = []
        for keys in mri_vols.keys():
            return_indices.append(mri_vols[keys])
        return return_indices

    def __getitem__(self, index):
        # Assuming your data is a list of pairs (image, segmentation)

        indices = self.mri_vols
        image = self.image_data[index, indices, :, :]
        segmentation = self.segmentation_data[index]

        # Specify device
        if self.device:
            image.to(device="cuda" if torch.cuda.is_available() else "cpu")
            segmentation.to(device="cuda" if torch.cuda.is_available() else "cpu")

        return {'image': image, 'segmentation': segmentation}
