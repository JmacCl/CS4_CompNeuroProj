import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms


class BraTS2020Data(Dataset):

    def __init__(self, img_path, seg_path):
        # Load your data from the .pt files
        self.image_data = torch.load(img_path)
        self.segmentation_data = torch.load(seg_path)

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index):
        # Assuming your data is a list of pairs (image, segmentation)
        image = self.image_data[index]
        segmentation = self.segmentation_data[index]

        return {'image': image, 'segmentation': segmentation}